#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import argparse
import copy
import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "1.0"


SEARCH_RUNTIME_SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "search_space_version": "2026-04-29",
    "compile_time_dimensions": [
        "dtype_a",
        "dtype_b",
        "dtype_c",
        "dtype_acc",
        "layout",
        "tile_m",
        "tile_n",
        "tile_k",
        "sg_m",
        "sg_n",
        "stages",
        "split_k",
        "grf_mode",
        "ilp_class",
        "instantiation_level",
        "runner",
        "benchmark_target",
    ],
    "runtime_dimensions": [
        "shape_id",
        "m",
        "n",
        "k",
        "raster_order",
        "swizzle_size",
        "barrier_interval",
        "k_unroll",
    ],
    "pruning_inputs": [
        "dpas_alignment",
        "safe_search_constraints",
        "phase_a_probe_results",
        "slm_limit_kb",
        "split_k_support",
    ],
    "microbench_guided_defaults": {
        "grf_mode": 256,
        "barrier_interval": 8,
        "k_unroll": 1,
        "raster_order": "heuristic",
        "swizzle_size": 1,
    },
}


SEED_KERNELS = {
    "bf16": [
        {
            "kernel_name": "BmgGemmBF16BF16FP32_RCR_5",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 128,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmBF16BF16FP32_RCR_7",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 128,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 8,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmBF16BF16FP32_RCR_9",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 64,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmBF16BF16FP32_RCR_16",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 16,
            "tile_n": 64,
            "tile_k": 32,
            "sg_m": 2,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmBF16BF16FP32_RCR_6",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 256,
            "tile_n": 256,
            "tile_k": 32,
            "sg_m": 8,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "03_bmg_gemm_streamk_splitk_bf16",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 64,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 4,
            "stages": 2,
            "split_k": 2,
            "runner": "streamk_example",
        },
    ],
    "f16": [
        {
            "kernel_name": "BmgGemmFP16FP16FP32_RCR_5",
            "layout": "rcr",
            "dtype_a": "f16",
            "dtype_b": "f16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 128,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmFP16FP16FP32_RCR_7",
            "layout": "rcr",
            "dtype_a": "f16",
            "dtype_b": "f16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 128,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 8,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmFP16FP16FP32_RCR_9",
            "layout": "rcr",
            "dtype_a": "f16",
            "dtype_b": "f16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 64,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmFP16FP16FP32_RCR_16",
            "layout": "rcr",
            "dtype_a": "f16",
            "dtype_b": "f16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 16,
            "tile_n": 64,
            "tile_k": 32,
            "sg_m": 2,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "BmgGemmFP16FP16FP32_RCR_6",
            "layout": "rcr",
            "dtype_a": "f16",
            "dtype_b": "f16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 256,
            "tile_n": 256,
            "tile_k": 32,
            "sg_m": 8,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
        },
        {
            "kernel_name": "03_bmg_gemm_streamk_splitk_f16",
            "layout": "rcr",
            "dtype_a": "f16",
            "dtype_b": "f16",
            "dtype_c": "f32",
            "dtype_acc": "f32",
            "tile_m": 8,
            "tile_n": 64,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 4,
            "stages": 2,
            "split_k": 2,
            "runner": "streamk_example",
        },
    ],
}


CSV_FIELDS = [
    "run_id",
    "stage",
    "attempt_index",
    "shape_id",
    "candidate_id",
    "compiler_profile_id",
    "status",
    "verify_status",
    "layout",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_acc",
    "m",
    "n",
    "k",
    "split_k",
    "avg_runtime_ms",
    "best_runtime_ms",
    "worst_runtime_ms",
    "avg_tflops",
    "avg_throughput",
    "max_error",
    "close_call_group",
    "failure_reason",
    "stdout_log",
]


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def default_constraints():
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "constraint_source": "default_bmg",
        "device_arch": "bmg",
        "limits": {
            "max_slm_kb": 128,
            "subgroup_size": 16,
            "max_split_k": 2,
            "max_stages": 3,
        },
        "allowed_values": {
            "tile_m": [8, 16, 32, 64, 128, 256],
            "tile_n": [64, 128, 256],
            "tile_k": [32, 64],
            "sg_m": [1, 2, 4, 8],
            "sg_n": [4, 8],
            "stages": [1, 2, 3],
            "split_k": [1, 2],
            "grf_mode": [256],
        },
        "blocked_rules": [],
    }


def resolve_executable(path, cwd=None):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    if cwd:
        joined = Path(cwd) / candidate
        if joined.exists():
            return joined.resolve()
    found = shutil.which(path)
    return Path(found) if found else None


def collect_environment_metadata(shell_init, benchmark_exe, streamk_example_exe, cwd=None):
    tracked_env = {}
    for name in (
        "ONEAPI_DEVICE_SELECTOR",
        "SYCL_PROGRAM_COMPILE_OPTIONS",
        "IGC_ExtraOCLOptions",
        "IGC_VectorAliasBBThreshold",
        "IGC_VISAOptions",
    ):
        value = os.environ.get(name)
        if value:
            tracked_env[name] = value
    benchmark_path = resolve_executable(benchmark_exe, cwd=cwd)
    streamk_path = resolve_executable(streamk_example_exe, cwd=cwd)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "hostname": socket.gethostname(),
        "node_id": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "proxy_bootstrap_method": shell_init or "inherited-environment",
        "executables": {
            "benchmark_exe": str(benchmark_path) if benchmark_path else benchmark_exe,
            "benchmark_available": bool(benchmark_path),
            "streamk_example_exe": str(streamk_path) if streamk_path else streamk_example_exe,
            "streamk_example_available": bool(streamk_path),
        },
        "effective_env": tracked_env,
    }


def apply_static_probe_constraints(base_constraints, env_caps):
    constraints = copy.deepcopy(base_constraints)
    constraints["constraint_source"] = "phase_a_static_probe"
    if not env_caps["executables"]["streamk_example_available"]:
        constraints["limits"]["max_split_k"] = 1
        constraints["allowed_values"]["split_k"] = [1]
    return constraints


def default_compiler_profiles():
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "profiles": [
            {
                "compiler_profile_id": "bmg.small_tile.default",
                "candidate_class": "small_tile",
                "description": "Default BMG profile for small tiles.",
                "selector": {"tile_m_max": 16, "sg_count_max": 8},
                "env": {
                    "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                    "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                    "IGC_VectorAliasBBThreshold": "100000000000",
                    "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
                },
                "cmake_flags": ["-DCMAKE_BUILD_TYPE=Release"],
            },
            {
                "compiler_profile_id": "bmg.medium_tile.default",
                "candidate_class": "medium_tile",
                "description": "Default BMG profile for medium tiles.",
                "selector": {"tile_m_min": 32, "tile_m_max": 64, "sg_count_max": 16},
                "env": {
                    "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                    "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                    "IGC_VectorAliasBBThreshold": "100000000000",
                    "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
                },
                "cmake_flags": ["-DCMAKE_BUILD_TYPE=Release"],
            },
            {
                "compiler_profile_id": "bmg.large_tile.default",
                "candidate_class": "large_tile",
                "description": "Default BMG profile for large tiles.",
                "selector": {"tile_m_min": 128, "sg_count_min": 16},
                "env": {
                    "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                    "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                    "IGC_VectorAliasBBThreshold": "100000000000",
                    "IGC_VISAOptions": "-perfmodel",
                    "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only",
                },
                "cmake_flags": ["-DCMAKE_BUILD_TYPE=Release"],
            },
        ],
    }


def default_shapes(dtype):
    base = [
        {"m": 1, "n": 4096, "k": 14336, "tags": ["decode"]},
        {"m": 8, "n": 4096, "k": 4096, "tags": ["decode"]},
        {"m": 64, "n": 4096, "k": 4096, "tags": ["prefill"]},
        {"m": 256, "n": 4096, "k": 8192, "tags": ["prefill"]},
    ]
    shapes = []
    for item in base:
        shapes.append(
            {
                "shape_id": f"rcr_{dtype}_{item['m']}_{item['n']}_{item['k']}",
                "layout": "rcr",
                "dtype_a": dtype,
                "dtype_b": dtype,
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": item["m"],
                "n": item["n"],
                "k": item["k"],
                "runtime_defaults": {
                    "raster_order": "heuristic",
                    "swizzle_size": 1,
                    "barrier_interval": 8 if item["m"] >= 64 else 0,
                    "k_unroll": 1,
                },
                "tags": item["tags"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "shape_set_id": f"default_{dtype}_decode_prefill",
        "source": "predefined",
        "shapes": shapes,
    }


def dry_run_shapes(dtype):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "shape_set_id": f"dry_run_{dtype}",
        "source": "dry_run",
        "shapes": [
            {
                "shape_id": f"dry_run_rcr_{dtype}_1_64_32",
                "layout": "rcr",
                "dtype_a": dtype,
                "dtype_b": dtype,
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 64,
                "k": 32,
                "runtime_defaults": {
                    "raster_order": "heuristic",
                    "swizzle_size": 1,
                    "barrier_interval": 0,
                    "k_unroll": 1,
                },
                "tags": ["dry_run"],
            }
        ],
    }


def candidate_class(tile_m, sg_count):
    if tile_m <= 16 and sg_count <= 8:
        return "small_tile"
    if tile_m >= 128 or sg_count >= 16:
        return "large_tile"
    return "medium_tile"


def select_compiler_profile_id(profiles, tile_m, sg_count):
    chosen = None
    for profile in profiles["profiles"]:
        selector = profile.get("selector", {})
        if "tile_m_min" in selector and tile_m < selector["tile_m_min"]:
            continue
        if "tile_m_max" in selector and tile_m > selector["tile_m_max"]:
            continue
        if "sg_count_min" in selector and sg_count < selector["sg_count_min"]:
            continue
        if "sg_count_max" in selector and sg_count > selector["sg_count_max"]:
            continue
        chosen = profile["compiler_profile_id"]
        break
    if chosen:
        return chosen
    return profiles["profiles"][0]["compiler_profile_id"]


def candidate_id_for(seed):
    return (
        f"{seed['layout']}_{seed['dtype_a']}{seed['dtype_b']}{seed['dtype_c']}"
        f"_tm{seed['tile_m']}_tn{seed['tile_n']}_tk{seed['tile_k']}"
        f"_sg{seed['sg_m']}x{seed['sg_n']}_st{seed['stages']}_sk{seed['split_k']}"
    )


def ilp_class(seed):
    ilp = (seed["tile_m"] // max(seed["sg_m"], 1) // 8) * (seed["tile_n"] // max(seed["sg_n"], 1) // 16)
    if ilp >= 16:
        return "ilp16"
    if ilp >= 8:
        return "ilp8"
    return "ilp4"


def kernel_catalog_entry(dtype, seed):
    entry = copy.deepcopy(seed)
    entry.setdefault("runner", "benchmark")
    entry["kernel_id"] = seed["kernel_name"]
    entry["instantiation_level"] = 0
    entry["benchmark_target"] = "cutlass_benchmarks_gemm_sycl" if entry["runner"] == "benchmark" else "03_bmg_gemm_streamk"
    entry["grf_mode"] = 256
    entry["ilp_class"] = ilp_class(entry)
    entry["runtime_defaults"] = {
        "raster_order": "heuristic",
        "swizzle_size": 1,
        "barrier_interval": 8 if entry["sg_m"] * entry["sg_n"] >= 4 else 0,
        "k_unroll": 1,
    }
    entry["allowed_runtime_sweeps"] = ["shape_id", "m", "n", "k", "raster_order", "swizzle_size", "barrier_interval", "k_unroll"]
    entry["source"] = "seed_catalog_level0"
    entry["dtype_family"] = dtype
    return entry


def build_kernel_catalog(dtypes=None, allowed_runners=("benchmark",)):
    selected_dtypes = dtypes if dtypes is not None else sorted(SEED_KERNELS.keys())
    catalog = []
    for dtype in selected_dtypes:
        for seed in SEED_KERNELS.get(dtype, []):
            entry = kernel_catalog_entry(dtype, seed)
            if entry["runner"] not in allowed_runners:
                continue
            catalog.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "catalog_version": "level0-seed-catalog",
        "instantiation_levels": {
            "0": "existing validated benchmark-backed kernels",
            "1": "expanded tile and subgroup layouts",
            "2": "full autotuning catalog including copy/epilogue variants",
        },
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "kernels": catalog,
    }


def blocked(seed, constraints):
    if seed["sg_m"] * seed["sg_n"] > 32:
        return True
    allowed = constraints["allowed_values"]
    for key in ("tile_m", "tile_n", "tile_k", "sg_m", "sg_n", "stages", "split_k", "grf_mode"):
        if seed[key] not in allowed[key]:
            return True
    for rule in constraints.get("blocked_rules", []):
        match = rule.get("match", {})
        if all(seed.get(name) == value for name, value in match.items()):
            return True
    return False


def generate_candidate_space(shapes_doc, constraints, profiles, allowed_runners=("benchmark",)):
    seen = set()
    candidates = []
    dtypes = sorted({shape["dtype_a"] for shape in shapes_doc["shapes"]})
    kernel_catalog = build_kernel_catalog(dtypes=dtypes, allowed_runners=allowed_runners)
    for seed in kernel_catalog["kernels"]:
        if blocked(seed, constraints):
            continue
        ident = candidate_id_for(seed)
        if ident in seen:
            continue
        seen.add(ident)
        sg_count = seed["sg_m"] * seed["sg_n"]
        klass = candidate_class(seed["tile_m"], sg_count)
        candidates.append(
            {
                "candidate_id": ident,
                "kernel_name": seed["kernel_name"],
                "kernel_id": seed["kernel_id"],
                "layout": seed["layout"],
                "dtype_a": seed["dtype_a"],
                "dtype_b": seed["dtype_b"],
                "dtype_c": seed["dtype_c"],
                "dtype_acc": seed["dtype_acc"],
                "tile_m": seed["tile_m"],
                "tile_n": seed["tile_n"],
                "tile_k": seed["tile_k"],
                "sg_m": seed["sg_m"],
                "sg_n": seed["sg_n"],
                "stages": seed["stages"],
                "split_k": seed["split_k"],
                "runner": seed.get("runner", "benchmark"),
                "benchmark_target": seed["benchmark_target"],
                "grf_mode": seed["grf_mode"],
                "ilp_class": seed["ilp_class"],
                "instantiation_level": seed["instantiation_level"],
                "runtime_defaults": seed["runtime_defaults"],
                "allowed_runtime_sweeps": seed["allowed_runtime_sweeps"],
                "candidate_class": klass,
                "compiler_profile_id": select_compiler_profile_id(profiles, seed["tile_m"], sg_count),
                "filters_applied": ["kernel_catalog", constraints["constraint_source"]],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "device_arch": constraints["device_arch"],
        "constraint_source": constraints["constraint_source"],
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "kernel_catalog": {
            "catalog_version": kernel_catalog["catalog_version"],
            "kernel_count": len(kernel_catalog["kernels"]),
        },
        "candidates": candidates,
    }


def build_candidate_build_manifest(candidate_space):
    variants = []
    for candidate in candidate_space["candidates"]:
        variants.append(
            {
                "candidate_id": candidate["candidate_id"],
                "kernel_id": candidate["kernel_id"],
                "benchmark_target": candidate["benchmark_target"],
                "runner": candidate["runner"],
                "compile_time_variant": {
                    "layout": candidate["layout"],
                    "dtype_a": candidate["dtype_a"],
                    "dtype_b": candidate["dtype_b"],
                    "dtype_c": candidate["dtype_c"],
                    "dtype_acc": candidate["dtype_acc"],
                    "tile_m": candidate["tile_m"],
                    "tile_n": candidate["tile_n"],
                    "tile_k": candidate["tile_k"],
                    "sg_m": candidate["sg_m"],
                    "sg_n": candidate["sg_n"],
                    "stages": candidate["stages"],
                    "split_k": candidate["split_k"],
                    "grf_mode": candidate["grf_mode"],
                    "ilp_class": candidate["ilp_class"],
                    "instantiation_level": candidate["instantiation_level"],
                },
                "runtime_sweep": {
                    "allowed_fields": candidate["allowed_runtime_sweeps"],
                    "defaults": candidate["runtime_defaults"],
                },
                "compiler_profile_id": candidate["compiler_profile_id"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "device_arch": candidate_space["device_arch"],
        "constraint_source": candidate_space["constraint_source"],
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "variants": variants,
    }


def choose_candidates_for_shape(shape, candidates):
    matched = []
    for candidate in candidates:
        if candidate["layout"] != shape["layout"]:
            continue
        if candidate["dtype_a"] != shape["dtype_a"] or candidate["dtype_b"] != shape["dtype_b"]:
            continue
        if candidate["split_k"] > 1 and shape["n"] < 16384 and shape["k"] < 8192:
            continue
        if shape["m"] <= 8 and candidate["tile_m"] <= 16:
            matched.append(candidate)
        elif 16 < shape["m"] <= 128 and candidate["tile_m"] <= 64:
            matched.append(candidate)
        elif shape["m"] > 128 and candidate["tile_m"] >= 16:
            matched.append(candidate)
    return matched or [
        candidate
        for candidate in candidates
        if candidate["layout"] == shape["layout"] and candidate["dtype_a"] == shape["dtype_a"]
    ]


def build_phase_a_probe_entries(shapes_doc, candidate_space):
    shape_map = {shape["shape_id"]: shape for shape in shapes_doc["shapes"]}
    candidates = candidate_space["candidates"]
    non_splitk = [candidate for candidate in candidates if candidate["split_k"] == 1]
    splitk = [candidate for candidate in candidates if candidate["split_k"] > 1]
    selected = []
    if non_splitk:
        selected.append(("small", min(non_splitk, key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"])), "rcr_" + non_splitk[0]["dtype_a"] + "_8_4096_4096"))
        medium = [candidate for candidate in non_splitk if 16 <= candidate["tile_m"] <= 64]
        if medium:
            selected.append(("medium", min(medium, key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"])), "rcr_" + medium[0]["dtype_a"] + "_64_4096_4096"))
        selected.append(("large", max(non_splitk, key=lambda item: (item["tile_m"], item["tile_n"])), "rcr_" + non_splitk[0]["dtype_a"] + "_256_4096_8192"))
    if splitk:
        selected.append(("splitk", splitk[0], "rcr_" + splitk[0]["dtype_a"] + "_1_4096_14336"))

    entries = []
    seen = set()
    for probe_class, candidate, shape_id in selected:
        if shape_id not in shape_map:
            continue
        key = (candidate["candidate_id"], shape_id)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "bm_name": f"{candidate['candidate_id']}__{shape_id}__probe__0",
                "stage": "probe",
                "attempt_index": 0,
                "probe_class": probe_class,
                "shape": shape_map[shape_id],
                "candidate": candidate,
            }
        )
    return entries


def blocked_rule_for_row(row):
    return {
        "rule_id": f"probe.blocked.{row['candidate_id']}",
        "match": {
            "tile_m": int(re.search(r"_tm(\d+)_", row["candidate_id"]).group(1)),
            "tile_n": int(re.search(r"_tn(\d+)_", row["candidate_id"]).group(1)),
            "tile_k": int(re.search(r"_tk(\d+)_", row["candidate_id"]).group(1)),
            "sg_m": int(re.search(r"_sg(\d+)x", row["candidate_id"]).group(1)),
            "sg_n": int(re.search(r"x(\d+)_st", row["candidate_id"]).group(1)),
            "split_k": int(row["split_k"]),
        },
    }


def apply_run_probe_constraints(static_constraints, probe_rows):
    constraints = copy.deepcopy(static_constraints)
    constraints["constraint_source"] = "phase_a_run_probe"
    if not any(row["status"] == "pass" and int(row["split_k"]) > 1 for row in probe_rows):
        constraints["limits"]["max_split_k"] = 1
        constraints["allowed_values"]["split_k"] = [1]
    failures = [row for row in probe_rows if row["status"] != "pass"]
    existing_ids = {rule.get("rule_id") for rule in constraints.get("blocked_rules", [])}
    for row in failures:
        rule = blocked_rule_for_row(row)
        if rule["rule_id"] not in existing_ids:
            constraints["blocked_rules"].append(rule)
            existing_ids.add(rule["rule_id"])
    return constraints


def build_screening_entries(shapes_doc, candidate_space):
    entries = []
    for shape in shapes_doc["shapes"]:
        for candidate in choose_candidates_for_shape(shape, candidate_space["candidates"]):
            bm_name = f"{candidate['candidate_id']}__{shape['shape_id']}__screening__0"
            entries.append(
                {
                    "bm_name": bm_name,
                    "stage": "screening",
                    "attempt_index": 0,
                    "shape": shape,
                    "candidate": candidate,
                }
            )
    return entries


def generate_confirmation_entries(rows, candidate_space, shapes_doc, top_k, confirm_runs):
    shape_map = {shape["shape_id"]: shape for shape in shapes_doc["shapes"]}
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidate_space["candidates"]}
    grouped = defaultdict(list)
    for row in rows:
        if row["stage"] == "screening" and row["status"] == "pass":
            grouped[row["shape_id"]].append(row)
    entries = []
    for shape_id, shape_rows in grouped.items():
        ranked = sorted(shape_rows, key=lambda row: float(row["avg_tflops"] or 0.0), reverse=True)[:top_k]
        for attempt_index in range(confirm_runs):
            for row in ranked:
                candidate_id = row["candidate_id"]
                candidate = candidate_map[candidate_id]
                shape = shape_map[shape_id]
                bm_name = f"{candidate_id}__{shape_id}__confirm__{attempt_index}"
                entries.append(
                    {
                        "bm_name": bm_name,
                        "stage": "confirm",
                        "attempt_index": attempt_index,
                        "shape": shape,
                        "candidate": candidate,
                    }
                )
    return entries


def write_config(entries, config_path):
    metadata = {}
    with open(config_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            candidate = entry["candidate"]
            shape = entry["shape"]
            line = (
                f"{candidate['kernel_name']} "
                f"--bm_name={entry['bm_name']} "
                f"--m={shape['m']} --n={shape['n']} --k={shape['k']}"
            )
            handle.write(f"{line}\n")
            metadata[entry["bm_name"]] = {
                "shape_id": shape["shape_id"],
                "candidate_id": candidate["candidate_id"],
                "compiler_profile_id": candidate["compiler_profile_id"],
                "stage": entry["stage"],
                "attempt_index": entry["attempt_index"],
                "layout": shape["layout"],
                "dtype_a": shape["dtype_a"],
                "dtype_b": shape["dtype_b"],
                "dtype_c": shape["dtype_c"],
                "dtype_acc": shape["dtype_acc"],
                "m": shape["m"],
                "n": shape["n"],
                "k": shape["k"],
                "kernel_name": candidate["kernel_name"],
                "split_k": candidate["split_k"],
                "runner": candidate.get("runner", "benchmark"),
            }
    return metadata


def shell_join(command):
    return " ".join(shlex.quote(part) for part in command)


def run_benchmark(command, log_path, cwd=None, shell_init=None):
    if shell_init:
        payload = f"{shell_init} && {shell_join(command)}"
        process = subprocess.run(
            ["bash", "-lc", payload],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    else:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(process.stdout)
    return process


def parse_metric(line, key):
    match = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
    return match.group(1) if match else ""


def parse_streamk_example_log(log_path, metadata_by_bm_name, run_id):
    rows = []
    bm_name = next(iter(metadata_by_bm_name))
    metadata = metadata_by_bm_name[bm_name]
    text = Path(log_path).read_text(encoding="utf-8")
    status = "pass" if "Disposition: Passed" in text else "fail"
    verify_status = status
    failure_reason = "" if status == "pass" else text.strip().splitlines()[-1] if text.strip() else "missing output"
    perf_match = re.search(r"Cutlass GEMM Performance:\s+\[([0-9.]+)\]TFlop/s\s+\(([0-9.]+)\)ms", text)
    avg_tflops = perf_match.group(1) if perf_match else ""
    avg_runtime_ms = perf_match.group(2) if perf_match else ""
    rows.append(
        {
            "run_id": run_id,
            "stage": metadata["stage"],
            "attempt_index": metadata["attempt_index"],
            "shape_id": metadata["shape_id"],
            "candidate_id": metadata["candidate_id"],
            "compiler_profile_id": metadata["compiler_profile_id"],
            "status": status,
            "verify_status": verify_status,
            "layout": metadata["layout"],
            "dtype_a": metadata["dtype_a"],
            "dtype_b": metadata["dtype_b"],
            "dtype_c": metadata["dtype_c"],
            "dtype_acc": metadata["dtype_acc"],
            "m": metadata["m"],
            "n": metadata["n"],
            "k": metadata["k"],
            "split_k": metadata.get("split_k", 1),
            "avg_runtime_ms": avg_runtime_ms,
            "best_runtime_ms": avg_runtime_ms,
            "worst_runtime_ms": avg_runtime_ms,
            "avg_tflops": avg_tflops,
            "avg_throughput": "",
            "max_error": "",
            "close_call_group": "",
            "failure_reason": failure_reason,
            "stdout_log": str(log_path),
        }
    )
    return rows


def parse_benchmark_log(log_path, metadata_by_bm_name, run_id):
    rows = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "manual_time" not in line and "ERROR" not in line:
                continue
            stripped = line.strip()
            parts = stripped.split()
            if not parts:
                continue
            token = parts[0]
            segments = token.split("/")
            if len(segments) < 2:
                continue
            bm_name = segments[1]
            metadata = metadata_by_bm_name.get(bm_name)
            if not metadata:
                continue
            failure = any(marker in stripped for marker in ("ERROR OCCURRED", "ERROR", "Disposition Failed"))
            row = {
                "run_id": run_id,
                "stage": metadata["stage"],
                "attempt_index": metadata["attempt_index"],
                "shape_id": metadata["shape_id"],
                "candidate_id": metadata["candidate_id"],
                "compiler_profile_id": metadata["compiler_profile_id"],
                "status": "fail" if failure else "pass",
                "verify_status": "fail" if failure else "pass",
                "layout": metadata["layout"],
                "dtype_a": metadata["dtype_a"],
                "dtype_b": metadata["dtype_b"],
                "dtype_c": metadata["dtype_c"],
                "dtype_acc": metadata["dtype_acc"],
                "m": metadata["m"],
                "n": metadata["n"],
                "k": metadata["k"],
                "split_k": metadata.get("split_k", 1),
                "avg_runtime_ms": parse_metric(stripped, "avg_runtime_ms"),
                "best_runtime_ms": parse_metric(stripped, "best_runtime_ms"),
                "worst_runtime_ms": parse_metric(stripped, "worst_runtime_ms"),
                "avg_tflops": parse_metric(stripped, "avg_tflops"),
                "avg_throughput": parse_metric(stripped, "avg_throughput"),
                "max_error": "",
                "close_call_group": "",
                "failure_reason": stripped if failure else "",
                "stdout_log": str(log_path),
            }
            rows.append(row)
    return rows


def run_entries_with_benchmark(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None):
    metadata = write_config(entries, config_path)
    write_json(manifest_path, metadata)
    command = [exe, f"--config_file={config_path}"]
    result = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init)
    rows = parse_benchmark_log(log_path, metadata, run_id=entries[0]["stage"]) if entries else []
    if result.returncode != 0 and not rows:
        raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {log_path}")
    return rows, command


def run_entries_with_streamk_example(entries, logs_dir, exe, cwd=None, shell_init=None):
    rows = []
    commands = []
    for entry in entries:
        candidate = entry["candidate"]
        shape = entry["shape"]
        bm_name = entry["bm_name"]
        metadata = {
            bm_name: {
                "shape_id": shape["shape_id"],
                "candidate_id": candidate["candidate_id"],
                "compiler_profile_id": candidate["compiler_profile_id"],
                "stage": entry["stage"],
                "attempt_index": entry["attempt_index"],
                "layout": shape["layout"],
                "dtype_a": shape["dtype_a"],
                "dtype_b": shape["dtype_b"],
                "dtype_c": shape["dtype_c"],
                "dtype_acc": shape["dtype_acc"],
                "m": shape["m"],
                "n": shape["n"],
                "k": shape["k"],
                "kernel_name": candidate["kernel_name"],
                "split_k": candidate["split_k"],
                "runner": candidate.get("runner", "streamk_example"),
            }
        }
        log_path = logs_dir / f"{bm_name}.log"
        command = [
            exe,
            "--splitk",
            f"--dtype={candidate['dtype_a']}",
            f"--splits={candidate['split_k']}",
            f"--m={shape['m']}",
            f"--n={shape['n']}",
            f"--k={shape['k']}",
            "--iterations=20",
            "--verify=1",
        ]
        result = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init)
        parsed = parse_streamk_example_log(log_path, metadata, run_id=entry["stage"])
        if result.returncode != 0 and not parsed:
            raise RuntimeError(f"StreamK example subprocess failed with return code {result.returncode}. See {log_path}")
        rows.extend(parsed)
        commands.append(shell_join(command))
    return rows, commands


def run_phase_a_probe(args, shapes_doc, base_constraints, profiles, reports_dir, configs_dir, manifests_dir, logs_dir):
    env_caps = collect_environment_metadata(
        shell_init=args.shell_init,
        benchmark_exe=args.benchmark_exe,
        streamk_example_exe=args.streamk_example_exe,
        cwd=args.cwd,
    )
    static_constraints = apply_static_probe_constraints(base_constraints, env_caps)
    static_candidate_space = generate_candidate_space(
        shapes_doc,
        static_constraints,
        profiles,
        allowed_runners=("benchmark", "streamk_example"),
    )
    probe_rows = []
    probe_logs = []
    probe_commands = []

    probe_entries = build_phase_a_probe_entries(shapes_doc, static_candidate_space)
    effective_probe_mode = args.probe_mode
    if effective_probe_mode == "auto":
        effective_probe_mode = "static" if args.skip_run else "run"

    if effective_probe_mode == "run" and not args.skip_run and probe_entries:
        probe_benchmark_entries = [entry for entry in probe_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
        probe_streamk_entries = [entry for entry in probe_entries if entry["candidate"].get("runner") == "streamk_example"]
        if probe_benchmark_entries:
            probe_log = logs_dir / "probe.log"
            probe_config = configs_dir / "probe.in"
            probe_manifest = manifests_dir / "probe_manifest.json"
            rows, command = run_entries_with_benchmark(
                probe_benchmark_entries,
                probe_config,
                probe_manifest,
                probe_log,
                args.benchmark_exe,
                cwd=args.cwd,
                shell_init=args.shell_init,
            )
            probe_rows.extend(rows)
            probe_logs.append(str(probe_log))
            probe_commands.append(shell_join(command))
        if probe_streamk_entries:
            rows, commands = run_entries_with_streamk_example(
                probe_streamk_entries,
                logs_dir,
                args.streamk_example_exe,
                cwd=args.cwd,
                shell_init=args.shell_init,
            )
            probe_rows.extend(rows)
            probe_logs.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in probe_streamk_entries)
            probe_commands.extend(commands)

    constraints = apply_run_probe_constraints(static_constraints, probe_rows) if probe_rows else static_constraints
    env_caps["probe_mode"] = effective_probe_mode
    env_caps["constraint_source"] = constraints["constraint_source"]
    env_caps["probe_results"] = [
        {
            "candidate_id": row["candidate_id"],
            "shape_id": row["shape_id"],
            "status": row["status"],
            "avg_tflops": row["avg_tflops"],
            "split_k": row["split_k"],
        }
        for row in probe_rows
    ]
    verified_hw_caps_path = reports_dir / "verified_hw_caps.json"
    write_json(verified_hw_caps_path, env_caps)
    return constraints, env_caps, verified_hw_caps_path, probe_rows, probe_logs, probe_commands


def write_results_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def median_or_nan(values):
    numeric = [float(value) for value in values if str(value) != ""]
    if not numeric:
        return math.nan
    return statistics.median(numeric)


def build_dispatch_table(rows, shapes_doc, top_k, confirm_runs, close_call_threshold):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "pass" and row["verify_status"] == "pass":
            grouped[row["shape_id"]].append(row)
    shapes = {shape["shape_id"]: shape for shape in shapes_doc["shapes"]}
    entries = []
    for shape_id, shape_rows in grouped.items():
        confirm_rows = [row for row in shape_rows if row["stage"] == "confirm"]
        selection_rows = confirm_rows if confirm_rows else [row for row in shape_rows if row["stage"] == "screening"]
        by_candidate = defaultdict(list)
        for row in selection_rows:
            by_candidate[row["candidate_id"]].append(row)
        ranked = []
        for candidate_id, candidate_rows in by_candidate.items():
            ranked.append(
                {
                    "candidate_id": candidate_id,
                    "compiler_profile_id": candidate_rows[0]["compiler_profile_id"],
                    "median_tflops": median_or_nan(row["avg_tflops"] for row in candidate_rows),
                    "median_runtime_ms": median_or_nan(row["avg_runtime_ms"] for row in candidate_rows),
                    "samples": len(candidate_rows),
                }
            )
        ranked.sort(key=lambda item: item["median_tflops"], reverse=True)
        if not ranked:
            continue
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        gap = None
        close_call = False
        if runner_up and runner_up["median_tflops"] > 0:
            gap = ((winner["median_tflops"] - runner_up["median_tflops"]) / runner_up["median_tflops"]) * 100.0
            close_call = gap < close_call_threshold
        screening_rank = 1
        shape = shapes[shape_id]
        entries.append(
            {
                "shape_key": {
                    "layout": shape["layout"],
                    "dtype_a": shape["dtype_a"],
                    "dtype_b": shape["dtype_b"],
                    "dtype_c": shape["dtype_c"],
                    "dtype_acc": shape["dtype_acc"],
                    "m": shape["m"],
                    "n": shape["n"],
                    "k": shape["k"],
                },
                "shape_id": shape_id,
                "candidate_id": winner["candidate_id"],
                "compiler_profile_id": winner["compiler_profile_id"],
                "status": "pass",
                "selected_metric": round(winner["median_tflops"], 6),
                "runner_up_candidate_id": runner_up["candidate_id"] if runner_up else "",
                "runner_up_gap_percent": round(gap, 6) if gap is not None else "",
                "close_call": close_call,
                "evidence": {
                    "confirm_median_runtime_ms": (
                        round(winner["median_runtime_ms"], 6) if not math.isnan(winner["median_runtime_ms"]) else ""
                    ),
                    "confirm_median_tflops": round(winner["median_tflops"], 6),
                    "screening_rank": screening_rank,
                    "confirm_samples": winner["samples"],
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "dispatch_id": f"intel_gemm_profiler_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "selection_policy": {
            "screening_top_k": top_k,
            "confirm_runs": confirm_runs,
            "metric": "confirm_median_tflops" if confirm_runs else "screening_avg_tflops",
            "close_call_threshold_percent": close_call_threshold,
        },
        "entries": entries,
    }


def build_run_summary(rows, dispatch_table, build_command, log_paths):
    passed = sum(1 for row in rows if row["status"] == "pass")
    failed = sum(1 for row in rows if row["status"] == "fail")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "rows": len(rows),
        "passed": passed,
        "failed": failed,
        "dispatch_entries": len(dispatch_table["entries"]),
        "benchmark_command": build_command,
        "logs": log_paths,
    }


def build_phase_a_summary(verified_hw_caps, constraints, probe_rows):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "probe_mode": verified_hw_caps.get("probe_mode", "off"),
        "constraint_source": constraints["constraint_source"],
        "probe_results": len(probe_rows),
        "successful_probe_results": sum(1 for row in probe_rows if row["status"] == "pass"),
        "allowed_values": constraints["allowed_values"],
        "limits": constraints["limits"],
        "blocked_rules": constraints.get("blocked_rules", []),
    }


def build_phase_b_summary(candidate_space, dispatch_table, summary):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "candidate_count": len(candidate_space["candidates"]),
        "catalog_version": candidate_space.get("kernel_catalog", {}).get("catalog_version", ""),
        "dispatch_entries": len(dispatch_table["entries"]),
        "rows": summary["rows"],
        "passed": summary["passed"],
        "failed": summary["failed"],
    }


def workflow(args):
    workspace = ensure_dir(Path(args.workspace).resolve())
    inputs_dir = ensure_dir(workspace / "inputs")
    generated_dir = ensure_dir(workspace / "generated")
    configs_dir = ensure_dir(generated_dir / "configs")
    manifests_dir = ensure_dir(generated_dir / "manifests")
    logs_dir = ensure_dir(workspace / "logs")
    reports_dir = ensure_dir(workspace / "reports")

    profiles = read_json(args.compiler_profiles_json) if args.compiler_profiles_json else default_compiler_profiles()
    dry_run_mode = getattr(args, "dry_run", False)
    if args.shapes_json:
        shapes_doc = read_json(args.shapes_json)
    else:
        shapes_doc = dry_run_shapes(args.dtype) if dry_run_mode else default_shapes(args.dtype)
    base_constraints = read_json(args.constraints_json) if args.constraints_json else default_constraints()
    top_k = min(args.top_k, 1) if dry_run_mode else args.top_k
    confirm_runs = 0 if dry_run_mode else args.confirm_runs
    probe_mode = "off" if dry_run_mode else args.probe_mode

    probe_rows = []
    probe_logs = []
    probe_commands = []
    benchmark_commands = []
    if args.constraints_json or probe_mode == "off":
        constraints = copy.deepcopy(base_constraints)
        env_caps = collect_environment_metadata(
            shell_init=args.shell_init,
            benchmark_exe=args.benchmark_exe,
            streamk_example_exe=args.streamk_example_exe,
            cwd=args.cwd,
        )
        env_caps["probe_mode"] = "dry_run_off" if dry_run_mode else ("off" if probe_mode == "off" else "external_constraints")
        env_caps["constraint_source"] = constraints["constraint_source"]
        env_caps["probe_results"] = []
        verified_hw_caps_path = reports_dir / "verified_hw_caps.json"
        write_json(verified_hw_caps_path, env_caps)
    else:
        constraints, env_caps, verified_hw_caps_path, probe_rows, probe_logs, probe_commands = run_phase_a_probe(
            args,
            shapes_doc,
            base_constraints,
            profiles,
            reports_dir,
            configs_dir,
            manifests_dir,
            logs_dir,
        )

    constraints_path = inputs_dir / "safe_search_constraints.json"
    profiles_path = inputs_dir / "compiler_profiles.json"
    shapes_path = inputs_dir / "gemm_target_shapes.json"
    write_json(constraints_path, constraints)
    write_json(profiles_path, profiles)
    write_json(shapes_path, shapes_doc)

    search_runtime_schema_path = inputs_dir / "search_runtime_schema.json"
    write_json(search_runtime_schema_path, SEARCH_RUNTIME_SCHEMA)
    kernel_catalog = build_kernel_catalog(
        dtypes=sorted({shape["dtype_a"] for shape in shapes_doc["shapes"]}),
        allowed_runners=("benchmark", "streamk_example"),
    )
    kernel_catalog_path = reports_dir / "kernel_catalog.json"
    write_json(kernel_catalog_path, kernel_catalog)

    candidate_space = generate_candidate_space(shapes_doc, constraints, profiles, allowed_runners=("benchmark",))
    candidate_space_path = reports_dir / "gemm_candidate_space.json"
    write_json(candidate_space_path, candidate_space)
    safe_candidates_path = reports_dir / "bmg_safe_candidates.json"
    write_json(safe_candidates_path, candidate_space)
    build_manifest_path = reports_dir / "candidate_build_manifest.json"
    write_json(build_manifest_path, build_candidate_build_manifest(candidate_space))

    screening_entries = build_screening_entries(shapes_doc, candidate_space)
    screening_config = configs_dir / "screening.in"
    screening_manifest = manifests_dir / "screening_manifest.json"

    all_rows = list(probe_rows)
    log_paths = list(probe_logs)
    benchmark_commands.extend(probe_commands)
    if not args.skip_run:
        screening_benchmark_entries = [entry for entry in screening_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
        screening_streamk_entries = [entry for entry in screening_entries if entry["candidate"].get("runner") == "streamk_example"]
        screening_rows = []
        if screening_benchmark_entries:
            screening_log = logs_dir / "screening.log"
            rows, command = run_entries_with_benchmark(
                screening_benchmark_entries,
                screening_config,
                screening_manifest,
                screening_log,
                args.benchmark_exe,
                cwd=args.cwd,
                shell_init=args.shell_init,
            )
            screening_rows.extend(rows)
            log_paths.append(str(screening_log))
            benchmark_commands.append(shell_join(command))
        if screening_streamk_entries:
            rows, commands = run_entries_with_streamk_example(
                screening_streamk_entries,
                logs_dir,
                args.streamk_example_exe,
                cwd=args.cwd,
                shell_init=args.shell_init,
            )
            screening_rows.extend(rows)
            log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in screening_streamk_entries)
            benchmark_commands.extend(commands)
        all_rows.extend(screening_rows)

        if confirm_runs > 0:
            confirm_entries = generate_confirmation_entries(
                screening_rows,
                candidate_space,
                shapes_doc,
                top_k=top_k,
                confirm_runs=confirm_runs,
            )
            if confirm_entries:
                confirm_config = configs_dir / "confirm.in"
                confirm_manifest = manifests_dir / "confirm_manifest.json"
                confirm_benchmark_entries = [entry for entry in confirm_entries if entry["candidate"].get("runner", "benchmark") == "benchmark"]
                confirm_streamk_entries = [entry for entry in confirm_entries if entry["candidate"].get("runner") == "streamk_example"]
                confirm_rows = []
                if confirm_benchmark_entries:
                    confirm_log = logs_dir / "confirm.log"
                    rows, command = run_entries_with_benchmark(
                        confirm_benchmark_entries,
                        confirm_config,
                        confirm_manifest,
                        confirm_log,
                        args.benchmark_exe,
                        cwd=args.cwd,
                        shell_init=args.shell_init,
                    )
                    confirm_rows.extend(rows)
                    log_paths.append(str(confirm_log))
                    benchmark_commands.append(shell_join(command))
                if confirm_streamk_entries:
                    rows, commands = run_entries_with_streamk_example(
                        confirm_streamk_entries,
                        logs_dir,
                        args.streamk_example_exe,
                        cwd=args.cwd,
                        shell_init=args.shell_init,
                    )
                    confirm_rows.extend(rows)
                    log_paths.extend(str(logs_dir / f"{entry['bm_name']}.log") for entry in confirm_streamk_entries)
                    benchmark_commands.extend(commands)
                all_rows.extend(confirm_rows)

    results_csv = reports_dir / "gemm_profile_results.csv"
    write_results_csv(all_rows, results_csv)

    dispatch_table = build_dispatch_table(
        all_rows,
        shapes_doc,
        top_k=top_k,
        confirm_runs=confirm_runs,
        close_call_threshold=args.close_call_threshold,
    )
    dispatch_path = reports_dir / "gemm_dispatch_table.json"
    write_json(dispatch_path, dispatch_table)
    optimal_dispatch_path = reports_dir / "optimal_dispatch_table.json"
    write_json(optimal_dispatch_path, dispatch_table)

    summary = build_run_summary(all_rows, dispatch_table, benchmark_commands, log_paths)
    summary_path = reports_dir / "run_summary.json"
    write_json(summary_path, summary)
    phase_a_summary_path = reports_dir / "phase_a_summary.json"
    write_json(phase_a_summary_path, build_phase_a_summary(env_caps, constraints, probe_rows))
    phase_b_summary_path = reports_dir / "phase_b_summary.json"
    write_json(phase_b_summary_path, build_phase_b_summary(candidate_space, dispatch_table, summary))
    return {
        "workspace": str(workspace),
        "search_runtime_schema": str(search_runtime_schema_path),
        "kernel_catalog": str(kernel_catalog_path),
        "candidate_space": str(candidate_space_path),
        "build_manifest": str(build_manifest_path),
        "safe_candidates": str(safe_candidates_path),
        "verified_hw_caps": str(verified_hw_caps_path),
        "results_csv": str(results_csv),
        "dispatch_table": str(dispatch_path),
        "optimal_dispatch_table": str(optimal_dispatch_path),
        "phase_a_summary": str(phase_a_summary_path),
        "phase_b_summary": str(phase_b_summary_path),
        "summary": str(summary_path),
        "dry_run": dry_run_mode,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Intel GEMM profiler MVP runner for non-legacy registered RCR kernels.")
    parser.add_argument("--workspace", required=True, help="Workspace directory for generated files and reports.")
    parser.add_argument(
        "--benchmark-exe",
        default="./build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl",
        help="Benchmark executable to run.",
    )
    parser.add_argument(
        "--streamk-example-exe",
        default="./build/examples/03_bmg_gemm_streamk/03_bmg_gemm_streamk",
        help="StreamK example executable used for split-k candidates.",
    )
    parser.add_argument("--cwd", default=None, help="Working directory for the benchmark subprocess.")
    parser.add_argument(
        "--shell-init",
        default="",
        help="Optional shell snippet executed before the benchmark command, e.g. 'source /home/intel/.bashrc && source /opt/intel/oneapi/setvars.sh'.",
    )
    parser.add_argument("--dtype", choices=sorted(SEED_KERNELS.keys()), default="bf16", help="Default dtype preset.")
    parser.add_argument(
        "--probe-mode",
        choices=["auto", "off", "static", "run"],
        default="auto",
        help="Phase A constraint probe mode. 'auto' runs representative probes unless --skip-run is set.",
    )
    parser.add_argument("--shapes-json", default="", help="Optional path to gemm_target_shapes.json.")
    parser.add_argument("--constraints-json", default="", help="Optional path to safe_search_constraints.json.")
    parser.add_argument("--compiler-profiles-json", default="", help="Optional path to compiler_profiles.json.")
    parser.add_argument("--skip-run", action="store_true", help="Only emit generated artifacts without invoking the benchmark.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a minimal benchmark-backed screening smoke with a tiny shape set and no confirmation.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k candidates kept for confirmation.")
    parser.add_argument("--confirm-runs", type=int, default=3, help="Number of confirmation attempts for top-k candidates.")
    parser.add_argument(
        "--close-call-threshold",
        type=float,
        default=3.0,
        help="Gap threshold in percent for close-call labeling.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    outputs = workflow(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
