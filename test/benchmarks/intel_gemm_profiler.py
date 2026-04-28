#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "1.0"


SEED_KERNELS = {
    "bf16": [
        {
            "kernel_name": "PvcGemmBF16BF16FP32_RCR_5",
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
            "kernel_name": "PvcGemmBF16BF16FP32_RCR_7",
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
            "kernel_name": "PvcGemmBF16BF16FP32_RCR_9",
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
            "kernel_name": "PvcGemmBF16BF16FP32_RCR_16",
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
            "kernel_name": "PvcGemmBF16BF16FP32_RCR_6",
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
    ],
    "f16": [
        {
            "kernel_name": "PvcGemmFP16FP16FP32_RCR_5",
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
            "kernel_name": "PvcGemmFP16FP16FP32_RCR_7",
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
            "kernel_name": "PvcGemmFP16FP16FP32_RCR_9",
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
            "kernel_name": "PvcGemmFP16FP16FP32_RCR_16",
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
            "max_split_k": 1,
            "max_stages": 3,
        },
        "allowed_values": {
            "tile_m": [8, 16, 32, 64, 128, 256],
            "tile_n": [64, 128, 256],
            "tile_k": [32, 64],
            "sg_m": [1, 2, 4, 8],
            "sg_n": [4, 8],
            "stages": [1, 2, 3],
            "split_k": [1],
        },
        "blocked_rules": [],
    }


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


def blocked(seed, constraints):
    if seed["sg_m"] * seed["sg_n"] > 32:
        return True
    allowed = constraints["allowed_values"]
    for key in ("tile_m", "tile_n", "tile_k", "sg_m", "sg_n", "stages", "split_k"):
        if seed[key] not in allowed[key]:
            return True
    for rule in constraints.get("blocked_rules", []):
        match = rule.get("match", {})
        if all(seed.get(name) == value for name, value in match.items()):
            return True
    return False


def generate_candidate_space(shapes_doc, constraints, profiles):
    seen = set()
    candidates = []
    dtypes = sorted({shape["dtype_a"] for shape in shapes_doc["shapes"]})
    for dtype in dtypes:
        for seed in SEED_KERNELS.get(dtype, []):
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
                    "candidate_class": klass,
                    "compiler_profile_id": select_compiler_profile_id(profiles, seed["tile_m"], sg_count),
                    "filters_applied": ["seed_kernel", constraints["constraint_source"]],
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "device_arch": constraints["device_arch"],
        "constraint_source": constraints["constraint_source"],
        "candidates": candidates,
    }


def choose_candidates_for_shape(shape, candidates):
    matched = []
    for candidate in candidates:
        if candidate["layout"] != shape["layout"]:
            continue
        if candidate["dtype_a"] != shape["dtype_a"] or candidate["dtype_b"] != shape["dtype_b"]:
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


def workflow(args):
    workspace = ensure_dir(Path(args.workspace).resolve())
    inputs_dir = ensure_dir(workspace / "inputs")
    generated_dir = ensure_dir(workspace / "generated")
    configs_dir = ensure_dir(generated_dir / "configs")
    manifests_dir = ensure_dir(generated_dir / "manifests")
    logs_dir = ensure_dir(workspace / "logs")
    reports_dir = ensure_dir(workspace / "reports")

    constraints = read_json(args.constraints_json) if args.constraints_json else default_constraints()
    profiles = read_json(args.compiler_profiles_json) if args.compiler_profiles_json else default_compiler_profiles()
    shapes_doc = read_json(args.shapes_json) if args.shapes_json else default_shapes(args.dtype)

    constraints_path = inputs_dir / "safe_search_constraints.json"
    profiles_path = inputs_dir / "compiler_profiles.json"
    shapes_path = inputs_dir / "gemm_target_shapes.json"
    write_json(constraints_path, constraints)
    write_json(profiles_path, profiles)
    write_json(shapes_path, shapes_doc)

    candidate_space = generate_candidate_space(shapes_doc, constraints, profiles)
    candidate_space_path = reports_dir / "gemm_candidate_space.json"
    write_json(candidate_space_path, candidate_space)

    screening_entries = build_screening_entries(shapes_doc, candidate_space)
    screening_config = configs_dir / "screening.in"
    screening_manifest = manifests_dir / "screening_manifest.json"
    screening_metadata = write_config(screening_entries, screening_config)
    write_json(screening_manifest, screening_metadata)

    all_rows = []
    log_paths = []
    benchmark_command = [args.benchmark_exe, f"--config_file={screening_config}"]
    if not args.skip_run:
        screening_log = logs_dir / "screening.log"
        result = run_benchmark(benchmark_command, screening_log, cwd=args.cwd, shell_init=args.shell_init)
        log_paths.append(str(screening_log))
        screening_rows = parse_benchmark_log(screening_log, screening_metadata, run_id="screening")
        if result.returncode != 0 and not screening_rows:
            raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {screening_log}")
        all_rows.extend(screening_rows)

        if args.confirm_runs > 0:
            confirm_entries = generate_confirmation_entries(
                screening_rows,
                candidate_space,
                shapes_doc,
                top_k=args.top_k,
                confirm_runs=args.confirm_runs,
            )
            if confirm_entries:
                confirm_config = configs_dir / "confirm.in"
                confirm_manifest = manifests_dir / "confirm_manifest.json"
                confirm_metadata = write_config(confirm_entries, confirm_config)
                write_json(confirm_manifest, confirm_metadata)
                confirm_log = logs_dir / "confirm.log"
                confirm_command = [args.benchmark_exe, f"--config_file={confirm_config}"]
                result = run_benchmark(confirm_command, confirm_log, cwd=args.cwd, shell_init=args.shell_init)
                log_paths.append(str(confirm_log))
                confirm_rows = parse_benchmark_log(confirm_log, confirm_metadata, run_id="confirm")
                if result.returncode != 0 and not confirm_rows:
                    raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {confirm_log}")
                all_rows.extend(confirm_rows)

    results_csv = reports_dir / "gemm_profile_results.csv"
    write_results_csv(all_rows, results_csv)

    dispatch_table = build_dispatch_table(
        all_rows,
        shapes_doc,
        top_k=args.top_k,
        confirm_runs=args.confirm_runs,
        close_call_threshold=args.close_call_threshold,
    )
    dispatch_path = reports_dir / "gemm_dispatch_table.json"
    write_json(dispatch_path, dispatch_table)

    summary = build_run_summary(all_rows, dispatch_table, benchmark_command, log_paths)
    summary_path = reports_dir / "run_summary.json"
    write_json(summary_path, summary)
    return {
        "workspace": str(workspace),
        "candidate_space": str(candidate_space_path),
        "results_csv": str(results_csv),
        "dispatch_table": str(dispatch_path),
        "summary": str(summary_path),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Intel GEMM profiler MVP runner for legacy registered RCR kernels.")
    parser.add_argument("--workspace", required=True, help="Workspace directory for generated files and reports.")
    parser.add_argument(
        "--benchmark-exe",
        default="./build/benchmarks/gemm/legacy/cutlass_benchmarks_gemm_sycl_legacy",
        help="Benchmark executable to run.",
    )
    parser.add_argument("--cwd", default=None, help="Working directory for the benchmark subprocess.")
    parser.add_argument(
        "--shell-init",
        default="",
        help="Optional shell snippet executed before the benchmark command, e.g. 'source /home/intel/.bashrc && source /opt/intel/oneapi/setvars.sh'.",
    )
    parser.add_argument("--dtype", choices=sorted(SEED_KERNELS.keys()), default="bf16", help="Default dtype preset.")
    parser.add_argument("--shapes-json", default="", help="Optional path to gemm_target_shapes.json.")
    parser.add_argument("--constraints-json", default="", help="Optional path to safe_search_constraints.json.")
    parser.add_argument("--compiler-profiles-json", default="", help="Optional path to compiler_profiles.json.")
    parser.add_argument("--skip-run", action="store_true", help="Only emit generated artifacts without invoking the benchmark.")
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
