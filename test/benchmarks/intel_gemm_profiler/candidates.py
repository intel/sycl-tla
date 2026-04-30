#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy

from .catalog import SEED_KERNELS, build_kernel_catalog
from .constraints import blocked
from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
from .utils import now_iso


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
                "runtime_defaults": {"raster_order": "heuristic", "swizzle_size": 1, "barrier_interval": 8 if item["m"] >= 64 else 0, "k_unroll": 1},
                "tags": item["tags"],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "generated_at": now_iso(), "shape_set_id": f"default_{dtype}_decode_prefill", "source": "predefined", "shapes": shapes}


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
                "runtime_defaults": {"raster_order": "heuristic", "swizzle_size": 1, "barrier_interval": 0, "k_unroll": 1},
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
    available_profiles = [profile for profile in profiles["profiles"] if profile.get("probe_status") not in {"fail", "timeout"}]
    for profile in profiles["profiles"]:
        if profile.get("probe_status") in {"fail", "timeout"}:
            continue
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
    fallback_profiles = available_profiles or profiles["profiles"]
    return fallback_profiles[0]["compiler_profile_id"]


def candidate_id_for(seed):
    return f"{seed['layout']}_{seed['dtype_a']}{seed['dtype_b']}{seed['dtype_c']}_tm{seed['tile_m']}_tn{seed['tile_n']}_tk{seed['tile_k']}_sg{seed['sg_m']}x{seed['sg_n']}_st{seed['stages']}_sk{seed['split_k']}"


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
                "candidate_class": candidate_class(seed["tile_m"], sg_count),
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
        "kernel_catalog": {"catalog_version": kernel_catalog["catalog_version"], "kernel_count": len(kernel_catalog["kernels"])},
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
                "runtime_sweep": {"allowed_fields": candidate["allowed_runtime_sweeps"], "defaults": candidate["runtime_defaults"]},
                "compiler_profile_id": candidate["compiler_profile_id"],
            }
        )
    return {"schema_version": SCHEMA_VERSION, "generated_at": now_iso(), "device_arch": candidate_space["device_arch"], "constraint_source": candidate_space["constraint_source"], "search_runtime_schema": SEARCH_RUNTIME_SCHEMA, "variants": variants}


def choose_candidates_for_shape(shape, candidates):
    matched = []
    for candidate in candidates:
        if candidate["layout"] != shape["layout"]:
            continue
        if candidate["dtype_a"] != shape["dtype_a"] or candidate["dtype_b"] != shape["dtype_b"]:
            continue
        if candidate["split_k"] > 1 and shape["n"] < 16384 and shape["k"] < 8192:
            continue
        matched.append(candidate)
    return matched or [candidate for candidate in candidates if candidate["layout"] == shape["layout"] and candidate["dtype_a"] == shape["dtype_a"]]


def select_probe_shape(shapes_doc, dtype, layout, target_m, target_n, target_k, predicate=None):
    pool = [shape for shape in shapes_doc["shapes"] if shape["dtype_a"] == dtype and shape["layout"] == layout]
    if predicate:
        filtered = [shape for shape in pool if predicate(shape)]
        if filtered:
            pool = filtered
    if not pool:
        return None
    return min(pool, key=lambda shape: (abs(shape["m"] - target_m), abs(shape["n"] - target_n), abs(shape["k"] - target_k), shape["m"], shape["n"], shape["k"]))


def build_phase_a_probe_entries(shapes_doc, candidate_space):
    candidates = candidate_space["candidates"]
    non_splitk = [candidate for candidate in candidates if candidate["split_k"] == 1]
    splitk = [candidate for candidate in candidates if candidate["split_k"] > 1]
    selected = []
    if non_splitk:
        small_candidate = min(non_splitk, key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"]))
        selected.append(("small", small_candidate, select_probe_shape(shapes_doc, small_candidate["dtype_a"], small_candidate["layout"], 8, 4096, 4096, predicate=lambda shape: shape["m"] <= 8)))
        medium = [candidate for candidate in non_splitk if 16 <= candidate["tile_m"] <= 64]
        if medium:
            medium_candidate = min(medium, key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"]))
            selected.append(("medium", medium_candidate, select_probe_shape(shapes_doc, medium_candidate["dtype_a"], medium_candidate["layout"], 64, 4096, 4096, predicate=lambda shape: 8 < shape["m"] < 128)))
        large_candidate = max(non_splitk, key=lambda item: (item["tile_m"], item["tile_n"]))
        selected.append(("large", large_candidate, select_probe_shape(shapes_doc, large_candidate["dtype_a"], large_candidate["layout"], 256, 4096, 8192, predicate=lambda shape: shape["m"] >= 128)))
    if splitk:
        splitk_candidate = splitk[0]
        selected.append(("splitk", splitk_candidate, select_probe_shape(shapes_doc, splitk_candidate["dtype_a"], splitk_candidate["layout"], 1, 4096, 14336, predicate=lambda shape: shape["n"] >= 16384 or shape["k"] >= 8192)))
    entries = []
    seen = set()
    for probe_class, candidate, shape in selected:
        if shape is None:
            continue
        key = (candidate["candidate_id"], shape["shape_id"])
        if key in seen:
            continue
        seen.add(key)
        entries.append({"bm_name": f"{candidate['candidate_id']}__{shape['shape_id']}__probe__0", "stage": "probe", "attempt_index": 0, "probe_class": probe_class, "shape": shape, "candidate": candidate})
    return entries


def build_dpas_probe_entry(shapes_doc, candidate_space):
    benchmark_candidates = [candidate for candidate in candidate_space["candidates"] if candidate.get("runner", "benchmark") == "benchmark" and candidate["split_k"] == 1]
    if not benchmark_candidates:
        return None
    baseline_candidate = min(benchmark_candidates, key=lambda item: (item["tile_m"], item["sg_m"] * item["sg_n"], item["tile_n"], item["tile_k"]))
    dtype_shapes = [shape for shape in shapes_doc["shapes"] if shape["dtype_a"] == baseline_candidate["dtype_a"] and shape["layout"] == baseline_candidate["layout"]]
    if not dtype_shapes:
        return None
    baseline_shape = min(dtype_shapes, key=lambda item: (item["k"], item["m"], item["n"]))
    return {"bm_name": f"{baseline_candidate['candidate_id']}__{baseline_shape['shape_id']}__dpas_probe__0", "stage": "dpas_probe", "attempt_index": 0, "probe_class": "dpas_baseline", "shape": baseline_shape, "candidate": baseline_candidate}


def build_compiler_profile_probe_entries(shapes_doc, candidate_space, profiles):
    probe_entries = build_phase_a_probe_entries(shapes_doc, candidate_space)
    probe_entry_by_class = {
        "small_tile": next((entry for entry in probe_entries if entry["probe_class"] == "small" and entry["candidate"].get("runner", "benchmark") == "benchmark"), None),
        "medium_tile": next((entry for entry in probe_entries if entry["probe_class"] == "medium" and entry["candidate"].get("runner", "benchmark") == "benchmark"), None),
        "large_tile": next((entry for entry in probe_entries if entry["probe_class"] == "large" and entry["candidate"].get("runner", "benchmark") == "benchmark"), None),
    }
    compiler_probe_entries = []
    for profile in profiles["profiles"]:
        base_entry = probe_entry_by_class.get(profile.get("candidate_class"))
        if base_entry is None:
            continue
        entry = copy.deepcopy(base_entry)
        entry["stage"] = "compiler_profile_probe"
        entry["probe_class"] = profile["candidate_class"]
        entry["compiler_profile_probe_id"] = profile["compiler_profile_id"]
        entry["compiler_profile_id"] = profile["compiler_profile_id"]
        entry["bm_name"] = f"{entry['candidate']['candidate_id']}__{entry['shape']['shape_id']}__compiler_probe__{profile['compiler_profile_id'].replace('.', '_')}"
        compiler_probe_entries.append(entry)
    return compiler_probe_entries


def build_screening_entries(shapes_doc, candidate_space):
    entries = []
    for shape in shapes_doc["shapes"]:
        for candidate in choose_candidates_for_shape(shape, candidate_space["candidates"]):
            entries.append({"bm_name": f"{candidate['candidate_id']}__{shape['shape_id']}__screening__0", "stage": "screening", "attempt_index": 0, "shape": shape, "candidate": candidate})
    return entries


def generate_confirmation_entries(rows, candidate_space, shapes_doc, top_k, confirm_runs):
    shape_map = {shape["shape_id"]: shape for shape in shapes_doc["shapes"]}
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidate_space["candidates"]}
    grouped = {}
    for row in rows:
        if row["stage"] == "screening" and row["status"] == "pass":
            grouped.setdefault(row["shape_id"], []).append(row)
    entries = []
    for shape_id, shape_rows in grouped.items():
        ranked = sorted(shape_rows, key=lambda row: float(row["avg_tflops"] or 0.0), reverse=True)[:top_k]
        for attempt_index in range(confirm_runs):
            for row in ranked:
                candidate_id = row["candidate_id"]
                entries.append({"bm_name": f"{candidate_id}__{shape_id}__confirm__{attempt_index}", "stage": "confirm", "attempt_index": attempt_index, "shape": shape_map[shape_id], "candidate": candidate_map[candidate_id]})
    return entries


def write_config(entries, config_path):
    metadata = {}
    with open(config_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            candidate = entry["candidate"]
            shape = entry["shape"]
            handle.write(f"{candidate['kernel_name']} --bm_name={entry['bm_name']} --m={shape['m']} --n={shape['n']} --k={shape['k']}\n")
            metadata[entry["bm_name"]] = {
                "shape_id": shape["shape_id"],
                "candidate_id": candidate["candidate_id"],
                "compiler_profile_id": entry.get("compiler_profile_id", candidate["compiler_profile_id"]),
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
