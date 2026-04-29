#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import math
from pathlib import Path

from .utils import read_json

DEFAULT_HW_REFERENCE_SPECS_PATH = Path(__file__).resolve().parent.parent / "intel_gemm_hw_reference_specs.json"
DEFAULT_HW_SPEC_IDS = {"bmg": "bmg_g21"}
ELEMENT_BYTES = {"bf16": 2, "f16": 2, "f32": 4}


def load_hw_reference_specs(path=DEFAULT_HW_REFERENCE_SPECS_PATH):
    return read_json(path)


def resolve_hw_reference_spec(device_arch="bmg", hw_spec_id="", path=DEFAULT_HW_REFERENCE_SPECS_PATH):
    specs = load_hw_reference_specs(path)
    resolved_id = hw_spec_id or DEFAULT_HW_SPEC_IDS.get(device_arch, device_arch)
    if resolved_id not in specs:
        raise KeyError(f"Unknown hardware reference spec '{resolved_id}'.")
    spec = copy.deepcopy(specs[resolved_id])
    spec["device_id"] = resolved_id
    return spec


def _peak_tflops_for_dtype(dtype, hw_spec):
    return hw_spec.get(f"peak_{dtype}_tflops", hw_spec.get("peak_xmx_tflops", 0.0))


def analyze_efficiency(shape, candidate, hw_spec):
    m = shape["m"]
    n = shape["n"]
    k = shape["k"]
    tile_m = candidate["tile_m"]
    tile_n = candidate["tile_n"]
    tile_k = candidate.get("tile_k", 32)
    sg_count = max(candidate["sg_m"] * candidate["sg_n"], 1)
    stages = candidate.get("stages", 2)
    dtype = candidate["dtype_a"]
    element_bytes = ELEMENT_BYTES.get(dtype, 2)

    grid_m = max(math.ceil(m / tile_m), 1)
    grid_n = max(math.ceil(n / tile_n), 1)
    total_tiles = grid_m * grid_n

    concurrent_sgs_per_xe_core = max(
        int(hw_spec.get("concurrent_sgs_per_xe_core_256grf", hw_spec.get("eus_per_xe_core", 1))),
        1,
    )
    max_concurrent_sgs = max(int(hw_spec.get("xe_cores", 1)), 1) * concurrent_sgs_per_xe_core
    max_concurrent_wg = max_concurrent_sgs // sg_count
    waves = math.ceil(total_tiles / max(max_concurrent_wg, 1))
    wave_efficiency = total_tiles / (waves * max(max_concurrent_wg, 1))

    tile_util_m = m / (grid_m * tile_m)
    tile_util_n = n / (grid_n * tile_n)
    tile_efficiency = tile_util_m * tile_util_n

    flops = 2.0 * m * n * k
    bytes_read = max((m * k + k * n) * element_bytes, 1)
    arithmetic_intensity = flops / bytes_read
    peak_tflops = _peak_tflops_for_dtype(dtype, hw_spec)
    measured_bw = hw_spec.get("measured_read_bw_gbps", hw_spec.get("peak_memory_bw_gbps", 0.0))
    ridge_point = (peak_tflops * 1e3 / measured_bw) if peak_tflops and measured_bw else math.inf
    is_compute_bound = arithmetic_intensity > ridge_point

    slm_kb = (tile_m * tile_k + tile_n * tile_k) * element_bytes * stages / 1024.0
    slm_limit_kb = hw_spec.get("slm_per_xe_core_kb", 64)
    slm_ok = slm_kb <= slm_limit_kb

    acc_bytes = max(tile_m // max(candidate["sg_m"], 1), 1) * max(tile_n // max(candidate["sg_n"], 1), 1) * 4
    grf_limit = hw_spec.get("grf_bytes_per_thread", 8192)
    grf_pressure = (acc_bytes / grf_limit) if grf_limit else math.inf

    xe_cores_per_wg = math.ceil(sg_count / concurrent_sgs_per_xe_core)
    cross_xe_core_penalty = 1.0 if xe_cores_per_wg <= 1 else 0.85 ** (xe_cores_per_wg - 1)
    if xe_cores_per_wg > 2:
        cross_xe_core_penalty *= 0.5

    if not slm_ok or grf_pressure > 1.0 or max_concurrent_wg == 0:
        bounds = (0.01, 0.10)
    elif is_compute_bound:
        gemm_efficiency = hw_spec.get("peak_gemm_efficiency", 0.92)
        base = wave_efficiency * tile_efficiency * cross_xe_core_penalty
        bounds = (max(0.01, base * 0.25), min(1.0, base * gemm_efficiency))
    else:
        mem_limited_tflops = arithmetic_intensity * measured_bw / 1e3 if measured_bw else 0.0
        expected_ratio = (mem_limited_tflops / peak_tflops) if peak_tflops else 0.0
        bounds = (max(0.001, expected_ratio * 0.3), min(1.0, expected_ratio * 1.1))

    root_cause_hints = []
    if not slm_ok:
        root_cause_hints.append(f"slm {slm_kb:.1f}KB exceeds {slm_limit_kb}KB")
    if grf_pressure > 1.0:
        root_cause_hints.append(f"grf pressure {grf_pressure:.2f} exceeds per-thread budget")
    if xe_cores_per_wg > 1:
        root_cause_hints.append(
            f"sg_count={sg_count} spans {xe_cores_per_wg} Xe-cores in 256-GRF mode (limit {concurrent_sgs_per_xe_core} SGs/Xe-core)"
        )
    if max_concurrent_wg == 0:
        root_cause_hints.append("workgroup cannot be scheduled concurrently under current occupancy model")

    return {
        "peak_tflops": peak_tflops,
        "measured_bw_gbps": measured_bw,
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "is_compute_bound": is_compute_bound,
        "slm_kb": slm_kb,
        "slm_ok": slm_ok,
        "grf_pressure": grf_pressure,
        "xe_cores_per_wg": xe_cores_per_wg,
        "max_concurrent_wg": max_concurrent_wg,
        "wave_efficiency": wave_efficiency,
        "tile_efficiency": tile_efficiency,
        "cross_xe_core_penalty": cross_xe_core_penalty,
        "min_expected_efficiency": bounds[0],
        "max_expected_efficiency": bounds[1],
        "root_cause_hints": root_cause_hints,
    }


def compute_efficiency_bounds(shape, candidate, hw_spec):
    analysis = analyze_efficiency(shape, candidate, hw_spec)
    return analysis["min_expected_efficiency"], analysis["max_expected_efficiency"]


def detect_probe_anomalies(probe_rows, shapes_doc, candidate_space, hw_spec):
    shape_map = {shape["shape_id"]: shape for shape in shapes_doc.get("shapes", [])}
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidate_space.get("candidates", [])}
    anomalies = []
    auto_block_rules = []
    peak_tflops = hw_spec.get("peak_xmx_tflops", 0.0)

    for row in probe_rows:
        if row.get("status") != "pass" or not row.get("avg_tflops"):
            continue
        candidate = candidate_map.get(row["candidate_id"])
        shape = shape_map.get(row["shape_id"])
        if candidate is None or shape is None:
            continue

        actual_tflops = float(row["avg_tflops"])
        analysis = analyze_efficiency(shape, candidate, hw_spec)
        min_eff = analysis["min_expected_efficiency"]
        max_eff = analysis["max_expected_efficiency"]
        actual_eff = (actual_tflops / analysis["peak_tflops"]) if analysis["peak_tflops"] else 0.0

        spec_anomaly = None
        if actual_eff < min_eff * 0.5:
            spec_anomaly = "severely_below_spec"
        elif actual_eff < min_eff:
            spec_anomaly = "below_spec"
        elif max_eff > 0 and actual_eff > max_eff * 1.5:
            spec_anomaly = "above_expected"

        cross_candidates = []
        for other_row in probe_rows:
            if other_row["candidate_id"] == row["candidate_id"]:
                continue
            if other_row.get("status") != "pass" or not other_row.get("avg_tflops"):
                continue
            other_candidate = candidate_map.get(other_row["candidate_id"])
            other_shape = shape_map.get(other_row["shape_id"])
            if other_candidate is None or other_shape is None:
                continue
            if (
                shape["layout"] == other_shape["layout"]
                and shape["dtype_a"] == other_shape["dtype_a"]
                and shape["m"] > other_shape["m"]
                and candidate["tile_m"] > other_candidate["tile_m"]
                and actual_tflops < float(other_row["avg_tflops"]) * 0.5
            ):
                cross_candidates.append((float(other_row["avg_tflops"]), other_row["candidate_id"]))
        cross_anomaly = None
        if cross_candidates:
            _, reference_candidate_id = max(cross_candidates, key=lambda item: item[0])
            cross_anomaly = f"large_tile_slower_than_{reference_candidate_id}"

        if not spec_anomaly and not cross_anomaly:
            continue

        root_cause_hint = "; ".join(analysis["root_cause_hints"])
        auto_action = "blocked" if spec_anomaly == "severely_below_spec" or cross_anomaly else "reported"
        anomaly = {
            "candidate_id": row["candidate_id"],
            "shape_id": row["shape_id"],
            "actual_tflops": actual_tflops,
            "actual_efficiency": round(actual_eff, 4),
            "expected_efficiency_range": [round(min_eff, 4), round(max_eff, 4)],
            "spec_anomaly": spec_anomaly,
            "cross_anomaly": cross_anomaly,
            "root_cause_hint": root_cause_hint,
            "auto_action": auto_action,
        }
        anomalies.append(anomaly)

        if auto_action == "blocked":
            auto_block_rules.append(
                {
                    "rule_id": f"probe.auto_block.anomaly.{row['candidate_id']}",
                    "match": {
                        "tile_m": candidate["tile_m"],
                        "tile_n": candidate["tile_n"],
                        "tile_k": candidate["tile_k"],
                        "sg_m": candidate["sg_m"],
                        "sg_n": candidate["sg_n"],
                        "split_k": candidate["split_k"],
                    },
                    "reason": cross_anomaly or spec_anomaly,
                    "evidence_tflops": actual_tflops,
                }
            )

    return {
        "hw_spec": hw_spec.get("device_id", ""),
        "hw_spec_calibration_status": hw_spec.get("calibration_status", "unknown"),
        "peak_tflops": peak_tflops,
        "anomalies": anomalies,
        "auto_block_rules": auto_block_rules,
    }
