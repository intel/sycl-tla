#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime

from .schemas import CSV_FIELDS, SCHEMA_VERSION
from .utils import now_iso


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
            ranked.append({"candidate_id": candidate_id, "compiler_profile_id": candidate_rows[0]["compiler_profile_id"], "median_tflops": median_or_nan(row["avg_tflops"] for row in candidate_rows), "median_runtime_ms": median_or_nan(row["avg_runtime_ms"] for row in candidate_rows), "samples": len(candidate_rows)})
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
        shape = shapes[shape_id]
        entries.append(
            {
                "shape_key": {"layout": shape["layout"], "dtype_a": shape["dtype_a"], "dtype_b": shape["dtype_b"], "dtype_c": shape["dtype_c"], "dtype_acc": shape["dtype_acc"], "m": shape["m"], "n": shape["n"], "k": shape["k"]},
                "shape_id": shape_id,
                "candidate_id": winner["candidate_id"],
                "compiler_profile_id": winner["compiler_profile_id"],
                "status": "pass",
                "selected_metric": round(winner["median_tflops"], 6),
                "runner_up_candidate_id": runner_up["candidate_id"] if runner_up else "",
                "runner_up_gap_percent": round(gap, 6) if gap is not None else "",
                "close_call": close_call,
                "evidence": {"confirm_median_runtime_ms": round(winner["median_runtime_ms"], 6) if not math.isnan(winner["median_runtime_ms"]) else "", "confirm_median_tflops": round(winner["median_tflops"], 6), "screening_rank": 1, "confirm_samples": winner["samples"]},
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "dispatch_id": f"intel_gemm_profiler_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "selection_policy": {"screening_top_k": top_k, "confirm_runs": confirm_runs, "metric": "confirm_median_tflops" if confirm_runs else "screening_avg_tflops", "close_call_threshold_percent": close_call_threshold},
        "entries": entries,
    }


def build_run_summary(rows, dispatch_table, build_command, log_paths):
    passed = sum(1 for row in rows if row["status"] == "pass")
    failed = sum(1 for row in rows if row["status"] != "pass")
    return {"schema_version": SCHEMA_VERSION, "generated_at": now_iso(), "rows": len(rows), "passed": passed, "failed": failed, "dispatch_entries": len(dispatch_table["entries"]), "benchmark_command": build_command, "logs": log_paths}


def build_phase_a_summary(verified_hw_caps, constraints, probe_rows):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "probe_mode": verified_hw_caps.get("probe_mode", "off"),
        "constraint_source": constraints["constraint_source"],
        "dpas_baseline_probe": verified_hw_caps.get("dpas_baseline_probe", {}),
        "compiler_flags_probe": verified_hw_caps.get("compiler_flags_probe", {}),
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
