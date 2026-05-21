#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime

from .hw_specs import analyze_efficiency
from .schemas import CSV_FIELDS, REPORT_TRACKED_DIMENSIONS, SCHEMA_VERSION
from .utils import now_iso


def write_results_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def median_or_nan(values):
    numeric = numeric_values(values)
    if not numeric:
        return math.nan
    return statistics.median(numeric)


def numeric_values(values):
    numeric = []
    for value in values:
        if str(value) == "":
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            numeric.append(parsed)
    return numeric


def pstdev_or_nan(values):
    numeric = numeric_values(values)
    if len(numeric) < 2:
        return math.nan
    return statistics.pstdev(numeric)


def round_or_empty(value):
    return round(value, 6) if not math.isnan(value) else ""


def tracked_metadata(source):
    return {field: source.get(field, "") for field in REPORT_TRACKED_DIMENSIONS if field in source}


def summarize_dimension_values(items):
    items = list(items)
    summary = {}
    for field in REPORT_TRACKED_DIMENSIONS:
        counts = defaultdict(int)
        for item in items:
            value = item.get(field, "")
            if value == "":
                value = "<empty>"
            counts[str(value)] += 1
        if counts:
            summary[field] = {
                "unique_count": len(counts),
                "values": dict(sorted(counts.items())),
            }
    return summary


def build_candidate_coverage_report(candidate_space):
    candidates = candidate_space.get("candidates", [])
    exceptions = candidate_space.get("candidate_exceptions", [])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "candidate_count": len(candidates),
        "candidate_exception_count": len(exceptions),
        "accepted_dimension_values": summarize_dimension_values(candidates),
        "exception_dimension_values": summarize_dimension_values(exceptions),
        "exception_reasons": candidate_space.get("candidate_exception_summary", []),
    }


def build_dispatch_table(rows, shapes_doc, top_k, confirm_runs, close_call_threshold, candidate_space=None, hw_spec=None, low_efficiency_threshold=0.4):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "pass" and row["verify_status"] == "pass":
            grouped[row["shape_id"]].append(row)
    shapes = {shape["shape_id"]: shape for shape in shapes_doc["shapes"]}
    candidates = {candidate["candidate_id"]: candidate for candidate in candidate_space.get("candidates", [])} if candidate_space and hw_spec else {}
    entries = []
    for shape_id, shape_rows in grouped.items():
        screening_rows = [row for row in shape_rows if row["stage"] == "screening"]
        screening_ranked = sorted(
            screening_rows,
            key=lambda row: float(row["avg_tflops"] or 0.0),
            reverse=True,
        )
        screening_rank_by_candidate = {
            row["candidate_id"]: index + 1 for index, row in enumerate(screening_ranked)
        }
        confirm_rows = [row for row in shape_rows if row["stage"] == "confirm"]
        selection_rows = confirm_rows if confirm_rows else [row for row in shape_rows if row["stage"] == "screening"]
        selection_stage = "confirm" if confirm_rows else "screening"
        by_candidate = defaultdict(list)
        for row in selection_rows:
            by_candidate[row["candidate_id"]].append(row)
        ranked = []
        for candidate_id, candidate_rows in by_candidate.items():
            median_tflops = median_or_nan(row["avg_tflops"] for row in candidate_rows)
            median_runtime_ms = median_or_nan(row["avg_runtime_ms"] for row in candidate_rows)
            if math.isnan(median_tflops):
                continue
            tflops_stdev = pstdev_or_nan(row["avg_tflops"] for row in candidate_rows)
            runtime_stdev = pstdev_or_nan(row["avg_runtime_ms"] for row in candidate_rows)
            ranked.append(
                {
                    "candidate_id": candidate_id,
                    "compiler_profile_id": candidate_rows[0]["compiler_profile_id"],
                    "median_tflops": median_tflops,
                    "median_runtime_ms": median_runtime_ms,
                    "tflops_stdev": tflops_stdev,
                    "runtime_stdev": runtime_stdev,
                    "tflops_cv_percent": (tflops_stdev / median_tflops) * 100.0 if median_tflops > 0 and not math.isnan(tflops_stdev) else math.nan,
                    "samples": len(candidate_rows),
                    "expected_samples": confirm_runs if selection_stage == "confirm" else 1,
                    "screening_rank": screening_rank_by_candidate.get(candidate_id, ""),
                }
            )
        ranked.sort(key=lambda item: item["median_tflops"], reverse=True)
        if not ranked:
            continue
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        winner_row = by_candidate[winner["candidate_id"]][0]
        runner_up_row = by_candidate[runner_up["candidate_id"]][0] if runner_up else None
        gap = None
        close_call = False
        if runner_up and runner_up["median_tflops"] > 0:
            gap = ((winner["median_tflops"] - runner_up["median_tflops"]) / runner_up["median_tflops"]) * 100.0
            close_call = gap < close_call_threshold
        shape = shapes[shape_id]
        shape_key = {
            "layout": shape["layout"],
            "dtype_a": shape["dtype_a"],
            "dtype_b": shape["dtype_b"],
            "dtype_c": shape["dtype_c"],
            "dtype_d": shape.get("dtype_d", shape["dtype_c"]),
            "dtype_acc": shape["dtype_acc"],
            "m": shape["m"],
            "n": shape["n"],
            "k": shape["k"],
            "batch_count": shape.get("batch_count", 1),
        }
        selected_efficiency = ""
        peak_tflops = ""
        efficiency_warning = ""
        if hw_spec and winner["candidate_id"] in candidates:
            analysis = analyze_efficiency(shape, candidates[winner["candidate_id"]], hw_spec)
            peak_tflops = round(analysis["peak_tflops"], 6)
            if analysis["peak_tflops"] > 0:
                selected_efficiency = round(winner["median_tflops"] / analysis["peak_tflops"], 6)
                if (
                    analysis["is_compute_bound"]
                    and selected_efficiency < low_efficiency_threshold
                    and selected_efficiency < analysis["min_expected_efficiency"]
                ):
                    efficiency_warning = (
                        f"winner_efficiency_below_{int(low_efficiency_threshold * 100)}pct_peak"
                    )
        entries.append(
            {
                "shape_key": shape_key,
                "shape_id": shape_id,
                "candidate_id": winner["candidate_id"],
                "compiler_profile_id": winner["compiler_profile_id"],
                "status": "pass",
                "selected_metric": round(winner["median_tflops"], 6),
                "selected_efficiency": selected_efficiency,
                "peak_tflops": peak_tflops,
                "efficiency_warning": efficiency_warning,
                "selected_candidate_metadata": tracked_metadata(winner_row),
                "runner_up_candidate_id": runner_up["candidate_id"] if runner_up else "",
                "runner_up_candidate_metadata": tracked_metadata(runner_up_row) if runner_up_row else {},
                "runner_up_gap_percent": round(gap, 6) if gap is not None else "",
                "close_call": close_call,
                "evidence": {
                    "selection_stage": selection_stage,
                    "selection_metric": "median_tflops" if selection_stage == "confirm" else "avg_tflops",
                    "confirm_median_runtime_ms": round_or_empty(winner["median_runtime_ms"]),
                    "confirm_median_tflops": round_or_empty(winner["median_tflops"]),
                    "confirm_runtime_stdev_ms": round_or_empty(winner["runtime_stdev"]),
                    "confirm_tflops_stdev": round_or_empty(winner["tflops_stdev"]),
                    "confirm_tflops_cv_percent": round_or_empty(winner["tflops_cv_percent"]),
                    "screening_rank": winner["screening_rank"],
                    "confirm_samples": winner["samples"] if selection_stage == "confirm" else 0,
                    "expected_confirm_samples": confirm_runs if selection_stage == "confirm" else 0,
                    "confirm_complete": selection_stage != "confirm" or winner["samples"] >= confirm_runs,
                    "runner_up_median_tflops": round_or_empty(runner_up["median_tflops"]) if runner_up else "",
                    "runner_up_median_runtime_ms": round_or_empty(runner_up["median_runtime_ms"]) if runner_up else "",
                    "runner_up_samples": runner_up["samples"] if runner_up else 0,
                    "runner_up_screening_rank": runner_up["screening_rank"] if runner_up else "",
                    "ranked_candidates": [
                        {
                            "candidate_id": item["candidate_id"],
                            "candidate_metadata": tracked_metadata(by_candidate[item["candidate_id"]][0]),
                            "median_tflops": round_or_empty(item["median_tflops"]),
                            "median_runtime_ms": round_or_empty(item["median_runtime_ms"]),
                            "samples": item["samples"],
                            "screening_rank": item["screening_rank"],
                            "confirm_complete": selection_stage != "confirm" or item["samples"] >= confirm_runs,
                        }
                        for item in ranked[:top_k]
                    ],
                },
            }
        )
    close_calls = sum(1 for entry in entries if entry["close_call"])
    entries_with_confirm = [entry for entry in entries if entry["evidence"]["selection_stage"] == "confirm"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "dispatch_id": f"intel_gemm_profiler_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "selection_policy": {"screening_top_k": top_k, "confirm_runs": confirm_runs, "metric": "confirm_median_tflops" if confirm_runs else "screening_avg_tflops", "close_call_threshold_percent": close_call_threshold},
        "selection_summary": {
            "entries": len(entries),
            "entries_with_confirmation": len(entries_with_confirm),
            "incomplete_confirmation_entries": sum(
                1 for entry in entries_with_confirm if not entry["evidence"]["confirm_complete"]
            ),
            "close_calls": close_calls,
        },
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
        "hw_reference_spec_id": verified_hw_caps.get("hw_reference_spec_id", ""),
        "constraint_source": constraints["constraint_source"],
        "dpas_baseline_probe": verified_hw_caps.get("dpas_baseline_probe", {}),
        "compiler_flags_probe": verified_hw_caps.get("compiler_flags_probe", {}),
        "anomaly_report": verified_hw_caps.get("anomaly_report", {"anomalies": [], "auto_block_rules": []}),
        "probe_results": len(probe_rows),
        "successful_probe_results": sum(1 for row in probe_rows if row["status"] == "pass"),
        "allowed_values": constraints["allowed_values"],
        "limits": constraints["limits"],
        "blocked_rules": constraints.get("blocked_rules", []),
        "probe_feedback": constraints.get("probe_feedback", {}),
    }


def build_phase_b_summary(candidate_space, dispatch_table, summary):
    low_efficiency_warnings = [
        {
            "shape_id": entry["shape_id"],
            "candidate_id": entry["candidate_id"],
            "selected_efficiency": entry["selected_efficiency"],
            "peak_tflops": entry["peak_tflops"],
            "warning": entry["efficiency_warning"],
        }
        for entry in dispatch_table["entries"]
        if entry.get("efficiency_warning")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "candidate_count": len(candidate_space["candidates"]),
        "catalog_version": candidate_space.get("kernel_catalog", {}).get("catalog_version", ""),
        "dispatch_entries": len(dispatch_table["entries"]),
        "rows": summary["rows"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "candidate_dimension_coverage": build_candidate_coverage_report(candidate_space),
        "selected_dimension_values": summarize_dimension_values(
            [entry.get("selected_candidate_metadata", {}) for entry in dispatch_table["entries"]]
        ),
        "low_efficiency_warnings": low_efficiency_warnings,
    }


def build_reference_comparison(dispatch_table, reference_doc):
    reference_by_shape = {entry["shape_id"]: entry for entry in reference_doc.get("entries", []) if entry.get("supported", True)}
    dispatch_by_shape = {entry["shape_id"]: entry for entry in dispatch_table.get("entries", [])}
    entries = []
    matched = 0
    missing_dispatch = 0
    for shape_id, reference in reference_by_shape.items():
        dispatch_entry = dispatch_by_shape.get(shape_id)
        if dispatch_entry is None:
            missing_dispatch += 1
            entries.append(
                {
                    "shape_id": shape_id,
                    "reference_provider": reference["reference_provider"],
                    "reference_tflops": reference["reference_tflops"],
                    "selected_candidate_id": "",
                    "selected_candidate_metadata": {},
                    "selected_tflops": "",
                    "selected_vs_reference_ratio": "",
                    "status": "missing_dispatch",
                }
            )
            continue
        matched += 1
        selected_tflops = dispatch_entry["selected_metric"]
        reference_tflops = reference["reference_tflops"]
        ratio = round(selected_tflops / reference_tflops, 6) if reference_tflops else ""
        entries.append(
            {
                "shape_id": shape_id,
                "reference_provider": reference["reference_provider"],
                "reference_tflops": reference_tflops,
                "selected_candidate_id": dispatch_entry["candidate_id"],
                "selected_candidate_metadata": dispatch_entry.get("selected_candidate_metadata", {}),
                "selected_tflops": selected_tflops,
                "selected_vs_reference_ratio": ratio,
                "status": "matched",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "dataset_id": reference_doc.get("dataset_id", ""),
        "summary": {
            "reference_entries": len(reference_by_shape),
            "matched": matched,
            "missing_dispatch": missing_dispatch,
        },
        "entries": entries,
    }
