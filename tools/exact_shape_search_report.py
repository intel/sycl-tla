#!/usr/bin/env python3
"""Summarize remote exact-shape search results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterable


MERGED_FIELDS = [
    "shape_tag",
    "kernel",
    "tflops",
    "avg_runtime_ms",
    "total_runtime_ms",
    "measure_iters",
    "warmup_iters",
    "latency_source",
    "status",
    "gpu",
    "m",
    "n",
    "k",
    "layout",
    "runner",
    "scheduler_family",
    "decomposition_mode",
    "streamk_mode",
    "reduction_mode",
    "tile_m",
    "tile_n",
    "tile_k",
    "sg_m",
    "sg_n",
    "stages",
    "split_k",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_d",
    "dtype_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exact-shape search results.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by remote_exact_shape_search.sh")
    parser.add_argument("--shape-tag", default="", help="Optional shape tag like 8192_384_3584. Defaults to all shapes.")
    parser.add_argument("--output-dir", default="", help="Optional report output directory. Defaults under run_dir/reports.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_shape_tags(run_dir: Path, requested_shapes: Path, explicit_shape_tag: str) -> list[str]:
    if explicit_shape_tag:
        return [explicit_shape_tag]
    if requested_shapes.exists():
        payload = load_json(requested_shapes)
        return [f"{shape['m']}_{shape['n']}_{shape['k']}" for shape in payload.get("shapes", [])]
    results_dir = run_dir / "results"
    return sorted(path.name for path in results_dir.iterdir() if path.is_dir())


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def derive_latency_fields(row: dict) -> dict:
    avg_runtime_ms = safe_float(row.get("avg_runtime_ms"))
    total_runtime_ms = safe_float(row.get("total_runtime_ms"))
    measure_iters = safe_int(row.get("measure_iters"))
    warmup_iters = safe_int(row.get("warmup_iters"))

    if avg_runtime_ms is not None or total_runtime_ms is not None:
        if measure_iters is None and avg_runtime_ms is not None and total_runtime_ms is not None and avg_runtime_ms > 0:
            measure_iters = int(round(total_runtime_ms / avg_runtime_ms))
        return {
            "avg_runtime_ms": "" if avg_runtime_ms is None else f"{avg_runtime_ms:.6f}",
            "total_runtime_ms": "" if total_runtime_ms is None else f"{total_runtime_ms:.6f}",
            "measure_iters": "" if measure_iters is None else str(measure_iters),
            "warmup_iters": "" if warmup_iters is None else str(warmup_iters),
            "latency_source": row.get("latency_source") or "reported",
        }

    tflops = safe_float(row.get("tflops"))
    m = safe_int(row.get("m"))
    n = safe_int(row.get("n"))
    k = safe_int(row.get("k"))
    if tflops is None or tflops <= 0 or m is None or n is None or k is None:
        return {
            "avg_runtime_ms": "",
            "total_runtime_ms": "",
            "measure_iters": "",
            "warmup_iters": "",
            "latency_source": "",
        }

    measure_iters = measure_iters or 100
    warmup_iters = warmup_iters or 100
    total_flops = 2.0 * float(m) * float(n) * float(k)
    avg_runtime_ms = (total_flops / (tflops * 1.0e12)) * 1.0e3
    total_runtime_ms = avg_runtime_ms * float(measure_iters)
    return {
        "avg_runtime_ms": f"{avg_runtime_ms:.6f}",
        "total_runtime_ms": f"{total_runtime_ms:.6f}",
        "measure_iters": str(measure_iters),
        "warmup_iters": str(warmup_iters),
        "latency_source": "derived_from_tflops",
    }


def read_rows(csv_paths: Iterable[Path], kernel_metadata: dict[str, dict], shape_tag: str) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(csv_paths):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                kernel = row.get("kernel", "")
                merged = {
                    "shape_tag": shape_tag,
                    "kernel": kernel,
                    "tflops": row.get("tflops", ""),
                    "status": row.get("status", ""),
                    "gpu": row.get("gpu", ""),
                    "m": row.get("m", ""),
                    "n": row.get("n", ""),
                    "k": row.get("k", ""),
                    "avg_runtime_ms": row.get("avg_runtime_ms", ""),
                    "total_runtime_ms": row.get("total_runtime_ms", ""),
                    "measure_iters": row.get("measure_iters", ""),
                    "warmup_iters": row.get("warmup_iters", ""),
                    "latency_source": row.get("latency_source", ""),
                }
                merged.update(kernel_metadata.get(kernel, {}))
                merged.update(derive_latency_fields(merged))
                rows.append(merged)
    return rows


def ok_rows(rows: Iterable[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        if row.get("status") != "OK":
            continue
        tflops = safe_float(row.get("tflops"))
        if tflops is None:
            continue
        avg_runtime_ms = safe_float(row.get("avg_runtime_ms"))
        total_runtime_ms = safe_float(row.get("total_runtime_ms"))
        ranked.append(
            {
                **row,
                "_tflops": tflops,
                "_avg_runtime_ms": avg_runtime_ms,
                "_total_runtime_ms": total_runtime_ms,
            }
        )
    return ranked


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MERGED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MERGED_FIELDS})


def trim_rank_rows(rows: list[dict]) -> list[dict]:
    trimmed = []
    for row in rows:
        copy = {field: row.get(field, "") for field in MERGED_FIELDS}
        trimmed.append(copy)
    return trimmed


def summarize_numeric(rows: list[dict], field: str) -> dict:
    values = [safe_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {}
    return {
        "min": round(min(values), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "max": round(max(values), 6),
    }


def summarize_shape(run_dir: Path, shape_tag: str, output_dir: Path) -> dict:
    kernel_metadata_path = run_dir / "kernel_metadata.json"
    kernel_metadata = load_json(kernel_metadata_path) if kernel_metadata_path.exists() else {}
    rows = read_rows((run_dir / "results" / shape_tag).glob("*.csv"), kernel_metadata, shape_tag)
    status_counts = Counter(row.get("status", "") for row in rows)
    ranked_ok = ok_rows(rows)
    ranked_desc = sorted(ranked_ok, key=lambda row: row["_tflops"], reverse=True)
    ranked_asc = sorted(ranked_ok, key=lambda row: row["_tflops"])
    ranked_latency = sorted(
        [row for row in ranked_ok if row["_total_runtime_ms"] is not None],
        key=lambda row: row["_total_runtime_ms"],
    )
    ranked_latency_desc = list(reversed(ranked_latency))
    ranked_rcr = [row for row in ranked_desc if row.get("layout") == "rcr"]
    ranked_rcr_latency = [row for row in ranked_latency if row.get("layout") == "rcr"]

    merged_csv = output_dir / "merged_results.csv"
    ranked_by_tflops_csv = output_dir / "ranked_by_tflops.csv"
    ranked_by_total_runtime_csv = output_dir / "ranked_by_total_runtime.csv"
    top5_csv = output_dir / "top5.csv"
    worst5_csv = output_dir / "worst5.csv"
    top5_rcr_csv = output_dir / "top5_rcr.csv"
    fastest5_latency_csv = output_dir / "fastest5_latency.csv"
    slowest5_latency_csv = output_dir / "slowest5_latency.csv"
    fastest5_rcr_latency_csv = output_dir / "fastest5_rcr_latency.csv"
    write_csv(merged_csv, rows)
    write_csv(ranked_by_tflops_csv, trim_rank_rows(ranked_desc))
    write_csv(ranked_by_total_runtime_csv, trim_rank_rows(ranked_latency))
    write_csv(top5_csv, trim_rank_rows(ranked_desc[:5]))
    write_csv(worst5_csv, trim_rank_rows(ranked_asc[:5]))
    write_csv(top5_rcr_csv, trim_rank_rows(ranked_rcr[:5]))
    write_csv(fastest5_latency_csv, trim_rank_rows(ranked_latency[:5]))
    write_csv(slowest5_latency_csv, trim_rank_rows(ranked_latency_desc[:5]))
    write_csv(fastest5_rcr_latency_csv, trim_rank_rows(ranked_rcr_latency[:5]))

    summary = {
        "shape_tag": shape_tag,
        "row_count": len(rows),
        "ok_row_count": len(ranked_ok),
        "status_counts": dict(sorted(status_counts.items())),
        "latency_stats": {
            "avg_runtime_ms": summarize_numeric(ranked_ok, "avg_runtime_ms"),
            "total_runtime_ms": summarize_numeric(ranked_ok, "total_runtime_ms"),
        },
        "kernel_metadata_path": str(kernel_metadata_path),
        "merged_results_csv": str(merged_csv),
        "ranked_by_tflops_csv": str(ranked_by_tflops_csv),
        "ranked_by_total_runtime_csv": str(ranked_by_total_runtime_csv),
        "top5_csv": str(top5_csv),
        "worst5_csv": str(worst5_csv),
        "top5_rcr_csv": str(top5_rcr_csv),
        "fastest5_latency_csv": str(fastest5_latency_csv),
        "slowest5_latency_csv": str(slowest5_latency_csv),
        "fastest5_rcr_latency_csv": str(fastest5_rcr_latency_csv),
        "top5": trim_rank_rows(ranked_desc[:5]),
        "worst5": trim_rank_rows(ranked_asc[:5]),
        "top5_rcr": trim_rank_rows(ranked_rcr[:5]),
        "fastest5_latency": trim_rank_rows(ranked_latency[:5]),
        "slowest5_latency": trim_rank_rows(ranked_latency_desc[:5]),
        "fastest5_rcr_latency": trim_rank_rows(ranked_rcr_latency[:5]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    requested_shapes = run_dir / "requested_shapes.json"
    shape_tags = iter_shape_tags(run_dir, requested_shapes, args.shape_tag)
    if not shape_tags:
        raise SystemExit(f"No shape tags found in {run_dir}")

    base_output = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "reports")
    summaries = []
    for shape_tag in shape_tags:
        result_dir = run_dir / "results" / shape_tag
        if not result_dir.is_dir():
            raise SystemExit(f"Result dir not found for shape {shape_tag}: {result_dir}")
        output_dir = base_output / shape_tag
        summaries.append(summarize_shape(run_dir, shape_tag, output_dir))

    print(json.dumps({"run_dir": str(run_dir), "shape_summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
