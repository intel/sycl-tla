#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

# Intel GEMM Profiler

Intel GEMM Profiler is the repository's Python-side orchestration layer for Intel/BMG GEMM search, screening, confirmation, exact-shape dispatch selection, and product-style artifact export.

It works together with:

- `test/benchmarks/intel_gemm_profiler.py`: compatibility CLI entrypoint
- `test/benchmarks/intel_gemm_profiler/`: profiler package implementation
- `tools/remote_exact_shape_search.sh`: remote exact-shape launcher
- `tools/remote_exact_shape_search_ctl.py`: local remote-control wrapper
- `tools/exact_shape_search_report.py`: exact-shape result merger and ranking generator

## What it covers

- old search standards kept reproducible:
  - `baseline`
  - `expanded_bmg`
  - `layered_exhaustive`
- new additive scheduler search:
  - `bruteforce_scheduler`
  - `layered_bmg_scheduler_expanded`
- benchmark-backed GEMM screening and confirmation
- candidate build manifest / preflight batch routing
- dispatch-table generation for exact shapes
- remote exact-shape batch search on B70/BMG
- merged reporting with TFLOPS and latency rankings

## Main artifacts

Typical workflow output lives under `<workspace>/reports/`:

- `kernel_catalog.json`
- `gemm_candidate_space.json`
- `candidate_coverage_report.json`
- `candidate_build_manifest.json`
- `candidate_build_plan.json`
- `candidate_build_summary.json`
- `candidate_build_preflight_summary.json`
- `gemm_profile_results.csv`
- `gemm_dispatch_table.json`
- `optimal_dispatch_table.json`
- `phase_a_summary.json`
- `phase_b_summary.json`
- `run_summary.json`
- `gemm_product_bundle_manifest.json`

Remote exact-shape output lives under `<run_dir>/reports/<shape_tag>/`:

- `merged_results.csv`
- `ranked_by_tflops.csv`
- `ranked_by_total_runtime.csv`
- `top5.csv`
- `worst5.csv`
- `top5_rcr.csv`
- `fastest5_latency.csv`
- `slowest5_latency.csv`
- `fastest5_rcr_latency.csv`
- `summary.json`

## Quick start

### 1. Local skip-run smoke

```bash
python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /tmp/profiler_smoke \
  --dtype bf16 \
  --search-strategy layered_exhaustive \
  --kernel-catalog-source layered_bmg \
  --max-shapes 1 \
  --skip-run
```

### 2. Scheduler-expanded smoke

```bash
python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /tmp/profiler_scheduler_smoke \
  --dtype bf16 \
  --bruteforce-scheduler-search \
  --max-shapes 1 \
  --skip-run
```

### 2.1 B70 scheduler-bootstrap SG filter

Use the external constraints JSON below when you want a narrow scheduler bootstrap band
that keeps only `SG=2x8` and `SG=8x2` while leaving the rest of the current BMG tile and
stage ranges intact:

```bash
python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /tmp/profiler_scheduler_bootstrap \
  --dtype bf16 \
  --bruteforce-scheduler-search \
  --constraints-json test/benchmarks/constraints_b70_scheduler_bootstrap.json \
  --max-shapes 1 \
  --skip-run
```

This file is intended as a **scheduler bootstrap** filter, not as the repository-wide B70
default. Large 4k-style Ali shapes still show strong `SG=8x4` winners, so use this config
when you want a narrow starting band rather than a fully validated general-purpose filter.

### 3. Remote exact-shape run

```bash
export EXACT_SHAPE_REMOTE_PASSWORD='***'

python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key sync
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key launch \
  --run-id shape_search_example \
  --shapes 8192x384x3584 \
  --layouts rcr,rrr \
  --kernel-catalog-source layered_bmg_scheduler_expanded \
  --batch-size 1 \
  --gpu-ids 0,1,2,3,4,5,6,7
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key status
```

### 4. Generate exact-shape report

```bash
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key report \
  --run-dir /root/.../shape_search_example \
  --shape-tag 8192_384_3584
```

### 5. Remote sampled profiler validation

Use this flow when you want to validate that the current profiler path still works on a
specific GPU with:

- the latest `bruteforce_scheduler` search logic
- preflight batch builds
- aggregate candidate build
- sampled regular GEMM and scheduler cases

This is the exact style used to validate the compile-scheduling change on GPUs `4` and `6`
for shape `8192x76032x8192`.

#### 5.1 Prepare a shared benchmark build cache on the remote host

The profiler's fresh workspace build must see a usable Google Benchmark build directory.
Before running sampled validation, make sure the remote path passed through
`--googlebenchmark-build-dir` contains:

- `_deps/googlebenchmark-build/src/libbenchmark.a`

If you do not already have a known-good build tree, create one once on the remote host:

```bash
cd /root/cutlass_profile_device7_b70_2500mhz/sycl-tla
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
export CC=icx
export CXX=icpx

cmake -S . -B /root/cutlass_profile_device7_b70_2500mhz/shared_benchmark_cache \
  -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
  -DDPCPP_HOST_COMPILER=g++-13 \
  -DCUTLASS_ENABLE_BENCHMARKS=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /root/cutlass_profile_device7_b70_2500mhz/shared_benchmark_cache \
  --target cutlass_benchmarks_gemm_sycl \
  --parallel 64
```

Then use:

```text
/root/cutlass_profile_device7_b70_2500mhz/shared_benchmark_cache/_deps/googlebenchmark-build
```

as the value of `--googlebenchmark-build-dir`.

#### 5.2 Generate the sampled validation inputs locally

The sampled validation uses two small input files:

1. a single-shape `shapes.json`
2. a `compiled-kernel-list` containing a representative regular/scheduler sample

Generate both from the current catalog:

```bash
python3 - <<'PY'
from pathlib import Path
import json
import sys

repo = Path("/path/to/sycl-tla")
sys.path.insert(0, str(repo / "test/benchmarks"))
import intel_gemm_profiler as p

out_dir = repo / "out" / "remote_profiler_validation"
out_dir.mkdir(parents=True, exist_ok=True)

shape_doc = {
    "schema_version": p.SCHEMA_VERSION,
    "generated_at": p.now_iso(),
    "shape_set_id": "remote_gpu46_8192_76032_8192",
    "source": "manual_remote_validation",
    "shapes": [
        {
            "shape_id": "rcr_bf16_8192_76032_8192",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_d": "f32",
            "dtype_acc": "f32",
            "m": 8192,
            "n": 76032,
            "k": 8192,
        },
        {
            "shape_id": "rrr_bf16_8192_76032_8192",
            "layout": "rrr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_d": "f32",
            "dtype_acc": "f32",
            "m": 8192,
            "n": 76032,
            "k": 8192,
        },
    ],
}
(out_dir / "shape_8192_76032_8192.json").write_text(json.dumps(shape_doc, indent=2) + "\n", encoding="utf-8")

space = p.generate_candidate_space(
    shape_doc,
    p.default_constraints(),
    p.default_compiler_profiles(),
    allowed_runners=("benchmark",),
    catalog_source="layered_bmg_scheduler_expanded",
    prefilter_strategy="none",
)
bench = [c for c in space["candidates"] if c.get("runner", "benchmark") == "benchmark"]

def pick(**want):
    for c in bench:
        if all(c.get(k) == v for k, v in want.items()):
            return c["kernel_id"]
    raise RuntimeError(f"sample kernel not found for {want}")

kernel_ids = [
    pick(layout="rcr", streamk_mode="", sg_m=8, sg_n=2, stages=1),
    pick(layout="rcr", streamk_mode="", sg_m=4, sg_n=4, stages=2),
    pick(layout="rcr", streamk_mode="", sg_m=2, sg_n=8, stages=3),
    pick(layout="rrr", streamk_mode="", sg_m=8, sg_n=2, stages=1),
    pick(layout="rrr", streamk_mode="", sg_m=4, sg_n=4, stages=2),
    pick(layout="rrr", streamk_mode="", sg_m=2, sg_n=8, stages=3),
    pick(layout="rcr", streamk_mode="streamk", sg_m=8, sg_n=4, stages=2),
    pick(layout="rcr", streamk_mode="data_parallel", sg_m=4, sg_n=8, stages=2),
    pick(layout="rcr", streamk_mode="splitk", sg_m=8, sg_n=2, stages=3),
    pick(layout="rrr", streamk_mode="streamk", sg_m=8, sg_n=2, stages=1),
    pick(layout="rrr", streamk_mode="data_parallel", sg_m=2, sg_n=8, stages=3),
    pick(layout="rrr", streamk_mode="splitk", sg_m=4, sg_n=4, stages=2),
]

(out_dir / "sample_kernels_8192_76032_8192.list").write_text(
    "\n".join(f"^{kernel_id}$" for kernel_id in kernel_ids) + "\n",
    encoding="utf-8",
)

print(out_dir / "shape_8192_76032_8192.json")
print(out_dir / "sample_kernels_8192_76032_8192.list")
PY
```

#### 5.3 Copy the two input files to the remote host

```bash
scp out/remote_profiler_validation/shape_8192_76032_8192.json \
  root@10.239.11.149:/root/cutlass_profile_device7_b70_2500mhz/validation_inputs/
scp out/remote_profiler_validation/sample_kernels_8192_76032_8192.list \
  root@10.239.11.149:/root/cutlass_profile_device7_b70_2500mhz/validation_inputs/
```

#### 5.4 Run the sampled validation on one GPU

Example for GPU `4`:

```bash
cd /root/cutlass_profile_device7_b70_2500mhz/sycl-tla
export ZE_AFFINITY_MASK=4

python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /root/cutlass_profile_device7_b70_2500mhz/validation_runs/profiler_gpu4_8192_76032_8192_sample \
  --dtype bf16 \
  --shapes-json /root/cutlass_profile_device7_b70_2500mhz/validation_inputs/shape_8192_76032_8192.json \
  --probe-mode off \
  --search-strategy bruteforce_scheduler \
  --compiled-kernel-list /root/cutlass_profile_device7_b70_2500mhz/validation_inputs/sample_kernels_8192_76032_8192.list \
  --candidate-build-batch-size 3 \
  --candidate-build-parallelism 2 \
  --run-candidate-build-preflight \
  --use-candidate-build-preflight-benchmarks \
  --build-candidate-benchmark \
  --top-k 4 \
  --confirm-runs 1 \
  --benchmark-entry-chunk-size 4 \
  --cmake-source-dir /root/cutlass_profile_device7_b70_2500mhz/sycl-tla \
  --benchmark-build-dir /root/cutlass_profile_device7_b70_2500mhz/validation_runs/profiler_gpu4_8192_76032_8192_sample/build/candidate_benchmarks \
  --googlebenchmark-dir /root/cutlass_profile_device7_b70_2500mhz/sycl-tla/_deps/googlebenchmark-src \
  --googlebenchmark-build-dir /root/cutlass_profile_device7_b70_2500mhz/shared_benchmark_cache/_deps/googlebenchmark-build \
  --cmake-cxx-compiler icpx \
  --shell-init 'source /opt/intel/oneapi/compiler/2025.3/env/vars.sh' \
  --cwd /root/cutlass_profile_device7_b70_2500mhz/sycl-tla \
  --timeout 1800 \
  --build-timeout 1800
```

Repeat on GPU `6` by changing:

```text
ZE_AFFINITY_MASK=6
--workspace /root/.../profiler_gpu6_8192_76032_8192_sample
```

#### 5.5 What files to inspect after the run

Under `<workspace>/reports/` the main checkpoints are:

- `candidate_build_plan.json`
- `candidate_build_preflight_summary.json`
- `candidate_build_summary.json`
- `scheduler_bruteforce_plan.json`
- `gemm_profile_results.csv`
- `phase_b_summary.json`
- `run_summary.json`

Use these checks first:

```bash
python3 - <<'PY'
from pathlib import Path
import json

reports = Path("/root/cutlass_profile_device7_b70_2500mhz/validation_runs/profiler_gpu4_8192_76032_8192_sample/reports")
plan = json.loads((reports / "candidate_build_plan.json").read_text())
pre = json.loads((reports / "candidate_build_preflight_summary.json").read_text())
agg = json.loads((reports / "candidate_build_summary.json").read_text())

print({
    "aggregate_build_parallelism": plan["build_parallelism"],
    "preflight_build_parallelism": plan["batch_build_parallelism"],
    "preflight_status": pre["status"],
    "preflight_passed_batches": pre["passed_batches"],
    "aggregate_build_status": agg["status"],
    "selected_kernel_count": agg["selected_kernel_count"],
})
PY
```

#### 5.6 How to analyze the sampled results

Summarize pass/fail coverage by `layout × streamk_mode`:

```bash
python3 - <<'PY'
from pathlib import Path
import csv
from collections import Counter

csv_path = Path("/root/cutlass_profile_device7_b70_2500mhz/validation_runs/profiler_gpu4_8192_76032_8192_sample/reports/gemm_profile_results.csv")
rows = list(csv.DictReader(csv_path.open()))
counts = Counter((row["layout"], row["streamk_mode"] or "<regular>", row["status"]) for row in rows)
for key, value in sorted(counts.items()):
    print(key, value)
PY
```

List the top kernel per `layout × streamk_mode` bucket:

```bash
python3 - <<'PY'
from pathlib import Path
import csv

csv_path = Path("/root/cutlass_profile_device7_b70_2500mhz/validation_runs/profiler_gpu4_8192_76032_8192_sample/reports/gemm_profile_results.csv")
rows = [row for row in csv.DictReader(csv_path.open()) if row["status"] == "pass"]
buckets = {}
for row in rows:
    key = (row["layout"], row["streamk_mode"] or "<regular>")
    score = float(row["avg_tflops"] or 0.0)
    if key not in buckets or score > buckets[key][0]:
        buckets[key] = (score, row["kernel_id"], row["candidate_id"])
for key, value in sorted(buckets.items()):
    print({"layout": key[0], "streamk_mode": key[1], "avg_tflops": value[0], "kernel_id": value[1], "candidate_id": value[2]})
PY
```

For the current 12-kernel sample and `top-k=4`, `confirm-runs=1`, the expected healthy output is
roughly:

- build stage: preflight pass, aggregate build pass
- result rows: around `20`
- both regular GEMM and scheduler modes present
- both `rcr` and `rrr` present
- no non-pass statuses in the sampled buckets

## Current exact-shape reporting behavior

- future runs write latency fields directly into per-batch CSV:
  - `avg_runtime_ms`
  - `total_runtime_ms`
  - `measure_iters`
  - `warmup_iters`
- old runs that only recorded `tflops` are still supported:
  - report generation backfills latency from `m*n*k*l` and measured TFLOPS
  - those rows are marked with `latency_source=derived_from_tflops`

## Validation status for this delivery

The current delivery was checked with:

- `python3 test/python/cutlass/test_intel_gemm_profiler.py`
- `python3 test/python/cutlass/test_exact_shape_search_report.py`

## Scheduler brute-force implementation notes

The repository now emits `reports/scheduler_bruteforce_plan.json` for profiler runs. This
plan makes the scheduler brute-force configuration explicit:

- whether the run is using the `bruteforce_scheduler` profile
- the effective `kernel_catalog_source`
- preflight/per-batch benchmark routing
- candidate counts for the preserved regular GEMM space and the widened BF16 scheduler space
- scheduler search axes (`layout`, `streamk_mode`, `sg`, `stages`)

It also emits:

- `reports/regular_gemm_full_config.csv` — deduplicated full regular GEMM config list
- `reports/regular_gemm_gap_scan.json` — duplicate-removal and exhaustive-coverage scan for regular GEMM
- `reports/scheduler_bruteforce_full_config.csv` — deduplicated full scheduler config list
- `reports/scheduler_bruteforce_gap_scan.json` — duplicate-removal and missing-mode scan

Use these two files to audit:

- whether repeated regular GEMM configs were removed
- whether the current regular GEMM exhaustive space is missing legal tile/sg/stage combinations
- whether repeated scheduler configs were removed
- whether each base compile-time config has all three scheduler modes
- whether the current brute-force search still has obvious completeness gaps

For the intended full scheduler brute-force path, the effective configuration is:

- `--search-strategy bruteforce_scheduler` or `--bruteforce-scheduler-search`
- `--kernel-catalog-source layered_bmg_scheduler_expanded`
- `--prefilter none`
- `--candidate-build-batch-size 1`
- `--candidate-build-parallelism N` with each batch build auto-capped to roughly `host_vcpus / N` compile jobs
- `--run-candidate-build-preflight`
- `--use-candidate-build-preflight-benchmarks`
- `python3 test/benchmarks/intel_gemm_profiler.py --max-shapes 1 --skip-run ...` for layered exhaustive smoke
- `python3 test/benchmarks/intel_gemm_profiler.py --max-shapes 1 --skip-run --bruteforce-scheduler-search` for scheduler-expanded smoke

For an operator-oriented step-by-step procedure, see `OPERATION_MANUAL.md` in the same directory.
