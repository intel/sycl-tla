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
- `python3 test/benchmarks/intel_gemm_profiler.py --max-shapes 1 --skip-run ...` for layered exhaustive smoke
- `python3 test/benchmarks/intel_gemm_profiler.py --max-shapes 1 --skip-run --bruteforce-scheduler-search` for scheduler-expanded smoke

For an operator-oriented step-by-step procedure, see `OPERATION_MANUAL.md` in the same directory.
