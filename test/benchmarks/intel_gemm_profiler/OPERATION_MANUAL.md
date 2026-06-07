#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

# Intel GEMM Profiler Operation Manual

This document is the operator handoff for running the Intel/BMG GEMM profiler workflow and the remote exact-shape search flow.

## 1. Scope

Use this manual when you need to:

1. generate candidate space and build artifacts
2. run local screening / confirmation workflows
3. launch remote exact-shape searches on B70/BMG
4. generate merged TFLOPS / latency reports
5. pull merged results back to a local machine for numerical analysis

## 2. Important invariants

1. **Keep the old search standards available.**
   - `baseline`
   - `expanded_bmg`
   - `layered_exhaustive`

2. **Scheduler expansion is additive.**
   - use `--bruteforce-scheduler-search` only when you explicitly want widened scheduler `sg/stages`

3. **Exact-shape remote compile must stay one-kernel-per-compile.**
   - use `--batch-size 1`

4. **Remote scheduler-expanded runs must keep root-repo sync and worker-repo sync consistent.**
   - use `tools/remote_exact_shape_search_ctl.py sync`
   - do not hand-edit a remote worker repo

5. **For old exact-shape runs, latency in the report can be derived rather than natively recorded.**
   - look at `latency_source`

## 3. Local workflow

### 3.1 Minimal artifact-only smoke

```bash
python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /tmp/profiler_smoke \
  --dtype bf16 \
  --search-strategy layered_exhaustive \
  --kernel-catalog-source layered_bmg \
  --max-shapes 1 \
  --skip-run
```

Expected result:

- `<workspace>/reports/` is populated
- no benchmark subprocess is launched

### 3.2 Scheduler-expanded artifact smoke

```bash
python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /tmp/profiler_scheduler_smoke \
  --dtype bf16 \
  --bruteforce-scheduler-search \
  --max-shapes 1 \
  --skip-run
```

Expected result:

- scheduler-expanded catalog is emitted
- `candidate_build_manifest.json` includes scheduler-expanded benchmark wiring

### 3.3 Full benchmark-backed workflow

Example:

```bash
python3 test/benchmarks/intel_gemm_profiler.py \
  --workspace /tmp/profiler_run \
  --dtype bf16 \
  --probe-mode off \
  --search-strategy layered_exhaustive \
  --kernel-catalog-source layered_bmg \
  --cmake-source-dir /path/to/sycl-tla \
  --benchmark-build-dir /path/to/sycl-tla/build-bench \
  --googlebenchmark-dir /path/to/googlebenchmark-src \
  --cmake-cxx-compiler icpx \
  --build-candidate-benchmark \
  --timeout 900
```

Primary outputs:

- `gemm_profile_results.csv`
- `gemm_dispatch_table.json`
- `run_summary.json`

## 4. Remote exact-shape workflow

### 4.1 Prepare

```bash
cd sycl-tla
export EXACT_SHAPE_REMOTE_PASSWORD='***'
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key sync
```

### 4.2 Launch

Example for the scheduler-expanded exact shape:

```bash
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key launch \
  --run-id shape_search_8192_384_3584_sched_expanded \
  --shapes 8192x384x3584 \
  --layouts rcr,rrr \
  --kernel-catalog-source layered_bmg_scheduler_expanded \
  --batch-size 1 \
  --gpu-ids 0,1,2,3,4,5,6,7
```

### 4.3 Monitor

```bash
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key status \
  --run-dir /root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_8192_384_3584_sched_expanded
```

Check:

- status markers exist under `<run_dir>/status/`
- CSV count keeps increasing
- failed batch count stays `0`

### 4.4 Stop

```bash
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key stop \
  --run-dir /root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_8192_384_3584_sched_expanded
```

## 5. Reporting

### 5.1 Generate report

```bash
python3 tools/remote_exact_shape_search_ctl.py --accept-new-host-key report \
  --run-dir /root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_8192_384_3584_sched_expanded_20260606_2200 \
  --shape-tag 8192_384_3584
```

### 5.2 Report outputs

Under `<run_dir>/reports/<shape_tag>/`:

- `merged_results.csv`: all kernels merged into one CSV
- `ranked_by_tflops.csv`: all OK kernels sorted by TFLOPS descending
- `ranked_by_total_runtime.csv`: all OK kernels sorted by total runtime ascending
- `top5.csv`: top TFLOPS rows
- `worst5.csv`: lowest TFLOPS rows
- `top5_rcr.csv`: top TFLOPS rows limited to RCR
- `fastest5_latency.csv`: lowest total runtime rows
- `slowest5_latency.csv`: highest total runtime rows
- `fastest5_rcr_latency.csv`: lowest total runtime rows limited to RCR
- `summary.json`: row counts, status counts, TFLOPS rankings, latency stats, report file paths

### 5.3 Latency semantics

- `avg_runtime_ms`: single kernel average runtime per measured iteration
- `total_runtime_ms`: total measured runtime across the direct-run measurement loop
- `measure_iters`: current measurement iteration count
- `warmup_iters`: current warmup iteration count
- `latency_source`:
  - `reported`: latency came from the run itself
  - `derived_from_tflops`: latency was backfilled during report generation

For a fixed single shape, **TFLOPS ranking and total-runtime ranking are mathematically equivalent** because total work is constant.

## 6. Pulling merged results back to local

Example:

```bash
local_dir=/mnt/c/work/src/cutlas_profile/out/exact_shape_analysis/shape_search_8192_384_3584_sched_expanded_20260606_2200
mkdir -p "$local_dir"

scp root@10.239.11.149:/root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_8192_384_3584_sched_expanded_20260606_2200/reports/8192_384_3584/merged_results.csv \
  "$local_dir/all_kernels_8192_384_3584.csv"
scp root@10.239.11.149:/root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_8192_384_3584_sched_expanded_20260606_2200/reports/8192_384_3584/summary.json \
  "$local_dir/summary.json"
```

This delivery already pulled the current merged CSV to:

`/mnt/c/work/src/cutlas_profile/out/exact_shape_analysis/shape_search_8192_384_3584_sched_expanded_20260606_2200/all_kernels_8192_384_3584.csv`

## 7. Validation checklist before push

Run:

```bash
python3 test/python/cutlass/test_intel_gemm_profiler.py
python3 test/python/cutlass/test_exact_shape_search_report.py
python3 test/benchmarks/intel_gemm_profiler.py --workspace /tmp/profiler_smoke --dtype bf16 --search-strategy layered_exhaustive --kernel-catalog-source layered_bmg --max-shapes 1 --skip-run
python3 test/benchmarks/intel_gemm_profiler.py --workspace /tmp/profiler_scheduler_smoke --dtype bf16 --bruteforce-scheduler-search --max-shapes 1 --skip-run
```

Push only after:

- unit tests pass
- skip-run artifact smokes pass
- exact-shape report regeneration succeeds for the target run
- untracked cache/build junk is excluded from the commit
