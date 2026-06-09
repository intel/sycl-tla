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

### 3.4 Remote sampled profiler validation

Use this flow when you want a **small but real** validation run on a specific remote GPU that
still exercises:

1. `bruteforce_scheduler`
2. preflight batch builds
3. aggregate candidate build
4. sampled regular GEMM and scheduler benchmark cases

This was the validation style used for GPUs `4` and `6` on shape `8192x76032x8192`.

#### 3.4.1 Required remote prerequisite

The remote `--googlebenchmark-build-dir` must point to a directory that already contains:

```text
_deps/googlebenchmark-build/src/libbenchmark.a
```

If not, create a shared cache once:

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

#### 3.4.2 Generate the two sampled-validation input files

Locally:

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
        {"shape_id": "rcr_bf16_8192_76032_8192", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_d": "f32", "dtype_acc": "f32", "m": 8192, "n": 76032, "k": 8192},
        {"shape_id": "rrr_bf16_8192_76032_8192", "layout": "rrr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_d": "f32", "dtype_acc": "f32", "m": 8192, "n": 76032, "k": 8192},
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
PY
```

Copy them to the remote machine:

```bash
scp out/remote_profiler_validation/shape_8192_76032_8192.json \
  root@10.239.11.149:/root/cutlass_profile_device7_b70_2500mhz/validation_inputs/
scp out/remote_profiler_validation/sample_kernels_8192_76032_8192.list \
  root@10.239.11.149:/root/cutlass_profile_device7_b70_2500mhz/validation_inputs/
```

#### 3.4.3 Run on one GPU

GPU `4` example:

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

For GPU `6`, change:

```text
ZE_AFFINITY_MASK=6
--workspace /root/.../profiler_gpu6_8192_76032_8192_sample
```

#### 3.4.4 Analyze the results

The most important outputs are:

- `candidate_build_preflight_summary.json`
- `candidate_build_summary.json`
- `scheduler_bruteforce_plan.json`
- `gemm_profile_results.csv`
- `phase_b_summary.json`
- `run_summary.json`

Quick health check:

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
})
PY
```

Coverage by `layout × streamk_mode × status`:

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

Healthy output for the current sample should look like:

1. preflight build passes
2. aggregate candidate build passes
3. both regular GEMM and scheduler modes appear
4. both `rcr` and `rrr` appear
5. the current 12-kernel sample yields about `20` pass rows with `top-k=4`, `confirm-runs=1`

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
