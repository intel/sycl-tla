# Intel B70 exact-shape remote search runbook

This runbook captures the operational lessons from the exact-shape B70 search flow so the same failures do not repeat.

## Scope

- target platform: Intel B70 / BMG
- target workflow: exact-shape batch search using benchmark-backed GEMM kernels
- current exact-shape defaults:
  - shapes: `2048x384x3584`, `8192x384x3584`
  - dtype: `bf16 / bf16 / f32`
  - layouts: `rcr`, `rrr`
  - execution mode: **shape-serial, multi-GPU**

## Persistent tooling

Use the repository scripts instead of ad hoc remote shell snippets:

1. remote launch: `tools/remote_exact_shape_search.sh`
2. remote status: `tools/remote_exact_shape_search_status.sh`
3. remote stop: `tools/remote_exact_shape_search_stop.sh`
4. local controller: `tools/remote_exact_shape_search_ctl.py`

Example:

```bash
python3 tools/remote_exact_shape_search_ctl.py sync
python3 tools/remote_exact_shape_search_ctl.py launch --skip-remote-repo-sync
python3 tools/remote_exact_shape_search_ctl.py status
python3 tools/remote_exact_shape_search_ctl.py stop
```

## Search contract for this run

- shapes:
  - `2048x384x3584`
  - `8192x384x3584`
- dtypes:
  - `A=bf16`
  - `B=bf16`
  - `C/D/acc=f32`
- layouts:
  - `rcr`
  - `rrr`
- kernel count:
  - **1772 kernels per shape**
  - **886 batches per shape** with `BATCH_SIZE=2`
  - GPU `0-4` split those 886 batches within a single shape
  - two shapes are never active at the same time

## Compile-time and runtime environment

The fixed perf environment for this workflow is:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
```

Notes:

- `IGC_ExtraOCLOptions` must use the **single-dash** form `-cl-intel-256-GRF-per-thread`
- the launcher now reapplies the same perf env:
  - once after oneAPI env setup
  - again before worker `cmake` / `make`
  - again before each worker runtime loop
- `ZE_AFFINITY_MASK=$gpu` is set per worker so runtime execution stays pinned to the assigned GPU

## End-to-end runnable procedure

### Option A: local controller (recommended)

```bash
cd sycl-tla

export EXACT_SHAPE_REMOTE_PASSWORD='your-remote-password'
python3 tools/remote_exact_shape_search_ctl.py sync
python3 tools/remote_exact_shape_search_ctl.py stop
python3 tools/remote_exact_shape_search_ctl.py launch --skip-remote-repo-sync
python3 tools/remote_exact_shape_search_ctl.py status
```

Notes:

- the controller uses SSH known-host verification by default
- if the remote host is not already in `known_hosts`, either add it first with `ssh` or pass `--accept-new-host-key`
- the controller reads the password from `EXACT_SHAPE_REMOTE_PASSWORD` unless `--password` is provided explicitly

### Option B: direct remote invocation

```bash
cd /root/cutlass_profile_device7_b70_2500mhz/sycl-tla

bash tools/remote_exact_shape_search_stop.sh
mkdir -p /root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_manual

nohup env \
  RUN_ID=shape_search_manual \
  GPU_IDS=0,1,2,3,4 \
  SHAPES='2048x384x3584;8192x384x3584' \
  SKIP_SYNC=1 \
  STOP_EXISTING=1 \
  bash tools/remote_exact_shape_search.sh \
  > /root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_manual/launcher.log 2>&1 < /dev/null &

bash tools/remote_exact_shape_search_status.sh \
  /root/cutlass_profile_device7_b70_2500mhz/screen_runs/shape_search_manual
```

## What to check after launch

1. `run_meta.txt`
   - verify `perf_env_IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread`
   - verify the other perf env lines match the fixed values above
2. `status`
   - `current_shape=2048_384_3584` first
   - `marker_status=running`
3. early result health
   - `csv_count > 0`
   - `status[OK] > 0`
   - `status[TIMEOUT] = 0`
   - `failed_gpuX = 0`
4. worker progress
   - `worker_2048_384_3584_gpu*.log` continues printing `batch_xxxx done`
5. shape transition
   - after the first shape finishes:
     - `2048_384_3584.done` exists
     - `completed_shapes=2048_384_3584`
     - `current_shape=8192_384_3584`

## Non-negotiable invariants

1. **Shapes are serial, not parallel.**
   - finish the first shape completely
   - only then start the second shape

2. **GPU sharding is allowed only within one shape.**
   - GPUs `0-4` can process disjoint batch subsets
   - two different shapes must never be searched at the same time

3. **Worker builds must reuse shared benchmark build artifacts.**
   - source `_deps` alone is not enough
   - worker build directories also need access to the prebuilt `googlebenchmark-build`

4. **Fresh worker configure must disable CUTLASS tests.**
   - the benchmark search does not need unit tests
   - enabling tests on a fresh configure can reintroduce `GTest::gtest` failures

5. **Do not source oneAPI env scripts under `set -u` without temporarily disabling nounset.**
   - `env/vars.sh` may reference unset variables internally
   - under `set -u`, this can terminate the launcher before any useful log is written

6. **Reapply perf env for both compile and runtime.**
   - current fixed values:
     - `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
     - `SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"`
     - `IGC_VectorAliasBBThreshold=10000`
     - `IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"`
   - use the single-dash `-cl-intel-256-GRF-per-thread` form
   - do not rely on one-time shell setup only; worker build and worker runtime should both inherit the same perf env

7. **Persist state markers inside `run_dir/status/`.**
   - `current_shape`
   - `${shape}.status`
   - `${shape}.done`
   - `completed_shapes.txt`
   - `worker_pids.txt`
   - these markers make status checks deterministic instead of inferring everything from logs only

8. **Each launch starts from a clean run directory.**
   - reusing the same `RUN_ID` must not reuse old CSVs/logs/status files
   - the launcher now clears prior per-run artifacts before starting a fresh run

## Why the earlier full validation succeeded but this exact-shape launcher failed

The successful 886-batch validation reused a known-good build tree in-place and compiled batches there. That flow already had:

- valid `_deps`
- valid `googlebenchmark-build`
- valid benchmark link dependencies

The new exact-shape launcher introduced a different execution model:

- new per-GPU worker worktrees
- new per-worker CMake configure/build trees

That changed the failure surface.

## Exact root causes from this failure

### 1. `source /opt/intel/oneapi/compiler/.../env/vars.sh` under `set -u`

Symptom:

- launcher exited immediately
- almost no logs were written

Root cause:

- the launcher used `set -euo pipefail`
- oneAPI `env/vars.sh` is not safe to source under `nounset`

Persistent fix:

- temporarily `set +u` around `source .../env/vars.sh`
- restore `set -u` afterward

### 2. Worker build reused `_deps` source tree but not `googlebenchmark-build`

Symptom:

```text
No rule to make target '_deps/googlebenchmark-build/src/libbenchmark.a'
```

Root cause:

- the benchmark executable still links against Google Benchmark static libraries
- reusing only `repo/_deps/googlebenchmark-src` headers is insufficient
- each worker build must also see the prebuilt benchmark library artifacts

Persistent fix:

- symlink worker build `_deps/googlebenchmark-build` to the known-good template build
- optionally reuse `googletest-build` as well when present

### 3. Fresh worker configure tried to configure unit tests

Symptom:

```text
Target "cutlass_test_unit_infra" links to:
  GTest::gtest
but the target was not found
```

Root cause:

- this was not a profiler benchmark runtime dependency
- it came from top-level CUTLASS CMake entering `test/unit/...`
- the worker build was a fresh CMake configure, unlike the previously validated in-place build flow

Persistent fix:

- set `-DCUTLASS_ENABLE_TESTS=OFF` for worker builds

### 4. Broad stop logic matched the launcher itself

Symptom:

- relaunch exited immediately after adding a stop-existing-runs phase

Root cause:

- process matching was too broad and included the current launcher PID

Persistent fix:

- exclude the current launcher PID explicitly from stop logic
- restrict process matching to known exact-shape launcher commands

## Google Benchmark vs GTest in this workflow

`cutlass_benchmarks_gemm_sycl` is **not** a pure “static link only, zero code usage” case.

In this exact-shape path:

- the direct path does **not** call `::benchmark::RunSpecifiedBenchmarks()`
- but the code still includes and uses Google Benchmark types/macros:
  - `#include <benchmark/benchmark.h>`
  - `::benchmark::State`
  - `CUTLASS_CREATE_GEMM_BENCHMARK(...)`
  - `CUTLASS_BENCHMARK(...)`
- legacy config-file mode still explicitly calls:
  - `::benchmark::Initialize(...)`
  - `::benchmark::RunSpecifiedBenchmarks()`
  - `::benchmark::Shutdown()`

So the accurate statement is:

- **Google Benchmark is still a code-level dependency of the benchmark target**
- **the exact-shape direct mode avoids the legacy benchmark event loop**
- **GTest is not required for this workflow**

## Why GTest should not be part of this workflow

The **profiler benchmark path itself does not require GTest**. If `GTest::gtest` appears, it means the build is accidentally configuring unit-test targets as part of a fresh top-level CMake run.

So the correct policy for this workflow is:

- **Google Benchmark: required**
  - because `cutlass_benchmarks_gemm_sycl` links against benchmark runtime libraries
- **GTest: not required**
  - because this workflow is not building or running CUTLASS unit tests

That is why the persistent fix is to keep:

- shared `googlebenchmark-build`
- `CUTLASS_ENABLE_TESTS=OFF`

## Required preflight checks before trusting a new launcher

1. `bash -n tools/remote_exact_shape_search.sh`
2. `bash -n tools/remote_exact_shape_search_status.sh`
3. launch one debug run in foreground with `STOP_EXISTING=0 SKIP_SYNC=1`
4. confirm all worker `cmake_gpu*.log` files exist
5. confirm worker build path contains:

```text
workers/gpuX/build/_deps/googlebenchmark-build/src/libbenchmark.a
```

6. confirm status script reports:
   - process list
   - current shape
   - marker status
   - nonzero CSV count once screening starts

## Manual validation checklist after the run

1. confirm both shapes appear in `completed_shapes.txt`
2. confirm both `status/<shape>.done` markers exist
3. confirm there are no `COMPILE FAIL`, `LINK FAIL`, or `TIMEOUT` rows
4. inspect the top-performing kernels from both shape result directories
5. verify the final run used the expected perf env from `run_meta.txt`
6. archive `run_meta.txt`, `requested_shapes.json`, and the final `results/` directory alongside any report or review notes

## Operational rule going forward

If a new exact-shape launcher changes:

- worktree layout
- worker build directory structure
- `_deps` reuse
- CMake flags
- stop/restart logic

then it must be validated with:

1. one foreground debug launch
2. one status-script check
3. one early-batch smoke

before any full exact-shape run is trusted.
