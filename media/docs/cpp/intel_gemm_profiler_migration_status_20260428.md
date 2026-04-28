# Intel GEMM Profiler Migration Status (2026-04-28)

## Summary

Today's output is **planning convergence plus remote build bring-up**. The profiler MVP implementation has not started yet.

Confirmed decisions:

1. **Phase 1 stays GEMM-only**
   - target the CUTLASS `GemmOperationProfiler` workflow first
   - defer BlockScaledGemm to Phase 2
   - defer GroupedGemm / Attention further

2. **Do not recreate `Manifest + Operation` on Intel**
   - do not mirror `generator.py -> Manifest -> operation->run()`
   - use `candidate generation -> batch build -> subprocess run -> parse result -> select_best`

3. **The Intel GEMM search dimensions are narrowed to**
   - `tile_m / tile_n / tile_k`
   - `sg_m / sg_n`
   - `stages`
   - `split_k`

4. **Correctness stays inside the benchmark binary**
   - keep binary-internal verify
   - use oneDNN / reference path as the correctness baseline

## Investigation completed

### CUTLASS profiler side

The following paths were reviewed to lock the reusable workflow:

- `tools/profiler/src/gemm_operation_profiler.cu`
- `tools/profiler/src/operation_profiler.cu`
- `tools/profiler/src/options.cu`
- `tools/profiler/include/cutlass/profiler/gpu_timer.h`
- `tools/profiler/src/gpu_timer.cpp`

The reusable part is the control-flow skeleton:

```text
problem -> candidate -> verify -> profile -> select_best -> report
```

### Intel / SYCL side

The following assets were reviewed:

- `examples/00_bmg_gemm`
- `examples/03_bmg_gemm_streamk`
- `examples/11_xe20_cutlass_library`
- `tools/util/include/cutlass/util/sycl_timer.hpp`
- `benchmarks/gemm/main.cpp`
- `benchmarks/gemm/benchmarks_sycl.hpp`
- `test/benchmarks/run_benchmarks.py`

Two key facts were confirmed:

1. **The existing examples already support binary-internal correctness verification**
2. **The existing benchmark flow is based on compile-time registered kernels, not a CUTLASS-style manifest**

## B60 node bring-up

### Fresh remote workspace

A new isolated remote workspace was created instead of reusing any existing checkout:

```text
/home/intel/cutlas_profile_task_20260428_1617
```

Layout:

```text
source: /home/intel/cutlas_profile_task_20260428_1617/sycl-tla
build : /home/intel/cutlas_profile_task_20260428_1617/build
```

### Built artifact

The following benchmark binary was built successfully:

```text
/home/intel/cutlas_profile_task_20260428_1617/build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
```

`--help` returns successfully, confirming that the benchmark executable is present and runnable.

## Environment notes confirmed during bring-up

### `test_env.md` points to a stale workdir

The directory recorded in `test_env.md`:

```text
/home/intel/sycl_profile
```

does not exist on the node and should not be used as the active workspace.

### Non-interactive shells must source `.bashrc` to inherit proxy

For later automation, remote commands should use:

```bash
bash -lc 'source /home/intel/.bashrc && <oneAPI env> && <cmd>'
```

Reason:

- a non-interactive shell does not inherit the proxy by default
- `source /home/intel/.bashrc` is required to restore `https_proxy`
- this is more suitable for automation than relying on `bash -il`

### Extra CMake switches were required for benchmark dependency handling

To stop the benchmark dependency chain from pulling gtest, the successful build used:

```text
-DBENCHMARK_ENABLE_TESTING=OFF
-DBENCHMARK_ENABLE_GTEST_TESTS=OFF
-DBENCHMARK_USE_BUNDLED_GTEST=OFF
```

## Current execution status

Completed:

- CUTLASS GEMM profiler architecture analysis
- Intel GEMM asset inventory
- B60 fresh workspace creation
- remote build of `cutlass_benchmarks_gemm_sycl`

In progress:

- `lock-data-schemas`

Not started yet:

- Phase A probe design
- GEMM candidate/result/report schema finalization
- binary/subprocess workflow detailing
- profiler MVP implementation

## Next step

The next concrete step is to lock the schemas so that probe output, candidate generation, benchmark execution, and result aggregation share a stable interface.
