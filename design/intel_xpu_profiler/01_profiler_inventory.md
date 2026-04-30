# Profiler inventory

This document is the source-of-truth inventory for `tools/profiler`.

## Build-listed sources

The current target is defined by `tools/profiler/CMakeLists.txt` via `CUTLASS_TOOLS_PROFILER_SOURCES`.

| Source | Class | Notes |
|---|---|---|
| `src/main.cpp` | Pure C++ | Main entry wrapper |
| `src/cutlass_profiler.cu` | Mechanical | Profiler bootstrap, device selection |
| `src/options.cu` | Mechanical | CLI parsing plus CUDA-first defaults |
| `src/performance_report.cpp` | Pure C++ | Report formatting/output |
| `src/enumerated_types.cpp` | Pure C++ | Enum/string tables |
| `src/gpu_timer.cpp` | Mechanical | Timing backend, now partly prepared for SYCL |
| `src/device_allocation.cu` | Mechanical | Largest memory/tensor utility file |
| `src/device_context.cu` | Mechanical | Device enumeration/properties |
| `src/cublas_helpers.cu` | Remove/Disable | Short-term skip, future oneMKL hook point |
| `src/cudnn_helpers.cpp` | Remove/Disable | Short-term skip, future oneDNN hook point |
| `src/problem_space.cpp` | Pure C++ | Problem argument expansion |
| `src/operation_profiler.cu` | Mechanical + remove subpaths | Shared timing/profile loop; delete CUDA Graph path |
| `src/gemm_operation_profiler.cu` | Mechanical + provider cleanup | GEMM mainline |
| `src/grouped_gemm_operation_profiler.cu` | Mechanical | Grouped GEMM variant |
| `src/block_scaled_gemm_operation_profiler.cu` | Mechanical / phased enable | BlockScaled variant |
| `src/blockwise_gemm_operation_profiler.cu` | Mechanical / phased enable | Blockwise variant |
| `src/rank_k_operation_profiler.cu` | Mechanical | BLAS-like op profiler |
| `src/rank_2k_operation_profiler.cu` | Mechanical | BLAS-like op profiler |
| `src/trmm_operation_profiler.cu` | Mechanical | BLAS-like op profiler |
| `src/symm_operation_profiler.cu` | Mechanical | BLAS-like op profiler |
| `src/conv2d_operation_profiler.cu` | Mechanical + disable vendor path | cuDNN-backed verification branch exists |
| `src/conv3d_operation_profiler.cu` | Mechanical + disable vendor path | cuDNN-backed verification branch exists |
| `src/sparse_gemm_operation_profiler.cu` | Mechanical / phased enable | Sparse GEMM variant |

## Profiler files present but not in current target

| Source | Status | Notes |
|---|---|---|
| `src/performance_result.cu` | Not build-listed | Stub TU including `performance_result.h`; should still be documented so it is not forgotten |

## Headers to track with source migration

Headers under `tools/profiler/include/cutlass/profiler/`:

- `block_scaled_gemm_operation_profiler.h`
- `blockwise_gemm_operation_profiler.h`
- `conv2d_operation_profiler.h`
- `conv3d_operation_profiler.h`
- `cublas_helpers.h`
- `cudnn_helpers.h`
- `cutlass_profiler.h`
- `debug.h`
- `device_allocation.h`
- `device_context.h`
- `enumerated_types.h`
- `gemm_operation_profiler.h`
- `gpu_timer.h`
- `grouped_gemm_operation_profiler.h`
- `operation_profiler.h`
- `options.h`
- `performance_report.h`
- `performance_result.h`
- `problem_space.h`
- `rank_2k_operation_profiler.h`
- `rank_k_operation_profiler.h`
- `reduction_operation_profiler.h`
- `sparse_gemm_operation_profiler.h`
- `symm_operation_profiler.h`
- `trmm_operation_profiler.h`

## Quick dependency scan

A simple string scan across build-listed sources shows:

- `options.cu` is CUDA/provider heavy.
- `device_allocation.cu` is large but mostly memory/tensor mechanics.
- `operation_profiler.cu` contains the CUDA Graph and stream-heavy shared loop.
- `gemm_operation_profiler.cu` is the biggest verification/provider hotspot.
- `cublas_helpers.cu` and `cudnn_helpers.cpp` are explicit vendor helper islands.

## Completeness rules

1. Every build-listed source above must appear in one of the per-file analysis docs.
2. Files not needed for stage-1 Intel GEMM must still be marked explicitly as:
   - deferred,
   - disabled in build,
   - or still ported mechanically.
3. No file is implicitly ignored because it is not on the initial GEMM path.
