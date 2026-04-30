# `operation_profiler.cu` analysis

## Files

- `tools/profiler/src/operation_profiler.cu`
- `tools/profiler/include/cutlass/profiler/operation_profiler.h`

## Classification

**Mechanical** for the mainline, plus **Remove/Disable** for CUDA Graph and occupancy-specific logic.

## Current role

This is the shared profile loop for all operation profilers:

- predict iteration count
- warmup
- timed region
- multi-device support
- CUDA Graph timing path

## Main replacements

| CUDA today | Intel/SYCL target |
|---|---|
| `cudaStream_t` | `sycl::queue*` |
| `cudaStreamSynchronize` | `queue->wait()` |
| `cudaDeviceSynchronize` | `compat::wait()` |
| event timing | `GpuTimer` / `SYCLTimer` |

## Non-1:1 points

1. **CUDA Graph path**
   - no Intel equivalent
   - recommended action: compile out / disable the whole path

2. **occupancy-driven logic**
   - no Intel analogue needed
   - remove instead of redesign

3. **stream creation/destruction**
   - translate to queue creation/ownership

## Migration strategy

1. keep `predict_iters()` / warmup / timed loop structure,
2. port the no-graph path first,
3. remove graph code behind SYCL guards,
4. keep multi-device scaffolding only if needed by stage-1 scope.

## Validation

- syntax compile
- single-queue profile path
- iteration prediction sanity
- B60 timing on one CUTLASS op
