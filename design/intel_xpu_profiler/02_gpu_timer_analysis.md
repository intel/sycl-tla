# `gpu_timer` analysis

## Files

- `tools/profiler/include/cutlass/profiler/gpu_timer.h`
- `tools/profiler/src/gpu_timer.cpp`
- supporting reuse:
  - `tools/util/include/cutlass/util/sycl_timer.hpp`
  - `tools/util/include/cutlass/util/sycl_event_manager.hpp`

## Classification

**Mechanical**, mostly already prepared.

## Current role

`GpuTimer` is the shared profiler timer used by `operation_profiler.cu`.

## CUDA/SYCL status

This file is no longer the main blocker:

- SYCL queue-aware timing support was added.
- wall-clock fallback and event-based timing were both hardened.
- stream handling now uses `sycl::queue*` instead of a fake integer placeholder.

## Migration notes

For full profiler migration, this file mainly needs:

1. stay on the current SYCL-aware implementation,
2. ensure callers pass real `sycl::queue*`,
3. remove any remaining build assumptions that treat the profiler as CUDA-only.

## Non-1:1 points

- None beyond the already-resolved timing policy choice:
  - event profiling when available,
  - wall-clock fallback otherwise.

## Validation

- syntax compile with and without `CUTLASS_SYCL_PROFILING_ENABLED`
- profiler call-sites in `operation_profiler.cu`
- B60 timing sanity on real kernels
