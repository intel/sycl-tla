# `blockwise_gemm_operation_profiler.cu` analysis

## Files

- `tools/profiler/src/blockwise_gemm_operation_profiler.cu`
- `tools/profiler/include/cutlass/profiler/blockwise_gemm_operation_profiler.h`

## Classification

**Mechanical** with scope decision.

## Current role

Profiles blockwise GEMM variants, including extra layout/scale plumbing not present in plain GEMM.

## Migration notes

- profiler-layer changes are still mostly `cuda* -> compat::* / sycl::*`
- the important decision is whether blockwise GEMM is:
  - stage-1 Intel XPU scope, or
  - documented but gated off initially

## Validation

Only after plain GEMM path is stable; otherwise this becomes noise during bring-up.
