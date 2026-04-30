# `block_scaled_gemm_operation_profiler.cu` analysis

## Files

- `tools/profiler/src/block_scaled_gemm_operation_profiler.cu`
- `tools/profiler/include/cutlass/profiler/block_scaled_gemm_operation_profiler.h`

## Classification

**Mechanical** with phased enablement.

## Current role

Profiles block-scaled GEMM variants and their extra scale-factor metadata.

## Migration notes

- the profiler infrastructure port is still mostly mechanical,
- but stage-1 Intel XPU may choose not to enable this operation until plain GEMM is proven.

## Required doc output

The final migration notes must say one of two things explicitly:

1. port now, or
2. defer in build/CLI/tests for stage-1.

Implicit omission is not allowed.
