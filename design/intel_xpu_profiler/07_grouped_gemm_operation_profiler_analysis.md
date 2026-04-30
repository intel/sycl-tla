# `grouped_gemm_operation_profiler.cu` analysis

## Files

- `tools/profiler/src/grouped_gemm_operation_profiler.cu`
- `tools/profiler/include/cutlass/profiler/grouped_gemm_operation_profiler.h`

## Classification

Mostly **Mechanical**.

## Current role

Grouped GEMM extends the GEMM profiler pattern to grouped problem lists and grouped arguments/workspaces.

## Migration expectation

Because it reuses the same profiler infrastructure:

- timer logic follows `operation_profiler.cu`
- workspace logic follows GEMM/device allocation
- provider behavior follows GEMM reference paths

## Main checks

1. grouped-specific argument packing
2. any queue/stream parameter threading
3. grouped verification path under SYCL

## Stage recommendation

Port after plain GEMM, but document now so it is not omitted.
