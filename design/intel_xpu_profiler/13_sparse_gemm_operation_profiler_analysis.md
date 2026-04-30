# `sparse_gemm_operation_profiler.cu` analysis

## Files

- `tools/profiler/src/sparse_gemm_operation_profiler.cu`
- `tools/profiler/include/cutlass/profiler/sparse_gemm_operation_profiler.h`

## Classification

**Mechanical** with explicit scope decision.

## Notes

This file follows the standard profiler model, but sparse GEMM is not part of the immediate Intel B60 GEMM tuning goal.

## Guidance

- include it in inventory and migration planning,
- decide explicitly whether to disable in stage-1 build,
- never omit it silently.
