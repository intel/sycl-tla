# `rank_k` / `rank_2k` profiler analysis

## Files

- `tools/profiler/src/rank_k_operation_profiler.cu`
- `tools/profiler/src/rank_2k_operation_profiler.cu`
- related headers:
  - `rank_k_operation_profiler.h`
  - `rank_2k_operation_profiler.h`

## Classification

Mostly **Mechanical**.

## Notes

These files follow the same broad profiler pattern:

- parse problem arguments
- initialize workspace
- run CUTLASS provider
- verify with provider/reference path

## Migration guidance

- port mechanically after base infrastructure is working
- if not needed in stage-1 build, disable explicitly in tests/CLI rather than ignoring them
