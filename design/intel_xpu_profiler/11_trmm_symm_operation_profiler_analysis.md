# `trmm` / `symm` profiler analysis

## Files

- `tools/profiler/src/trmm_operation_profiler.cu`
- `tools/profiler/src/symm_operation_profiler.cu`
- related headers:
  - `trmm_operation_profiler.h`
  - `symm_operation_profiler.h`

## Classification

Mostly **Mechanical**, with provider cleanup.

## Notes

The porting pattern is similar to RankK-style files. The main concern is vendor-provider dependence in verification defaults.

## Required decisions

1. keep stage-1 disabled but documented, or
2. port mechanically with reference-provider-only validation.

Either choice must be reflected in:

- build rules
- provider defaults
- executable tests
