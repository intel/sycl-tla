# Build system analysis

## Files

- `tools/profiler/CMakeLists.txt`

## Classification

**Build wiring**.

## Current issues

1. many profiler sources are still `.cu`
2. CUDA libraries are linked unconditionally:
   - `cudart`
   - `cuda_driver`
3. executable tests default to cuBLAS/cuDNN-based verification on several operations

## Required migration work

1. switch Intel-targeted profiler sources from `.cu` compilation assumptions to SYCL-capable compilation
2. gate CUDA-only libs behind non-SYCL conditions
3. gate vendor helper sources or providers by backend
4. revisit executable test options for Intel builds

## Important completeness note

`performance_result.cu` exists in `src/` but is not in `CUTLASS_TOOLS_PROFILER_SOURCES`. The build doc should call out whether that is intentional dead weight or a missing source-list entry.
