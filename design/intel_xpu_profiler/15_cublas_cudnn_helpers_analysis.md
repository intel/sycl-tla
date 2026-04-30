# `cublas_helpers` / `cudnn_helpers` analysis

## Files

- `tools/profiler/src/cublas_helpers.cu`
- `tools/profiler/src/cudnn_helpers.cpp`
- related headers:
  - `cublas_helpers.h`
  - `cudnn_helpers.h`

## Classification

**Remove/Disable** for stage-1 Intel XPU.

## Reason

There is no 1:1 compat-layer equivalent for:

- cuBLAS Lt helper stack
- cuDNN helper stack

The right short-term move is not to redesign these now; it is to:

1. disable them in the Intel path,
2. switch verification to reference providers,
3. adjust CLI/build/tests so they do not request unavailable vendors.

## Future extension points

- oneMKL for GEMM-like vendor reference
- oneDNN for convolution-like vendor reference

## Required cleanup

- provider defaults in source and CMake tests
- report labels and expectation management
