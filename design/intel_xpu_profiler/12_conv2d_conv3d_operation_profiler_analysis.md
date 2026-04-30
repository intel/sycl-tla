# `conv2d` / `conv3d` profiler analysis

## Files

- `tools/profiler/src/conv2d_operation_profiler.cu`
- `tools/profiler/src/conv3d_operation_profiler.cu`
- related headers:
  - `conv2d_operation_profiler.h`
  - `conv3d_operation_profiler.h`

## Classification

**Mechanical** for common profiler plumbing, plus **Remove/Disable** for cuDNN validation defaults.

## Current role

These are profiler frontends for convolution operations, with stronger cuDNN ties than the GEMM path.

## Main issue

The CUDA runtime migration is still straightforward, but the vendor helper side is not:

- `cudnn_helpers.cpp`
- cuDNN-based verification defaults in CMake/tests

## Stage recommendation

Document fully now, but likely disable for stage-1 Intel GEMM bring-up unless there is an explicit XPU convolution requirement.
