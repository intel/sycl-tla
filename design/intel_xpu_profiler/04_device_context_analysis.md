# `device_context.cu` analysis

## Files

- `tools/profiler/src/device_context.cu`
- `tools/profiler/include/cutlass/profiler/device_context.h`

## Classification

**Mechanical**.

## Current role

This file abstracts:

- device enumeration
- current device tracking
- device property queries
- profiler-facing device metadata

## Main replacements

| CUDA today | Intel/SYCL target |
|---|---|
| `cudaGetDeviceCount` | `compat::device_count()` |
| `cudaGetDevice` | `compat::current_device_id()` |
| `cudaSetDevice` | `compat::set_device(id)` |
| `cudaGetDeviceProperties` | `compat::get_device(id).get_info<...>()` |
| `cudaDeviceSynchronize` | `compat::wait()` |

## Design impact

Minimal. The file should retain its shape and just translate device/property access to the compat layer.

## Risks

- matching CUDA property fields to SYCL device info fields
- preserving output/report semantics expected by `options.cu` and reports

## Validation

- build on SYCL
- print/query device info on B60
- confirm option parsing and report headers still work
