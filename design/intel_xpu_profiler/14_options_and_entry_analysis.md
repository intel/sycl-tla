# `options` and entry analysis

## Files

- `tools/profiler/src/options.cu`
- `tools/profiler/src/cutlass_profiler.cu`
- `tools/profiler/src/main.cpp`
- related headers:
  - `options.h`
  - `cutlass_profiler.h`

## Classification

`main.cpp`: **Pure C++**  
`options.cu` and `cutlass_profiler.cu`: **Mechanical** with provider/default cleanup.

## Current role

- parse CLI options
- enumerate devices/providers
- select operations
- start profiler execution

## Main replacements

| CUDA today | Intel/SYCL target |
|---|---|
| `cudaGetDeviceCount` | `compat::device_count()` |
| `cudaGetDeviceProperties` | `compat::get_device(...).get_info<...>()` |
| `cudaSetDevice` | `compat::set_device(...)` |

## Non-1:1 points

- provider default values currently assume cuBLAS/cuDNN in several places
- version/device strings may need Intel-specific wording

## Validation

- `cutlass_profiler --help`
- one Intel device detected correctly
- provider lists and defaults make sense under SYCL-only build
