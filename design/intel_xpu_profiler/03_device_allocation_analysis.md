# `device_allocation.cu` analysis

## Files

- `tools/profiler/src/device_allocation.cu`
- `tools/profiler/include/cutlass/profiler/device_allocation.h`

## Classification

**Mechanical** with large edit volume.

## Current role

This file owns the dynamic tensor/device memory utilities used by profiler workspaces:

- device allocation/free
- host/device copies
- random initialization
- tensor compare/save helpers

## Why it looks bigger than it really is

The file is about 90 KB, but most of the size is:

- `NumericTypeID` switch dispatch,
- typed fill/copy helpers,
- tensor utility plumbing.

The core migration surface is much smaller.

## Main 1:1 replacements

| CUDA today | Intel/SYCL target |
|---|---|
| `cudaMalloc` | `sycl::malloc_device(..., queue)` |
| `cudaFree` | `sycl::free(ptr, queue)` |
| `cudaMemcpy` | `compat::memcpy(...)` or `queue.memcpy(...).wait()` |
| `cudaMemset` | `compat::memset(...)` |
| `cudaGetDevice` | `compat::current_device_id()` |
| `cudaSetDevice` | `compat::set_device(id)` |

## Non-1:1 points

Very few:

1. queue ownership must be made explicit,
2. error handling currently written around `cudaError_t` needs adaptation,
3. every typed init/compare path still needs compile verification under SYCL.

## Main risks

- hidden host/device copy assumptions
- typed fill helpers compiling differently under SYCL
- save/compare paths using device context implicitly

## Validation

1. syntax compile after replacement
2. allocate/copy/fill smoke tests
3. compile every switch-dispatched typed path
4. real profiler workspace allocation on B60
