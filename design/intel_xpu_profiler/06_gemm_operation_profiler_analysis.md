# `gemm_operation_profiler.cu` analysis

## Files

- `tools/profiler/src/gemm_operation_profiler.cu`
- `tools/profiler/include/cutlass/profiler/gemm_operation_profiler.h`

## Classification

**Mechanical** for the main structure, plus **provider cleanup**.

## Current role

This is the core operation-specific profiler for GEMM:

- argument parsing
- workspace setup
- `can_implement` / initialize / run
- verification
- result reporting

## Why it matters

This file is the stage-1 Intel XPU priority because it is the bridge between:

- CUTLASS operation objects
- problem space
- verification providers
- runtime/performance result output

## Mechanical parts

- `cudaSetDevice` / `cudaDeviceSynchronize`
- stream passing
- workspace copies
- reference path plumbing

## Non-1:1 parts

1. **cuBLAS provider**
   - short-term disable
   - use `ReferenceDevice` / `ReferenceHost`

2. **provider defaults / report semantics**
   - CLI and reports must remain coherent after disabling vendor providers

3. **mixed dtype / reference edge cases**
   - logic stays, but needs SYCL compile validation

## Validation

1. build with vendor provider disabled
2. run GEMM verification with device/host reference
3. run one B60 GEMM profile end-to-end
