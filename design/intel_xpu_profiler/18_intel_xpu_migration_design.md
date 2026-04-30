# Intel XPU profiler migration design

## Objective

Deliver a complete Intel XPU migration plan for `tools/profiler` without missing non-GEMM profiler files, while keeping the work split correctly between:

1. **mechanical profiler porting**, and
2. **design-heavy generator/search-space expansion**.

## Core judgment

### Profiler port

The profiler port is mostly:

- `cuda* -> compat::* / sycl::*`
- `cudaStream_t -> sycl::queue*`
- provider/build cleanup

Only a few branches need design decisions:

1. remove/disable CUDA Graph timing
2. disable cuBLAS/cuDNN helper paths for stage-1
3. remove occupancy-specific CUDA logic

### Generator/search-space work

This is the real design area:

- wider Intel Xe variant enumeration
- benchmark/codegen/catalog alignment
- turning generated library space into runnable searchable space

## Phased execution

### Phase 1: Documentation and completeness gate

- produce the full inventory
- produce per-file migration handbooks
- classify every profiler file as Mechanical / Remove / Build wiring

### Phase 2: Stage-1 Intel GEMM profiler port

Prioritize:

1. `gpu_timer`
2. `device_allocation`
3. `device_context`
4. `operation_profiler`
5. `gemm_operation_profiler`
6. `options` / `cutlass_profiler` / build wiring

Temporarily disable:

- `cublas_helpers`
- `cudnn_helpers`
- CUDA Graph path
- occupancy-only CUDA logic

### Phase 3: Broader profiler family cleanup

Port or explicitly gate:

- grouped GEMM
- block scaled/blockwise GEMM
- RankK/Rank2K
- TRMM/SYMM
- Conv2d/Conv3d
- Sparse GEMM

### Phase 4: Generator/search-space expansion

- expand Intel Xe compile-time dimensions
- align generator output with benchmark-backed runnable candidates
- increase searchable kernel coverage for B60 cases

## Validation matrix

1. **Static compile validation**
   - oneAPI syntax compile for touched sources
2. **Python/tooling regression**
   - existing Intel benchmark/search tests
3. **Executable validation**
   - `cutlass_profiler` startup and GEMM run on Intel build
4. **B60 real-hardware validation**
   - one or more GEMM cases with reference verification
5. **Dataset validation**
   - tie back to the Ali GEMM dataset workflow

## Deliverables

1. complete `design/intel_xpu_profiler/` document set
2. Intel XPU-capable `cutlass_profiler` stage-1 GEMM path
3. explicit disabled/gated policy for unsupported vendor/helper paths
4. follow-on generator/search-space expansion plan tied to B60 best-kernel search
