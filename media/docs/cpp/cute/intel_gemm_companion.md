# Intel SYCL GEMM Companion

## Purpose

This document provides **SYCL execution model notes** that complement the CuTe GEMM tutorial.
Read the tutorial first:

> 📖 [0x_gemm_tutorial.md](0x_gemm_tutorial.md)
>
> **Also useful:**
> [intel_overview.md](intel_overview.md) for Intel-specific component map
> · [xe_2d_copy.md](xe_2d_copy.md) for copy atom naming reference

This companion does **not** repeat tutorial content.  It explains how to translate each tutorial
concept into a SYCL submission structure for Intel Xe and shows where Intel-specific copy and MMA
primitives plug in.

---

## Translating tutorial concepts to SYCL

The GEMM tutorial uses CUDA terminology.  The table below maps each concept to its SYCL equivalent
as used in SYCL\*TLA examples (e.g., `examples/sgemm_1_sycl.cpp`,
`examples/bgemm_bmg_legacy.cpp`).

| Tutorial concept | SYCL\*TLA equivalent | Notes |
|-----------------|---------------------|-------|
| `__global__ void kernel(...)` | SYCL kernel submitted via `compat::launch<gemm_kernel, GemmKernelName>(grid, block, ...)` | Two template params: kernel function and kernel name type (for SYCL named kernels). Queue defaults to `get_default_queue()`; pass explicitly as 3rd arg if needed. |
| `blockIdx.x`, `blockIdx.y` | `BlockIdxX()`, `BlockIdxY()` (from `include/cutlass/gpu_generics.h`) | Portable wrappers over `compat::work_group_id::x/y()`. Used in all Xe kernel examples. |
| `threadIdx.x` | `ThreadIdxX()` (from `include/cutlass/gpu_generics.h`) | Portable wrapper over `compat::local_id::x()`. |
| `__shared__ T smem[N]` | `sycl::ext::oneapi::experimental::work_group_scratch_size<sizeof(T[N])>` | Declared via a kernel property; accessed through a pointer obtained from `sycl::ext::oneapi::experimental::get_work_group_scratch_memory()`. |
| `dim3 grid(gx, gy)` | `compat::dim3{gx, gy, 1}` | Passed to `compat::launch<>`. |
| `__launch_bounds__(N)` | `sycl::ext::oneapi::experimental::sub_group_size<16>` + work-group size in `nd_range` | Intel Xe always uses 16-wide subgroups for XMX dispatch. |

### Kernel submission skeleton

```cpp
// Host side — two template parameters: kernel function + kernel name type
auto grid  = compat::dim3{ceil_div(N, BLK_N), ceil_div(M, BLK_M), 1};
auto block = compat::dim3{SubgroupSize * SubgroupsPerGroup, 1, 1};

// Without explicit queue (uses get_default_queue()):
compat::launch<gemm_kernel, GemmKernelName>(grid, block, args...);

// With explicit queue:
compat::launch<gemm_kernel, GemmKernelName>(grid, block, queue, args...);
```

```cpp
// Device side (kernel properties)
auto props = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::sub_group_size<16>
};
```

### Production kernel launch (with kernel properties)

The skeleton above uses the tutorial-style `compat::launch` API, which is suitable for simple
kernels like `sgemm_1_sycl.cpp`.  Production Intel Xe GEMM kernels (e.g.,
`examples/00_bmg_gemm/`, all `GemmUniversalAdapter`-based examples) use the **experimental
launch API** which bundles kernel properties (subgroup size, scratch memory) into a launch policy:

```cpp
namespace sycl_exp = sycl::ext::oneapi::experimental;

auto sycl_grid  = compat::dim3{grid_x, grid_y, 1};
auto sycl_block = compat::dim3{block_x, 1, 1};

compat::experimental::launch_properties launch_props{
    sycl_exp::work_group_scratch_size(shared_mem_bytes),
};
auto kernel_props = compat::experimental::kernel_properties{
    sycl_exp::sub_group_size<16>
};
compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};

// Two template params: device_kernel wrapper + kernel name type
auto event = compat::experimental::launch<
    cutlass::device_kernel<GemmKernel>, GemmKernel>(policy, kernel_params);
```

**When to use which:**

| Pattern | Use when |
|---------|----------|
| `compat::launch<K, KName>(grid, block, args...)` | Simple CuTe tutorial kernels, custom one-file kernels |
| `compat::experimental::launch<device_kernel<K>, K>(policy, params)` | `GemmUniversalAdapter`-based kernels, production GEMM/attention, any kernel needing SLM scratch or kernel properties |

The experimental API is what all `examples/00_bmg_gemm/` through `examples/13_bmg_gemm_bias/`
and Flash Attention examples use internally via `GemmUniversalAdapter`.

---

## Where Intel-specific copy primitives plug in

The standard CuTe GEMM flow is:

```
make_tiled_copy  →  partition  →  cute::copy  →  TiledMMA  →  cute::gemm  →  epilogue store
```

For Intel Xe, the copy atoms and MMA atom are replaced with Xe hardware primitives.
The pattern from `bgemm_bmg_legacy.cpp`:

```cpp
// A-matrix: load with transposed layout (LD_T) for XMX A operand
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, TA>, TA>{}, ...);

// B-matrix: load with VNNI-packed layout (LD_V) for XMX B operand
TiledCopy copyB = make_tiled_copy(
    Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_V, TB>, TB>{}, ...);

// C/D-matrix: store with row-major layout (ST_N)
TiledCopy copyC = make_tiled_copy(
    Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N, TC>, TC>{}, ...);

// MMA: Xe Matrix Extension atom
TiledMMA mmaC = TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, ...>::TiledMMA{};
```

### Layout conventions for Xe operands

| Operand | Recommended layout | Reason |
|---------|--------------------|--------|
| A matrix | `LD_T` (column-major / transposed) | XMX expects A in column-major order |
| B matrix | `LD_V` (VNNI-packed, row-major) | XMX requires VNNI packing for B |
| C / D matrix | `ST_N` (row-major) | Output is row-major |

Using `LD_N` for the B matrix is a common mistake that produces incorrect results or severe
performance degradation.  Always use `LD_V` for B on Intel Xe.

### Epilogue wiring

The CuTe GEMM flow ends with an epilogue that reads the accumulator, optionally loads C, applies
a fusion (e.g., linear combination, bias, activation), and stores D.  The standard Intel Xe
epilogue pattern from `examples/00_bmg_gemm/legacy/00_bmg_gemm.cpp`:

```cpp
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

// Epilogue fusion — D = alpha * acc + beta * C
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementComputeEpilogue,
    ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,    // dispatch policy — must match mainloop
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    ElementAccumulator, cutlass::gemm::TagToStrideC_t<LayoutC>,  // C
    ElementOutput,      cutlass::gemm::TagToStrideC_t<LayoutD>,  // D
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,   // C load atom (for beta * C)
    XE_2D_U32x8x16_ST_N, void, void>;  // D store atom
```

Key points:
- The epilogue dispatch policy (`IntelXeXMX16`) must match the mainloop dispatch policy.
- `FusionCallbacks` wraps the epilogue operation and connects it to the tile shape.
- C is loaded with `XE_2D_U32x8x16_LD_N` (row-major read) and D is stored with
  `XE_2D_U32x8x16_ST_N` (row-major write).
- For grouped GEMM, use `IntelXeXMX16Group` and the array epilogue variant.
- For fused epilogues (bias + activation, softmax, etc.), replace `LinearCombination` with
  the appropriate fusion op from `cutlass/epilogue/fusion/xe_callbacks.hpp`.

See `examples/05_bmg_gemm_with_epilogues/` for fused epilogue examples (bias+ReLU, split-K,
dequantization).

---

## GEMM flow diagram with Intel primitives

```
Layout ──► Tensor ──► Tile ──► Copy ──► MMA ──► Store
  │           │         │        │        │        │
  │           │         │        │        │        └─ XE_2D_U32x8x16_ST_N
  │           │         │        │        └────────── XE_8x16x16_F32BF16BF16F32_TT
  │           │         │        └─────────────────── XE_2D_U16x16x16_LD_T  (A)
  │           │         │                             XE_2D_U16x32x32_LD_V  (B)
  │           │         └──────────────────────────── make_shape(Int<256>{}, Int<256>{}, Int<32>{})
  │           └────────────────────────────────────── make_tensor(gmem_ptr, shape, stride)
  └────────────────────────────────────────────────── make_stride(Int<1>{}, ldA)
```

---

## Performance notes

For detailed tile-size selection, pipeline depth guidance, subgroup sizing, and common pitfalls, see the
[Intel Performance Tuning Guide](intel_performance_guide.md).

**Quick reference for this GEMM flow:**

- **Tile shape:** Start with `Shape<_256, _256, _32>` for BF16 on BMG/PVC.
- **Subgroup layout:** `Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>` gives 32 subgroups per
  work-group (512 threads with `SubgroupSize = 16`).
- **Pipeline stages:** `PipelineStages = 2` is the standard starting point.
- **CollectiveBuilder:** For standard GEMM, prefer `CollectiveBuilder` over manual wiring —
  see the [performance guide](intel_performance_guide.md#start-simple-collectivebuilder).

---

## Further reading

- [xe_2d_copy.md](xe_2d_copy.md) — Full reference for all `XE_2D_*` copy atoms
- [intel_performance_guide.md](intel_performance_guide.md) — Tuning checklist, CollectiveBuilder, and common pitfalls
- [0t_mma_atom.md](0t_mma_atom.md) — CuTe MMA atom concept background
- [examples/README.md](../../../../examples/README.md) — Full SYCL\*TLA example directory (Intel GPU, device-agnostic, NVIDIA SYCL)
- [examples/cute/tutorial/](../../../../examples/cute/tutorial/) — CuTe tutorial examples (including `sgemm_1_sycl.cpp`)
- [test/unit/cute/intel_xe/](../../../../test/unit/cute/intel_xe/) — CuTe unit tests for Xe copy and MMA atoms
