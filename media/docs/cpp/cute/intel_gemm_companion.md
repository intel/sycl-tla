# Intel SYCL GEMM Companion

## Purpose

This document provides **SYCL execution model notes** that complement the CuTe GEMM tutorial.
Read the tutorial first:

> 📖 [0x_gemm_tutorial.md](0x_gemm_tutorial.md)

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
| `__global__ void kernel(...)` | SYCL kernel submitted via `compat::launch<gemm_kernel>(queue, grid, block, ...)` | The `compat` shim maps CUDA launch parameters to `sycl::nd_range`. |
| `blockIdx.x`, `blockIdx.y` | `compat::work_group_id::x()`, `compat::work_group_id::y()` | Called inside the kernel body. |
| `threadIdx.x` | `compat::local_id::x()` | Local lane index within the work-group. |
| `__shared__ T smem[N]` | `sycl::ext::oneapi::experimental::work_group_scratch_size<sizeof(T[N])>` | Declared via a kernel property; accessed through a pointer obtained from `sycl::ext::oneapi::experimental::get_work_group_scratch_memory()`. |
| `dim3 grid(gx, gy)` | `compat::dim3{gx, gy, 1}` | Passed to `compat::launch<>`. |
| `__launch_bounds__(N)` | `sycl::ext::oneapi::experimental::sub_group_size<16>` + work-group size in `nd_range` | Intel Xe always uses 16-wide subgroups for XMX dispatch. |

### Kernel submission skeleton

```cpp
// Host side
auto grid  = compat::dim3{ceil_div(N, BLK_N), ceil_div(M, BLK_M), 1};
auto block = compat::dim3{SubgroupSize * SubgroupsPerGroup, 1, 1};

compat::launch<gemm_kernel>(queue, grid, block, args...);
```

```cpp
// Device side (kernel properties)
auto props = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::sub_group_size<16>
};
```

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

## Performance notes: tile sizes and subgroup layout

### Common tile shapes

| Data type | Tile shape (M × N × K) | MMA atom |
|-----------|------------------------|---------|
| BF16 | `(256, 256, 32)` | `XE_8x16x16_F32BF16BF16F32_TT` |
| FP16 | `(256, 256, 32)` | `XE_8x16x16_F32F16F16F32_TT` |
| FP8 | `(256, 256, 32)` | `XE_8x16x16_F32F8F8F32_TT` |

Start with `(256, 256, 32)` for BF16 on BMG and PVC.  Reduce M or N if the compiler reports
register spill.

### Subgroup layout

The subgroup layout tensor controls how subgroups tile the output:

```cpp
// Example: 8 subgroups along M × 4 subgroups along N
Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
```

This means 32 subgroups per work-group.  Combined with `SubgroupSize = 16` this gives a
work-group size of `32 × 16 = 512` threads, which is a typical Xe GEMM work-group.

### Pipeline stages

```cpp
static constexpr int PipelineStages = 2;
```

`PipelineStages = 2` is the standard starting point.  It overlaps one iteration of 2D block loads
with the XMX compute of the previous iteration.  Increase to 3 for higher-latency HBM
configurations, but verify that register pressure remains acceptable.

---

## Further reading

- [xe_2d_copy.md](xe_2d_copy.md) — Full reference for all `XE_2D_*` copy atoms
- [intel_performance_guide.md](intel_performance_guide.md) — Tuning checklist and common pitfalls
- [0t_mma_atom.md](0t_mma_atom.md) — CuTe MMA atom concept background
