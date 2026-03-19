# Intel SYCL GEMM Companion

## Overview

This document walks through a **complete Intel Xe GEMM** step by step, using
[`examples/00_bmg_gemm/00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp)
as the narrative thread.  Each section explains the Intel-specific API in the context where it
actually appears in that example.

> **Related reading:**
> [intel_overview.md](intel_overview.md) — Intel component map and recommended reading order  
> [xe_2d_copy.md](xe_2d_copy.md) — Full reference for all 2D block copy atoms  
> [intel_performance_guide.md](intel_performance_guide.md) — Tile-size and pipeline-depth tuning

---

## 1. Architecture context

Intel Xe GPUs expose a two-level parallelism hierarchy:

| Level | SYCL term | CUTLASS term | Size |
|-------|-----------|--------------|------|
| GPU thread group | work-group | block / CTA | configurable |
| SIMD lane group | sub-group | subgroup | **always 16 lanes** on Xe |

**XMX (Xe Matrix Extensions)** is Intel's hardware matrix engine — the Xe equivalent of CUDA
Tensor Cores.  The fundamental compute instruction is **DPAS** (Dot Product Accumulate Systolic),
which multiplies and accumulates a small matrix tile using the XMX units.

A single DPAS instruction computes:

```
D[M×16] = A[M×K] × B[K×16] + C[M×16]
```

where `K = 256 / max(sizeof_bits(TypeA), sizeof_bits(TypeB))` (e.g. K=16 for BF16, K=8 for TF32).

CUTLASS abstracts DPAS through the `XE_DPAS_TT` MMA atom defined in
`include/cute/arch/mma_xe.hpp`.

---

## 2. Step 1 — Element types and copy atoms

From [`00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp) lines ~334–351:

```cpp
using ElementAccumulator     = float;         // accumulator type
using ElementComputeEpilogue = float;         // epilogue compute type
using ElementInputA          = bfloat16_t;    // A matrix element type
using ElementInputB          = bfloat16_t;    // B matrix element type
using ElementOutput          = float;         // D matrix element type

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// [New Copy Atom] When left as void, MainloopXeL1Staged automatically selects
// appropriate 2D block copy operations for A and B.  You can also set these
// explicitly to XE_LOAD_2D, XE_LOAD_2D_VNNI, or XE_LOAD_2D_TRANSPOSE.
using GmemTiledCopyA = void; // auto-deduced by get_block_2d_copy_A()
using GmemTiledCopyB = void; // auto-deduced by get_block_2d_copy_B()
```

### 2D block copy atoms (`include/cute/arch/copy_xe_2d.hpp`)

The framework auto-selects one of these atoms for each operand when `GmemTiledCopyA/B = void`:

| Atom | Template | Direction | Use case |
|------|----------|-----------|----------|
| `XE_LOAD_2D<Bits,H,W>` | `<int Bits, int Height, int Width, int BlockWidth=Width>` | gmem → regs | Standard 2D block load, row-major |
| `XE_LOAD_2D_VNNI<Bits,H,W>` | `<int Bits, int Height, int Width, int BlockWidth=Width>` | gmem → regs | VNNI-packed load for XMX B operand (BF16/FP16 only) |
| `XE_LOAD_2D_TRANSPOSE<Bits,H,W>` | `<int Bits, int Height, int Width>` | gmem → regs | Transposed load for XMX A operand (32-bit or 64-bit elements only) |
| `XE_PREFETCH_2D<Bits,H,W>` | `<int Bits, int Height, int Width>` | gmem → L1/L2 | Prefetch hint (no register output) |
| `XE_STORE_2D<Bits,H,W>` | `<int Bits, int Height, int Width>` | regs → gmem | 2D block store for epilogue output |

All atoms share a common base `XE_Copy_Op_2D_Base<Bits, Height, Width, Count, Transpose>` that
carries `CopyBits`, `AtomWidth`, `AtomHeight`, `BlockCount`, and `Transposing` as static members.

---

## 3. Step 2 — Workgroup tile shape

From [`00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp) line ~354:

```cpp
// Workgroup-level tile: each work-group processes (M=256, N=256, K=32) at a time
using TileShape = Shape<_256, _256, _32>;
```

`TileShape` is a compile-time `(M, N, K)` triple describing how much of the output matrix a
single work-group owns.  Choosing a good tile shape involves:

- **M and N:** Must be divisible by the subgroup MMA tile.  `_256` gives 8×4=32 subgroups for
  BF16 with the 8×4 subgroup layout below.
- **K:** Must be a multiple of the DPAS K dimension (16 for BF16).  `_32` = 2 DPAS K-steps.
- **Memory footprint:** Larger tiles increase register pressure and reuse; smaller tiles reduce
  occupancy.  See the [performance guide](intel_performance_guide.md) for tuning guidance.

---

## 4. Step 3 — Build the TiledMMA

From [`00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp) lines ~356–367:

```cpp
// TiledMMAHelper constructs a TiledMMA where each sub-group operates on a single
// contiguous chunk of the workgroup TileShape.
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
>::TiledMMA;
```

### `XE_DPAS_TT<M, TypeD, TypeA, TypeB, TypeC>` (`include/cute/arch/mma_xe.hpp`)

Template parameters:

| Parameter | Role | Example value |
|-----------|------|---------------|
| `M` | Output rows per DPAS (1–8) | `8` |
| `TypeD` | Accumulator / output type | `float` |
| `TypeA` | A-operand element type | `bfloat16_t` |
| `TypeB` | B-operand element type (defaults to `TypeA`) | (same) |
| `TypeC` | C-operand element type (defaults to `TypeD`) | (same) |

The K dimension is **auto-computed** by the base class:

```cpp
static constexpr int K = 256 / cute::max(sizeof_bits_v<TypeA>, sizeof_bits_v<TypeB>);
// BF16 → K=16,  FP16 → K=16,  TF32 → K=8,  INT8 → K=32,  INT4 → K=64
```

Type aliases in `namespace cute::dpas_type` make declarations concise:

```cpp
namespace dpas_type {
  using f    = float;
  using tf32 = tfloat32_t;
  using bf   = bfloat16_t;
  using hf   = half_t;
  using u8   = uint8_t;
  using s8   = int8_t;
  using u4   = uint4_t;
  using s4   = int4_t;
}
```

### `TiledMMAHelper` (`include/cute/atom/mma_atom.hpp`)

`TiledMMAHelper` takes three arguments:

1. **`MMA_Atom<XE_DPAS_TT<...>>`** — the single-DPAS hardware atom.
2. **`Layout<TileShape>`** — the CTA-level tile the work-group owns.
3. **Subgroup layout** — how subgroups are arranged within the work-group.

In `Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>`:
- `_8` subgroups along M, `_4` along N → 32 subgroups per work-group.
- Stride `<_4, _1, _0>` means subgroups are laid out **row-major** in the (M, N) plane, which
  places adjacent subgroups in the same N column together — beneficial for B-matrix locality.
- `_1` in K means each subgroup iterates over the full K range (no K parallelism within the CTA).

`TiledMMAHelper` computes the permutations and thread assignments so that each subgroup works on
a contiguous `(SG_M, SG_N, SG_K)` tile, exposing data reuse across DPAS calls within a subgroup.

---

## 5. Step 4 — Dispatch policies

From [`00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp) lines ~369–373:

```cpp
// PipelineStages controls how many K-blocks ahead to prefetch from A and B.
constexpr int PipelineStages = 2;

// MainloopXeL1Staged uses L1-staged prefetching for the mainloop.
// For older copy/MMA atoms, use MainloopIntelXeXMX16 as the dispatch policy.
using GEMMDispatchPolicy     = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
```

| Policy | Header | Purpose |
|--------|--------|---------|
| `MainloopXeL1Staged<Stages>` | `cutlass/gemm/dispatch_policy.hpp` | New API: prefetches `Stages` K-blocks into L1 ahead of compute |
| `MainloopIntelXeXMX16` | `cutlass/gemm/dispatch_policy.hpp` | Legacy API: used with the older `XE_2D_U*` copy atoms |
| `IntelXeGeneric` | `cutlass/epilogue/dispatch_policy.hpp` | Epilogue policy for Intel Xe; works with `XE_STORE_2D` |

---

## 6. Step 5 — Assemble the GEMM types

From [`00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp) lines ~375–423:

```cpp
// Epilogue fusion: D = alpha * (A*B) + beta * C
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementComputeEpilogue,
    ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

// FusionCallbacks ties EpilogueOp to the dispatch policy and tile shape
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy, EpilogueOp, TileShape,
    decltype(tile_shape(TiledMma()))>;

// Collective epilogue: handles C/D loads and stores
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy, TileShape,
    void,                                    // Epilogue tile (void = automatic)
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC>,  // C stride
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>,  // D stride
    FusionCallbacks,
    void,   // C load atom  (void = auto-select XE_LOAD_2D)
    void>;  // D store atom (void = auto-select XE_STORE_2D)

// Collective mainloop: K-loop with prefetch
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy, TileShape,
    ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputB, cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A copy, smem layout, smem copy, transform
    GmemTiledCopyB, void, void, cute::identity   // B copy, smem layout, smem copy, transform
>;

// Full kernel: mainloop + epilogue
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,   // runtime problem shape (M, N, K, L)
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Adapter: wraps the kernel and manages launch + workspace
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

---

## 7. Step 6 — Mainloop: the K-loop (`include/cutlass/gemm/collective/xe_mma.hpp`)

Inside `CollectiveMma::operator()`, the framework executes the following steps for each K-block:

**Setup (before the K-loop):**

```cpp
// 1. Build TiledCopy objects from the global tensor.
//    get_block_2d_copy_A/B selects XE_LOAD_2D_TRANSPOSE (A) and XE_LOAD_2D_VNNI (B)
//    when GmemTiledCopyA/B is void.
auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, mainloop.mA_mkl(_,_,batch_idx));
auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, mainloop.mB_nkl(_,_,batch_idx));

// 2. Allocate register fragments.
//    partition_sg_fragment_A/B returns a SubgroupTensor — a register-resident tensor
//    whose shape matches exactly one subgroup's share of the MMA operand.
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

// Copy fragments (potentially different layout from MMA fragments)
auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

// 3. Partition the global tensor for element-wise access by the copy.
Tensor tAgA = thr_copy_a.partition_S(gA);
Tensor tBgB = thr_copy_b.partition_S(gB);

// 4. Build prefetch TiledCopy instances (no register output — L1/L2 hint only).
auto prefetch_a = make_block_2d_prefetch(copy_a);
auto prefetch_b = make_block_2d_prefetch(copy_b);
```

**K-loop body:**

```cpp
for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
    barrier_arrive(barrier_scope);

    // 5. Load current K-block from global memory into register fragments.
    copy(copy_a, tAgA(_,_,_,k_tile), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

    // Issue prefetch for a future K-block (PipelineStages ahead).
    if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_,_,_,prefetch_k));
        prefetch(prefetch_b, pBgB(_,_,_,prefetch_k));
    }

    // 6. Shuffle data from copy layout to MMA layout.
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    // 7. Execute DPAS: accumulate into accumulators.
    cute::gemm(tiled_mma, tCrA, tCrB, accum);

    barrier_wait(barrier_scope);
}
```

---

## 8. Step 7 — Accumulator allocation (`include/cutlass/gemm/kernel/xe_gemm.hpp`)

Before calling the mainloop, the kernel allocates and zeroes the accumulator tensor:

```cpp
TiledMma tiled_mma;

// partition_fragment_C allocates a register-resident tensor for the (M, N) output tile.
// The shape is determined by the TiledMMA and the CTA tile shape.
Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
clear(accumulators);   // zero-initialize before the K-loop
```

`partition_fragment_C` returns a `SubgroupTensor` in register memory (`rmem`) with shape matching
the subgroup's share of the accumulator.  `clear()` fills it with zero before the first DPAS.

---

## 9. Step 8 — Epilogue

After the K-loop completes, the kernel calls `CollectiveEpilogue::operator()`:

```cpp
CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
epilogue(problem_shape_MNKL, subgroup_shape, blk_coord_mnkl,
         accumulators, tiled_mma, thread_idx);
```

With `EpilogueDispatchPolicy = IntelXeGeneric`, the epilogue:

1. Optionally loads C from global memory using an auto-selected `XE_LOAD_2D` atom.
2. Applies the `LinearCombination` fusion (`D = alpha * acc + beta * C`).
3. Stores D to global memory using an auto-selected `XE_STORE_2D` atom.

For fused epilogues (bias, ReLU, dequantization), replace `LinearCombination` with a fusion op
from `cutlass/epilogue/fusion/xe_callbacks.hpp`.  See `examples/05_bmg_gemm_with_epilogues/` for
complete examples.

---

## 10. GEMM flow diagram with Intel primitives {#gemm-flow-diagram-with-intel-primitives}

```
Global A (row-major)  --[XE_LOAD_2D_TRANSPOSE]--> Register fragments tCrA (SubgroupTensor)
                                                                              |
Global B (row-major)  --[XE_LOAD_2D_VNNI]-------> Register fragments tCrB --+
                                                                              |
                                     [reorder(tArA --> tCrA, tBrB --> tCrB)] |
                                                                              v
                                                         cute::gemm(tiled_mma, tCrA, tCrB, accum)
                                                         -- XE_DPAS_TT<8, float, bfloat16_t> -->
                                                                              |
                                                                        accumulators
                                                                              |
                                                         [Epilogue: LinearCombination]
                                                         D = alpha * accum + beta * C
                                                                              |
Global D (row-major)  <--[XE_STORE_2D]--------------------------------------------
```

Prefetch path (runs `PipelineStages` K-blocks ahead, fills L1 cache):

```
Global A/B --[XE_PREFETCH_2D]--> L1/L2 cache  (no register allocation)
```

---

## 11. Supported type combinations

All combinations declared in `include/cute/arch/mma_xe.hpp` via `CUTE_DECLARE_XE_DPAS_TT`:

| TypeD | TypeA | TypeB | TypeC | K | Notes |
|-------|-------|-------|-------|---|-------|
| `float` | `tfloat32_t` | `tfloat32_t` | `float` | 8 | TF32 |
| `float` | `bfloat16_t` | `bfloat16_t` | `float` | 16 | BF16×BF16→F32 (most common) |
| `bfloat16_t` | `bfloat16_t` | `bfloat16_t` | `float` | 16 | BF16 accum into BF16 |
| `float` | `bfloat16_t` | `bfloat16_t` | `bfloat16_t` | 16 | C in BF16 |
| `bfloat16_t` | `bfloat16_t` | `bfloat16_t` | `bfloat16_t` | 16 | All BF16 |
| `float` | `half_t` | `half_t` | `float` | 16 | FP16×FP16→F32 |
| `float` | `half_t` | `half_t` | `half_t` | 16 | C in FP16 |
| `half_t` | `half_t` | `half_t` | `float` | 16 | FP16 accum |
| `half_t` | `half_t` | `half_t` | `half_t` | 16 | All FP16 |
| `uint32_t` | `uint8_t` | `uint8_t` | `uint32_t` | 32 | U8×U8→U32 |
| `int32_t` | `uint8_t` | `uint8_t` | `int32_t` | 32 | U8×U8→I32 |
| `int32_t` | `uint8_t` | `int8_t` | `int32_t` | 32 | U8×S8→I32 |
| `int32_t` | `int8_t` | `uint8_t` | `int32_t` | 32 | S8×U8→I32 |
| `int32_t` | `int8_t` | `int8_t` | `int32_t` | 32 | S8×S8→I32 |
| `uint32_t` | `uint8_t` | `uint4_t` | `uint32_t` | 32 | U8×U4 |
| `int32_t` | `uint8_t` | `uint4_t` | `int32_t` | 32 | U8×U4 |
| `int32_t` | `uint8_t` | `int4_t` | `int32_t` | 32 | U8×S4 |
| `int32_t` | `int8_t` | `uint4_t` | `int32_t` | 32 | S8×U4 |
| `int32_t` | `int8_t` | `int4_t` | `int32_t` | 32 | S8×S4 |
| `uint32_t` | `uint4_t` | `uint8_t` | `uint32_t` | 64 | U4×U8 |
| `int32_t` | `uint4_t` | `uint8_t` | `int32_t` | 64 | U4×U8 |
| `int32_t` | `uint4_t` | `int8_t` | `int32_t` | 64 | U4×S8 |
| `int32_t` | `int4_t` | `uint8_t` | `int32_t` | 64 | S4×U8 |
| `int32_t` | `int4_t` | `int8_t` | `int32_t` | 64 | S4×S8 |
| `uint32_t` | `uint4_t` | `uint4_t` | `uint32_t` | 64 | U4×U4 |
| `int32_t` | `uint4_t` | `uint4_t` | `int32_t` | 64 | U4×U4 |
| `int32_t` | `uint4_t` | `int4_t` | `int32_t` | 64 | U4×S4 |
| `int32_t` | `int4_t` | `uint4_t` | `int32_t` | 64 | S4×U4 |
| `int32_t` | `int4_t` | `int4_t` | `int32_t` | 64 | S4×S4 |

Use the `dpas_type` namespace aliases (`f`, `bf`, `hf`, `tf32`, `u8`, `s8`, `u4`, `s4`) to
reference these types concisely in template instantiations.

---

## Further reading

- [xe_2d_copy.md](xe_2d_copy.md) — Full reference for all `XE_LOAD_2D` / `XE_STORE_2D` atoms
- [intel_performance_guide.md](intel_performance_guide.md) — Tuning checklist and common pitfalls
- [0t_mma_atom.md](0t_mma_atom.md) — CuTe MMA atom concept background
- [examples/00_bmg_gemm/](../../../../examples/00_bmg_gemm/) — The example walked through here
- [examples/README.md](../../../../examples/README.md) — Full SYCL\*TLA example directory
- [test/unit/cute/intel_xe/](../../../../test/unit/cute/intel_xe/) — CuTe unit tests for Xe atoms
