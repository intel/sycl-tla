# Intel Xe GPU GEMM Companion

## Overview

This document walks through a **complete Intel Xe GEMM kernel** step by step, using the
pure CuTe tutorial
[`examples/cute/tutorial/xe_gemm.cpp`](../../../../examples/cute/tutorial/xe_gemm.cpp)
as the narrative thread.  Every code snippet comes directly from that tutorial; there are no
CUTLASS collective or kernel-level abstractions here — only CuTe atoms, tensors, and primitives.

> **Related reading:**
> [10_intel_overview.md](10_intel_overview.md) — Intel component map and recommended reading order
> [xe_2d_copy.md](xe_2d_copy.md) — Full reference for all 2D block copy atoms
> [12_intel_performance_guide.md](12_intel_performance_guide.md) — Tile-size and pipeline-depth tuning

---

## 1. Xe Architecture Primer

To understand the code that follows, you need a mental model of (1) the hardware
and (2) the memory system it exposes to CuTe.

### Execution hierarchy

```
GPU
 └─ Xe-core  (≈ an SM)
      └─ EU thread  (≈ a CUDA warp)
           └─ XMX systolic array  (≈ Tensor Core)
```

| Hardware      | SYCL term   | CuTe term  | Notes |
|---------------|-------------|------------|-------|
| Xe-core       | work-group  | CTA / block | one tile of the output matrix |
| EU thread     | sub-group   | subgroup    | **always 16 lanes** on Xe |
| XMX           | —           | `XE_DPAS_TT` atom | executes DPAS instruction |

### Memory hierarchy

```
Global Memory  (GDDR6 on BMG / HBM2e on PVC)
  └─ L1 cache  (per Xe-core, ~256 KB)
       └─ GRF  (general register file, 256 registers × 64 B when using large-GRF mode)
```

All MMA operands **must** reside in GRF before executing DPAS.  The data path in this
tutorial is:

```
Global Memory ──prefetch──▶ L1 cache ──copy (2D block load)──▶ GRF
```

The 2D block load instructions read a rectangular sub-matrix from global memory (or L1 if
prefetched) directly into registers, with optional VNNI packing or transposition.

> For full architecture details see [10_intel_overview.md](10_intel_overview.md).

---

## 2. MMA atom and tile shape selection

The first decision is which MMA atom to use and how to tile the problem across the
work-group.

Reference: [`xe_gemm.cpp` lines 220–261](../../../../examples/cute/tutorial/xe_gemm.cpp)

### Selecting the MMA atom

```cpp
auto op = choose_mma_op<TA, TB, TC>();
// Returns  XE_DPAS_TT<8, TC, TA, TB>{}  when the combination is supported,
// otherwise falls back to BF16 or FP16.
```

`XE_DPAS_TT` is the CuTe atom wrapping a single DPAS instruction.  A DPAS computes
`D[M×16] += A[M×K] × B[K×16]` where `K = 256 / max(sizeof_bits(TypeA), sizeof_bits(TypeB))`
(e.g. K = 16 for BF16, K = 8 for TF32).

Reference: `include/cute/arch/mma_xe.hpp` for all supported type combinations.

### Choosing workgroup tile and subgroup layout

```cpp
using WGTile = Shape<_256, _256, _K>;                                // workgroup tile (M,N,K)
using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;     // 8×4 subgroup grid, n-major
```

- **WGTile** — each work-group computes a 256 × 256 output tile, processing `_K` elements
  along the reduction dimension per mainloop iteration.
- **SGLayout** — arranges 32 subgroups (8 along M, 4 along N) inside the work-group.
  The stride `<_4,_1,_0>` makes the grid n-major so adjacent subgroups share B-operand cache
  lines.

### Building the TiledMMA

```cpp
using MMA = typename TiledMMAHelper<
    MMA_Atom<decltype(op)>,     // single-DPAS atom
    Layout<WGTile>,             // workgroup tile
    SGLayout                    // subgroup arrangement
>::TiledMMA;
```

`TiledMMAHelper` fuses these three ingredients into a single `TiledMMA` object that knows
how to partition the output tile across subgroups and map each subgroup's data to DPAS calls.

---

## 3. Creating TiledCopy instances

CuTe provides factory functions that automatically select the right 2D block copy variant
(`XE_LOAD_2D`, `XE_LOAD_2D_VNNI`, `XE_LOAD_2D_TRANSPOSE`, `XE_PREFETCH_2D`, or
`XE_STORE_2D`) based on the element type, layout, and MMA requirements.

Reference: [`xe_gemm.cpp` lines 140–168](../../../../examples/cute/tutorial/xe_gemm.cpp)

```cpp
auto copy_a = make_block_2d_copy_A(mma, A);       // load A tiles
auto copy_b = make_block_2d_copy_B(mma, B);       // load B tiles
auto copy_c = make_block_2d_copy_D(mma, C);       // store output
```

For prefetch (no register output — L1/L2 hint only):

```cpp
auto prefetch_a = make_block_2d_prefetch(copy_a);
auto prefetch_b = make_block_2d_prefetch(copy_b);
```

The auto-selection logic lives in `copy_traits_xe_2d.hpp` and considers:
- Whether the operand needs VNNI packing (B with BF16/FP16)
- Whether the matrix is transposed (A with 32-bit or 64-bit elements)
- Whether this is a load, store, or prefetch

Reference: `include/cute/atom/copy_traits_xe_2d.hpp`

---

## 4. Tiling the problem and partitioning to threads

Before the K-loop, the kernel tiles the global problem into workgroup-sized blocks
and slices those blocks down to per-work-item fragments.

Reference: [`xe_gemm.cpp` lines 118–170](../../../../examples/cute/tutorial/xe_gemm.cpp)

### Tiling to workgroup level

```cpp
auto wg_tile = mma.tile_mnk();
Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m, _));  // (BLK_M, BLK_K, k)
Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n, _));  // (BLK_N, BLK_K, k)
Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1,X>{});         // (BLK_M, BLK_N)
```

`local_tile` splits each global tensor into workgroup-sized blocks.  `gA` and `gB` get a
trailing mode `k` that indexes K-tiles for the mainloop to iterate over.

### Slicing to thread level

```cpp
auto thr_mma    =    mma.get_slice(local_id);
auto thr_copy_a = copy_a.get_slice(local_id);
```

`.get_slice(local_id)` distributes work across the work-items inside a work-group.

### Allocating register fragments

```cpp
// MMA fragments — layout matches what DPAS expects
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

// Copy fragments — layout matches what the 2D block load produces
auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

// Accumulator (M×N output tile per subgroup)
Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));
```

Copy fragments and MMA fragments may have **different register layouts**; the `reorder()`
call in the K-loop bridges the two.

---

## 5. Prefetch warmup

Before entering the main loop, the kernel prefetches the first few K-tiles into L1 cache
so the first iterations don't stall on global memory latency.

Reference: [`xe_gemm.cpp` lines 180–186](../../../../examples/cute/tutorial/xe_gemm.cpp)

```cpp
const int prefetch_dist = 3;
clear(tCrC);                  // zero the accumulators

CUTE_UNROLL
for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
}
```

`prefetch()` issues `XE_PREFETCH_2D` instructions that bring data into L1 without
allocating any registers.

---

## 6. The K-loop (mainloop)

Each iteration performs: load → prefetch → reorder → compute → barrier.

Reference: [`xe_gemm.cpp` lines 189–211](../../../../examples/cute/tutorial/xe_gemm.cpp)

```cpp
for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);           // 1

    copy(copy_a, tAgA(_,_,_,k_tile), tArA); // 2  Global/L1 → copy registers
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));  // 3  prime L1 ahead
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

    reorder(tArA, tCrA);                    // 4  copy layout → MMA layout
    reorder(tBrB, tCrB);

    gemm(mma, tCrA, tCrB, tCrC);           // 5  DPAS accumulate

    barrier_wait(barrier_scope);            // 6
}
```

| Step                | CuTe primitive     | Hardware action |
|---------------------|--------------------|-----------------|
| 1. barrier arrive   | `barrier_arrive()` | Split barrier — loosely synchronises subgroups within the work-group |
| 2. load             | `copy()`           | 2D block load (`XE_LOAD_2D_TRANSPOSE` for A, `XE_LOAD_2D_VNNI` for B) |
| 3. prefetch         | `prefetch()`       | `XE_PREFETCH_2D` — warms L1 for a future K-tile |
| 4. reorder          | `reorder()`        | Register-level shuffle to convert copy layout → MMA layout |
| 5. DPAS             | `gemm()`           | Executes XMX systolic multiply-accumulate |
| 6. barrier wait     | `barrier_wait()`   | Completes the split barrier cycle |

Reference: `include/cute/util/xe_split_barrier.hpp` for split barrier implementation.

---

## 7. Store

After the K-loop, the accumulator is written to global memory with a single 2D block store.

Reference: [`xe_gemm.cpp` line 213](../../../../examples/cute/tutorial/xe_gemm.cpp)

```cpp
copy(copy_c, tCrC, tCgC);   // registers → global D via XE_STORE_2D
```

This tutorial computes a raw C = A × B with no epilogue fusion.  For fused epilogues
(alpha/beta scaling, bias add, activation) see the CUTLASS-level examples:
[`examples/00_bmg_gemm/`](../../../../examples/00_bmg_gemm/) and
[`examples/05_bmg_gemm_with_epilogues/`](../../../../examples/05_bmg_gemm_with_epilogues/).

---

## 8. Kernel launch

The kernel is launched as a 2D SYCL `nd_range` with Intel-specific kernel properties.

Reference: [`xe_gemm.cpp` lines 268–283](../../../../examples/cute/tutorial/xe_gemm.cpp)

```cpp
sycl::range<2> local  = {size(mma), 1};
sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                          local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
};

Q.parallel_for<GemmCuteName<TA, TB, layoutA, layoutB>>(
    sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) { gemm_device(A, B, C, mma); }
);
```

- **Local size** = `size(mma)` work-items — typically 32 subgroups × 16 lanes = 512.
- **Global size** tiles the N and M dimensions of the problem over workgroups.
- **`sub_group_size<16>`** — required for Xe DPAS (always 16 lanes).
- **`grf_size<256>`** — selects 256-register (large GRF) mode, giving each EU thread
  256 × 64 B = 16 KB of register space.

---

## 9. CuTe GEMM flow diagram {#gemm-flow-diagram-with-intel-primitives}

```
Global A/B  ──prefetch()──▶  L1 cache
     │                          │
     └──copy() (2D block load)──┘──▶  Copy fragments (GRF)
                                           │
                                       reorder()
                                           │
                                       MMA fragments (GRF)
                                           │
                                       gemm()  ── DPAS (XMX) ──▶  Accumulator (GRF)
                                                                        │
                                                                   copy() (2D block store)
                                                                        │
                                                                   Global D
```

All data movement and compute use CuTe primitives: `prefetch()`, `copy()`, `reorder()`,
and `gemm()`.  The underlying hardware atoms (`XE_PREFETCH_2D`, `XE_LOAD_2D_*`,
`XE_DPAS_TT`, `XE_STORE_2D`) are selected automatically by the factory functions.

---

## Further reading

- [xe_2d_copy.md](xe_2d_copy.md) — Full reference for `XE_LOAD_2D` / `XE_STORE_2D` atoms
- [12_intel_performance_guide.md](12_intel_performance_guide.md) — Tuning checklist (tile sizes, pipeline depth, register pressure)
- [10_intel_overview.md](10_intel_overview.md) — Intel component map and reading order
- [0t_mma_atom.md](0t_mma_atom.md) — CuTe MMA atom concept background
- [`examples/cute/tutorial/xe_gemm.cpp`](../../../../examples/cute/tutorial/xe_gemm.cpp) — The tutorial walked through here
- [`examples/cute/tutorial/xe_gemm_slm.cpp`](../../../../examples/cute/tutorial/xe_gemm_slm.cpp) — SLM (shared local memory) variant with double-buffering
- [`examples/00_bmg_gemm/`](../../../../examples/00_bmg_gemm/) — CUTLASS-level GEMM with collective mainloop and epilogue
- [`examples/README.md`](../../../../examples/README.md) — Full SYCL\*TLA example directory
- [`test/unit/cute/intel_xe/`](../../../../test/unit/cute/intel_xe/) — CuTe unit tests for Xe atoms
