# CuTe in SYCL\*TLA — Intel Overview

## CuTe in SYCL\*TLA (What it is)

> **Prerequisite:** If you are brand new to CuTe, read the
> [quickstart](00_quickstart.md) first for a high-level orientation.
> Note: the quickstart currently uses CUDA/NVCC terminology inherited from upstream CUTLASS —
> the concepts apply identically to SYCL. Substitute `sub_group` for `warp`,
> `work-group` for `threadblock`, and Intel DPC++ for NVCC.  A SYCL-first rewrite of the
> quickstart is planned.

CuTe in SYCL\*TLA is a collection of C++ SYCL template abstractions for defining and operating on
hierarchically multidimensional layouts of threads and data.

The two central objects are:

- **`Layout`**: a compile-time mapping from a logical coordinate space to a flat index.
  Layouts compose naturally — slicing, tiling, and transposing are pure algebra.
- **`Tensor`**: a `Layout` paired with a pointer to storage.  CuTe `Tensor`s handle all
  index arithmetic for you; you work in logical coordinates.

Together these building blocks let you express complex GEMM tiling hierarchies (global→SLM→register)
and epilogue fusions without hand-writing index calculations.

## Concept map

```
Layout ──► Layout Algebra ──► Tensor ──► Algorithms ──► Atoms ──► GEMM tutorial
                                                           │
                                                           ▼
                                                  Intel Xe Extensions
                                               (xe_2d_copy, XMX atoms)
```

> See the [Intel SYCL GEMM Companion](intel_gemm_companion.md#gemm-flow-diagram-with-intel-primitives)
> for this flow annotated with specific Intel Xe atom names.

## What's Intel-specific

The following components are unique to the Intel Xe path in this repository and are **not** part of
the upstream NVIDIA CUTLASS CuTe:

| Component | Location | Purpose |
|-----------|----------|---------|
| **Xe 2D block loads/stores/prefetch** | `xe_2d_copy.md`, `include/cute/arch/copy_xe_2d.hpp` | Hardware 2D block operations (`XE_LOAD_2D`, `XE_STORE_2D`, `XE_LOAD_2D_TRANSPOSE`, `XE_LOAD_2D_VNNI`, `XE_PREFETCH_2D`) — see [xe_2d_copy.md](xe_2d_copy.md) for naming reference, [intel_gemm_companion.md](intel_gemm_companion.md) for usage patterns |
| **XMX MMA atoms** (`XE_DPAS_TT`) | `include/cute/arch/mma_xe.hpp` (current), `include/cute/arch/mma_xe_legacy.hpp` (legacy `XE_8x16x16_*` structs) | Xe Matrix Extension compute atoms — see [intel_gemm_companion.md](intel_gemm_companion.md) for wiring patterns |
| **`SubgroupTensor`** | `include/cute/tensor_sg.hpp` | Intel-specific tensor type that scatters/gathers across subgroup lanes |
| **`TiledMMAHelper`** | `include/cute/atom/mma_atom.hpp` | Helper that constructs a `TiledMMA` from an Xe MMA atom and subgroup tile shape |

> **Legacy 2D copy API:** The legacy per-type-and-shape named structs (e.g., `XE_2D_U32x8x16_LD_N`)
> are documented in the [Legacy 2D Copy API](#legacy-2d-copy-api) section below.

## Intel Xe atoms and GEMM helpers

This section is a comprehensive reference for all Intel Xe building blocks an engineer
needs when writing GEMM kernels and using CuTe on Intel hardware.  Each subsection
covers one atom or helper: the header file location, the key template parameters, and
a minimal usage example.

---

### 1. MMA atoms — `XE_DPAS_TT`

**Header:** `include/cute/arch/mma_xe.hpp`

```cpp
template <int M, typename TypeD, typename TypeA, typename TypeB = TypeA, typename TypeC = TypeD>
struct XE_DPAS_TT;
```

| Parameter | Meaning |
|-----------|---------|
| `M` | Number of output rows; hardware supports 1–8. |
| `TypeD` | Accumulator output type (destination). |
| `TypeA` | A-matrix element type. |
| `TypeB` | B-matrix element type (defaults to `TypeA`). |
| `TypeC` | Accumulator input type (defaults to `TypeD`). |

`K` is derived automatically: `K = 256 / max(sizeof_bits(TypeA), sizeof_bits(TypeB))`.
The fixed N dimension for all CUTLASS-supported GPUs (PVC and later) is 16.

All type parameters use aliases from the `cute::dpas_type` namespace:

| Alias | Underlying type |
|-------|----------------|
| `f` | `float` (FP32) |
| `tf32` | `tfloat32_t` |
| `bf` | `bfloat16_t` |
| `hf` | `half_t` (FP16) |
| `ud` | `uint32_t` |
| `d` | `int32_t` |
| `u8` | `uint8_t` |
| `s8` | `int8_t` |
| `u4` | `uint4_t` |
| `s4` | `int4_t` |

#### Available type combinations

All specialisations declared in the header (using `CUTE_DECLARE_XE_DPAS_TT`):

| TypeD | TypeA | TypeB | TypeC | Description |
|-------|-------|-------|-------|-------------|
| `f` | `tf32` | `tf32` | `f` | TF32 × TF32 → FP32 |
| `f` | `bf` | `bf` | `f` | BF16 × BF16 → FP32 |
| `bf` | `bf` | `bf` | `f` | BF16 × BF16 → BF16 (FP32 accum input) |
| `f` | `bf` | `bf` | `bf` | BF16 × BF16 → FP32 (BF16 accum input) |
| `bf` | `bf` | `bf` | `bf` | BF16 × BF16 → BF16 |
| `f` | `hf` | `hf` | `f` | FP16 × FP16 → FP32 |
| `f` | `hf` | `hf` | `hf` | FP16 × FP16 → FP32 (FP16 accum input) |
| `hf` | `hf` | `hf` | `f` | FP16 × FP16 → FP16 (FP32 accum input) |
| `hf` | `hf` | `hf` | `hf` | FP16 × FP16 → FP16 |
| `ud` | `u8` | `u8` | `ud` | U8 × U8 → U32 |
| `d` | `u8` | `u8` | `d` | U8 × U8 → S32 |
| `d` | `u8` | `s8` | `d` | U8 × S8 → S32 |
| `d` | `s8` | `u8` | `d` | S8 × U8 → S32 |
| `d` | `s8` | `s8` | `d` | S8 × S8 → S32 |
| `ud` | `u8` | `u4` | `ud` | U8 × U4 → U32 |
| `d` | `u8` | `u4` | `d` | U8 × U4 → S32 |
| `d` | `u8` | `s4` | `d` | U8 × S4 → S32 |
| `d` | `s8` | `u4` | `d` | S8 × U4 → S32 |
| `d` | `s8` | `s4` | `d` | S8 × S4 → S32 |
| `ud` | `u4` | `u8` | `ud` | U4 × U8 → U32 |
| `d` | `u4` | `u8` | `d` | U4 × U8 → S32 |
| `d` | `u4` | `s8` | `d` | U4 × S8 → S32 |
| `d` | `s4` | `u8` | `d` | S4 × U8 → S32 |
| `d` | `s4` | `s8` | `d` | S4 × S8 → S32 |
| `ud` | `u4` | `u4` | `ud` | U4 × U4 → U32 |
| `d` | `u4` | `u4` | `d` | U4 × U4 → S32 |
| `d` | `u4` | `s4` | `d` | U4 × S4 → S32 |
| `d` | `s4` | `u4` | `d` | S4 × U4 → S32 |
| `d` | `s4` | `s4` | `d` | S4 × S4 → S32 |

#### Usage example

```cpp
#include <cute/arch/mma_xe.hpp>

// BF16 × BF16 → FP32 DPAS with 8 output rows (M=8, N=16, K=16)
using MmaOp = cute::XE_DPAS_TT<8, cute::dpas_type::f,
                                   cute::dpas_type::bf,
                                   cute::dpas_type::bf,
                                   cute::dpas_type::f>;

// Wrap in an MMA_Atom before passing to TiledMMAHelper or TiledMMA
using Atom = cute::MMA_Atom<MmaOp>;
```

> **Legacy note:** Existing examples may use the older named structs
> (`XE_8x16x16_F32BF16BF16F32_TT`, `XE_4x16x16_F32BF16BF16F32_TT`, etc.)
> from `include/cute/arch/mma_xe_legacy.hpp`.  These continue to work but
> `XE_DPAS_TT` is the current API for all new development.

---

### 2. Copy atoms — 2D block operations

**Header:** `include/cute/arch/copy_xe_2d.hpp`

Intel Xe provides hardware-accelerated 2D block load/store/prefetch instructions.
CuTe exposes five atom types:

```cpp
template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D;           // 2D block load (row-major)

template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D_VNNI;      // 2D block load with VNNI (packed) transform — for B matrix

template <int Bits, int Height, int Width>
struct XE_LOAD_2D_TRANSPOSE; // 2D block load with hardware transpose — for column-major A

template <int Bits, int Height, int Width>
struct XE_PREFETCH_2D;       // 2D block prefetch (no data returned)

template <int Bits, int Height, int Width>
struct XE_STORE_2D;          // 2D block store
```

| Parameter | Meaning |
|-----------|---------|
| `Bits` | Bits per element in the underlying memory operation (e.g., 16 for FP16/BF16, 32 for FP32). |
| `Height` | Number of rows (elements in the stride dimension of global memory). |
| `Width` | Number of columns (elements in the contiguous dimension of global memory). |
| `BlockWidth` | Optional sub-tiling of the width dimension into register blocks (default = `Width`). |

Each atom is parameterised on element width, not on the C++ type, so the same atom covers
any two types that share the same bit-width (e.g., `Bits=16` works for both `half_t` and
`bfloat16_t`).

`XE_LOAD_2D` exposes a `PREFETCH` member alias:
```cpp
using Prefetch = XE_LOAD_2D<Bits, Height, Width>::PREFETCH;  // → XE_PREFETCH_2D<Bits, Height, Width>
```

**Constraint:** `XE_STORE_2D` requires `Height <= 8`.

**`XE_LOAD_2D_VNNI` constraint:** `Bits` must be 8 or 16.

#### Usage example

```cpp
#include <cute/arch/copy_xe_2d.hpp>

// Load a 32-row × 32-column BF16 block from global memory (row-major A matrix)
using GmemCopyA = cute::XE_LOAD_2D<16, 32, 32>;

// Load a 32-row × 32-column BF16 block with VNNI transform (B matrix, DPAS-ready)
using GmemCopyB = cute::XE_LOAD_2D_VNNI<16, 32, 32>;

// Store an 8-row × 16-column FP32 block to global memory (epilogue output)
using GmemStoreC = cute::XE_STORE_2D<32, 8, 16>;
```

> **Full reference:** See [xe_2d_copy.md](xe_2d_copy.md) for the complete naming
> conventions, layout rules, and hardware constraints.

> **Legacy API:** The older per-type-and-shape named structs (e.g.,
> `XE_2D_U16x32x32_LD_N`, `XE_2D_U16x32x32_LD_V`) are documented in the
> [Legacy 2D Copy API](#legacy-2d-copy-api) section below.

---

### 3. `TiledMMAHelper`

**Header:** `include/cute/atom/mma_atom.hpp`

`TiledMMAHelper` constructs a `TiledMMA` from three ingredients:

```cpp
template <class MMA_Atom,   // e.g. MMA_Atom<XE_DPAS_TT<8, ...>>
          class CTALayout,  // Layout<WGTileShape>  — work-group tile
          class WarpLayout> // Layout<SGShape, SGStride> — subgroup arrangement
struct TiledMMAHelper {
  using TiledMMA = cute::TiledMMA<MMA_Atom, WarpLayout, /*computed Permutation*/>;
};
```

| Parameter | Meaning |
|-----------|---------|
| `MMA_Atom` | The wrapped DPAS atom, e.g. `MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>`. |
| `CTALayout` | `Layout<Shape<M, N, K>>` — the work-group (CTA) tile shape. |
| `WarpLayout` | A `Layout` that describes how subgroups tile the work-group, e.g. `Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>` for an 8×4 row-major arrangement. |

`TiledMMAHelper` automatically computes the permutation that makes each subgroup
operate on a single contiguous chunk of the work-group tile.

#### Usage example

```cpp
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_xe.hpp>
using namespace cute;

// Work-group tile: 256 × 256 × 32
using WGTile   = Shape<_256, _256, _32>;

// 8×4 subgroup layout (n-major arrangement for cache-line efficiency)
using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

// Assemble the TiledMMA
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>,
    Layout<WGTile>,
    SGLayout
>::TiledMMA;
```

Pass `TiledMma{}` to the collective mainloop or use it directly in a device kernel
via `get_slice` / `partition_A` / `partition_B` / `partition_C`.

---

### 4. `SubgroupTensor`

**Header:** `include/cute/tensor_sg.hpp`

`SubgroupTensor` distributes tensor storage across the lanes of an Intel subgroup.
It is the Intel equivalent of the per-thread register tile in CUDA CUTLASS: instead of
each work-item holding independent storage, `SubgroupTensor` models the collective
ownership of a tile by the whole subgroup (16 work-items).

It is used internally by CuTe's partitioning and copy machinery when targeting Intel Xe
copy atoms and MMA atoms.  Kernels written against the collective `CollectiveMma` API
(e.g., via `CollectiveBuilder`) do not need to instantiate `SubgroupTensor` directly;
it appears as a register-file abstraction when inspecting tensor types in device code.

## Recommended reading order

For engineers new to SYCL\*TLA CuTe, we recommend this sequence:

1. **[00_quickstart.md](00_quickstart.md)** — What CuTe is (see CUDA-first note above)
2. **This page** — Intel-specific context and concept map
3. **[examples/00_bmg_gemm](../../../../../examples/00_bmg_gemm/)** — Basic Intel GEMM example (start here for runnable code)
4. **[examples/01_bmg_gemm_with_collective_builder](../../../../../examples/01_bmg_gemm_with_collective_builder/)** — GEMM using the CollectiveBuilder API
5. **[intel_gemm_companion.md](intel_gemm_companion.md)** — SYCL / Intel Xe companion notes explaining the GEMM flow
6. **[xe_2d_copy.md](xe_2d_copy.md)** — Intel copy atom reference
7. **[intel_performance_guide.md](intel_performance_guide.md)** — Tuning and optimization
8. **[All examples](../../../../../examples/README.md)** — Full list of Intel programming examples

## Quick navigation (jump to any topic)

| Goal | Start here |
|------|-----------|
| **Learn CuTe concepts** | [01_layout.md](01_layout.md) → [02_layout_algebra.md](02_layout_algebra.md) → [03_tensor.md](03_tensor.md) → [04_algorithms.md](04_algorithms.md) |
| **Implement a GEMM** | [0x_gemm_tutorial.md](0x_gemm_tutorial.md) |
| **Explore compute atoms** | [0t_mma_atom.md](0t_mma_atom.md) |
| **Optimize memory movement on Intel** | [xe_2d_copy.md](xe_2d_copy.md) |
| **Tune for Intel GPU performance** | [intel_performance_guide.md](intel_performance_guide.md) |
| **SYCL GEMM companion notes** | [intel_gemm_companion.md](intel_gemm_companion.md) |
| **Run Intel examples** | [examples/README.md](../../../../../examples/README.md) — start with `00_bmg_gemm` or `01_bmg_gemm_with_collective_builder` |

> **Key concept:** Layout algebra ([02_layout_algebra.md](02_layout_algebra.md)) is the most important
> concept in CuTe — it powers all tiling, partitioning, and thread-to-data mapping. Functions like
> `logical_divide`, `composition`, and `complement` are how CuTe slices a global problem into
> per-subgroup work. If you read only one concept page, make it that one.

## Legacy 2D Copy API

> ⚠️ **Deprecation notice:** The legacy 2D copy API described below may be deprecated in a future
> release. New code should use the current API (`XE_LOAD_2D`, `XE_STORE_2D`, `XE_LOAD_2D_TRANSPOSE`,
> `XE_LOAD_2D_VNNI`, `XE_PREFETCH_2D`) defined in `include/cute/arch/copy_xe_2d.hpp`.

The legacy copy atoms use per-type-and-shape named structs (e.g., `XE_2D_U32x8x16_LD_N`,
`XE_2D_U16x8x16_LD_N`) defined in separate headers per element width:

| Header | Description |
|--------|-------------|
| `include/cute/arch/copy_xe_legacy_U16.hpp` | Legacy 16-bit 2D block load/store atoms |
| `include/cute/arch/copy_xe_legacy_U32.hpp` | Legacy 32-bit 2D block load/store atoms |
| `include/cute/arch/copy_xe_legacy_U8.hpp`  | Legacy 8-bit 2D block load/store atoms |
| `include/cute/arch/copy_xe_legacy_U64.hpp` | Legacy 64-bit 2D block load/store atoms |
| `include/cute/arch/copy_xe_legacy_U4.hpp`  | Legacy 4-bit 2D block load/store atoms |

Traits for these atoms live in `include/cute/atom/copy_traits_xe_legacy.hpp`. Existing code using
`XE_2D_*` atoms (e.g., in `xe_epilogue_legacy.hpp`) continues to work but should be migrated to
the new API when possible.

