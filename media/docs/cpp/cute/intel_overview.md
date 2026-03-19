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

### Intel Xe MMA atoms

Xe MMA atoms use the `XE_DPAS_TT<M, TypeD, TypeA, TypeB, TypeC>` template
(defined in `include/cute/arch/mma_xe.hpp`), where `M` is the number of output rows
and types use the `dpas_type` namespace aliases (`f`, `bf`, `hf`, `tf32`, `u8`, `s8`, `u4`, `s4`).
K is computed automatically as `256 / max(sizeof_bits(TypeA), sizeof_bits(TypeB))`.

For example, `XE_DPAS_TT<8, dpas_type::f, dpas_type::bf, dpas_type::bf, dpas_type::f>` performs
a BF16 × BF16 → FP32 DPAS with 8 output rows.

> **Legacy note:** Existing examples may use the older named structs
> (`XE_8x16x16_F32BF16BF16F32_TT`, `XE_4x16x16_F32BF16BF16F32_TT`, etc.)
> from `include/cute/arch/mma_xe_legacy.hpp`. These continue to work but
> `XE_DPAS_TT` is the current API for new development.

### SubgroupTensor

`SubgroupTensor` (from `include/cute/tensor_sg.hpp`) distributes tensor storage across the lanes of
an Intel subgroup.  It is the Intel equivalent of the per-thread register tile used in CUDA CUTLASS.

### TiledMMAHelper

`TiledMMAHelper` (from `include/cute/atom/mma_atom.hpp`) wraps the low-level `MMA_Atom` with
subgroup tile size information to produce the `TiledMMA` object used in GEMM kernels.

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

