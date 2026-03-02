# CuTe in SYCL\*TLA — Intel Overview

## CuTe in SYCL\*TLA (What it is)

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

## What's Intel-specific

The following components are unique to the Intel Xe path in this repository and are **not** part of
the upstream NVIDIA CUTLASS CuTe:

| Component | Location | Purpose |
|-----------|----------|---------|
| **Xe 2D block loads/stores/prefetch** | `xe_2d_copy.md`, `include/cute/arch/copy_xe_legacy_U16.hpp`, `include/cute/arch/copy_xe_legacy_U32.hpp`, `include/cute/arch/copy_xe_2d.hpp` (new unified API) | Hardware 2D block operations (XE_2D_*_LD_N/T/V, XE_2D_*_ST_N) |
| **XMX MMA atoms** (`XE_8x16x16_*`) | `include/cute/arch/mma_xe_legacy.hpp` | Xe Matrix Extension compute atoms for BF16, FP16, FP8, INT8 |
| **`SubgroupTensor`** | `include/cute/tensor_sg.hpp` | Intel-specific tensor type that scatters/gathers across subgroup lanes |
| **`TiledMMAHelper`** | `include/cute/atom/mma_atom.hpp` | Helper that constructs a `TiledMMA` from an Xe MMA atom and subgroup tile shape |

> **Legacy vs. new 2D copy API:** The table above lists both the legacy and new copy headers.
>
> - **Legacy API** (`copy_xe_legacy_U16.hpp`, `copy_xe_legacy_U32.hpp`): Uses named structs per
>   size/type/layout combination — e.g., `XE_2D_U16x32x32_LD_V`, `XE_2D_U32x8x16_ST_N`.
>   All existing examples and tests in this repository use the legacy API.
> - **New unified API** (`copy_xe_2d.hpp`): Parameterized templates —
>   e.g., `XE_LOAD_2D<Bits, Height, Width>`. This is the future direction and supports
>   new atom features like subtiling and size-1 fragments.
>
> For new kernel development, check whether the new API covers your use case.  For understanding
> existing code and examples, refer to the legacy headers.

### Intel Xe MMA atoms

Xe MMA atoms follow the naming convention `XE_8x16x16_<AccumType><AType><BType><CType>_<Layout>`.
For example `XE_8x16x16_F32BF16BF16F32_TT` accumulates FP32 from BF16 A and BF16 B operands.
These are defined in `include/cute/arch/mma_xe_legacy.hpp`.

### SubgroupTensor

`SubgroupTensor` (from `include/cute/tensor_sg.hpp`) distributes tensor storage across the lanes of
an Intel subgroup.  It is the Intel equivalent of the per-thread register tile used in CUDA CUTLASS.

### TiledMMAHelper

`TiledMMAHelper` (from `include/cute/atom/mma_atom.hpp`) wraps the low-level `MMA_Atom` with
subgroup tile size information to produce the `TiledMMA` object used in GEMM kernels.

## Choose your path (engineer navigation)

| Goal | Start here |
|------|-----------|
| **Learn CuTe concepts** | [01_layout.md](01_layout.md) → [02_layout_algebra.md](02_layout_algebra.md) → [03_tensor.md](03_tensor.md) → [04_algorithms.md](04_algorithms.md) |
| **Implement a GEMM** | [0x_gemm_tutorial.md](0x_gemm_tutorial.md) |
| **Explore compute atoms** | [0t_mma_atom.md](0t_mma_atom.md) |
| **Optimize memory movement on Intel** | [xe_2d_copy.md](xe_2d_copy.md) |
| **Tune for Intel GPU performance** | [intel_performance_guide.md](intel_performance_guide.md) |
| **SYCL GEMM companion notes** | [intel_gemm_companion.md](intel_gemm_companion.md) |

> **Key concept:** Layout algebra ([02_layout_algebra.md](02_layout_algebra.md)) is the most important
> concept in CuTe — it powers all tiling, partitioning, and thread-to-data mapping. Functions like
> `logical_divide`, `composition`, and `complement` are how CuTe slices a global problem into
> per-subgroup work. If you read only one concept page, make it that one.

If you are new to CuTe, start with the
[quickstart](00_quickstart.md) before reading this overview.
