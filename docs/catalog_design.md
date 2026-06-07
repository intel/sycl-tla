# B70 GEMM Catalog Design

## Overview

The screening catalog generates **1091 unique BF16 kernel configurations** from 3 sources,
packed into **546 batches** (2 kernels/batch, last batch has 1).

## Kernel Composition

| Group | Count | Source | Example |
|-------|-------|--------|---------|
| Hand-written | 9 | `benchmarks_sycl.hpp` direct `using` declarations | `BmgGemmBF16BF16FP32_RCR_18` |
| Gemm_ (RCR, SG exhaustive) | 76 | `.def` file × `BMG_SOURCE_GEMM_TILE_SG` + `BMG_DECLARE_EXPANDED_GEMM_TILES` | `RCR_Gemm_128x128x32_SG2x4` |
| Gemm_ (RRR, SG exhaustive) | 76 | Same as above but RRR layout | `RRR_Gemm_128x128x32_SG2x4` |
| GemmExhaustive (RCR, stages) | 783 | `exhaustive_regular_gemm_tile_candidates()` — full tile×sg×stages enumeration | `RCR_GemmExhaustive_128x256x32_SG2x8_ST2` |
| StreamK | 49 | shared `bmg_streamk_*.def` tile lists in full HPP | `RCR_StreamK_128x128x32` |
| DataParallel | 49 | same shared tile lists, different scheduler | `RCR_DataParallel_128x128x32` |
| SplitK | 49 | same shared tile lists, different scheduler | `RCR_SplitK_128x128x32` |

## Generation Pipeline

```
catalog.py
├── generated_expanded_streamk_kernel_catalog()
│   ├── Hand-written kernels (9)
│   ├── Gemm_ SG exhaustive via .def file (76 RCR + 76 RRR)
│   └── StreamK/DP/SplitK via shared bmg_streamk_*.def tile lists (49 each)
│
├── exhaustive_regular_gemm_tile_candidates()
│   └── GemmExhaustive: tile_m × tile_n × tile_k × sg_m × sg_n × stages (783 RCR)
│
└── Filter: bf16 only, exclude 03_bmg_example kernels
    → 1091 kernels → 546 batches
```

## Tile / Subgroup Search Space

### Gemm_ (SG exhaustive)
```
tile_m:  {8, 16, 32, 64, 128, 256, 512}
tile_n:  {32, 64, 96, 128, 192, 256, 512}
tile_k:  {16, 32, 64}
sg_m:    {1, 2, 4, 8}
sg_n:    {2, 4, 8}
stages:  {2}  (default)
filter:  is_valid_xe2_tile_sg() + valid_subgroup_sizes ∈ {16, 32}
```

### GemmExhaustive (tile×sg×stages)
```
tile_m:  {8, 16, 32, 64, 128, 256, 512}
tile_n:  {32, 64, 96, 128, 192, 256, 512}
tile_k:  {16, 32, 64}
sg_m:    {1, 2, 4, 8}
sg_n:    {2, 4, 8}
stages:  {1, 2, 3}
filter:  is_valid_xe2_tile_sg() + valid_subgroup_sizes ∈ {16, 32}
```

### StreamK/DP/SplitK
```
tile_m/n/k: from expanded streamk set (subset of Gemm tiles)
sg:         fixed 8×4 (StreamK template hardcode)
Note:       Generates all 3 scheduler variants per tile
```

## Compilation

Each batch compiles into a single binary:
1. `gen_mini_hpp.py` copies full `benchmarks_sycl.hpp` + adds batch-specific `BMG_DECLARE_*` calls
2. `cmake make -j128` compiles `main.cpp.o` (~2 min, perf flags baked via env vars)
3. `icpx` manually links (cmake fails at link due to GB stub)
4. Binary runs kernels sequentially via `run_direct()`

## Scheduler Types

| Scheduler | KernelSchedule | Behavior |
|-----------|---------------|----------|
| Gemm | `KernelXe` | Static tile assignment |
| StreamK | `KernelXeCooperative` | Dynamic tile dispatch, K/chunk streaming |
| DataParallel | `KernelXeCooperative` | Splits M×N grid, no reduction |
| SplitK | `KernelXeCooperative` | Splits K dimension, partial sums + reduce |

All four share the same `CollectiveMma<TileShape, TiledMma>` compute kernel.
Only `KernelSchedule` and `Scheduler` enum differ.
