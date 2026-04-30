# Intel XPU profiler migration docs

This directory documents the full `tools/profiler` migration to Intel XPU/SYCL.

## Goals

1. Cover **all profiler code**, not just GEMM mainline files.
2. Separate **mechanical porting work** from the few real design decisions.
3. Provide file-by-file migration handbooks that can be executed without re-discovery.

## Document map

| File | Purpose |
|---|---|
| `01_profiler_inventory.md` | Complete source/header/build inventory and classification |
| `02_gpu_timer_analysis.md` | `GpuTimer` / `SYCLTimer` / event timing |
| `03_device_allocation_analysis.md` | Device memory, tensor init, compare, copies |
| `04_device_context_analysis.md` | Device enumeration and device properties |
| `05_operation_profiler_analysis.md` | Shared profiling loop, timing, graphs, streams |
| `06_gemm_operation_profiler_analysis.md` | GEMM-specific workspace, verification, profiling |
| `07_grouped_gemm_operation_profiler_analysis.md` | Grouped GEMM profiler |
| `08_block_scaled_gemm_operation_profiler_analysis.md` | BlockScaled GEMM profiler |
| `09_blockwise_gemm_operation_profiler_analysis.md` | Blockwise GEMM profiler |
| `10_rank_k_rank_2k_operation_profiler_analysis.md` | RankK and Rank2K profilers |
| `11_trmm_symm_operation_profiler_analysis.md` | TRMM and SYMM profilers |
| `12_conv2d_conv3d_operation_profiler_analysis.md` | Conv2d and Conv3d profilers |
| `13_sparse_gemm_operation_profiler_analysis.md` | Sparse GEMM profiler |
| `14_options_and_entry_analysis.md` | CLI/options, main entry, profiler bootstrap |
| `15_cublas_cudnn_helpers_analysis.md` | Vendor helper files and disable strategy |
| `16_build_system_analysis.md` | `tools/profiler/CMakeLists.txt` and build wiring |
| `17_generator_search_gap_analysis.md` | Generator/search-space gap vs current Intel path |
| `18_intel_xpu_migration_design.md` | Consolidated migration design |

## Working rules

Profiler migration work is split into three labels:

1. **Mechanical**: mostly `cuda* -> compat::* / sycl::*`.
2. **Remove/Disable**: paths with no Intel equivalent, e.g. CUDA Graph, cuBLAS/cuDNN helpers.
3. **Build wiring**: source suffixes, target dependencies, provider/test defaults.

The intent is to make this directory a completeness gate: no profiler source should be migrated without a matching note here, and no file should be omitted because it is not on the GEMM critical path.
