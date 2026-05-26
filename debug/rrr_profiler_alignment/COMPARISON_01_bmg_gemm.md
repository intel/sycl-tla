# Profiler RRR_6 vs Example 01 (collective_builder) — Detailed Code Comparison

## 23 Debug Package Files

| # | File | Description |
|---|------|-------------|
| 1 | `README.md` | Full package documentation |
| 2 | `profiler_benchmarks_sycl.hpp.modified` | **Actual compiled** benchmarks_sycl.hpp from B70 remote (with RRR_6 TiledMMAHelper) |
| 3 | `benchmarks_sycl.hpp` | Local sycl-tla benchmarks_sycl.hpp (before patch) |
| 4 | `rrr6_kernel_diff.patch` | Exact diff adding RRR_6 TiledMMAHelper |
| 5 | `profiler_main.cpp.modified` | **Actual compiled** main.cpp (GB-free profiler entry) |
| 6 | `main.cpp` | Local main.cpp |
| 7 | `main_cpp_diff.patch` | main.cpp diff |
| 8 | `benchmark_runner.hpp` | Profiler's run_direct() method |
| 9 | `gemm_configuration_sycl.hpp` | GemmConfiguration wrapper template |
| 10 | `CMakeCache.txt` | Full cmake cache from working batch_001 build |
| 11 | `CMakeLists_benchmarks_gemm.txt` | CMake build rules |
| 12 | `build_flags.make` | Actual compiler/linker flags |
| 13 | `build_configure.log` | cmake configure output |
| 14 | `build_cmake_output.txt` | cmake command used |
| 15 | `cutlass_benchmark_filter.hpp` | Kernel filter (only RRR_6 enabled) |
| 16 | `runtime_env.sh` | Runtime env vars for B70 GPU7 @ 2500MHz |
| 17 | `run_cmd.sh` | Run command |
| 18 | `inner_source_00_bmg_gemm.cpp` | Inner-source 00_bmg_gemm reference (122 TFLOPS) |
| 19 | `inner_source_benchmarks_sycl.hpp` | Inner-source kernel registry |
| 20 | `inner_source_benchmark_runner.hpp` | Inner-source benchmark runner |
| 21 | `inner_source_gemm_configuration_sycl.hpp` | Inner-source GemmConfiguration |
| 22 | `ali_dataset.py` | Modified — RCR+RRR dual layout generation |
| 23 | `build_ali_gemm_dataset.py` | Modified — `--layouts` flag |

---

## Structural Comparison: Profiler RRR_6 vs Example 01 (collective_builder)

### Architecture Overview

```
Example 01:           Profiler RRR_6:
=================     =================
CollectiveBuilder     GemmConfiguration
  ├─ auto MMA atom      ├─ explicit TiledMMAHelper<MMAAtom>
  ├─ auto copy ops      ├─ void GmemTiledCopy (auto)
  ├─ auto schedule      ├─ MainloopXeL1Staged<2, KernelXeDefault>
  └─ auto stages        └─ IntelXeGeneric epilogue
       │                      │
       ▼                      ▼
CollectiveMainloop     CollectiveMainloop (CollectiveMma)
CollectiveEpilogue     CollectiveEpilogue
       │                      │
       ▼                      ▼
GemmUniversal<>        GemmUniversal<>
       │                      │
       ▼                      ▼
GemmUniversalAdapter   GemmUniversalAdapter
```

### Detailed Type-by-Type Comparison

#### 1. Mainloop Dispatch Policy

| | Example 01 (CollectiveBuilder) | Profiler (GemmConfiguration) |
|---|---|---|
| **Type** | `KernelScheduleAuto` (builder selects) | `MainloopXeL1Staged<2, KernelXeDefault>` |
| **Stages** | Auto-selected (optimized for tile/problem) | Hardcoded `PipelineStages=2` |
| **Kernel Schedule** | Auto-selected | Hardcoded `KernelXeDefault` |

**Impact**: Example 01 lets the builder pick optimal pipeline stages and schedule. Profiler always uses 2 stages with default schedule. For RRR (RowMajor B), the optimal stage count might differ from RCR.

#### 2. MMA Subgroup Layout

| | Example 01 | Profiler RRR_6 |
|---|---|---|
| **Approach** | Builder auto-selects (from arch + dtype + tile) | Explicit `TiledMMAHelper<MMAAtom, Layout<256x256x32>, Layout<SG8x4>>` |
| **Atom** | Builder internal | `MMAAtom = XE_8x16x16_F32BF16BF16F32_TT` |

After builder expansion, both should produce the same MMA atom and SG layout. **Should be equivalent.**

#### 3. Epilogue Dispatch Policy

| | Example 01 | Profiler |
|---|---|---|
| **Type** | `EpilogueScheduleAuto` (builder selects) | `IntelXeGeneric` (hardcoded) |
| **Fusion** | `LinCombEltAct<ReLu>` | `LinearCombination` (no activation) |
| **Tile** | `EpilogueTileAuto` | `void` (auto) |

**Impact**: Different epilogue dispatch policies may affect epilogue tile sizing and memory access patterns.

#### 4. Stride Types

| | Example 01 | Profiler |
|---|---|---|
| **StrideA extract** | `typename Gemm::GemmKernel::StrideA` (from GemmUniversal) | `TagToStrideA_t<LayoutA>` (converts RowMajor → Stride tuple) |
| **StrideB** | Same pattern | `TagToStrideB_t<LayoutB>` |

**Should be equivalent** — both produce packed row-major strides.

#### 5. GmemTiledCopy (Global Memory Copy Atoms)

| | Example 01 | Profiler |
|---|---|---|
| **CopyA** | Builder auto-selects | `void` → selects copy atom automatically |
| **CopyB** | Builder auto-selects | `void` → selects copy atom automatically |

**Should be equivalent** for the default (void) case — the collective builder selects based on dtype/alignment.

#### 6. GemmUniversal Arguments

| | Example 01 | Profiler |
|---|---|---|
| **Mode** | `kGemm` | `kGemm` |
| **ProblemShape** | `{m, n, k, 1}` | `{options.m, options.n, options.k, options.l}` |
| **Mainloop** | `{A_ptr, stride_A, B_ptr, stride_B}` | Same pattern |
| **Epilogue** | `{{alpha, beta}, C_ptr, stride_C, D_ptr, stride_D}` | Same pattern |

**Identical.**

#### 7. Runtime Pattern

| | Example 01 | Profiler run_direct() |
|---|---|---|
| **Warmup** | 1 run + wait | 100 runs + wait |
| **Measure** | N runs, 1 wait, timer.seconds()/N | 100 runs, 1 wait, timer.seconds()/100 |
| **Timer** | `GPU_Clock` (SYCLTimer → chrono) | Same `GPU_Clock` |

**Identical timing approach.**

### Critical Difference Summary

The **primary difference** is in how the collective mainloop and epilogue are constructed:

| Aspect | Example 01 | Profiler | Risk |
|--------|-----------|----------|------|
| Dispatch Policy | `CollectiveBuilder` auto-selects | `MainloopXeL1Staged<2, KernelXeDefault>` | **HIGH** — fixed 2 stages may be suboptimal for RRR |
| Epilogue Policy | `EpilogueScheduleAuto` | `IntelXeGeneric` | **MEDIUM** |
| Kernel Schedule | `KernelScheduleAuto` | `KernelXeDefault` | **MEDIUM** |
| MMA/Copy selection | Builder auto | Explicit TiledMMAHelper + void copies | LOW |

### Actionable Fix

To completely align profiler RRR_6 with example 01, the `GemmConfiguration` template should use `CollectiveBuilder` instead of manually constructing `CollectiveMma` with `MainloopXeL1Staged`. This would:

1. Auto-select the optimal pipeline stages for RRR layout
2. Auto-select the optimal kernel schedule
3. Auto-select the optimal epilogue dispatch policy

### Specific Code Change

In `gemm_configuration_sycl.hpp`, replace:

```cpp
using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages, KernelSchedule>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

using CollectiveMainloop = collective::CollectiveMma<
    GEMMDispatchPolicy, TileShape,
    ElementA, StrideA, ElementB, StrideB, TiledMma,
    GmemTiledCopyA, void, void, identity,
    GmemTiledCopyB, void, void, identity>;
```

With:

```cpp
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, /*alignment=*/sizeof(ElementA),
    ElementB, LayoutB, /*alignment=*/sizeof(ElementB),
    ElementAccumulator,
    TileShape, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

And similarly for `CollectiveEpilogue`.
