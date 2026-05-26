# RRR Profiler vs Example (00_bmg_gemm) Alignment Debug Package

## Purpose
This directory contains all source code, build configuration, and runtime configuration needed to
reproduce and debug the RRR profiler kernel performance gap vs the inner-source `00_bmg_gemm` example.

## Performance Summary

| Binary | Kernel Type | Tile | SG | Performance @ 8192³ |
|--------|------------|------|-----|---------------------|
| Inner-source `00_bmg_gemm` | TiledMMAHelper + XE_DPAS_TT<8,float,bf16> | 256×256×32 | 8×4 | **122 TFLOPS** |
| Profiler `RRR_6` (TiledMMAHelper) | TiledMMAHelper + MMAAtom (XE_8x16x16_F32BF16BF16F32_TT) | 256×256×32 | 8×4 | **69.6 TFLOPS** (one build), 4.4 TFLOPS (rebuilds) ⚠️ |
| Profiler `RRR_TileShape_512_256_32` | TiledMMAHelper + XE_DPAS_TT<8,float,bf16> | 512×256×32 | 8×4 | 1.7 TFLOPS (tile too large) |

## Key Files

### Profiler Source (sycl-tla)
- `benchmarks_sycl.hpp` — Kernel type definitions. RRR_6 added at line ~264 (before BmgTile_1)
- `main.cpp` — GB-free profiler entry point using `run_direct()`
- `benchmark_runner.hpp` — Contains `run_direct()` method (line 1159)
- `gemm_configuration_sycl.hpp` — `GemmConfiguration` template wrapping `GemmUniversalAdapter`
- `CMakeLists_benchmarks_gemm.txt` — CMake build rules for benchmarks/gemm
- `cutlass_benchmark_filter.hpp` — Kernel filter enabling only RRR_6

### Inner-Source Reference (libraries.ai.cutlass.internal, master_next)
- `inner_source_00_bmg_gemm.cpp` — The 122 TFLOPS example
- `inner_source_benchmarks_sycl.hpp` — Inner-source kernel registry
- `inner_source_gemm_configuration_sycl.hpp` — Inner-source GemmConfiguration
- `inner_source_benchmark_runner.hpp` — Inner-source benchmark runner

### Build Configuration
- `CMakeCache.txt` — Full cmake cache from working batch_001 build
- `build_flags.make` — Actual compiler/linker flags used
- `build_configure.log` — cmake configure output
- `build_cmake_output.txt` — cmake command used

### Runtime Configuration
- `runtime_env.sh` — Environment variables for B70 GPU 7 at 2500MHz
- `run_cmd.sh` — Command to run the profiler kernel

## Build Command
```bash
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
cmake -S <sycl-tla-root> -B <build-dir> \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
  -DDPCPP_HOST_COMPILER=g++-13 \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DCUTLASS_NVCC_ARCHS= \
  -DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON \
  -DCUTLASS_BENCHMARK_EXHAUSTIVE_GEMM=ON \
  -DCUTLASS_KERNEL_FILTER_FILE=<build-dir>/benchmarks/gemm/cutlass_benchmark_filter.hpp \
  -DGOOGLETEST_DIR=<deps>/googletest-src \
  -DGOOGLEBENCHMARK_DIR=<deps>/googlebenchmark-src
make -C <build-dir> cutlass_benchmarks_gemm_sycl -j4
```

## Run Command
```bash
source runtime_env.sh
./cutlass_benchmarks_gemm_sycl --kernel=BmgGemmBF16BF16FP32_RRR_6 --m=8192 --n=8192 --k=4096
```

## RRR_6 Kernel Definition (added to benchmarks_sycl.hpp)
```cpp
// RRR kernel using TiledMMAHelper (auto-tuned tile) — matches inner-source approach
using BmgGemmBF16BF16FP32_RRR_TiledMMAHelper_256x256x32_Tile = typename TiledMMAHelper<
    MMAAtom,                              // MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>
    Layout<Shape<_256, _256, _32>>,       // Tile shape
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>  // SG 8×4
>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_6 = Gemm_Bench_BF16FP32_RRR<
    Shape<_256, _256, _32>,
    BmgGemmBF16BF16FP32_RRR_TiledMMAHelper_256x256x32_Tile,
    void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RRR_6);
```

## Known Differences vs Inner-Source Example

1. **DispatchPolicy**: Profiler uses `MainloopXeL1Staged<2, KernelXeDefault>`, example uses `dispatch_2d_stage_padding`
2. **Stride Type**: Profiler's `GemmConfiguration` uses `TagToStrideA_t` conversion, example uses raw `Stride<int64_t, _1, int64_t>`
3. **Epilogue**: Profiler uses `CollectiveEpilogue<IntelXeGeneric, ...>`, example uses `XeEpilogue<...>`
4. **GemmKernel wrapping**: Profiler goes through `GemmConfiguration → GemmUniversalAdapter`, example uses `GemmUniversalAdapter` directly
5. **Build Target**: Example built with `intel_gpu_bmg_g21`, profiler with `intel_gpu_bmg_g31`
6. **MMA Atom**: Profiler uses `XE_8x16x16_F32BF16BF16F32_TT`, example uses `XE_DPAS_TT<8, float, bfloat16_t>` (should be equivalent)

## Build System Issue
The profiler binary gives inconsistent results (69.6 vs 4.4 TFLOPS) depending on build state.
IGC cache ("Compilation from IR - skipping loading of FCL") may be reusing stale cached code
from previous kernel configurations. A clean build from scratch (no cached IR) is needed for
reliable results.
