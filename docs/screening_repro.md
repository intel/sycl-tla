# B70 GEMM Screening — Reproduction Guide

## Prerequisites

- B70 (BMG G31) hardware with Xe driver
- Intel oneAPI 2025.3 compiler
- cmake 3.28+, Python 3.12+
- 256+ CPU cores, 1TB RAM recommended

## Architecture

```
catalog.py ──→ 1091 kernel names ──→ 546 batches (2/batch)
                  ↓
gen_mini_hpp.py ──→ mini benchmarks_sycl.hpp (copies full HPP + batch declares)
                  ↓
cmake make ──→ main.cpp.o (~2 min, perf flags baked)
                  ↓
icpx link ──→ binary (~1s)
                  ↓
GPU screen ──→ CSV results (30s per kernel, sequential)
```

## File Map

| File | Role |
|------|------|
| `tools/gen_mini_hpp.py` | Generate per-batch mini HPP |
| `tools/reboot_screen.sh` | Main pipeline: build + screen |
| `test/.../catalog.py` | Generate exhaustive kernel list |
| `test/.../constraints.py` | Tile/SG constraints (B70: sg product ∈ {16,32}) |
| `cmake/googlebenchmark_stub.cmake` | GB stub for cmake (avoids library build) |
| `benchmarks/gemm/CMakeLists.txt` | cmake config with exhaustive gemm filter |

## Quick Start

```bash
# 1. Clone and setup
cd /root/cutlass_profile_device7_b70_2500mhz
git clone https://github.com/tinafengfun/sycl-tla.git
cd sycl-tla

# 2. Restore cmake dependencies
WS=/root/cutlass_profile_device7_b70_2500mhz/screen_ws
BDIR=/path/to/existing/cmake/build/selected_kernel_batch_001
mkdir -p _deps/googlebenchmark-src/include
cp -r $BDIR/_deps/googlebenchmark-src/include/benchmark _deps/googlebenchmark-src/include/
cp -r $BDIR/_deps/googletest-src/googletest _deps/googletest-src/
cp -r $BDIR/_deps/googletest-src/googlemock _deps/googletest-src/

# 3. Dry-run (4 batches)
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
BATCHES=4 bash tools/reboot_screen.sh

# 4. Full screening (546 batches, ~8 hours)
nohup bash -c "source /opt/intel/oneapi/compiler/2025.3/env/vars.sh; BATCHES=all bash tools/reboot_screen.sh" > screen.log 2>&1 &

# 5. Monitor
tail -f screen.log
```

## Environment Variables

| Variable | Value | Required |
|----------|-------|----------|
| `SYCL_PROGRAM_COMPILE_OPTIONS` | `-ze-opt-large-register-file -gline-tables-only` | Before make |
| `IGC_VectorAliasBBThreshold` | `10000` | Before make |
| `IGC_ExtraOCLOptions` | `-cl-intel-256-GRF-per-thread` | Before make |
| `ZE_AFFINITY_MASK` | `5` or `7` | Before screening |
| `PARALLEL` | `3` | Optional: parallel builds |
| `BATCHES` | `4` (dry-run) or `all` | Batch count |

## Kernel Name Taxonomy

| Pattern | Example | How definitions work |
|---------|---------|---------------------|
| Hand-written | `RCR_18`, `RRR_6` | Direct `using` in full HPP |
| Gemm_ SG | `Gemm_128x256x32_SG4x4` | .def file + BMG_DECLARE_GEMM_TILE_SG |
| GemmExhaustive | `GemmExhaustive_..._ST2` | gen_mini: BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE |
| DataParallel | `DataParallel_128x128x16` | gen_mini: direct GemmConfiguration (bypass StreamK template) |
| StreamK | `StreamK_128x128x16` | gen_mini: direct GemmConfiguration |
| SplitK | `SplitK_128x128x16` | gen_mini: direct GemmConfiguration |

## Verification Results (2026-05-28)

- **767/1091 kernels passed** (70%)
- **Top**: RRR_Gemm_128x128x32_SG2x4 = **154.9 TFLOPS**
- 57 failures: 15 invalid SG patterns (fixed) + 42 StreamK template issue (fixed)
- RRR avg 84.7 vs RCR avg 75.4 TFLOPS (RRR +12%)

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| 4 TFLOPS | Perf flags not baked | Export env vars BEFORE make |
| `compiler_depend.make` missing | cmake hard include | `touch` to create |
| `no template named Gemm_Bench_BF16FP32_RCR_StreamK` | StreamK template conflict | gen_mini bypasses with direct GemmConfiguration |
| SG8x8 kernel compile fail | sg product 64 > HW max 32 | constraints: valid_subgroup_sizes=[16,32] |
| Results dir empty | dir deleted mid-run | mkdir in batch loop |

## Reproduce Top Result Manually

The #1 kernel `BmgGemmBF16BF16FP32_RRR_Gemm_128x128x32_SG2x4` = **154.9 TFLOPS** is macro-generated (not hand-written).

### Configuration
| Parameter | Value |
|-----------|-------|
| Data type | BF16 × BF16 → FP32 |
| Layout | RRR (A: RowMajor, B: RowMajor) |
| Tile | 128 × 128 × 32 |
| Subgroup | 2 × 4 |
| Atom | `XE_DPAS_TT<8, float, cute::bfloat16_t>` |
| Pipeline | 2 (default) |
| Problem | M=8192, N=4096, K=1536 |

### Step-by-step Reproduction

```bash
# 1. Setup environment
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh

cd /root/cutlass_profile_device7_b70_2500mhz/sycl-tla

# 2. Generate mini HPP for just this kernel
echo "BmgGemmBF16BF16FP32_RRR_Gemm_128x128x32_SG2x4" > /tmp/top1.txt
python3 tools/gen_mini_hpp.py --manifest /tmp/top1.txt --output /tmp/top1.hpp

# 3. Write minimal main.cpp
cat > benchmarks/gemm/main.cpp << 'EOF'
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include <iostream>
#include "benchmark_runner.hpp"
#if defined(SYCL_INTEL_TARGET)
#include "benchmarks_sycl.hpp"
#endif
int main(int argc, const char** argv) {
  cutlass::CommandLine cmd(argc, argv);
  std::string kernel; cmd.get_cmd_line_argument("kernel", kernel, std::string(""));
  if (kernel.empty()) { std::cerr << "--kernel=NAME" << std::endl; return 1; }
  register_gemm_benchmarks();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  cutlass::benchmark::GEMMOptions opts;
  cmd.get_cmd_line_argument("m", opts.m, 8192);
  cmd.get_cmd_line_argument("n", opts.n, 4096);
  cmd.get_cmd_line_argument("k", opts.k, 1536);
  opts.verify_library = 0;
  double tflops = 0; bool ok = false;
#define RUN(K) if (kernel == #K) { tflops = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct(opts, hw); ok = true; }
  RUN(BmgGemmBF16BF16FP32_RRR_Gemm_128x128x32_SG2x4)
#undef RUN
  if (!ok) { std::cerr << "NOT_FOUND" << std::endl; return 1; }
  std::cout << "RESULT: kernel=" << kernel << " median_tflops=" << tflops << std::endl;
  return 0;
}
EOF

# 4. Build
BDIR=/root/cutlass_profile_device7_b70_2500mhz/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001
GB_LIB=$BDIR/_deps/googlebenchmark-build/src/libbenchmark.a
CUTLASS_LIB=$BDIR/tools/library/libcutlass.a

rm -f $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o
touch $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts
make -C $BDIR cutlass_benchmarks_gemm_sycl -j128

# 5. Link (make fails at link — expected, do manual link)
OBJ=$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -O3 $OBJ -o /tmp/top1_bin $GB_LIB -L/lib64/stubs -Wl,-rpath,/lib64/stubs: $CUTLASS_LIB \
  -Wl,-rpath=/opt/intel/oneapi/mkl/2025.3/lib \
  /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_ilp64.so \
  /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_thread.so \
  /opt/intel/oneapi/mkl/2025.3/lib/libmkl_core.so \
  /opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so \
  -lm -ldl -lpthread /opt/intel/oneapi/compiler/2025.3/lib/libsycl.so

# 6. Run
ZE_AFFINITY_MASK=5 /tmp/top1_bin \
  --kernel=BmgGemmBF16BF16FP32_RRR_Gemm_128x128x32_SG2x4 \
  --m=8192 --n=4096 --k=1536
# Expected: ~154.9 TFLOPS
```

### Generated Kernel Type

The macro `BMG_DECLARE_GEMM_TILE_SG` expands to:
```cpp
using BmgGemmBF16BF16FP32_RRR_Gemm_TileShape_128x128x32 = Shape<_128, _128, _32>;

using BmgGemmBF16BF16FP32_RRR_Gemm_Tile_128x128x32_SG2x4 =
    TiledMMAHelper<
        MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
        Layout<Shape<_128, _128, _32>>,
        Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>
    >::TiledMMA;

using BmgGemmBF16BF16FP32_RRR_Gemm_128x128x32_SG2x4 =
    Gemm_Bench_BF16FP32_RRR<
        Shape<_128, _128, _32>,
        BmgGemmBF16BF16FP32_RRR_Gemm_Tile_128x128x32_SG2x4,
        void, void>;
```

The `Gemm_Bench_BF16FP32_RRR` template wraps `GemmConfiguration` with:
- Arch: `IntelXe`
- A: `bfloat16_t, RowMajor`
- B: `bfloat16_t, RowMajor` ← RRR key difference
- C/D: `float, RowMajor`
- Scheduler: `Gemm` (default)
- Epilogue: `LinearCombination<float,float,float,float>`
- PipelineStages: 2 (default)
