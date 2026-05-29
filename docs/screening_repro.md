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
