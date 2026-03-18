# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SYCL*TLA (SYCL Templates for Linear Algebra)** is a fork of NVIDIA CUTLASS that extends CUTLASS and CuTe API support to Intel GPUs through SYCL enablement. This is a header-only C++ template library for high-performance GEMM and fused epilogue kernels, applying hierarchical tiling and composable abstractions for dense linear algebra on Intel Xe GPUs (PVC and BMG).

**Key Architecture**: The project adapts CUTLASS 3.x's composition model for Intel GPUs:
```
GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>
```

- **CollectiveMainloop**: MMA tile computation using Intel DPAS (Dot Product Accumulate Systolic) instructions
- **CollectiveEpilogue**: Post-MMA epilogue with fusion support via visitor pattern
- **TileScheduler**: Work partitioning with dynamic load balancing (Stream-K)
- **CuTe**: Compile-time tensor layout library for composing layouts, strides, and copy operations

## Build System

### Environment Setup (Required First)

```bash
source /opt/intel/oneapi/setvars.sh
export CXX=icpx
export CC=icx
```

### CMake Configuration

**Standard Build (PVC - Intel Data Center GPU Max):**
```bash
rm -rf build && mkdir build && cd build
cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_pvc \
  -DCUTLASS_SYCL_RUNNING_CI=ON
ninja
```

**Build for BMG (Intel Arc B580):**
```bash
cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCUTLASS_SYCL_RUNNING_CI=ON
ninja
```

**With G++ as Host Compiler:**
```bash
cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_pvc \
  -DDPCPP_HOST_COMPILER=g++-13 \
  -DCUTLASS_SYCL_RUNNING_CI=ON
ninja
```

**CI-exact CMake command** (use as reference for local CI reproduction):
```bash
cmake -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DIGC_VERSION_MAJOR=2 \
  -DIGC_VERSION_MINOR=18 \
  -DCMAKE_CXX_FLAGS="-Werror" \
  -DCUTLASS_SYCL_RUNNING_CI=ON
```
For PVC, use `-DDPCPP_SYCL_TARGET=intel_gpu_pvc -DIGC_VERSION_MINOR=11`.

**Important Notes:**
- Always use a clean build directory (`rm -rf build`) to avoid CMake cache issues
- `-DDPCPP_SYCL_TARGET` determines hardware-specific intrinsics: `intel_gpu_pvc` for PVC, `intel_gpu_bmg_g21` for BMG
- Build time: ~10-20 minutes on 8-core machine
- If Intel oneAPI unavailable, CMake can catch syntax errors but linking will fail

### Runtime Environment Variables

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=100000000000
```

## Testing

### C++ Unit Tests

**Run all unit tests:**
```bash
cmake --build . --target test_unit -j 8
```

**Run single test suite:**
```bash
cmake --build . --target cutlass_test_unit_gemm_device -j 8
./test/unit/gemm/device/cutlass_test_unit_gemm_device
```

**Run specific test case:**
```bash
./test/unit/gemm/device/cutlass_test_unit_gemm_device --gtest_filter='*BF16*'
```

Test binaries are in `build/test/unit/<subdir>/`. Use `--gtest_filter` for granular testing.

### C++ Examples

```bash
cmake --build . --target test_examples -j 1
```

**Run single example:**
```bash
ninja 00_bmg_gemm
./examples/00_bmg_gemm/00_bmg_gemm
```

### Python Tests

**Setup:**
```bash
pip install -e .  # Install dependencies from pyproject.toml
```

**Run all Python tests:**
```bash
export CUTLASS_USE_SYCL=1
cd python && python3 -m pytest -q
```

**Run single test file:**
```bash
export CUTLASS_USE_SYCL=1
python3 -m pytest test/python/cutlass/gemm/gemm_bf16_xe20.py -v
```

**Runtime library path for Python tests:**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:build/tools/library
```

## Code Organization

### Core CUTLASS Components

```
include/cutlass/
├── arch/              # Intel GPU architecture features (instruction-level GEMMs)
├── gemm/              # GEMM kernels specialized for Intel GPUs
│   ├── kernel/        # xe_gemm.hpp - main GemmUniversal kernel
│   └── collective/    # xe_* - MMA collective implementations
├── epilogue/          # Epilogue operations for GEMM/conv (xe_visitor.hpp for fusion)
├── layout/            # Memory layout definitions
└── platform/          # SYCL-capable Standard Library components

include/cute/
├── algorithm/         # Core operations (copy, gemm, tuple operations)
├── arch/              # Intel GPU arch wrappers for copy/math instructions
├── atom/              # Copy and MMA atoms for Intel Xe
│   ├── mma_atom.hpp   # cute::Mma_Atom and cute::TiledMma
│   ├── copy_atom.hpp  # cute::Copy_Atom and cute::TiledCopy
│   └── *xe*.hpp       # Intel Xe specific meta-information
└── [Layout/Tensor]    # Core CuTe types
```

### Examples Structure

```
examples/
├── 00_bmg_gemm/                         # Basic GEMM (BF16/FP32)
├── 01_bmg_gemm_with_collective_builder/ # CollectiveBuilder usage
├── 02_bmg_gemm_mixed_dtype/             # Mixed precision + dequantization
├── 03_bmg_gemm_streamk/                 # Stream-K scheduler
├── 04_bmg_grouped_gemm/                 # Grouped GEMM
├── 05_bmg_gemm_with_epilogues/          # Epilogue fusion (GELU, ReLU, etc.)
├── 06_bmg_flash_attention/              # Flash Attention V2
├── 07_bmg_dual_gemm/                    # Fused dual GEMM
└── 08-10_*/                             # FP8 and grouped variants
```

### Python Components

```
python/
├── cutlass_cppgen/       # Python interface for compiling/running CUTLASS on Intel GPUs
├── cutlass_library/      # Kernel enumeration and C++ code generation
│   ├── generator.py      # Kernel generation/filtering logic
│   └── arch_constants.py # Architecture detection (INTEL_XE12=12, INTEL_XE20=20)
└── test/python/          # Python test suite
```

### Tests

```
test/unit/
├── cute/
│   └── intel_xe/           # Intel Xe-specific CuTe tests
├── gemm/device/            # GEMM device-level tests
└── flash_attention/        # Flash attention tests
```

## Architecture Constraints

### Intel Xe Hardware Specifics

**DPAS (Dot Product Accumulate Systolic):**
- M dimension: 1 ≤ M ≤ 8 (configurable)
- N dimension: Fixed at 16
- K dimension: Data-type dependent
- B matrix uses VNNI layout (hidden from user, exposed via `XE_DPAS_TT` atoms)

**Block 2D Copy Operations:**
- Alignment: Base address (64-byte), stride (16-byte), width (4-byte)
- Size limits: Width/pitch < 2^24 bytes, height < 2^24 elements
- Types: `XE_LOAD_2D`, `XE_LOAD_2D_VNNI`, `XE_LOAD_2D_TRANSPOSE`, `XE_STORE_2D`, `XE_PREFETCH_2D`
- Enable runtime checks: `-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1`

**Architecture Constants (Python):**
- `INTEL_XE12 = 12` (PVC - Ponte Vecchio)
- `INTEL_XE20 = 20` (BMG - Battlemage, Arc B580)
- `INTEL_XE_ARCH_MIN = 12`, `INTEL_XE_ARCH_MAX = 50` (range check)
- `CUDA_ARCH_MIN = 50` (distinguishes CUDA vs Intel Xe in generator)

### Coding Conventions

**Copyright Headers:**
- Preserve dual NVIDIA/Intel copyright headers on modified files
- Never remove or alter existing copyright notices

**SYCL vs CUDA:**
- Use SYCL-compatible code; avoid adding CUDA-only paths without `#ifdef` gating
- Check for target-conditional code: `#ifdef SYCL_INTEL_TARGET`

**Compiler Flags:**
- CI enforces `-Werror` - all warnings are errors
- Use `-ftemplate-backtrace-limit=0` for better error messages during development

## Key Files for Modifications

When making changes, consult these files:

- `python/cutlass_library/generator.py` - Kernel generation and filtering
- `python/cutlass_library/arch_constants.py` - Architecture detection
- `include/cutlass/gemm/kernel/xe_gemm.hpp` - Main Xe GemmUniversal kernel
- `include/cutlass/gemm/collective/` - MMA collective implementations
- `.github/workflows/intel_test.yml` - Authoritative CI build/test steps

## Common Issues

**CMake Cache:** Delete `build/` completely if seeing unexpected behavior
**Missing Intel Environment:** Build fails at linking - source `/opt/intel/oneapi/setvars.sh`
**Wrong SYCL Target:** Intrinsics are target-specific - match CI target or use conservative paths
**Python Imports:** Run `pip install -e .` from project root if import errors occur
**Layout Constraints:** Reuse existing epilogue code/tests to avoid violating layout requirements

## CI Validation

Primary workflows (`.github/workflows/`):
- `intel_test.yml` - Main CI for Intel targets (BMG + PVC)
- `intel_test_gpp_host.yml` - G++ host compiler builds
- `sycl_python_test.yml` - Python test workflow

## Documentation

See `media/docs/cpp/` for detailed documentation:
- `quickstart.md` - Building and running basics
- `functionality.md` - Supported features by architecture
- `xe_rearchitecture.md` - Intel Xe CuTe architecture redesign
- `cute/00_quickstart.md` - CuTe introduction
- `build/building_with_sycl_support.md` - SYCL build details

Python documentation: `media/docs/python/xe_cutlass_library.md`, `media/docs/python/xe_library_generation.md`

## Pull Request Process

Use PR templates in `.github/PULL_REQUEST_TEMPLATE/`:
- Bug fix: `bug_fix.md`
- Performance: `performance.md`
- Feature: `feature.md`
- Refactoring: `refactoring.md`

**Quick PR creation:**
```bash
gh pr create --template .github/PULL_REQUEST_TEMPLATE/bug_fix.md
```

**PR Description Must Include:**
1. Summary of changes and modified files
2. Build steps executed locally (CMake command, environment variables)
3. Tests run and results (include test names and pass/fail counts)
4. Performance impact (if applicable) with benchmark data
5. Whether Intel oneAPI environment was available for full validation
