# SYCL\*TLA (SYCL Templates for Linear Algebra) - Copilot Instructions

This document provides comprehensive guidance for AI assistants and code analysts working with the SYCL\*TLA (Intel CUTLASS SYCL fork) repository.

**Repository**: `/localdisk/syk/cutlass-sycl-up`
**Project**: SYCL\*TLA - NVIDIA CUTLASS extended with Intel GPU support via SYCL
**Base Version**: NVIDIA CUTLASS 4.2.0
**Current Version**: 0.6.0

---

## 1. BUILD SYSTEM

### CMake Configuration

**Top-level file**: `/localdisk/syk/cutlass-sycl-up/CMakeLists.txt`
**SYCL-specific file**: `/localdisk/syk/cutlass-sycl-up/SYCL.cmake`
**Build script**: `/localdisk/syk/cutlass-sycl-up/build.sh`

### Build Command (Intel GPU - BMG/Battlemage)

```bash
mkdir build && cd build
CC=icx CXX=icpx cmake .. -GNinja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCUTLASS_ENABLE_BENCHMARKS=OFF \
  -DCMAKE_CXX_FLAGS="-g" \
  -DDPCPP_HOST_COMPILER=g++-13

# Build all
ninja

# Build specific target
ninja 00_bmg_gemm
```

### Build Command (Intel GPU - PVC/Ponte Vecchio)

```bash
CC=icx CXX=icpx cmake .. -GNinja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_pvc \
  -DDPCPP_HOST_COMPILER=g++-13
```

### Key CMake Options

| Option | Purpose | Default |
|--------|---------|---------|
| `CUTLASS_ENABLE_SYCL` | Enable SYCL backend | OFF |
| `DDPCPP_SYCL_TARGET` | Target GPU: `intel_gpu_pvc`, `intel_gpu_bmg_g21` | Required |
| `DDPCPP_HOST_COMPILER` | Host compiler (g++-13, etc.) | icpx default |
| `CUTLASS_ENABLE_BENCHMARKS` | Build benchmark suite | ON |
| `CUTLASS_ENABLE_TESTS` | Build unit tests | ON |
| `CUTLASS_ENABLE_EXAMPLES` | Build examples | ON |
| `CUTLASS_SYCL_PROFILING_ENABLED` | Enable SYCL event profiling | OFF |
| `CUTLASS_SYCL_RUNNING_CI` | CI environment mode | OFF |
| `CUTLASS_SYCL_BUILTIN_ENABLE` | Use builtin instead of SPIR-V | OFF |
| `CUTLASS_ENABLE_HEADERS_ONLY` | Header-only library mode | OFF |
| `CMAKE_BUILD_TYPE` | Build type (Debug/Release) | Release |

### Environment Variables for Performance

```bash
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export IGC_VectorAliasBBThreshold=100000000000
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

### Setup Intel oneAPI

```bash
source /opt/intel/oneapi/setvars.sh
```

### Build Targets

```bash
cmake --build . --target test_unit -j 8      # Run unit tests
cmake --build . --target test_examples -j 1  # Run examples
cmake --build . --target cutlass_benchmarks  # Build benchmarks
ninja [EXAMPLE_NAME]                         # Build specific example
```

---

## 2. TESTING

### Test Structure

**Test Directory**: `/localdisk/syk/cutlass-sycl-up/test/`
- `test/unit/` - GTest-based unit tests
  - `cute/` - CuTe layout/tensor tests
  - `gemm/device/` - GEMM device tests
  - `flash_attention/` - Flash Attention tests
  - `common/` - Test infrastructure
- `test/CMakeLists.txt` - Test configuration

### Running Tests

```bash
# Run all unit tests
cmake --build . --target test_unit

# Run specific test (after building)
cd build
./test/unit/[TEST_NAME]

# Run with ctest
ctest --output-on-failure
ctest -R test_pattern --verbose

# Run examples as tests
cmake --build . --target test_examples -j 1
```

### Test Frameworks Used

- **Google Test (GTest)** - Primary unit test framework
- **CTest** - CMake test driver
- **Custom infrastructure** - Located in `test/unit/common/` (filter_architecture.cpp, test_unit.cpp)

### CI Pipeline

**CI File**: `.github/workflows/intel_test.yml`

**CI Build Command** (from workflow):
```bash
cmake -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=${{ matrix.sycl_target }} \
  -DIGC_VERSION_MAJOR=${{ matrix.igc_version_major }} \
  -DIGC_VERSION_MINOR=${{ matrix.igc_version_minor }} \
  -DCMAKE_CXX_FLAGS="-Werror" \
  -DCUTLASS_SYCL_RUNNING_CI=ON

cmake --build . 
cmake --build . --target test_unit -j 8
cmake --build . --target test_examples -j 1
cmake --build . --target cutlass_benchmarks
```

**CI Tested Configurations**:
- BMG (Battlemage) with intel_gpu_bmg_g21 target
- PVC (Ponte Vecchio) with intel_gpu_pvc target
- Both RELEASE and NIGHTLY DPC++ compilers

### Running a Single Test Example

```bash
# Build
ninja 00_bmg_gemm

# Run
./examples/00_bmg_gemm/00_bmg_gemm
# Output: Disposition: Passed, Performance in TFlop/s
```

---

## 3. PYTHON INTERFACE

### Key Files

- **Generator**: `/localdisk/syk/cutlass-sycl-up/python/cutlass_library/generator.py` (lines 1-100 shown)
- **Architecture Constants**: `/localdisk/syk/cutlass-sycl-up/python/cutlass_library/arch_constants.py`
- **Project Config**: `/localdisk/syk/cutlass-sycl-up/pyproject.toml`

### Generator Module (generator.py)

Lines 1-100 contain:
```python
# Core imports: argparse, enum, itertools, logging, os.path, shutil, sys, copy
# from typing import Any, Dict, Optional, Sequence, Tuple

# Utility functions:
# - logging_prefix(indent_level) -> str
# - log_debug_line(line, indent_level) -> None

# Package detection mechanism:
# - CUTLASS_IGNORE_PACKAGE flag (builtins global)
# - Conditional imports: 
#   from cutlass_library.library import *
#   from cutlass_library.manifest import *
#   from cutlass_library.heuristics import *
#   from cutlass_library.emit_kernel_listing import emit_gemm_kernel_testlist
#   from cutlass_library.arch_constants import INTEL_XE12, INTEL_XE20, INTEL_XE35
```

**Architecture Support** (from arch_constants.py):
```python
# Constants
INTEL_XE_ARCH_MIN = 12  # PVC (Ponte Vecchio)
INTEL_XE_ARCH_MAX = 50
INTEL_XE12 = 12         # PVC - HPC architecture
INTEL_XE20 = 20         # BMG - Battlemage gaming architecture
INTEL_XE35 = 35         # Future CRI architecture
CUDA_ARCH_MIN = 50      # NVIDIA CUDA architectures

# Validation helpers
def is_intel_xe_arch(arch) -> bool
def is_cuda_arch(arch) -> bool
def get_arch_name(arch) -> str  # Human-readable names
```

### Python Layer Architecture

1. **Library Generation**: `generator.py` uses manifest.py to enumerate kernel configurations
2. **Architecture Constants**: Shared across manifest.py and gemm_operation.py
3. **Imports Strategy**: Supports both package and relative imports via `CUTLASS_IGNORE_PACKAGE`
4. **Dependencies**: networkx, numpy, pydot, scipy, treelib (from pyproject.toml)

### Python Build

```bash
# From pyproject.toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "sycl-tla"
version = "0.6.0"
requires-python = ">=3.8"
```

---

## 4. KEY ARCHITECTURE

### GEMM Kernel Directory

**Path**: `/localdisk/syk/cutlass-sycl-up/include/cutlass/gemm/kernel/`

**Key Xe-specific files** (Intel GPU kernels):
- `xe_gemm.hpp` (282 lines) - Main GemmUniversal specialization for Xe
- `xe_gemm_cooperative.hpp` (321 lines) - Cooperative kernel variant
- `xe_tile_scheduler_streamk.hpp` - Stream-K scheduling for Xe
- `xe_persistent_tile_scheduler_params_streamk.hpp` - Persistent scheduler params
- `xe_tile_scheduler_group.hpp` - Group-based tile scheduling

**Other kernel types** (NVIDIA, generic):
- `default_gemm.h` - Default GEMM template (42KB)
- `default_gemm_universal.h` - Universal GEMM interface
- `default_gemm_grouped.h` - Grouped GEMM operations
- `gemm_params.h` - Common GEMM parameter structures

### Core Abstractions (from xe_gemm.hpp)

```cpp
namespace cutlass::gemm::kernel {

template <
  class ProblemShape_,           // <M, N, K> or <M, N, K, L>
  class CollectiveMainloop_,     // Core computation (MMA operations)
  class CollectiveEpilogue_,     // Post-processing (epilogue fusion)
  class TileScheduler_            // Work distribution scheduler
>
class GemmUniversal<...>
{
  struct SharedStorage { };       // Shared memory layout
  struct Arguments { };           // Host-side kernel arguments
  struct Params { };              // Pre-computed device parameters
  // Actual kernel body implementation
};

} // namespace cutlass::gemm::kernel
```

### Dispatch Policy

**File**: `/localdisk/syk/cutlass-sycl-up/include/cutlass/gemm/dispatch_policy.hpp`

Defines kernel categorization and compilation policies:
- Kernel type detection templates (is_kernel_tag_of_v)
- Transform types: FastF32, InterleavedComplexTF32, MixedInput
- Architecture-specific dispatch variants

### Collective Operations

**Mainloop Collective** (`cutlass::gemm::collective`):
- Xe-specific MMA collectives: `xe_mma.hpp`, `xe_array_mma.hpp`
- Mixed input support: `xe_mma_mixed_input.hpp`, `xe_array_mma_mixed_input.hpp`
- FP8 support: `xe_mma_fp8_scaling.hpp`, `xe_array_mma_fp8_legacy.hpp`

**Epilogue Collective** (`cutlass::epilogue::collective`):
- Xe epilogue: `xe_epilogue.hpp`, `xe_array_epilogue.hpp`
- Visitor pattern: `xe_visitor.hpp`, `xe_visitor_softmax.hpp`, `xe_visitor_splitk.hpp`
- Callbacks: `xe_callbacks.hpp` - Custom post-processing functions

### CuTe Integration

**CuTe Layout**: `/localdisk/syk/cutlass-sycl-up/include/cute/`

Core abstractions:
```cpp
// Layout: Hierarchical multidimensional indexing
Layout layout = make_layout(...)

// Tensor: Layout + memory data
Tensor tensor = make_tensor(data_ptr, layout)

// Algorithms in cute/algorithm/
- gemm.hpp - Layout-based GEMM description
- copy.hpp - Data movement primitives
- reorder.hpp - VNNI and other tensor reorders
- subgroup_algorithms.hpp - Xe subgroup operations
- axpby.hpp - AXPY operations
```

---

## 5. EXISTING AI CONFIGS

**Result**: None found.

Checked for:
- `CLAUDE.md` - Not present
- `AGENTS.md` - Not present
- `.cursorrules` - Not present
- `.cursor/rules/` - Not present
- `.windsurfrules` - Not present
- `CONVENTIONS.md` - Not present
- `AIDER_CONVENTIONS.md` - Not present
- `.clinerules` - Not present
- `.cline_rules` - Not present
- `.github/copilot-instructions.md` - Not present

This is the first AI configuration file created for this repository.

---

## 6. CONVENTIONS

### Example Structure

**Examples Directory**: `/localdisk/syk/cutlass-sycl-up/examples/`

**Structure**:
```
examples/
├── 00_bmg_gemm/                           # Simple GEMM (Xe20/BMG)
├── 01_bmg_gemm_with_collective_builder/   # Using CollectiveBuilder pattern
├── 02_bmg_gemm_mixed_dtype/               # Mixed precision (dequantization)
├── 03_bmg_gemm_streamk/                   # Stream-K load balancing
├── 04_bmg_grouped_gemm/                   # Batch GEMM with varied sizes
├── 05_bmg_gemm_with_epilogues/            # Epilogue Visitor Tree (EVT) fusion
├── 06_bmg_flash_attention/                # Flash Attention V2
├── 07_bmg_dual_gemm/                      # Fused dual GEMM
├── 08_bmg_gemm_f8/                        # 8-bit float GEMM
├── 09_bmg_grouped_gemm_f8/                # FP8 grouped operations
├── 10_bmg_grouped_gemm_mixed_dtype/       # Mixed precision batch
├── 11_xe20_cutlass_library/               # CUTLASS library generation
│   ├── xe_20_cutlass_library_b16.cpp      # BF16 library example
│   └── CMakeLists.txt
├── common/                                # Shared utilities
├── cute/                                  # CuTe layout tutorials
├── generics/device_agnostic/              # Device-agnostic kernels
├── nv_sycl/                               # NVIDIA GPU examples
└── python/                                # Python interface examples
```

### Xe20 Example Pattern (from xe_20_cutlass_library_b16.cpp)

```cpp
// Standard headers
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

// CuTe tensor abstractions
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// Utilities
#include "cutlass/tensor_ref.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

// GEMM API
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

// Collective abstractions
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/activation.h"

// Dispatch and scheduling
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"

// Namespace alias
using namespace cute;

// Export visibility (for library compilation)
#ifdef __GNUC__
#define PT_EXPORT __attribute__((__visibility__("default")))
#else
#define PT_EXPORT
#endif
```

### Kernel Writing Pattern (Xe)

1. **Use CollectiveBuilder**: Abstracts MMA, data movement, epilogue
2. **Specify ProblemShape**: Template on <M, N, K> dimensions
3. **Choose TileScheduler**: StreamK for load balancing, default for sequential
4. **Define Epilogue**: Via Epilogue Visitor Tree for fusion opportunities
5. **Dispatch via GemmUniversal**: Single entry point for all architectures

### Architecture-Specific Markers

- **PVC (Ponte Vecchio)**: Use `intel_gpu_pvc` in DPCPP_SYCL_TARGET
- **BMG (Battlemage/Arc B580)**: Use `intel_gpu_bmg_g21` in DPCPP_SYCL_TARGET
- **File prefixes**: `xe_` for Intel GPU; `sm90_`, `sm100_` for NVIDIA
- **Conditional compilation**: `#ifdef SYCL_INTEL_TARGET` for Intel-specific code

---

## 7. README HIGHLIGHTS

**File**: `/localdisk/syk/cutlass-sycl-up/README.md`

### Project Overview

**Name**: SYCL\*TLA (SYCL Templates for Linear Algebra)

**Tagline**: NVIDIA CUTLASS extended with Intel GPU support via SYCL enablement

**Base**: Forked from NVIDIA CUTLASS 4.2.0

**Current Version**: 0.6.0

### Key Features

- **Header-only C++ template framework** for GEMM and fused epilogues
- **Hierarchical tiling** and composable policy abstractions
- **Efficient data-movement** primitives via CuTe
- **Mixed-precision support**: FP64, FP32, FP16, BF16, FP8 (E5M2, E4M3), 4-bit and 8-bit integers
- **Quantization variants**: Tensor-wise, channel-wise, group-wise
- **Epilogue fusion** for custom post-processing pipelines
- **Intel GPU optimization** for Xe-HPC (PVC) and Xe2 (BMG/Arc B580)

### Supported Hardware

| GPU | Architecture | Codename |
|-----|--------------|----------|
| Intel Data Center GPU Max Series | Xe-HPC | Ponte Vecchio (PVC) |
| Intel Arc GPU B580 Graphics | Xe2 | Battlemage (BMG) |

### Validated Configurations

| Platform | OS | DPC++ | G++ | Compute Runtime | IGC |
|----------|----|----- |-----|-----------------|-----|
| Xe-HPC | Ubuntu 22.04 | 2025.2+ | G++13 | 25.18 | 2.11 |
| Xe2 | Ubuntu 25.04 | 2025.2+ | G++13 | 25.35 | 2.18 |

### Quick Build

```bash
# Intel Data Center GPU Max
CC=icx CXX=icpx cmake .. -GNinja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_pvc

# Intel Arc B580
CC=icx CXX=icpx cmake .. -GNinja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21

ninja 00_bmg_gemm
./examples/00_bmg_gemm/00_bmg_gemm
# Output: Disposition: Passed, Performance: [TFlop/s]
```

### CuTe Core Library

**Purpose**: C++ SYCL template abstractions for hierarchical multidimensional tensor operations

**Key Concepts**:
- **Layout**: Compactly encodes shape, stride, memory layout
- **Tensor**: Layout + data pointer for hierarchical indexing
- **Composition**: Functional composition for tiling and partitioning

**Documentation**: `/localdisk/syk/cutlass-sycl-up/media/docs/cpp/cute/00_quickstart.md`

### Documentation Links

- Quick Start Guide: `media/docs/cpp/quickstart.md`
- Functionality: `media/docs/cpp/functionality.md`
- SYCL Build Support: `media/docs/cpp/build/building_with_sycl_support.md`
- CUTLASS 3.x Design: `media/docs/cpp/cutlass_3x_design.md`
- GEMM API 3.x: `media/docs/cpp/gemm_api_3x.md`
- Code Organization: `media/docs/cpp/code_organization.md`
- CuTe Documentation: `media/docs/cpp/cute/`

### Version Mapping

| SYCL\*TLA | NVIDIA CUTLASS | Release Date |
|-----------|----------------|--------------|
| 0.1 | 3.9 | - |
| 0.2 | 3.9.2 | - |
| 0.3 | 3.9.2 | - |
| 0.5 | 4.2.0 | - |
| 0.6 | 4.2.0 | 2025-11-03 |

### What's New in 0.6

**Major Architecture Changes**:
- **Flash Attention Reimplementation**: Complete rewrite with optimized Xe atoms
- **CUTLASS Library Generation**: Full support for Xe architecture in library pipeline

**Enhancements**:
- Python Operations Support with comprehensive test coverage
- CuTe Subgroup Extensions (broadcast, reduction for Xe)
- Enhanced 2D Copy Operations with subtiling
- 4-bit VNNI Reorders
- Batch GEMM and Grouped GEMM with new APIs

---

## 8. CI WORKFLOWS

### Workflow Files

**Location**: `.github/workflows/`

**Relevant CI Files**:
- `intel_test.yml` - Main Intel GPU testing (PVC, BMG)
- `cuda_test.yml` - CUDA/NVIDIA testing
- `nvidia_test.yml` - NVIDIA-specific tests
- `sycl_python_test.yml` - Python interface tests
- `blossom-ci.yml` - Pre-merge checks
- Others: CodeQL, Coverity, Nightly, Labeler

### Intel Test Pipeline (intel_test.yml)

**Configuration Matrix**:
```yaml
- compiler: RELEASE or NIGHTLY
  gpu: BMG or PVC
  intel_graphics: ROLLING or STAGING
  sycl_target: intel_gpu_bmg_g21 or intel_gpu_pvc
  igc_version_major: 2
  igc_version_minor: 18 (BMG) or 11 (PVC)
  runner: bmg108629-01 or pvc146162-01
```

**Build Steps**:

1. **Checkout**: `actions/checkout@v4.1.6`
2. **Install Graphics**: Custom action `.github/actions/install-intel-graphics`
3. **Install DPC++**: Custom action `.github/actions/install-dpcpp`
4. **Setup Environment**:
   ```bash
   source setvars.sh
   export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
   export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
   export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
   export IGC_VectorAliasBBThreshold=100000000000
   ```
5. **Build**:
   ```bash
   cmake -G Ninja \
     -DCUTLASS_ENABLE_SYCL=ON \
     -DDPCPP_SYCL_TARGET=${{ matrix.sycl_target }} \
     -DIGC_VERSION_MAJOR=${{ matrix.igc_version_major }} \
     -DIGC_VERSION_MINOR=${{ matrix.igc_version_minor }} \
     -DCMAKE_CXX_FLAGS="-Werror" \
     -DCUTLASS_SYCL_RUNNING_CI=ON
   cmake --build .
   ```
6. **Unit Tests**: `cmake --build . --target test_unit -j 8`
7. **Examples**: `cmake --build . --target test_examples -j 1`
8. **Benchmarks**: `cmake --build . --target cutlass_benchmarks`
9. **Cleanup**: Remove DPC++ installation and OneAPI packages

**Timeout**: 120 minutes per workflow run

**Concurrency**: Grouped by PR/branch, in-progress runs cancelled when new push arrives

### Test Execution

**Targets**:
```makefile
test_unit          # All unit tests via GTest
test_examples      # Example executables as tests
cutlass_benchmarks # Performance benchmarks
```

**CI Reports**:
- Pass/Fail status per architecture
- Compiler version compatibility
- Graphics driver compatibility

---

## 9. COMMON DEVELOPMENT WORKFLOWS

### Setting Up Development Environment

```bash
# 1. Clone repository
git clone https://github.com/intel/sycl-tla.git
cd sycl-tla

# 2. Set up Intel oneAPI
source /opt/intel/oneapi/setvars.sh

# 3. Create build directory
mkdir build && cd build

# 4. Configure for your GPU
CC=icx CXX=icpx cmake .. -GNinja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DDPCPP_HOST_COMPILER=g++-13 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Building and Testing

```bash
# Build everything
ninja

# Build specific example
ninja 00_bmg_gemm

# Run tests
ctest --output-on-failure

# Run single example
./examples/00_bmg_gemm/00_bmg_gemm
```

### Adding a New Kernel

1. **Use Collective Pattern**: Inherit from `GemmUniversal` or `FlashAttentionKernel`
2. **Choose Collectives**: Select `MainloopCollective` and `EpilogueCollective` from `collective/xe_*.hpp`
3. **Define TileShape**: Via `CollectiveBuilder<TileShape, ElementA, ElementB, ...>`
4. **Dispatch**: Override `DispatchPolicy` for custom scheduling
5. **Test**: Add to `test/unit/gemm/` or `examples/`

### File Organization Patterns

- **Headers**: `include/cutlass/gemm/kernel/xe_*.hpp`
- **Collectives**: `include/cutlass/gemm/collective/xe_*.hpp`
- **Epilogue**: `include/cutlass/epilogue/collective/xe_*.hpp`
- **Examples**: `examples/0X_bmg_*.cpp`
- **Tests**: `test/unit/*/xe_*.cpp` or `test/unit/*/test_*.cpp`

### Debug Flags

```bash
# Enable profiling
-DCUTLASS_SYCL_PROFILING_ENABLED=ON

# Verbose output
-DCMAKE_VERBOSE_MAKEFILE=ON

# Debug symbols
-DCMAKE_BUILD_TYPE=Debug

# CI environment (additional checks)
-DCUTLASS_SYCL_RUNNING_CI=ON
```

---

## 10. QUICK REFERENCE

### Project Structure
```
cutlass-sycl-up/
├── include/cutlass/               # Main library headers
│   ├── gemm/kernel/xe_*.hpp      # Xe GPU kernels
│   ├── gemm/collective/          # MMA and data movement
│   ├── epilogue/collective/      # Post-processing
│   └── cute/                     # Layout and tensor abstractions
├── test/                          # Unit tests and infrastructure
├── examples/                      # Example kernels and applications
├── python/                        # Python bindings and generators
├── cmake/                         # CMake build infrastructure
├── .github/workflows/             # CI configuration
├── CMakeLists.txt                # Top-level build config
├── SYCL.cmake                    # SYCL-specific build settings
├── build.sh                      # Quick build script
├── README.md                     # Project overview
└── pyproject.toml                # Python package config
```

### Key File Locations by Function

| Function | File |
|----------|------|
| GEMM kernels | `include/cutlass/gemm/kernel/xe_gemm.hpp` |
| MMA operations | `include/cutlass/gemm/collective/xe_mma.hpp` |
| Epilogue fusion | `include/cutlass/epilogue/collective/xe_epilogue.hpp` |
| CuTe layout | `include/cute/layout.hpp` |
| Build config | `CMakeLists.txt`, `SYCL.cmake` |
| Tests | `test/unit/gemm/device/`, `test/unit/cute/` |
| Examples | `examples/00_bmg_gemm/`, `examples/11_xe20_cutlass_library/` |

### Supported Data Types

- **Floats**: FP64, FP32, FP16, BF16
- **Integers**: int8_t, uint8_t, int4b_t, uint4b_t
- **FP8**: e5m2_t, e4m3_t
- **Quantization**: Zero-point, scale, channel-wise, group-wise

### SYCL Targets

- `intel_gpu_pvc` - Ponte Vecchio (Xe-HPC)
- `intel_gpu_bmg_g21` - Battlemage (Xe2)
- `nvptx64-nvidia-cuda` - NVIDIA CUDA (validation only)
- `spir64` - Generic SPIR-V (minimal support)

---

## 11. ADDITIONAL RESOURCES

### Documentation
- `/localdisk/syk/cutlass-sycl-up/media/docs/cpp/` - Comprehensive documentation
- `/localdisk/syk/cutlass-sycl-up/examples/README.md` - Example descriptions
- CHANGELOG-SYCL.md - SYCL-specific changes
- CHANGELOG.md - NVIDIA CUTLASS changes

### Related Repositories
- NVIDIA CUTLASS: https://github.com/nvidia/cutlass
- Intel SYCL-TLA: https://github.com/intel/sycl-tla
- DPC++ Compiler: https://github.com/intel/llvm

### Support
- GitHub Issues: https://github.com/intel/sycl-tla/issues
- SYCL Specification: https://www.khronos.org/sycl/
- oneAPI Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/

---

**Last Updated**: Generated for SYCL\*TLA 0.6.0
**Applicable To**: Intel GPU support (PVC, BMG), NVIDIA validation
**Compiler Requirement**: DPC++ 2025.1+, Clang 19.0+
**C++ Standard**: C++17 (required)
