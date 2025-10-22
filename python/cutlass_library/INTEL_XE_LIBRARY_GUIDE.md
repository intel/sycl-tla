# Intel SYCL*TLA Library Generation Guide

**Complete Reference for Intel Xe GPU Architecture Support**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Supported Kernel Types](#supported-kernel-types)
4. [Generated Libraries](#generated-libraries)
5. [Build & Usage](#build--usage)
6. [Implementation Details](#implementation-details)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Generate and Build Libraries

```bash
# Configure CMake for BMG (Xe2)
cd build
cmake .. -GNinja \
    -DCUTLASS_NVCC_ARCHS="" \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DSYCL_INTEL_TARGET \
    -DCUTLASS_LIBRARY_GENERATOR_ARCHS="20"

# Build all libraries
ninja cutlass_library

# Verify generated libraries
ls -lh tools/library/libcutlass_gemm_xe20_*.so
```

### Test Generation

```bash
cd python/cutlass_library
python3 test_simple_generation.py --build-dir ./test_build --arch 20
```

**Expected Output:**
```
✓ TEST PASSED - All files generated with .cpp extension!
Summary:
  - Generated 24 operations
  - .cpp files: 31
  - .cu files: 0
```

---

## Architecture Overview

### Supported Architectures

| GPU | Architecture | Compute Cap | Identifiers | File Ext | Arch Tag |
|-----|-------------|-------------|-------------|----------|----------|
| **BMG** (Battlemage/Xe2) | 20 | 12-50 | `20`, `bmg`, `xe2`, `intel_gpu_bmg_g21` | `.cpp` | `cutlass::arch::Xe20` |
| **PVC** (Ponte Vecchio) | 12 | 12-50 | `12`, `pvc`, `intel_gpu_pvc` | `.cpp` | `cutlass::arch::Xe12` |

### Technical Specifications

**BMG/Xe2:**
- Subgroup size: 16 threads
- DPAS instruction support
- FP16/BF16 instruction: [8, 16, 16] (M, N, K)
- FP8/INT8 instruction: [8, 16, 32] (M, N, K)

**Key Differences from CUDA:**
- Uses `.cpp` files (not `.cu`)
- Architecture prefix: `xe` (not `sm`)
- Compute capability range: 12-50 (vs 50-120 for CUDA)

---

## Supported Kernel Types

### ✅ Homogeneous Types (Regular GEMM)

All kernel types use the **same data type for A and B matrices**:

| Type | A × B → C/D | Accumulator | Math Inst | Tile Sizes | Alignment | Status |
|------|-------------|-------------|-----------|------------|-----------|--------|
| **FP16** | half × half → float | float | [8,16,16] | 256×256×32 | 8 | ✅ Built |
| **BF16** | bf16 × bf16 → float | float | [8,16,16] | 256×256×32 | 8 | ✅ Built |
| **FP8-E4M3** | e4m3 × e4m3 → float | float | [8,16,32] | 256×256×64 | 16 | ✅ Built |
| **FP8-E5M2** | e5m2 × e5m2 → float | float | [8,16,32] | 256×256×64 | 16 | ✅ Built |
| **INT8** | int8 × int8 → int32 | int32 | [8,16,32] | 256×256×64 | 16 | ✅ Built |

**Tile Size Variants:**
- 256×256×K (optimal for large matrices)
- 128×256×K (balanced)
- 256×128×K (balanced)
- 128×128×K (high occupancy)

**Layout Combinations:**
- RR (RowMajor × RowMajor → RowMajor)
- RC (RowMajor × ColumnMajor → RowMajor)
- CR (ColumnMajor × RowMajor → RowMajor)
- CC (ColumnMajor × ColumnMajor → RowMajor)

### ❌ Mixed Precision (Not Supported for Regular GEMM)

These require **Grouped GEMM** infrastructure (`KernelXePtrArrayCooperative`):

| Type | A × B → C/D | Why Not Supported |
|------|-------------|-------------------|
| FP16 × E4M3 → FP32 | half × e4m3 → float | Needs `MainloopIntelXeXMX16GroupMixedPrecision` |
| FP16 × E5M2 → FP32 | half × e5m2 → float | Needs `MainloopIntelXeXMX16GroupMixedPrecision` |
| BF16 × E4M3 → FP32 | bf16 × e4m3 → float | Needs `MainloopIntelXeXMX16GroupMixedPrecision` |
| BF16 × E5M2 → FP32 | bf16 × e5m2 → float | Needs `MainloopIntelXeXMX16GroupMixedPrecision` |
| FP16 × INT4 → FP32 | half × int4 → float | Needs `MainloopIntelXeXMX16GroupMixedPrecision` |

**Reason:** Regular library GEMMs use `MainloopIntelXeXMX16` which requires `ElementA == ElementB` (same input types).

---

## Generated Libraries

### Library Files

After successful build, you'll have:

```bash
$ ls -lh build/tools/library/libcutlass*.so
-rwxrwxr-x 186K libcutlass_gemm_xe20_gemm_bf16.so    # BF16 kernels
-rwxrwxr-x 186K libcutlass_gemm_xe20_gemm_e4m3.so    # FP8 E4M3 kernels
-rwxrwxr-x 186K libcutlass_gemm_xe20_gemm_e5m2.so    # FP8 E5M2 kernels
-rwxrwxr-x 186K libcutlass_gemm_xe20_gemm_f16.so     # FP16 kernels
-rwxrwxr-x 186K libcutlass_gemm_xe20_gemm_s8.so      # INT8 kernels
-rwxrwxr-x 186K libcutlass_gemm_xe20_gemm.so         # Generic library
-rwxrwxr-x  19K libcutlass.so                        # Main library
```

### Generated Kernel Count

**Per Data Type:**
- 4 kernels per tile size (RR, RC, CR, CC layouts)
- 4 tile sizes (256×256, 128×256, 256×128, 128×128)
- **Total: ~16 kernels per data type**

**Overall:**
- FP16: 4 kernels (1 tile size shown in generation)
- BF16: 4 kernels
- FP8 E4M3: 4 kernels
- FP8 E5M2: 4 kernels
- INT8: 4 kernels
- **Total: ~24 operations, 31 .cpp files**

### File Structure

```
build/tools/library/generated/gemm/20/
├── gemm/
│   ├── all_xe20_gemm_operations.cpp
│   └── cutlass3x_xe20_tensorop_gemm_256x256_32x0_*.cpp
├── gemm_bf16/
│   ├── all_xe20_gemm_bf16_gemm_operations.cpp
│   └── cutlass3x_xe20_tensorop_gemm_bf16_256x256_32x0_*.cpp
├── gemm_f16/
│   └── cutlass3x_xe20_tensorop_gemm_f16_256x256_32x0_*.cpp
├── gemm_e4m3/
│   └── cutlass3x_xe20_tensorop_gemm_e4m3_256x256_64x0_*.cpp
├── gemm_e5m2/
│   └── cutlass3x_xe20_tensorop_gemm_e5m2_256x256_64x0_*.cpp
└── gemm_s8/
    └── cutlass3x_xe20_tensorop_gemm_s8_256x256_64x0_*.cpp
```

### Kernel Naming Convention

**Format:**
```
cutlass3x_xe{arch}_{opclass}_{operation}_{dtype}_{tile}_{warp}_{layout}_align{N}
```

**Examples:**
```cpp
// FP16: 256×256×32, RowMajor×RowMajor→RowMajor, alignment 8
cutlass3x_xe20_tensorop_gemm_f16_256x256_32x0_nn_align8

// BF16: 256×256×32, RowMajor×ColumnMajor→RowMajor, alignment 8
cutlass3x_xe20_tensorop_gemm_bf16_256x256_32x0_nt_align8

// FP8 E4M3: 256×256×64, ColumnMajor×RowMajor→RowMajor, alignment 16
cutlass3x_xe20_tensorop_gemm_e4m3_256x256_64x0_tn_align16

// INT8: 256×256×64, ColumnMajor×ColumnMajor→RowMajor, alignment 16
cutlass3x_xe20_tensorop_gemm_s8_256x256_64x0_tt_align16
```

**Layout Codes:**
- `nn`: A=RowMajor (N), B=RowMajor (N)
- `nt`: A=RowMajor (N), B=ColumnMajor (T)
- `tn`: A=ColumnMajor (T), B=RowMajor (N)
- `tt`: A=ColumnMajor (T), B=ColumnMajor (T)

---

## Build & Usage

### CMake Configuration

**BMG (Xe2):**
```bash
cmake .. -GNinja \
    -DCUTLASS_NVCC_ARCHS="" \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DCUTLASS_LIBRARY_GENERATOR_ARCHS="20"
```

**PVC (Xe-HPC):**
```bash
cmake .. -GNinja \
    -DCUTLASS_NVCC_ARCHS="" \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DCUTLASS_LIBRARY_GENERATOR_ARCHS="12"
```

### Build Targets

```bash
# Build all libraries
ninja cutlass_library

# Build specific data type
ninja cutlass_library_gemm_xe20_gemm_bf16
ninja cutlass_library_gemm_xe20_gemm_f16
ninja cutlass_library_gemm_xe20_gemm_e4m3
ninja cutlass_library_gemm_xe20_gemm_e5m2
ninja cutlass_library_gemm_xe20_gemm_s8
```

### Python Generator (Direct)

```bash
cd build
python3 ../python/cutlass_library/generator.py \
    --operations=gemm \
    --architectures=20 \
    --build-dir=. \
    --curr-build-dir=.
```

### Using the Libraries

```cpp
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

// Initialize library
cutlass::library::initialize();

// Find operation
cutlass::library::Operation const *operation = 
    cutlass::library::find_gemm_operation(
        cutlass::library::Provider::kCUTLASS,
        cutlass::library::GemmKind::Gemm,
        cutlass::library::NumericTypeID::kF16,  // Element A
        cutlass::library::LayoutTypeID::kRowMajor,
        cutlass::library::NumericTypeID::kF16,  // Element B
        cutlass::library::LayoutTypeID::kColumnMajor,
        cutlass::library::NumericTypeID::kF32,  // Element C
        cutlass::library::LayoutTypeID::kRowMajor,
        cutlass::library::NumericTypeID::kF32   // Compute type
    );

// Execute operation
cutlass::Status status = operation->run(
    &arguments,
    host_workspace,
    device_workspace,
    stream
);
```

---

## Implementation Details

### Code Changes

**Modified Files:**

1. **`python/cutlass_library/generator.py`** (~230 lines added)
   - `GenerateXe_TensorOp_16b_DPAS_gemm()` - FP16/BF16 kernels
   - `GenerateXe_TensorOp_fp8_DPAS_gemm()` - FP8 kernels (E4M3, E5M2 only)
   - `GenerateXe_TensorOp_int8_DPAS_gemm()` - INT8 kernels
   - `GenerateXe_TensorOp_mixed_dtype_DPAS_gemm()` - Mixed precision (disabled for regular GEMM)
   - `GenerateIntelXe()` - Unified orchestrator for PVC and BMG

2. **`include/cutlass/gemm/collective/builders/xe_mma_builder.inl`** (~20 lines)
   - Added INT32 accumulator support
   - Added INT8 MMA atom: `XE_8x16x32_S32S8S8S32_TT`
   - Added FP8 MMA atoms: `XE_8x16x16_F32F16F16F32_TT` (with FP8→FP16 conversion)

3. **`include/cutlass/epilogue/collective/builders/xe_builder.inl`** (~5 lines)
   - Added INT32 support for ElementC

### Architecture Aliases

```cpp
// include/cutlass/arch/arch.h
namespace cutlass::arch {
    struct IntelXe { /* Base Intel Xe tag */ };
    using Xe20 = IntelXe;  // BMG/Xe2 alias
    using Xe12 = IntelXe;  // PVC alias
}
```

### CollectiveBuilder Constraints

```cpp
// xe_mma_builder.inl
static_assert(cute::is_any_of_v<ElementAccumulator, float, bfloat16_t, half_t, int32_t>,
    "Intel multi-stage pipeline requires ElementC to be of type float, bfloat, half, or int32");

static_assert(cute::is_any_of_v<ElementA, bfloat16_t, half_t, cute::float_e5m2_t, cute::float_e4m3_t, cute::int8_t>,
    "Supported A types: bf16, f16, e4m3, e5m2, int8");

static_assert(cute::is_any_of_v<ElementB, bfloat16_t, half_t, cute::float_e5m2_t, cute::float_e4m3_t, cute::int8_t, cute::uint4_t>,
    "Supported B types: bf16, f16, e4m3, e5m2, int8, int4");
```

**Note:** For regular GEMM, `MainloopIntelXeXMX16` requires `ElementA == ElementB`.

### MMA Atom Mapping

```cpp
// xe_mma_builder.inl - pick_mma_atom specializations
PICK_MMA(bfloat16_t, float, XE_8x16x16_F32BF16BF16F32_TT);
PICK_MMA(bfloat16_t, bfloat16_t, XE_8x16x16_BF16BF16BF16BF16_TT);
PICK_MMA(half_t, float, XE_8x16x16_F32F16F16F32_TT);
PICK_MMA(half_t, half_t, XE_8x16x16_F16F16F16F16_TT);
PICK_MMA(float_e4m3_t, float, XE_8x16x16_F32F16F16F32_TT);  // FP8→FP16 conversion
PICK_MMA(float_e5m2_t, float, XE_8x16x16_F32F16F16F32_TT);  // FP8→FP16 conversion
PICK_MMA(int8_t, int32_t, XE_8x16x32_S32S8S8S32_TT);        // Note: K=32
```

---

## Troubleshooting

### Issue: Mixed Precision Kernels Fail to Compile

**Error:**
```
error: no type named 'ElementA' in 'cutlass3x_xe20_tensorop_gemm_f16_e4m3_...'
```

**Cause:** Mixed precision (different A and B types) requires grouped GEMM mainloop.

**Solution:** Mixed precision is not supported for regular library generation. Use grouped GEMM examples instead:
```bash
# This works (grouped GEMM)
./examples/09_bmg_grouped_gemm_f8/09_bmg_grouped_gemm_f8

# Regular library only supports homogeneous types
```

### Issue: INT8 Kernels Fail to Build

**Error:**
```
error: unknown type name 'XE_8x16x16_S32S8S8S32_TT'
```

**Solution:** Use correct MMA atom name `XE_8x16x32_S32S8S8S32_TT` (K=32, not K=16).

### Issue: Wrong File Extension (.cu instead of .cpp)

**Cause:** Architecture not detected as Intel Xe.

**Solution:** Ensure compute capability is in range 12-50:
```bash
# Correct
cmake .. -DCUTLASS_LIBRARY_GENERATOR_ARCHS="20"  # BMG
cmake .. -DCUTLASS_LIBRARY_GENERATOR_ARCHS="12"  # PVC

# Wrong (will generate .cu files)
cmake .. -DCUTLASS_LIBRARY_GENERATOR_ARCHS="90"  # CUDA SM90
```

### Issue: No Operations Generated

**Cause:** Generator functions not called or architecture mismatch.

**Solution:** Check GenerateIntelXe is called:
```python
# generator.py
if arch in [12, 20]:
    GenerateIntelXe(manifest, cuda_version, arch=arch)
```

### Issue: Library Link Errors

**Error:**
```
undefined reference to `initialize_all_xe20_gemm_bf16_gemm_operations()`
```

**Solution:** Ensure library is built and linked:
```bash
ninja cutlass_library_gemm_xe20_gemm_bf16
# Link with: -lcutlass_gemm_xe20_gemm_bf16
```

---

## Performance Considerations

### Optimal Tile Sizes

| Matrix Size | Recommended Tile | Reason |
|-------------|------------------|--------|
| Large (4096+) | 256×256×K | Best occupancy, full XVE utilization |
| Medium (1024-4096) | 128×256×K or 256×128×K | Balanced performance |
| Small (<1024) | 128×128×K | Lower resource usage |

### Memory Alignment

Proper alignment is critical for Block 2D loads:
- **FP16/BF16:** 8-element alignment (16 bytes)
- **FP8:** 16-element alignment (16 bytes)
- **INT8:** 16-element alignment (16 bytes)
- **Output (INT32/FP32):** 4-8 element alignment

### Layout Preferences

- **NN (Row×Row):** Best for A and B both in RowMajor
- **NT (Row×Column):** Standard GEMM, B transposed
- **TN (Column×Row):** A transposed
- **TT (Column×Column):** Both transposed

---

## Summary

### ✅ What Works

- **5 data type libraries** built successfully (FP16, BF16, E4M3, E5M2, INT8)
- **~24 operations, 31 .cpp files** generated
- **All homogeneous type kernels** compile cleanly
- **INT32 accumulator** support for INT8
- **FP8 support** with automatic FP8→FP16 conversion in MMA

### ❌ Current Limitations

- **Mixed precision** (FP16×FP8, FP16×INT4) requires grouped GEMM infrastructure
- **Regular library** only supports ElementA == ElementB
- **No INT4 support** in regular GEMM (requires grouped GEMM)

### 📊 Quick Reference

| Feature | Value |
|---------|-------|
| Architecture Numbers | BMG=20, PVC=12 |
| File Extension | `.cpp` (not `.cu`) |
| Architecture Prefix | `xe` (not `sm`) |
| Compute Cap Range | 12-50 (Intel Xe) |
| Total Libraries | 7 (.so files) |
| Total Kernels | ~24 operations |
| Supported Types | FP16, BF16, E4M3, E5M2, INT8 |
| Mixed Precision | ❌ Not supported (use grouped GEMM) |

---

**Copyright © 2025 Intel Corporation. All rights reserved.**  
**SPDX-License-Identifier: BSD-3-Clause**

**Last Updated:** October 16, 2025
