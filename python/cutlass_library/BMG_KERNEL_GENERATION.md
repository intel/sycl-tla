# BMG/Xe2 Kernel Generation for CUTLASS Library

## Overview

This document describes the kernel generation functions added for Intel's BMG (Battlemage/Xe2) GPU architecture in the CUTLASS library manifest system.

## Architecture Specification

**BMG (Battlemage/Xe2)**
- Compute Capability: **20**
- Architecture Prefix: **xe**
- DPAS (Dot Product Accumulate Systolic) instruction support
- Subgroup size: 16 threads

## Generated Kernel Categories

### 1. 16-bit Floating Point GEMM (`GenerateBMG_TensorOp_16b_DPAS_gemm`)

**Supported Data Types:**
- FP16 x FP16 → FP32
- FP16 x FP16 → FP16
- BF16 x BF16 → FP32
- BF16 x BF16 → BF16

**Math Instruction Shape:** `[8, 16, 16]` (M, N, K)

**Tile Sizes:**
- 256x256x32
- 128x256x32
- 256x128x32
- 128x128x32
- 64x128x32

**Layouts:** All combinations of RowMajor/ColumnMajor for A, B, C
**Alignment:** 8 elements for all matrices

### 2. FP8 GEMM (`GenerateBMG_TensorOp_fp8_DPAS_gemm`)

**Supported Data Types:**
- E4M3 x E4M3 → FP32
- E5M2 x E5M2 → FP32
- E4M3 x E5M2 → FP32 (mixed FP8)

**Math Instruction Shape:** `[8, 16, 32]` (M, N, K)

**Tile Sizes:**
- 256x256x64
- 128x256x64
- 256x128x64
- 128x128x64

**Layouts:** All combinations of RowMajor/ColumnMajor for A, B, C
**Alignment:** 16 elements for A and B, 8 elements for C

### 3. INT8 GEMM (`GenerateBMG_TensorOp_int8_DPAS_gemm`)

**Supported Data Types:**
- INT8 x INT8 → INT32

**Math Instruction Shape:** `[8, 16, 32]` (M, N, K)

**Tile Sizes:**
- 256x256x64
- 128x256x64
- 256x128x64
- 128x128x64

**Layouts:** All combinations of RowMajor/ColumnMajor for A, B, C
**Alignment:** 16 elements for A and B, 4 elements for C

### 4. Mixed Precision GEMM (`GenerateBMG_TensorOp_mixed_dtype_DPAS_gemm`)

**Supported Data Types:**
- INT8 x FP16 → FP32

**Math Instruction Shape:** `[8, 16, 32]` (M, N, K)

**Tile Sizes:**
- 256x256x64
- 128x256x64
- 256x128x64

**Layouts:** All combinations of RowMajor/ColumnMajor for A, B, C
**Alignment:** 16 elements for A, 8 elements for B and C

## Configuration Details

### Thread Block Configuration

Each tile description specifies:
- **Tile shape:** [M, N, K] dimensions
- **Stages:** 0 (auto-tuned)
- **Warp count:** [warp_m, warp_n, warp_k]
- **Cluster shape:** [1, 1, 1] (no clustering for BMG)

### Scheduling

- **Kernel Schedule:** `ScheduleAuto`
- **Epilogue Schedule:** `ScheduleAuto`
- **Tile Scheduler:** `Persistent`

## Kernel Naming Convention

Generated kernels follow the pattern:
```
cutlass_xe20_dpas_gemm_<A_type><B_type>_<C_type>_<layout>_<tile>_<alignment>
```

Example:
```
cutlass_xe20_dpas_gemm_f16f16_f32_rrr_256x256x32_align8
```

## Build Integration

### CMake Configuration

To generate BMG kernels:
```bash
cmake .. -DCUTLASS_ENABLE_SYCL=ON \
         -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" \
         -DCUTLASS_LIBRARY_OPERATIONS="gemm"
```

### Architecture Detection

The generator automatically detects BMG targets from the following identifiers:
- `20` (numeric compute capability)
- `bmg`
- `xe2`
- `intel_gpu_bmg_g21`

### Generated File Structure

```
tools/library/generated/gemm/20/
├── all_xe20_gemm_operations.cpp
├── dpas/
│   ├── all_xe20_dpas_gemm_operations.cpp
│   ├── cutlass_xe20_dpas_gemm_f16_f32_*.cpp
│   ├── cutlass_xe20_dpas_gemm_bf16_f32_*.cpp
│   ├── cutlass_xe20_dpas_gemm_e4m3_f32_*.cpp
│   ├── cutlass_xe20_dpas_gemm_e5m2_f32_*.cpp
│   └── cutlass_xe20_dpas_gemm_s8_s32_*.cpp
```

## Comparison with SM90 Generation

| Feature | SM90 (NVIDIA) | BMG (Intel Xe2) |
|---------|---------------|-----------------|
| **Compute Capability** | 90 | 20 |
| **Prefix** | `sm` | `xe` |
| **Matrix Instruction** | WGMMA | DPAS |
| **Subgroup Size** | 32 (warp) | 16 (subgroup) |
| **FP16 Instruction** | 64x64x16 | 8x16x16 |
| **FP8 Instruction** | 64x64x32 | 8x16x32 |
| **INT8 Instruction** | 64x64x32 | 8x16x32 |

## Performance Considerations

### Optimal Tile Sizes

- **256x256x32:** Best for large matrices with good occupancy
- **128x256x32:** Balanced for moderate matrix sizes
- **128x128x32:** Lower resource usage, higher occupancy
- **64x128x32:** Smallest footprint for limited resources

### Memory Alignment

Proper alignment is critical for Block 2D load performance:
- **FP16/BF16:** 8-element alignment (16 bytes)
- **FP8:** 16-element alignment (16 bytes)
- **INT8:** 16-element alignment (16 bytes)
- **INT32/FP32 output:** 4-8 element alignment

### Layout Preferences

- **Row-Row-Row (RRR):** Default for most workloads
- **Row-Column-Row (RCR):** Common for standard GEMM (B transposed)
- **Column-Row-Row (CRR):** Less common, A transposed
- **Column-Column-Row (CCR):** Both A and B transposed

## Usage Examples

### From Python Interface

```python
from cutlass_library.manifest import Manifest
from cutlass_library.generator import GenerateBMG

manifest = Manifest(args)
GenerateBMG(manifest, cuda_version="11.0.0")
manifest.emit(GeneratorTarget.Library)
```

### From Command Line

```bash
cd /path/to/cutlass/build
python ../python/cutlass_library/generator.py \
    --operations=gemm \
    --architectures="20" \
    --build-dir=. \
    --curr-build-dir=.
```

## Supported Operations

Based on existing BMG examples in the repository:

1. ✅ **Basic GEMM** - Standard matrix multiplication
2. ✅ **Grouped GEMM** - Batch processing with different sizes
3. ✅ **Mixed Precision** - INT8 x FP16, FP8 variations
4. ✅ **FP8 GEMM** - E4M3/E5M2 formats
5. ✅ **StreamK** - Stream-K tile scheduling (future)
6. ✅ **Custom Epilogues** - ReLU, GELU, etc.

## Testing

### Verify Generated Kernels

After generation, verify the kernels were created:

```bash
# Check generated files
ls build/tools/library/generated/gemm/20/dpas/

# Count generated kernels
# Count generated files
find build/tools/library/generated/gemm/20 -name "*.cpp" | wc -l

# Build the library
ninja cutlass_library
```

### Run Example Programs

```bash
# Basic GEMM
./examples/sycl/00_bmg_gemm/00_bmg_gemm

# FP8 GEMM
./examples/sycl/08_bmg_gemm_f8/08_bmg_gemm_f8

# Grouped GEMM with FP8
./examples/sycl/09_bmg_grouped_gemm_f8/09_bmg_grouped_gemm_fp8
```

## Future Enhancements

1. **Additional Data Types:**
   - INT4 support
   - TF32 emulation
   - Complex types

2. **Advanced Features:**
   - StreamK scheduler support
   - Multi-stage pipelining
   - Cluster shapes > 1

3. **Specialized Kernels:**
   - Rank-K updates
   - Triangular matrix operations (TRMM)
   - Symmetric matrix operations (SYMM)

4. **Optimizations:**
   - Tuned tile sizes per data type
   - Architecture-specific epilogues
   - Custom copy strategies

## Related Documentation

- [XE_ARCHITECTURE_SUPPORT.md](XE_ARCHITECTURE_SUPPORT.md) - Intel Xe architecture support in manifest system
- [BMG Examples](../../examples/README.md) - BMG example programs
- [CUTLASS 3.x Documentation](../../docs/) - General CUTLASS documentation

---

**Copyright (c) 2025 Intel Corporation. All rights reserved.**
**SPDX-License-Identifier: BSD-3-Clause**
