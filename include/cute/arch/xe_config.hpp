/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
#include <cute/util/sycl_vec.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x)                                                 \
  inline x {                                                                   \
    CUTE_INVALID_CONTROL_PATH(                                                 \
        "Attempting to use a device built-in in host code.");                  \
  }
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL x
#else
#define SYCL_DEVICE_OCL(x)                                                     \
  inline x {                                                                   \
    CUTE_INVALID_CONTROL_PATH(                                                 \
        "Attempting to use a device built-in in host code.");                  \
  }
#endif


#undef __global
#define __global __attribute__((opencl_global))


#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_XE_ENABLED
#endif
 
#if defined(CUTE_ARCH_XE_ENABLED) && ((defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER < 20250200)) || defined(CUTLASS_SYCL_BUILTIN_ENABLE))
#define CUTE_ARCH_XE_BUILTIN_ENABLED
#elif defined(CUTE_ARCH_XE_ENABLED)
#define CUTE_ARCH_XE_SPIRV_ENABLED
#endif

#if defined(CUTE_ARCH_XE_BUILTIN_ENABLED)
namespace cute::detail
{
template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockPrefetch {
  static_assert(dependent_false<>, "Unsupported 2D Block Load Configuration.");
};
template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockLoad {
  static_assert(dependent_false<>, "Unsupported 2D Block Load Configuration.");
};
template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockTransform {
  static_assert(dependent_false<>, "Unsupported 2D Block Load Configuration.");
};
template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockTranspose {
  static_assert(dependent_false<>, "Unsupported 2D Block Load Configuration.");
};
template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockStore {
  static_assert(dependent_false<>, "Unsupported 2D Block Load Configuration.");
};
}
#endif
// SPIRV copy definitions
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockLoadINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* src_base_pointer, int memory_width, int memory_height,
    int memory_pitch,  cute::intel::coord_t coordinate, void* dst_pointer);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockLoadTransformINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* src_base_pointer, int memory_width, int memory_height,
    int memory_pitch,  cute::intel::coord_t coordinate, void* dst_pointer);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockLoadTransposeINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* src_base_pointer, int memory_width, int memory_height,
    int memory_pitch,  cute::intel::coord_t coordinate, void* dst_pointer);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockStoreINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount, 
    void* src_pointer, const void* dst_base_pointer, int memory_width,
    int memory_height, int memory_pitch,  cute::intel::coord_t coordinate);    
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockPrefetchINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* src_base_pointer, int memory_width, int memory_height,
    int memory_pitch,  cute::intel::coord_t coordinate);

#if defined(CUTE_ARCH_XE_SPIRV_ENABLED)
namespace cute::detail {
template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockLoad {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
#ifdef __SYCL_DEVICE_ONLY__
        __spirv_Subgroup2DBlockLoadINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
            srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate,
            static_cast<void *>(dstPointer));
#endif
    }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockTransform {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
          cute::intel::coord_t coordinate, T* dstPointer) {
#ifdef __SYCL_DEVICE_ONLY__
        __spirv_Subgroup2DBlockLoadTransformINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
          srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate,
          static_cast<void *>(dstPointer));
#endif
  }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockTranspose {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
          cute::intel::coord_t coordinate, T* dstPointer) {
#ifdef __SYCL_DEVICE_ONLY__
        __spirv_Subgroup2DBlockLoadTransposeINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
          srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate,
          static_cast<void *>(dstPointer));
#endif
  }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockPrefetch {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
#ifdef __SYCL_DEVICE_ONLY__
        __spirv_Subgroup2DBlockPrefetchINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
            srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate);
#endif
    }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockStore {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* dstBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
          cute::intel::coord_t coordinate, T* srcPointer) {
#ifdef __SYCL_DEVICE_ONLY__
        __spirv_Subgroup2DBlockStoreINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
            (void*)(srcPointer), dstBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate);
#endif
  }
};
} // namespace cute::detail end
#endif

enum class CacheControl {
  kDefault   = 0,
  kL1UC_L3UC = 1, // Override to L1 uncached and L3 uncached
  kL1UC_L3C  = 2, // Override to L1 uncached and L3 cached
  kL1C_L3UC  = 3, // Override to L1 cached and L3 uncached
  kL1C_L3C   = 4, // Override to L1 cached and L3 cached
  kL1S_L3UC  = 5, // Override to L1 streaming load and L3 uncached
  kL1S_L3C   = 6, // Override to L1 streaming load and L3 cached
  kL1IAR_L3C = 7, // Override to L1 invalidate-after-read, and L3 cached
};

namespace cute::detail{
  template <class DElement, class AElement, class BElement, class CElement>
  struct XeSubgroupMatrixMultiplyAccumulate {
    static_assert(dependent_false<>, "Unsupported MMA Configuration.");
  };
} // namespace cute::detail end

// mma_bf16
SYCL_DEVICE_OCL(cute::intel::float8 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short8 a, cute::intel::int8 b, cute::intel::float8 acc));
SYCL_DEVICE_OCL(cute::intel::float4 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short4 a, cute::intel::int8 b, cute::intel::float4 acc));
SYCL_DEVICE_OCL(cute::intel::float2 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short2 a, cute::intel::int8 b, cute::intel::float2 acc));
SYCL_DEVICE_OCL(              float intel_sub_group_bf16_bf16_matrix_mad_k16(              short a, cute::intel::int8 b,               float acc));
// mma_half
SYCL_DEVICE_OCL(cute::intel::float8 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short8 a, cute::intel::int8 b, cute::intel::float8 acc));
SYCL_DEVICE_OCL(cute::intel::float4 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short4 a, cute::intel::int8 b, cute::intel::float4 acc));
SYCL_DEVICE_OCL(cute::intel::float2 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short2 a, cute::intel::int8 b, cute::intel::float2 acc));
SYCL_DEVICE_OCL(              float intel_sub_group_f16_f16_matrix_mad_k16(              short a, cute::intel::int8 b,               float acc));
// mma_s8
SYCL_DEVICE_OCL(cute::intel::int8 intel_sub_group_i8_i8_matrix_mad_k32(cute::intel::short8 a, cute::intel::int8 b, cute::intel::int8 acc));
SYCL_DEVICE_OCL(cute::intel::int4 intel_sub_group_i8_i8_matrix_mad_k32(cute::intel::short4 a, cute::intel::int8 b, cute::intel::int4 acc));
SYCL_DEVICE_OCL(cute::intel::int2 intel_sub_group_i8_i8_matrix_mad_k32(cute::intel::short2 a, cute::intel::int8 b, cute::intel::int2 acc));
SYCL_DEVICE_OCL(              int intel_sub_group_i8_i8_matrix_mad_k32(              short a, cute::intel::int8 b,               int acc));
// mma_u8
SYCL_DEVICE_OCL(cute::intel::int8 intel_sub_group_u8_u8_matrix_mad_k32(cute::intel::ushort8 a, cute::intel::uint8 b, cute::intel::int8 acc));
SYCL_DEVICE_OCL(cute::intel::int4 intel_sub_group_u8_u8_matrix_mad_k32(cute::intel::ushort4 a, cute::intel::uint8 b, cute::intel::int4 acc));
SYCL_DEVICE_OCL(cute::intel::int2 intel_sub_group_u8_u8_matrix_mad_k32(cute::intel::ushort2 a, cute::intel::uint8 b, cute::intel::int2 acc));
SYCL_DEVICE_OCL(              int intel_sub_group_u8_u8_matrix_mad_k32(              ushort a, cute::intel::uint8 b,               int acc));
// mma_tf32
SYCL_DEVICE_OCL(cute::intel::float8 intel_sub_group_tf32_tf32_matrix_mad_k8(cute::intel::float4 a, cute::intel::float8 b, cute::intel::float8 acc));
SYCL_DEVICE_OCL(cute::intel::float4 intel_sub_group_tf32_tf32_matrix_mad_k8(cute::intel::float2 a, cute::intel::float8 b, cute::intel::float4 acc));
SYCL_DEVICE_OCL(cute::intel::float2 intel_sub_group_tf32_tf32_matrix_mad_k8(              float a, cute::intel::float8 b, cute::intel::float2 acc));
SYCL_DEVICE_OCL(              float intel_sub_group_tf32_tf32_matrix_mad_k8(              float a, cute::intel::float8 b,               float acc));
// mma_bfloat16 with bfloat16 accumulator:
SYCL_DEVICE_OCL(cute::intel::short8 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short8 a, cute::intel::int8 b, cute::intel::short8 acc));
SYCL_DEVICE_OCL(cute::intel::short4 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short4 a, cute::intel::int8 b, cute::intel::short4 acc));
SYCL_DEVICE_OCL(cute::intel::short2 intel_sub_group_bf16_bf16_matrix_mad_k16(cute::intel::short2 a, cute::intel::int8 b, cute::intel::short2 acc));
SYCL_DEVICE_OCL(              short intel_sub_group_bf16_bf16_matrix_mad_k16(              short a, cute::intel::int8 b,               short acc));
// mma_half with half accumulator:
SYCL_DEVICE_OCL(cute::intel::half8 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short8 a, cute::intel::int8 b, cute::intel::half8 acc));
SYCL_DEVICE_OCL(cute::intel::half4 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short4 a, cute::intel::int8 b, cute::intel::half4 acc));
SYCL_DEVICE_OCL(cute::intel::half2 intel_sub_group_f16_f16_matrix_mad_k16(cute::intel::short2 a, cute::intel::int8 b, cute::intel::half2 acc));
SYCL_DEVICE_OCL(        sycl::half intel_sub_group_f16_f16_matrix_mad_k16(              short a, cute::intel::int8 b,         sycl::half acc));

#if defined(CUTE_ARCH_XE_BUILTIN_ENABLED)
namespace cute::detail
{
template<>
struct XeSubgroupMatrixMultiplyAccumulate<float, bfloat16_t, bfloat16_t, float> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
#endif
    }
};
  
template<>
struct XeSubgroupMatrixMultiplyAccumulate<float, half_t, half_t, float> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return intel_sub_group_f16_f16_matrix_mad_k16(a, b, c);
#endif
    }
};

template<>
struct XeSubgroupMatrixMultiplyAccumulate<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
#endif
    }
};
  
template<>
struct XeSubgroupMatrixMultiplyAccumulate<half_t, half_t, half_t, half_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return intel_sub_group_f16_f16_matrix_mad_k16(a, b, c);
#endif
    }
};
  
template<>
struct XeSubgroupMatrixMultiplyAccumulate<int32_t, int8_t, int8_t, int32_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return intel_sub_group_i8_i8_matrix_mad_k32(a, b, c);
#endif
    }
};
  
template<>
struct XeSubgroupMatrixMultiplyAccumulate<int32_t, uint8_t, uint8_t, int32_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return intel_sub_group_u8_u8_matrix_mad_k32(a, b, c);
#endif
    }
};
  
template<>
struct XeSubgroupMatrixMultiplyAccumulate<float, tfloat32_t, tfloat32_t, float> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, c);
#endif
    }
};
} // namespace cute::detail end
#endif


// @brief spirv APIs for mma 
// @param dims K 
// @param ARegisters
// @param BRegisters 
// @param AccRegisters
// @param Operands code 
// @return DRegisters
SYCL_EXTERNAL cute::intel::float8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short8, cute::intel::int8, cute::intel::float8, int32_t);
SYCL_EXTERNAL cute::intel::float4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short4, cute::intel::int8, cute::intel::float4, int32_t);
SYCL_EXTERNAL cute::intel::float2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short2, cute::intel::int8, cute::intel::float2, int32_t);
SYCL_EXTERNAL               float __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t,               short, cute::intel::int8,               float, int32_t);

SYCL_EXTERNAL cute::intel::int8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short8, cute::intel::int8, cute::intel::int8, int32_t);
SYCL_EXTERNAL cute::intel::int4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short4, cute::intel::int8, cute::intel::int4, int32_t);
SYCL_EXTERNAL cute::intel::int2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short2, cute::intel::int8, cute::intel::int2, int32_t);
SYCL_EXTERNAL               int __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t,               short, cute::intel::int8,               int, int32_t);

SYCL_EXTERNAL cute::intel::int8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::ushort8, cute::intel::uint8, cute::intel::int8, int32_t);
SYCL_EXTERNAL cute::intel::int4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::ushort4, cute::intel::uint8, cute::intel::int4, int32_t);
SYCL_EXTERNAL cute::intel::int2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::ushort2, cute::intel::uint8, cute::intel::int2, int32_t);
SYCL_EXTERNAL               int __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t,               ushort, cute::intel::uint8,               int, int32_t);

SYCL_EXTERNAL cute::intel::float8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::float4, cute::intel::float8, cute::intel::float8, int32_t);
SYCL_EXTERNAL cute::intel::float4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::float2, cute::intel::float8, cute::intel::float4, int32_t);
SYCL_EXTERNAL cute::intel::float2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t,               float, cute::intel::float8, cute::intel::float2, int32_t);
SYCL_EXTERNAL               float __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t,               float, cute::intel::float8,               float, int32_t);

SYCL_EXTERNAL cute::intel::short8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short8, cute::intel::int8, cute::intel::short8, int32_t);
SYCL_EXTERNAL cute::intel::short4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short4, cute::intel::int8, cute::intel::short4, int32_t);
SYCL_EXTERNAL cute::intel::short2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short2, cute::intel::int8, cute::intel::short2, int32_t);
SYCL_EXTERNAL               short __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t,               short, cute::intel::int8,               short, int32_t);

SYCL_EXTERNAL cute::intel::half8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short8, cute::intel::int8, cute::intel::half8, int32_t);
SYCL_EXTERNAL cute::intel::half4 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short4, cute::intel::int8, cute::intel::half4, int32_t);
SYCL_EXTERNAL cute::intel::half2 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, cute::intel::short2, cute::intel::int8, cute::intel::half2, int32_t);
SYCL_EXTERNAL         sycl::half __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t,               short, cute::intel::int8,         sycl::half, int32_t);

struct SPIRV_MMAOperands {
  static constexpr int SPIRV_MatrixASigned = 0x1;
  static constexpr int SPIRV_MatrixBSigned = 0x2;
  static constexpr int SPIRV_MatrixAInt8 = 0x10;
  static constexpr int SPIRV_MatrixBInt8 = 0x20;
  static constexpr int SPIRV_MatrixAFp16 = 0x400;
  static constexpr int SPIRV_MatrixBFp16 = 0x800;
  static constexpr int SPIRV_MatrixABf16 = 0x1000;
  static constexpr int SPIRV_MatrixBBf16 = 0x2000;
  static constexpr int SPIRV_MatrixCBf16 = 0xC;
  static constexpr int SPIRV_MatrixATf32 = 0x100;
  static constexpr int SPIRV_MatrixBTf32 = 0x200;
};
#if defined(CUTE_ARCH_XE_SPIRV_ENABLED)
namespace cute::detail
{
template<>
struct XeSubgroupMatrixMultiplyAccumulate<float, bfloat16_t, bfloat16_t, float> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, SPIRV_MMAOperands::SPIRV_MatrixABf16 | SPIRV_MMAOperands::SPIRV_MatrixBBf16 );
#endif
    }
};

template<>
struct XeSubgroupMatrixMultiplyAccumulate<float, half_t, half_t, float> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, SPIRV_MMAOperands::SPIRV_MatrixAFp16 | SPIRV_MMAOperands::SPIRV_MatrixBFp16);
#endif
    }
};

template<>
struct XeSubgroupMatrixMultiplyAccumulate<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, SPIRV_MMAOperands::SPIRV_MatrixABf16 | SPIRV_MMAOperands::SPIRV_MatrixBBf16 |SPIRV_MMAOperands::SPIRV_MatrixCBf16);
#endif
    }
};

template<>
struct XeSubgroupMatrixMultiplyAccumulate<half_t, half_t, half_t, half_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, a, b, c, SPIRV_MMAOperands::SPIRV_MatrixAFp16 | SPIRV_MMAOperands::SPIRV_MatrixBFp16);
#endif
    }
};

template<>
struct XeSubgroupMatrixMultiplyAccumulate<int32_t, int8_t, int8_t, int32_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(32, a, b, c, SPIRV_MMAOperands::SPIRV_MatrixASigned | SPIRV_MMAOperands::SPIRV_MatrixBSigned | SPIRV_MMAOperands::SPIRV_MatrixAInt8 | SPIRV_MMAOperands::SPIRV_MatrixBInt8);
#endif
    }
};

template<>
struct XeSubgroupMatrixMultiplyAccumulate<int32_t, uint8_t, uint8_t, int32_t> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(32, a, b, c, SPIRV_MMAOperands::SPIRV_MatrixAInt8 | SPIRV_MMAOperands::SPIRV_MatrixBInt8);
#endif
    }
};

template<>
struct XeSubgroupMatrixMultiplyAccumulate<float, tfloat32_t, tfloat32_t, float> {
    template<typename ARegisters, typename BRegisters, typename CRegisters>
    CUTE_HOST_DEVICE
    auto operator()(ARegisters a, BRegisters b, CRegisters c) {
#ifdef __SYCL_DEVICE_ONLY__
     return __spirv_SubgroupMatrixMultiplyAccumulateINTEL(8, a, b, c, SPIRV_MMAOperands::SPIRV_MatrixATf32 | SPIRV_MMAOperands::SPIRV_MatrixBTf32);
#endif
    }
};
} // namespace cute::detail end
#endif
