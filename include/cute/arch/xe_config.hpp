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
#define CUTE_ARCH_COPY_XE_ENABLED
#define CUTE_ARCH_MMA_XE_ENABLED
#endif
 
#if defined(CUTE_ARCH_COPY_XE_ENABLED) && defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER < 20250200)
#define CUTE_ARCH_COPY_XE_BUILTIN_ENABLED
#define CUTE_ARCH_MMA_XE_BUILTIN_ENABLED
#elif defined(CUTE_ARCH_COPY_XE_ENABLED)
#define CUTE_ARCH_COPY_XE_SPIRV_ENABLED
#define CUTE_ARCH_MMA_XE_SPIRV_ENABLED
// #define CUTE_ARCH_MMA_XE_BUILTIN_ENABLED
#endif

// SPIRV copy definitions
#if defined(CUTE_ARCH_COPY_XE_SPIRV_ENABLED)
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

// template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
// struct XeSubgroup2DBlockPrefetch {
//     CUTE_HOST_DEVICE
//     void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
//             cute::intel::coord_t coordinate) {
// #ifdef __SYCL_DEVICE_ONLY__
//         __spirv_Subgroup2DBlockPrefetchINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
//             srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate);
// #endif
//     }
// };


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

namespace cute::detail
{
template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockPrefetch {
  // static_assert(dependent_false<>, "Unsupported 2D Block Load Configuration.");
};

#if defined(CUTE_ARCH_COPY_XE_BUILTIN_ENABLED)
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
#endif
}
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

#if defined(CUTE_ARCH_MMA_XE_SPIRV_ENABLED)
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

struct SPIRV_MMAOperands {
  static constexpr int SPIRV_MatrixASigned = 0x1;
  static constexpr int SPIRV_MatrixBSigned = 0x2;
  static constexpr int SPIRV_MatrixAInt8 = 0x10;
  static constexpr int SPIRV_MatrixBInt8 = 0x20;
  static constexpr int SPIRV_MatrixAFp16 = 0x400;
  static constexpr int SPIRV_MatrixBFp16 = 0x800;
  static constexpr int SPIRV_MatrixABf16 = 0x1000;
  static constexpr int SPIRV_MatrixBBf16 = 0x2000;
  static constexpr int SPIRV_MatrixATf32 = 0x100;
  static constexpr int SPIRV_MatrixBTf32 = 0x200;
};
#endif

namespace cute::detail{
  template <class DElement, class AElement, class BElement, class CElement>
  struct XeSubgroupMatrixMultiplyAccumulate {
    static_assert(dependent_false<>, "Unsupported MMA Configuration.");
  };
} // namespace cute::detail end
