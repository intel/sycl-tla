/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
#include "cute/config.hpp"

// TODO(Codeplay): This builtin is not available on SPIRV
SYCL_EXTERNAL extern "C"
cute::intel::uint2 __builtin_IB_subgroup_block_read_flat_transpose_u32_k2(
  intptr_t baseoffset, int width_minus_one, int height_minus_one,
  int pitch_minus_one, cute::intel::coord_t coord);

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

namespace cute::detail {

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockLoad {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
    __spirv_Subgroup2DBlockLoadINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
      srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate,
      static_cast<void *>(dstPointer));
    }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockTransform {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
          cute::intel::coord_t coordinate, T* dstPointer) {
    __spirv_Subgroup2DBlockLoadTransformINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
      srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate,
      static_cast<void *>(dstPointer));
  }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockTranspose {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
          cute::intel::coord_t coordinate, T* dstPointer) {
    __spirv_Subgroup2DBlockLoadTransposeINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
      srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate,
      static_cast<void *>(dstPointer));
  }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockPrefetch {
  CUTE_HOST_DEVICE
  void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
      __spirv_Subgroup2DBlockPrefetchINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
        srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate);
    }
};

template<int ElementSize, int BlockWidth, int BlockHeight, int BlockCount>
struct XeSubgroup2DBlockStore {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* dstBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
          cute::intel::coord_t coordinate, T* srcPointer) {
    __spirv_Subgroup2DBlockStoreINTEL(ElementSize, BlockWidth, BlockHeight, BlockCount,
      (void*)(srcPointer), dstBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate);
  }
};

template<>
struct XeSubgroup2DBlockTranspose<4, 2, 16, 1> {
  template<typename T>
  CUTE_HOST_DEVICE
  void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
          cute::intel::coord_t coordinate, T* dstPointer) {
    *reinterpret_cast<intel::uint2 *>(dstPointer) = __builtin_IB_subgroup_block_read_flat_transpose_u32_k2(
       reinterpret_cast<long>(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
  }
};

} // namespace cute::detail end
