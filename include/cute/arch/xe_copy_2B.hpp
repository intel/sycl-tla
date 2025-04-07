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

#include <cute/arch/copy.hpp>
#include <cute/config.hpp>
#include <cute/arch/xe_copy_4B.hpp>

SYCL_DEVICE_BUILTIN(cute::intel::ushort16 intel_subgroup_block_read_u16_m8k16v2(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord));

SYCL_DEVICE_BUILTIN(cute::intel::int8 intel_subgroup_block_read_transform_u16_k16(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord));

// U16 prefetch
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, enum CacheControl cache_control));

// 16 bits No transform No transpose
SYCL_DEVICE_BUILTIN(cute::intel::ushort __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort2 __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort4 __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort16 __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));

SYCL_DEVICE_BUILTIN(
    cute::intel::ushort2 __builtin_IB_subgroup_block_read_flat_u16_m1k16v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort4 __builtin_IB_subgroup_block_read_flat_u16_m2k16v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort8 __builtin_IB_subgroup_block_read_flat_u16_m4k16v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort16 __builtin_IB_subgroup_block_read_flat_u16_m8k16v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort32 __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::ushort64 __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));

// 16bits VNNI transform No transpose
SYCL_DEVICE_BUILTIN(
    cute::intel::uint8 __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::uint16 __builtin_IB_subgroup_block_read_flat_transform_u16_k32(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::uint16 __builtin_IB_subgroup_block_read_flat_transform_u16_k16v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    cute::intel::uint32 __builtin_IB_subgroup_block_read_flat_transform_u16_k32v2(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, cute::intel::coord_t coord));

// 16bits
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m1k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, cute::intel::ushort data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m2k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, cute::intel::ushort2 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m4k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, cute::intel::ushort4 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, cute::intel::coord_t coord, cute::intel::ushort8 data));

// 2D prefetch
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_1r16x2c(
    __global void* base_address, int width, int height, int pitch,
    cute::intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_2r16x2c(
    __global void* base_address, int width, int height, int pitch,
    cute::intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_4r16x2c(
    __global void* base_address, int width, int height, int pitch,
    cute::intel::coord_t coord));

namespace cute::detail {
#if defined(CUTE_ARCH_XE_BUILTIN_ENABLED)
template<>
struct XeSubgroup2DBlockLoad<2, 16, 1, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<ushort *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 2, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort2 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};


template<>
struct XeSubgroup2DBlockLoad<2, 16, 4, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort4 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 8, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort8 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 16, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort16 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 32, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort32 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 1, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort2 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m1k16v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 2, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort4 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m2k16v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 4, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort8 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m4k16v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 8, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort16 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m8k16v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 16, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort32 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockLoad<2, 16, 32, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::ushort64 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockTransform<2, 16, 16, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::uint8 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockTransform<2, 16, 32, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::uint16 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_transform_u16_k32(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockTransform<2, 16, 16, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::uint16 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_transform_u16_k16v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockTransform<2, 16, 32, 2> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* dstPointer) {
        *reinterpret_cast<intel::uint32 *>(dstPointer) =  __builtin_IB_subgroup_block_read_flat_transform_u16_k32v2(
           (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockStore<2, 16, 1, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* dstBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* srcPointer) {
        __builtin_IB_subgroup_block_write_flat_u16_m1k16v1(
           (intptr_t)(dstBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, *(ushort *)(srcPointer));
    }
};

template<>
struct XeSubgroup2DBlockStore<2, 16, 2, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* dstBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* srcPointer) {
        __builtin_IB_subgroup_block_write_flat_u16_m2k16v1(
           (intptr_t)(dstBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, *(intel::ushort2 *)(srcPointer));
    }
};

template<>
struct XeSubgroup2DBlockStore<2, 16, 4, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* dstBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* srcPointer) {
        __builtin_IB_subgroup_block_write_flat_u16_m4k16v1(
           (intptr_t)(dstBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, *(intel::ushort4 *)(srcPointer));
    }
};

template<>
struct XeSubgroup2DBlockStore<2, 16, 8, 1> {
    template<typename T>
    CUTE_HOST_DEVICE
    void operator()(const void* dstBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate, T* srcPointer) {
        __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(
           (intptr_t)(dstBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, *(intel::ushort8 *)(srcPointer));
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 8, 1> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
            (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, CacheControl::kL1C_L3C);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 16, 1> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
            (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, CacheControl::kL1C_L3C);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 32, 1> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
            (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, CacheControl::kL1C_L3C);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 1, 2> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        intel_sub_group_2d_block_prefetch_16b_1r16x2c(
            (__global void*)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 2, 2> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        intel_sub_group_2d_block_prefetch_16b_2r16x2c(
            (__global void*)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 4, 2> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        intel_sub_group_2d_block_prefetch_16b_4r16x2c(
            (__global void*)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 8, 2> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(
            (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, CacheControl::kL1C_L3C);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 16, 2> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
            (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, CacheControl::kL1C_L3C);
    }
};

template<>
struct XeSubgroup2DBlockPrefetch<2, 16, 32, 2> {
    CUTE_HOST_DEVICE
    void operator()(const void* srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
            cute::intel::coord_t coordinate) {
        __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
            (intptr_t)(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate, CacheControl::kL1C_L3C);
    }
};
#endif
}

namespace cute
{
struct XE_2D_U16x1x16_LD_N {
  using BlockShape = Shape<_1, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 1, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x2x16_LD_N {
  using BlockShape = Shape<_2, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 2, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x4x16_LD_N {
  using BlockShape = Shape<_4, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 4, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x8x16_LD_N {
  using BlockShape = Shape<_8, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 8, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 8, 1>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x16_LD_N {
  using BlockShape = Shape<_16, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 16, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 16, 1>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x16_LD_N {
  using BlockShape = Shape<_32, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 32, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 32, 1>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x1x32_LD_N {
  using BlockShape = Shape<_1, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 1, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 1, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x2x32_LD_N {
  using BlockShape = Shape<_2, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 2, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 2, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x4x32_LD_N {
  using BlockShape = Shape<_4, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 4, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 4, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x8x32_LD_N {
  using BlockShape = Shape<_8, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 8, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 8, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x32_LD_N {
  using BlockShape = Shape<_16, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 16, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 16, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x32_LD_N {
  using BlockShape = Shape<_32, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockLoad<2, 16, 32, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
      // __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
    detail::XeSubgroup2DBlockPrefetch<2, 16, 32, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x16_LD_V {
  using BlockShape = Shape<_16, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockTransform<2, 16, 16, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 16, 1>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x16_LD_V {
  using BlockShape = Shape<_32, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockTransform<2, 16, 32, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 32, 1>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x32_LD_V {
  using BlockShape = Shape<_16, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockTransform<2, 16, 16, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 16, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x32_LD_V {
  using BlockShape = Shape<_32, _32>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockTransform<2, 16, 32, 2>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(CUTE_ARCH_XE_ENABLED)
    detail::XeSubgroup2DBlockPrefetch<2, 16, 32, 2>{}(baseoffset, width, height, pitch, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x8_LD_T {
  using BlockShape = Shape<_8, _16>;
  using inst_dtype = uint32_t;

  static constexpr bool is_transpose = true;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 4");
    detail::XeSubgroup2DBlockTranspose<4, 4, 16, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x16x16_LD_T {
  using BlockShape = Shape<_16, _16>;
  using inst_dtype = uint32_t;

  static constexpr bool is_transpose = true;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_XE_ENABLED)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockTranspose<4, 8, 16, 1>{}(baseoffset, width, height, pitch, coord, dst);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x1x16_ST_N {
  using BlockShape = Shape<_1, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(CUTE_ARCH_XE_ENABLED)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockStore<2, 16, 1, 1>{}(baseoffset, width, height, pitch, coord, src);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x2x16_ST_N {
  using BlockShape = Shape<_2, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(CUTE_ARCH_XE_ENABLED)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockStore<2, 16, 2, 1>{}(baseoffset, width, height, pitch, coord, src);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x4x16_ST_N {
  using BlockShape = Shape<_4, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(CUTE_ARCH_XE_ENABLED)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockStore<2, 16, 4, 1>{}(baseoffset, width, height, pitch, coord, src);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x8x16_ST_N {
  using BlockShape = Shape<_8, _16>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(CUTE_ARCH_XE_ENABLED)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    detail::XeSubgroup2DBlockStore<2, 16, 8, 1>{}(baseoffset, width, height, pitch, coord, src);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};
} // end namespace cute
