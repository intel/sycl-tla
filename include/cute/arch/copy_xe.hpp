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
#include <cute/util/sycl_vec.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x) inline x { assert(false); }
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL x
#else
#define SYCL_DEVICE_OCL(x) inline x { assert(false); }
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
using namespace cute;

SYCL_DEVICE_BUILTIN(intel::ushort16 intel_subgroup_block_read_u16_m8k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord));

SYCL_DEVICE_BUILTIN(intel::int8 intel_subgroup_block_read_transform_u16_k16(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord));

// prefetch
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_uchar(
    const __attribute__((opencl_global)) uint8_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_ushort(
    const __attribute__((opencl_global)) uint16_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_uint(
    const __attribute__((opencl_global)) uint32_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_uint2(
    const __attribute__((opencl_global)) uint32_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_uint4(
    const __attribute__((opencl_global)) uint32_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_uint8(
    const __attribute__((opencl_global)) uint32_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_ulong(
    const __attribute__((opencl_global)) uint64_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_ulong2(
    const __attribute__((opencl_global)) uint64_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_ulong4(
    const __attribute__((opencl_global)) uint64_t *base, int immElemOff,
    enum CacheControl cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_lsc_prefetch_global_ulong8(
    const __attribute__((opencl_global)) uint64_t *base, int immElemOff,
    enum CacheControl cacheOpt));

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, enum CacheControl cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, enum CacheControl cache_control));

// 8bits No transform No transpose
SYCL_DEVICE_BUILTIN(ushort __builtin_IB_subgroup_block_read_flat_u8_m1k32v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort2 __builtin_IB_subgroup_block_read_flat_u8_m2k32v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort4 __builtin_IB_subgroup_block_read_flat_u8_m4k32v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort8 __builtin_IB_subgroup_block_read_flat_u8_m8k32v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort16 __builtin_IB_subgroup_block_read_flat_u8_m16k32v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort32 __builtin_IB_subgroup_block_read_flat_u8_m32k32v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

SYCL_DEVICE_BUILTIN(
    intel::ushort2 __builtin_IB_subgroup_block_read_flat_u8_m1k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort4 __builtin_IB_subgroup_block_read_flat_u8_m2k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort8 __builtin_IB_subgroup_block_read_flat_u8_m4k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort16 __builtin_IB_subgroup_block_read_flat_u8_m8k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort32 __builtin_IB_subgroup_block_read_flat_u8_m16k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort64 __builtin_IB_subgroup_block_read_flat_u8_m32k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 16 bits No transform No transpose
SYCL_DEVICE_BUILTIN(ushort __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort2 __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort4 __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort16 __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

SYCL_DEVICE_BUILTIN(
    intel::ushort2 __builtin_IB_subgroup_block_read_flat_u16_m1k16v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort4 __builtin_IB_subgroup_block_read_flat_u16_m2k16v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort8 __builtin_IB_subgroup_block_read_flat_u16_m4k16v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort16 __builtin_IB_subgroup_block_read_flat_u16_m8k16v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort32 __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ushort64 __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 32bits specific for tf32 No transform No transpose
SYCL_DEVICE_BUILTIN(
    uint __builtin_IB_subgroup_block_read_flat_u32_m1k8v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    uint __builtin_IB_subgroup_block_read_flat_u32_m2k8v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint2 __builtin_IB_subgroup_block_read_flat_u32_m4k8v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint4 __builtin_IB_subgroup_block_read_flat_u32_m8k8v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint8 __builtin_IB_subgroup_block_read_flat_u32_m16k8v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint16 __builtin_IB_subgroup_block_read_flat_u32_m32k8v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

SYCL_DEVICE_BUILTIN(
    intel::uint2 __builtin_IB_subgroup_block_read_flat_u32_m1k8v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint2 __builtin_IB_subgroup_block_read_flat_u32_m2k8v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint4 __builtin_IB_subgroup_block_read_flat_u32_m4k8v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint8 __builtin_IB_subgroup_block_read_flat_u32_m8k8v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint16 __builtin_IB_subgroup_block_read_flat_u32_m16k8v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint32 __builtin_IB_subgroup_block_read_flat_u32_m32k8v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 32bits No transform No transpose
SYCL_DEVICE_BUILTIN(uint __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint2 __builtin_IB_subgroup_block_read_flat_u32_m2k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint4 __builtin_IB_subgroup_block_read_flat_u32_m4k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint8 __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint16 __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint32 __builtin_IB_subgroup_block_read_flat_u32_m32k16v1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 8bits VNNI transform No transpose
SYCL_DEVICE_BUILTIN(
    intel::uint8 __builtin_IB_subgroup_block_read_flat_transform_u8_k32(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint16 __builtin_IB_subgroup_block_read_flat_transform_u8_k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint32 __builtin_IB_subgroup_block_read_flat_transform_u8_k32v4(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 16bits VNNI transform No transpose
SYCL_DEVICE_BUILTIN(
    intel::uint8 __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint16 __builtin_IB_subgroup_block_read_flat_transform_u16_k32(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint16 __builtin_IB_subgroup_block_read_flat_transform_u16_k16v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint32 __builtin_IB_subgroup_block_read_flat_transform_u16_k32v2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 32bits No transform Transpose
SYCL_DEVICE_BUILTIN(uint __builtin_IB_subgroup_block_read_flat_transpose_u32_k1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint2 __builtin_IB_subgroup_block_read_flat_transpose_u32_k2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint4 __builtin_IB_subgroup_block_read_flat_transpose_u32_k4(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::uint8 __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

SYCL_DEVICE_BUILTIN(
    intel::ushort8 __builtin_IB_subgroup_block_read_flat_transpose_u16_k8(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 64bits No transform Transpose
SYCL_DEVICE_BUILTIN(
    intel::ulong __builtin_IB_subgroup_block_read_flat_transpose_u64_k1(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ulong2 __builtin_IB_subgroup_block_read_flat_transpose_u64_k2(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ulong4 __builtin_IB_subgroup_block_read_flat_transpose_u64_k4(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));
SYCL_DEVICE_BUILTIN(
    intel::ulong4 __builtin_IB_subgroup_block_read_flat_transpose_u16_k16(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::coord_t coord));

// 8bits No transform No transpose
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u8_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uchar data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u8_m2k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uchar2 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u8_m4k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uchar4));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u8_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uchar8));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u8_m8k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uchar8));

// 16bits
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, ushort data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m2k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::ushort2 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m4k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::ushort4 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::ushort8 data));

// 32bits
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, uint data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m2k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uint2 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m4k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uint4 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::uint8 data));

#undef SYCL_DEVICE_BUILTIN

#define __global __attribute__((opencl_global))
SYCL_DEVICE_OCL(uint intel_sub_group_block_read(const __global uint *p));

// 8 bits No transform No transpose
SYCL_DEVICE_OCL(ushort intel_sub_group_block_read_8b_1r32c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort2 intel_sub_group_block_read_8b_2r32c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort4 intel_sub_group_block_read_8b_4r32c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort8 intel_sub_group_block_read_8b_8r32c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort16 intel_sub_group_block_read_8b_16r32c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

SYCL_DEVICE_OCL(intel::ushort2 intel_sub_group_block_read_8b_1r32x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort4 intel_sub_group_block_read_8b_2r32x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort8 intel_sub_group_block_read_8b_4r32x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort16 intel_sub_group_block_read_8b_8r32x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort32 intel_sub_group_block_read_8b_16r32x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort64 intel_sub_group_block_read_8b_32r32x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 16bits No transform No transpose
SYCL_DEVICE_OCL(ushort intel_sub_group_block_read_16b_1r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort2 intel_sub_group_block_read_16b_2r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort4 intel_sub_group_block_read_16b_4r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort8 intel_sub_group_block_read_16b_8r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort16 intel_sub_group_block_read_16b_16r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort32 intel_sub_group_block_read_16b_32r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

SYCL_DEVICE_OCL(intel::ushort2 intel_sub_group_block_read_16b_1r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort4 intel_sub_group_block_read_16b_2r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort8 intel_sub_group_block_read_16b_4r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort16 intel_sub_group_block_read_16b_8r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort32 intel_sub_group_block_read_16b_16r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ushort64 intel_sub_group_block_read_16b_32r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 32bits specific for tf32 No transform No transpose
SYCL_DEVICE_OCL(uint intel_sub_group_block_read_32b_1r8c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(uint intel_sub_group_block_read_32b_2r8c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint2 intel_sub_group_block_read_32b_4r8c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint4 intel_sub_group_block_read_32b_8r8c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint8 intel_sub_group_block_read_32b_16r8c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint16 intel_sub_group_block_read_32b_32r8c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

SYCL_DEVICE_OCL(intel::uint2 intel_sub_group_block_read_32b_1r8x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint2 intel_sub_group_block_read_32b_2r8x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint4 intel_sub_group_block_read_32b_4r8x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint8 intel_sub_group_block_read_32b_8r8x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint16 intel_sub_group_block_read_32b_16r8x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint32 intel_sub_group_block_read_32b_32r8x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 32bits No transform No transpose
SYCL_DEVICE_OCL(uint intel_sub_group_block_read_32b_1r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint2 intel_sub_group_block_read_32b_2r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint4 intel_sub_group_block_read_32b_4r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint8 intel_sub_group_block_read_32b_8r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint16 intel_sub_group_block_read_32b_16r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint32 intel_sub_group_block_read_32b_32r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 8bits VNNI transform No transpose
SYCL_DEVICE_OCL(intel::uint8 intel_sub_group_block_read_transform_8b_32r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint16 intel_sub_group_block_read_transform_8b_32r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint32 intel_sub_group_block_read_transform_8b_32r16x4c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 16bits VNNI transform No transpose
SYCL_DEVICE_OCL(intel::uint8 intel_sub_group_block_read_transform_16b_16r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint16 intel_sub_group_block_read_transform_16b_32r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint16 intel_sub_group_block_read_transform_16b_16r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint32 intel_sub_group_block_read_transform_16b_32r16x2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 32bits No transform Transpose
SYCL_DEVICE_OCL(uint intel_sub_group_block_read_transpose_32b_16r1c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint2 intel_sub_group_block_read_transpose_32b_16r2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint4 intel_sub_group_block_read_transpose_32b_16r4c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::uint8 intel_sub_group_block_read_transpose_32b_16r8c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 64bits No transform Transpose
SYCL_DEVICE_OCL(ulong intel_sub_group_block_read_transpose_64b_8r1c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ulong2 intel_sub_group_block_read_transpose_64b_8r2c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(intel::ulong4 intel_sub_group_block_read_transpose_64b_8r4c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord));

// 8bits store
SYCL_DEVICE_OCL(void intel_sub_group_block_write_8b_1r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::uchar data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_8b_2r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::uchar2 data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_8b_4r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::uchar4 data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_8b_8r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::uchar8 data));

// 16bits store
SYCL_DEVICE_OCL(void intel_sub_group_block_write_16b_1r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, ushort data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_16b_2r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::ushort2 data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_16b_4r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::ushort4 data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_16b_8r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::ushort8 data));

// 32bits store
SYCL_DEVICE_OCL(void intel_sub_group_block_write_32b_1r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, uint data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_32b_2r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::uint2 data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_32b_4r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::uint4 data));
SYCL_DEVICE_OCL(void intel_sub_group_block_write_32b_8r16c(
    const __global void *base_address, int width, int height, int pitch,
    intel::coord_t coord, intel::uint8 data));

// 2D prefetch
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_8b_1r32x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_8b_2r32x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_8b_4r32x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_8b_8r32x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_1r16x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_2r16x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_4r16x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_8r16x2c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_8b_32r16x1c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_16b_16r16x1c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
SYCL_DEVICE_OCL(void intel_sub_group_2d_block_prefetch_32b_16r8x1c(
    __global void* base_address, int width, int height, int pitch,
    intel::coord_t coord));
#undef SYCL_DEVICE_OCL

namespace cute
{
struct XE_2D_U8x1x32_LD_N {

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<ushort *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m1k32v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x2x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m2k32v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x2x32_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    __builtin_IB_subgroup_block_write_flat_u16_m2k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::ushort2 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x4x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m4k32v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x8x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m8k32v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x16x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m16k32v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U8x32x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m32k32v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x1x64_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m1k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_8b_1r32x2c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U8x2x64_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m2k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_8b_2r32x2c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U8x4x64_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m4k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_8b_4r32x2c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U8x8x64_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m8k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_8b_8r32x2c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U8x16x64_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m16k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U8x32x64_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::ushort64 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u8_m32k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x1x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<ushort *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x2x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x4x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x8x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x1x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m1k16v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_16b_1r16x2c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x2x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m2k16v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_16b_2r16x2c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x4x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m4k16v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_16b_4r16x2c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x8x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m8k16v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x32_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::ushort64 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      // __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
      __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_TF32x1x8_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<uint *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m1k8v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x2x8_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<uint *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m2k8v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x4x8_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m4k8v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x8x8_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m8k8v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x16x8_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m16k8v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x32x8_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m32k8v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x1x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m1k8v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x2x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m2k8v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x4x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m4k8v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x8x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m8k8v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x16x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m16k8v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_TF32x32x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m32k8v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x1x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<uint *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x2x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m2k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x4x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m4k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x8x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x16x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x32x16_LD_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_u32_m32k16v1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x32x16_LD_V {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::uint8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transform_u8_k32(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_8b_32r16x1c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U8x32x32_LD_V {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::uint16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transform_u8_k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x32x64_LD_V {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    *reinterpret_cast<intel::uint32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transform_u8_k32v4(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x16x16_LD_V {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::uint8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x16_LD_V {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::uint16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transform_u16_k32(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x32_LD_V {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::uint16 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transform_u16_k16v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x32x32_LD_V {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::uint32 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transform_u16_k32v2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          CacheControl::kL1C_L3C);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U32x16x1_LD_T {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<uint *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u32_k1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x16x2_LD_T {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u32_k2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x16x4_LD_T {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u32_k4(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x16x8_LD_T {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *reinterpret_cast<intel::uint8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
#if defined(SYCL_INTEL_TARGET)
      intel_sub_group_2d_block_prefetch_32b_16r8x1c(
          (__global void*)baseoffset, width - 1, height - 1, pitch - 1, coord);
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
  };
};

struct XE_2D_U16x16x8_LD_T {
  using inst_dtype = uint32_t;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 4");
    *reinterpret_cast<intel::uint4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u32_k4(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U64x8x1_LD_T {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 8, "Expected T to have size 8");
    *reinterpret_cast<ulong *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u64_k1(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U64x8x2_LD_T {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 8, "Expected T to have size 8");
    *reinterpret_cast<intel::ulong2 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u64_k2(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U64x8x4_LD_T {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 8, "Expected T to have size 8");
    *reinterpret_cast<intel::ulong4 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u64_k4(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x16x16_LD_T {
  using inst_dtype = uint32_t;
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *reinterpret_cast<intel::uint8 *>(dst) =
        __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
            (long)(baseoffset), width - 1, height - 1, pitch - 1, coord);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x1x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    __builtin_IB_subgroup_block_write_flat_u8_m1k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uchar *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x2x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    __builtin_IB_subgroup_block_write_flat_u8_m2k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uchar2 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x4x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    __builtin_IB_subgroup_block_write_flat_u8_m4k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uchar4 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x8x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    __builtin_IB_subgroup_block_write_flat_u8_m8k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uchar8 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U8x8x32_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 1, "Expected T to have size 1");
    __builtin_IB_subgroup_block_write_flat_u8_m8k16v2(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uchar8 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x1x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __builtin_IB_subgroup_block_write_flat_u16_m1k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(ushort *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x2x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __builtin_IB_subgroup_block_write_flat_u16_m2k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::ushort2 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x4x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __builtin_IB_subgroup_block_write_flat_u16_m4k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::ushort4 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U16x8x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 2, "Expected T to have size 2");
    __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::ushort8 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x1x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 4, "Expected T to have size 4");
    __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(uint *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x2x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    __builtin_IB_subgroup_block_write_flat_u32_m2k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uint2 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x4x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    __builtin_IB_subgroup_block_write_flat_u32_m4k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uint4 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

struct XE_2D_U32x8x16_ST_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, intel::coord_t coord,
                                    const T *src) {
#if defined(SYCL_INTEL_TARGET)
    // static_assert(sizeof(T) == 4, "Expected T to have size 4");
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
        (long)(baseoffset), width - 1, height - 1, pitch - 1, coord,
        *(intel::uint8 *)(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
#endif
  }
};

template<class S, class D = S>
struct XE_ATOMIC {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(is_same_v<S, float> || is_same_v<S, double> || is_same_v<S, int>);

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(S_ const& src, D_ & dst) {
    #if defined(SYCL_INTEL_TARGET)
      auto v = sycl::atomic_ref<D_, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(*&dst);
      v += static_cast<D_>(*&src);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
    #endif
  }
};

template <class S, class D = S>
struct XE_1D_LDSM {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(D) % sizeof(S) == 0,
    "dst failed to vectorize into registers");
  static constexpr size_t N = sizeof(D) / sizeof(S);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
    "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(const S_ &src, D_ &dst) {
    #if defined(SYCL_INTEL_TARGET)
      CUTE_STATIC_ASSERT(sizeof(S_) == sizeof(S));
      auto sg = sycl::ext::oneapi::experimental::this_nd_item<3>().get_sub_group();
      *(sycl::vec<S_, N>*)(&dst)
        = sg.load<N>(sycl::address_space_cast<sycl::access::address_space::local_space,
                  sycl::access::decorated::yes>(&*&src));
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
    #endif 
  }
};

template <class S, class D = S>
struct PREFETCH {
  using SRegisters = S[1];
  using DRegisters = D[1];

  template <class S_, class D_>
  CUTE_HOST_DEVICE static void copy(const S_ &src, D_ &dst) {
#if defined(SYCL_INTEL_TARGET)
    if constexpr(sizeof(D) == 1) {
      __builtin_IB_lsc_prefetch_global_uchar(
          (const __attribute__((opencl_global)) uint8_t *)(&*&src), 0, CacheControl::kL1C_L3C);
    }
    else if constexpr(sizeof(D) == 2) {
      __builtin_IB_lsc_prefetch_global_ushort(
          (const __attribute__((opencl_global)) uint16_t *)(&*&src), 0, CacheControl::kL1C_L3C);
    }
    else if constexpr(sizeof(D) == 4) {
      __builtin_IB_lsc_prefetch_global_uint(
          (const __attribute__((opencl_global)) uint32_t *)(&*&src), 0, CacheControl::kL1C_L3C);
    }
    else if constexpr(sizeof(D) == 8) {
      __builtin_IB_lsc_prefetch_global_uint2(
          (const __attribute__((opencl_global)) uint32_t *)(&*&src), 0, CacheControl::kL1C_L3C);
    }
    else if constexpr(sizeof(D) == 16) {
      __builtin_IB_lsc_prefetch_global_uint4(
          (const __attribute__((opencl_global)) uint32_t *)(&*&src), 0, CacheControl::kL1C_L3C);
    }
    else if constexpr(sizeof(D) == 32) {
      __builtin_IB_lsc_prefetch_global_uint8(
          (const __attribute__((opencl_global)) uint32_t *)(&*&src), 0, CacheControl::kL1C_L3C);
    }
    else if constexpr(sizeof(D) == 64) {
      __builtin_IB_lsc_prefetch_global_ulong8(
          (const __attribute__((opencl_global)) uint64_t *)(&*&src), 0, CacheControl::kL1C_L3C);
    }
#else
      CUTE_INVALID_CONTROL_PATH(
          "Trying to use block prefetch on non-PVC hardware");
#endif
    }
};

template <class S, class D = S>
struct XE_1D_LOAD_GLOBAL {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(D) % sizeof(S) == 0,
    "dst failed to vectorize into registers");
  static constexpr size_t N = sizeof(D) / sizeof(S);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
    "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(const S_ &src, D_ &dst) {
    #if defined(SYCL_INTEL_TARGET)
      CUTE_STATIC_ASSERT(sizeof(S_) == sizeof(S));
      CUTE_STATIC_ASSERT(sizeof(D_) == sizeof(D));
      auto sg = sycl::ext::oneapi::experimental::this_nd_item<3>().get_sub_group();
      *(sycl::vec<S_, N>*)(&dst) 
        = sg.load<N>(sycl::address_space_cast<sycl::access::address_space::global_space,
                  sycl::access::decorated::yes>(&*&src));
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
    #endif 
  }

  using PREFETCH = PREFETCH<S, D>;

};

template<class S, class D = S>
struct XE_1D_STSM {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(S) % sizeof(D) == 0,
      "src failed to vectorize into registers");
  static constexpr size_t N = sizeof(S) / sizeof(D);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
      "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(S_ const& src, D_ & dst) {
    #if defined(SYCL_INTEL_TARGET)
      auto sg = sycl::ext::oneapi::experimental::this_nd_item<3>().get_sub_group(); 
      sg.store<N>(sycl::address_space_cast<sycl::access::address_space::local_space,
            sycl::access::decorated::yes>(&*&dst), *(sycl::vec<D_, N>*)(&src));
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
    #endif
  }
};

template<class S, class D = S>
struct XE_1D_STORE_GLOBAL {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(S) % sizeof(D) == 0,
      "src failed to vectorize into registers");
  static constexpr size_t N = sizeof(S) / sizeof(D);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
      "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(S_ const& src, D_ &dst) {
    #if defined(SYCL_INTEL_TARGET)
      auto sg = sycl::ext::oneapi::experimental::this_nd_item<3>().get_sub_group(); 
      sg.store<N>(sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::yes>(&*&dst), *(sycl::vec<D_, N>*)(&src));
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-PVC hardware");
    #endif
  }
};
} // end namespace cute
