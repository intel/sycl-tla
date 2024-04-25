/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace cute
{

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x)                                                 \
  inline x { assert(false); }
#endif

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, uint8 data));
SYCL_DEVICE_BUILTIN(ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(uint8 __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));

/// Load A
SYCL_DEVICE_BUILTIN(ushort64 __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort32 __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort16 intel_subgroup_block_read_u16_m8k16v2(
    __global void *baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));

/// Load B
SYCL_DEVICE_BUILTIN(uint16 __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));

#undef SYCL_DEVICE_BUILTIN

struct XE_2D_U16X8X16X1X1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *(ushort8 *)dst = __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
        (long)baseoffset, width - 1, height - 1, pitch - 1, coord);
  }
};

struct XE_2D_U32X8X16X1X1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    *(uint8 *)dst = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord);                                      
  }
};

struct XE_2D_U16X16X16X1X1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *(uint8 *)dst = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord);                                      
  }
};

struct XE_2D_U16X8X16X4X2_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *(ushort64 *)dst = __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
        long(baseoffset), width - 1, height - 1, pitch - 1, coord);
  }
};

struct XE_2D_U16X8X16X2X2_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    *(ushort32*) dst = __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
        long(baseoffset), width - 1, height - 1, pitch - 1, coord);
  }
};

struct XE_2D_U16X8X16X1X2_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    ushort16 tmp = (intel_subgroup_block_read_u16_m8k16v2(
        (__global void *)baseoffset, width, height, pitch, coord));
    *(ushort16 *)dst = *reinterpret_cast<ushort16 *>(&tmp);
  }
};

struct XE_2D_U16X8X16X4X1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
      *(ushort32*) dst = __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
  }
};

struct XE_2D_U32X8X16X2X1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    uint16 tmp = __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
        long(baseoffset), width - 1, height - 1, pitch - 1, coord);
    *(uint16 *)dst = *reinterpret_cast<uint16 *>(&tmp);
  }
};

struct XE_2D_U16X16X16X2X1_LD_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    static_assert(sizeof(T) == 2, "Expected T to have size 2");
    uint16 tmp = __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
        long(baseoffset), width - 1, height - 1, pitch - 1, coord);
    *(uint16 *)dst = *reinterpret_cast<uint16 *>(&tmp);
  }
};

struct XE_2D_U32X8X16X1X1_ST_N
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, int2_ coord, const T *src) {
    static_assert(sizeof(T) == 4, "Expected T to have size 4");
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
        (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
        *(uint8 *)src);
  }
};

} // end namespace cute
