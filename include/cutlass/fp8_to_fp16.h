/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include <cutlass/half.h>
#include <cute/util/sycl_vec.hpp>

using uchar16 = cute::intel::uchar16;
using ushort16 = cute::intel::ushort16;

static inline ushort16 convert_ushort16(uchar16 x) {
    ushort16 result;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = static_cast<uint16_t>(x[i]);
    }
    return result;
}

static inline unsigned short E4M3_to_FP16(unsigned char xin) {
  unsigned char xa, sgn_x, nan_mask, den_mask;

  union {
      signed short i;
      _Float16 f;
  } x16, den_corr;

  xa = xin & 0x7f;
  sgn_x = xin ^ xa;

  // mask for NaN input
  nan_mask = (0x7e - xa) & 0x80;
  // mask for denormal / zero input
  den_mask = (((signed char)(xa - 8)) >> 7);

  // apply Nan correction
  xa += (nan_mask >> 1);
  // first denormal correction
  xa |= (den_mask & 8);
  den_mask &= 0x48;
  // exponent bias correction
  xa += 0x40;

  // zero-extend to 16 bits
  x16.i = xa;
  den_corr.i = den_mask;
  // FP16 format
  x16.i <<= 7;
  den_corr.i <<= 7;

  // apply correction for denormals/zero
  x16.f -= den_corr.f;

  // finally, apply the sign
  x16.i ^= (((signed short)sgn_x) << 8);

  return (unsigned short)x16.i;
}



static inline ushort16 E4M3_to_FP16_chunk16(uchar16 xin) {
  uchar16 xa = xin & 0x7F;
  uchar16 sgn_x = xin ^ xa;

  uchar16 zero_mask;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
      zero_mask[i] = (xa[i] == 0) ? 1 : 0;
  }
  uchar16 nan_mask = (0x7E - xa) & 0x80;
  uchar16 den_mask = ((xa - 8) >> 7) & 0x01;

  xa += (nan_mask >> 1);
  xa |= (den_mask & 8);
  den_mask &= 0x48;
  xa += 0x40 & ~(zero_mask * 0x40);

  ushort16 x16 = convert_ushort16(xa) << 7;
  ushort16 den_corr = convert_ushort16(den_mask & ~zero_mask) << 7;

  ushort16 result = x16 - den_corr;
  result &= ~(convert_ushort16(zero_mask) << 7);

  ushort16 sign_ext = convert_ushort16(sgn_x) << 8;
  result ^= sign_ext;

  return result;
}


template <int N>
static inline void E5M2_to_FP16(cutlass::Array<uint16_t, N> const &xin, cutlass::Array<uint32_t, N> &xout) {
  // Adapted from https://github.com/pytorch/pytorch/blob/dfcfad2112933cc34247421ac0a4d3f19a1806c1/c10/util/Float8_e5m2.h#L30-L43,
  // except that since 32-bit registers & int32 ALUs are used, convert 2 FP8 elements at a time.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; i += 4) {
    // Manually unroll since using CUTLASS_PRAGMA_UNROLL or #pragma unroll
    // is not helping here with DPCPP nightly dated March 25.
    // More recent DPCPP nightlies are currently not working with cutlass.
    // If preprocessor directive for loop unrolling worked in this case,
    // we could've determined unrolling factor at compile time, based on how many elements each work-item handled.
    // An unrolling factor of 8 would be desirable since shl instruction can pipeline 9 inputs at a time,
    // but the number of elements are usually divisible by 8, so perhaps 8 is sufficient.
    // I chose an unrolling factor of 4 in this PR because it can work with a multiple of 8 FP8 elements.
    // The subgroup tile to be converted in that case would have a multiple of 128 elements.
    uint32_t tmp0 = static_cast<uint32_t>(xin[i]);
    uint32_t tmp1 = static_cast<uint32_t>(xin[i + 1]);
    uint32_t tmp2 = static_cast<uint32_t>(xin[i + 2]);
    uint32_t tmp3 = static_cast<uint32_t>(xin[i + 3]);
    uint32_t lo0 = (tmp0 & 0x000000FF) << 8;
    uint32_t lo1 = (tmp1 & 0x000000FF) << 8;
    uint32_t lo2 = (tmp2 & 0x000000FF) << 8;
    uint32_t lo3 = (tmp3 & 0x000000FF) << 8;
    uint32_t hi0 = (tmp0 & 0x0000FF00) << 16;
    uint32_t hi1 = (tmp1 & 0x0000FF00) << 16;
    uint32_t hi2 = (tmp2 & 0x0000FF00) << 16;
    uint32_t hi3 = (tmp3 & 0x0000FF00) << 16;
    xout[i] = hi0 | lo0;
    xout[i + 1] = hi1 | lo1;
    xout[i + 2] = hi2 | lo2;
    xout[i + 3] = hi3 | lo3;
  }
}

template <int N>
static inline void E5M2_to_FP16(cutlass::Array<uint8_t, N> const &xin, cutlass::Array<uint16_t, N> &xout) {
  // Adapted from https://github.com/pytorch/pytorch/blob/dfcfad2112933cc34247421ac0a4d3f19a1806c1/c10/util/Float8_e5m2.h#L30-L43
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    xout[i] = (static_cast<uint16_t>(xin[i])) << 8;
  }
}
