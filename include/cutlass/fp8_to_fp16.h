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

#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/underscore.hpp>
#include <cute/util/sycl_vec.hpp>
#include <cutlass/detail/helper_macros.hpp>
#include <cutlass/half.h>

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
static inline void E5M2_to_FP16(cutlass::Array<uint32_t, N / 4> const &xin,
                                cutlass::Array<uint32_t, N / 2> &xout) {
  // Since 32-bit registers & int32 ALUs are used, convert 4 FP8 elements at a time.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0, j = 0; i < N / 2; i += 4, j += 2) {
    // Manually unroll since using CUTLASS_PRAGMA_UNROLL or #pragma unroll
    // is not working here with DPCPP nightly dated March 24.
    // More recent DPCPP nightlies were not working with cutlass at the time
    // this code was written. If preprocessor directive for loop unrolling
    // worked, we could've determined the unrolling factor at compile time,
    // based on how many elements each work-item handled.
    // An unrolling factor of 8 would be desirable (although shl instruction can
    // pipeline 9 inputs at a time, the inputs we usually handle are not a
    // multiple of 9). but this code uses an unrolling factor of 4 so that it
    // can work with a multiple of 8 FP8 elements. The subgroup tile to be
    // converted in that case would have a multiple of 128 elements, so we would
    // also be able to cover small subgroup-level tiles such as 8x16, which
    // would not have been possible with an unrolling factor of 8.
    uint32_t tmp0 = xin[j];
    uint32_t tmp1 = xin[j + 1];
    uint32_t first = (tmp0 & 0x000000FF) << 8;
    uint32_t second = (tmp0 & 0x0000FF00) << 16;
    uint32_t third = (tmp0 & 0x00FF0000) >> 8;
    uint32_t fourth = (tmp0 & 0xFF000000);
    uint32_t fifth = (tmp1 & 0x000000FF) << 8;
    uint32_t sixth = (tmp1 & 0x0000FF00) << 16;
    uint32_t seventh = (tmp1 & 0x00FF0000) >> 8;
    uint32_t eighth = (tmp1 & 0xFF000000);
    xout[i] = first | second;
    xout[i + 1] = third | fourth;
    xout[i + 2] = fifth | sixth;
    xout[i + 3] = seventh | eighth;
  }
}

template <int N>
static inline void E5M2_to_FP16(cutlass::Array<uint8_t, N> const &xin,
                                cutlass::Array<uint16_t, N> &xout) {
  // Adapted from
  // https://github.com/pytorch/pytorch/blob/dfcfad2112933cc34247421ac0a4d3f19a1806c1/c10/util/Float8_e5m2.h#L30-L43
  // Using something as simple as the following code surprisingly
  // leads to poor performance.
  // CUTLASS_PRAGMA_UNROLL
  // for (int i = 0; i < num_elements; i++) {
  //   reinterpret_cast<uint16_t*>(pDst)[i] =
  //   (static_cast<uint16_t>((pSrc[i]))) << 8;
  // }
  // The root-cause is unknown, but private memory use is seen in that case.
  // We're using a workaround that doesn't use private memory.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    xout[i] = (static_cast<uint16_t>(xin[i])) << 8;
  }
}

template <typename ElementA, class EngineIn, class EngineOut, class LayoutIn,
          class LayoutOut, class... Ts>
CUTLASS_DEVICE void
convert_FP8_to_FP16(cute::Tensor<EngineIn, LayoutIn> const &in,
                    cute::Tensor<EngineOut, LayoutOut> &out) {

  static_assert(cute::is_rmem<EngineIn>::value,
                "Input tensor for A conversion must come from registers");
  static_assert(cute::is_rmem<EngineOut>::value,
                "Output tensor for A conversion must come from registers");
  static_assert(cute::cosize_v<LayoutIn> == cute::cosize_v<LayoutOut>);
  static_assert(cute::size_v<LayoutIn> == cute::cosize_v<LayoutIn>);
  static_assert(cute::size_v<LayoutOut> == cute::cosize_v<LayoutOut>);

  using SrcType = typename EngineIn::value_type;
  using DstType = typename EngineOut::value_type;

  static_assert(std::is_same_v<SrcType, uint8_t>,
                "Expected fp8 (E4M3) input as uint8_t");
  static_assert(std::is_same_v<DstType, cute::half_t>,
                "Expected fp16 output as half_t");

  auto const &src = in(cute::_, cute::_, cute::_);
  auto const &dst = out(cute::_, cute::_, cute::_);

  SrcType const *pSrc = src.data();
  DstType *pDst = dst.data();

  constexpr int num_elements = decltype(size(src))::value;

  // TODO(Codeplay): Move conversion to NumericArrayConverter
  if constexpr (std::is_same_v<ElementA, cute::float_e5m2_t>) {
    // May convert two FP8 elements at a time
    constexpr bool use_faster_conversion = num_elements >= 8;
    using src_dtype =
        std::conditional_t<use_faster_conversion, uint32_t, uint8_t>;
    using dst_dtype =
        std::conditional_t<use_faster_conversion, uint32_t, uint16_t>;
    using SrcArray =
        std::conditional_t<use_faster_conversion,
                           cutlass::Array<src_dtype, num_elements / 4>,
                           cutlass::Array<src_dtype, num_elements>>;
    using DstArray =
        std::conditional_t<use_faster_conversion,
                           cutlass::Array<dst_dtype, num_elements / 2>,
                           cutlass::Array<dst_dtype, num_elements>>;
    SrcArray const *pSrcArr = reinterpret_cast<SrcArray const *>(pSrc);
    DstArray *pDstArr = reinterpret_cast<DstArray *>(pDst);
    if constexpr (use_faster_conversion) {
      // convert 4 FP8 elements at a time
      E5M2_to_FP16<num_elements>(*pSrcArr, *pDstArr);
    } else {
      E5M2_to_FP16<num_elements>(*pSrcArr, *pDstArr);
    }
  } else {
    // E4M3 -> FP16 conversion
    constexpr int chunk_size = 16;
    constexpr int iters = num_elements / chunk_size;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < iters; ++i) {
      cute::intel::uchar16 src_vec;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < chunk_size; ++j) {
        src_vec[j] = pSrc[i * chunk_size + j];
      }
      cute::intel::ushort16 dst_vec;
      dst_vec = E4M3_to_FP16_chunk16(src_vec);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < chunk_size; ++j) {
        reinterpret_cast<uint16_t *>(pDst)[i * chunk_size + j] = dst_vec[j];
      }
    }
  }
}
