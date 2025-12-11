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
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_conversion.h>

// Helper device function for E4M3 -> BFLOAT16 bitwise conversion
CUTLASS_DEVICE inline uint16_t
fp8_e4m3_to_fp16_bitwise(uint8_t const& src) {
    // E4M3 (1-4-3) constants
    constexpr uint32_t e4m3_exp_bias = 7;
    // BFLOAT16 (1-8-7) constants
    constexpr uint32_t bf16_exp_bias = 127;

    // Unpack FP8 bits
    uint16_t sign = static_cast<uint16_t>(src & 0x80);
    uint16_t exponent = static_cast<uint16_t>(src & 0x78) >> 3;
    uint16_t mantissa = static_cast<uint16_t>(src & 0x07);

    // Reconstruct BFLOAT16 bits
    uint16_t bf16_sign = sign << 8;
    // Re-bias exponent and shift to BFLOAT16 position
    uint16_t bf16_exponent = (exponent - e4m3_exp_bias + bf16_exp_bias) << 7;
    // Shift mantissa to BFLOAT16 position
    uint16_t bf16_mantissa = mantissa << 4;

    return bf16_sign | bf16_exponent | bf16_mantissa;
}

// Helper device function for E5M2 -> BFLOAT16 bitwise conversion
CUTLASS_DEVICE inline uint16_t
fp8_e5m2_to_fp16_bitwise(uint8_t const& src) {
    // E5M2 (1-5-2) constants
    constexpr uint32_t e5m2_exp_bias = 15;
    // BFLOAT16 (1-8-7) constants
    constexpr uint32_t bf16_exp_bias = 127;

    // Unpack FP8 bits
    uint16_t sign = static_cast<uint16_t>(src & 0x80);
    uint16_t exponent = static_cast<uint16_t>(src & 0x7C) >> 2;
    uint16_t mantissa = static_cast<uint16_t>(src & 0x03);

    // Reconstruct BFLOAT16 bits
    uint16_t bf16_sign = sign << 8;
    // Re-bias exponent and shift to BFLOAT16 position
    uint16_t bf16_exponent = (exponent - e5m2_exp_bias + bf16_exp_bias) << 7;
    // Shift mantissa to BFLOAT16 position
    uint16_t bf16_mantissa = mantissa << 5;

    return bf16_sign | bf16_exponent | bf16_mantissa;
}


template <
    typename Encoding,
    int VectorizeSize = 8,
    typename SrcTensor,
    typename DstTensor
>
CUTLASS_DEVICE void
convert_and_descale(
    SrcTensor const& src,
    DstTensor& dst,
    float scale) {

    using SrcVec_u8 = sycl::vec<uint8_t, VectorizeSize>;
    using DstVec_u16 = sycl::vec<uint16_t, VectorizeSize>;

    auto src_ptr = reinterpret_cast<SrcVec_u8 const*>(src.data());
    auto dst_ptr = reinterpret_cast<DstVec_u16*>(dst.data());

    // Create a SCALAR bfloat16_t for scaling
    const cutlass::bfloat16_t scale_bf16 = static_cast<cutlass::bfloat16_t>(scale);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(src) / VectorizeSize; ++i) {
        SrcVec_u8 const src_vec_u8 = src_ptr[i];
        DstVec_u16 result_vec_u16;

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < VectorizeSize; ++j) {
            // 1. Convert FP8 bits to BFLOAT16 bits
            uint16_t val_bf16_bits;
            if constexpr (std::is_same_v<Encoding, cutlass::float_e4m3_t>) {
                val_bf16_bits = fp8_e4m3_to_fp16_bitwise(src_vec_u8[j]);
            } else {
                val_bf16_bits = fp8_e5m2_to_fp16_bitwise(src_vec_u8[j]);
            }

            // 2. Reinterpret bits as bfloat16_t to perform math
            cutlass::bfloat16_t val_bf16 = reinterpret_cast<cutlass::bfloat16_t const&>(val_bf16_bits);

            // 3. Apply scaling
            val_bf16 *= scale_bf16;

            // 4. Reinterpret back to bits for storage
            result_vec_u16[j] = reinterpret_cast<uint16_t const&>(val_bf16);
        }

        // 5. Store the final vector of bits
        dst_ptr[i] = result_vec_u16;
    }
}

template <typename EncodingType, typename TensorIn, typename TensorOut>
CUTLASS_DEVICE void
convert_FP8_to_FP16(TensorIn const &in,
                    TensorOut &out) {

  static_assert(cute::is_rmem<typename TensorIn::engine_type>::value,
                "Input tensor for A conversion must come from registers");
  static_assert(cute::is_rmem<typename TensorOut::engine_type>::value,
                "Output tensor for A conversion must come from registers");
  static_assert(cute::cosize_v<typename TensorIn::layout_type> == cute::cosize_v<typename TensorOut::layout_type>);
  static_assert(cute::size_v<typename TensorIn::layout_type> == cute::cosize_v<typename TensorIn::layout_type>);
  static_assert(cute::size_v<typename TensorOut::layout_type> == cute::cosize_v<typename TensorOut::layout_type>);

  using SrcType = typename TensorIn::value_type;
  using DstType = typename TensorOut::value_type;

  static_assert(std::is_same_v<SrcType, uint8_t>,
                "Expected fp8 input as uint8_t");
  static_assert(cute::is_any_of_v<EncodingType, cute::float_e5m2_t, cute::float_e4m3_t>,
                "Expected EncodingType to be float_e5m2_t or float_e4m3_t");

  constexpr int num_elements = decltype(size(in))::value;
  constexpr int fragment_size = std::is_same_v<EncodingType, cute::float_e5m2_t> ? 4 : 8;

  static_assert(num_elements % fragment_size == 0,
                "Currently, FP8 -> FP16 conversion is only supported when "
                "each work-item converts a multiple of fragment_size");

  auto in_frag = cute::recast<cutlass::Array<EncodingType, fragment_size>>(in);
  auto out_frag = cute::recast<cutlass::Array<DstType, fragment_size>>(out);

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < num_elements / fragment_size; ++i) {
    out_frag(i) = cutlass::NumericArrayConverter<DstType, EncodingType, fragment_size>{}(in_frag(i));
  }
}
