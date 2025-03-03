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

#include <cute/arch/mma_xe.hpp>

namespace cute {
// Mixed data type for MMA are offten used in quantization processing, like (int4_t/int8_t)  -->  (bf16/fp16)

// Use " XE_MIXED_DTYPE_8x16x16_F32F16F32_TT<int8_t, int4_t> " as example for GEMM(D = A x B + C)
//    "8x16x16"    : dpas MNK
//    "F32F16F32"  : the first 'F32' is data type of D, the second 'F16' is data type of DPAS (Xe required data type of A and B are same), the last 'F32' is data type of C
//    "TT"         : A transpose and B transpose
//    "int8_t"      : the data type of A in memory is int8_t, we must transform it to F16 before going to DPAS.
//    "int4_t"      : the data type of B in memory is int4_t, we must shuffle the data, then transform it to F16 before going to DPAS



// ====================  BF16 ====================
template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_8x16x16_F32BF16F32_TT : XE_8x16x16_F32BF16BF16F32_TT
{
  using MMA_Op = XE_8x16x16_F32BF16BF16F32_TT;
};

template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_4x16x16_F32BF16F32_TT : XE_4x16x16_F32BF16BF16F32_TT
{
  using MMA_Op = XE_4x16x16_F32BF16BF16F32_TT;
};

template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_2x16x16_F32BF16F32_TT : XE_2x16x16_F32BF16BF16F32_TT
{
  using MMA_Op = XE_2x16x16_F32BF16BF16F32_TT;
};

template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_1x16x16_F32BF16F32_TT : XE_1x16x16_F32BF16BF16F32_TT
{
  using MMA_Op = XE_1x16x16_F32BF16BF16F32_TT;
};




// ====================  F16 ====================
template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_8x16x16_F32F16F32_TT : XE_8x16x16_F32F16F16F32_TT
{
  using MMA_Op = XE_8x16x16_F32F16F16F32_TT;
};

template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_4x16x16_F32F16F32_TT : XE_4x16x16_F32F16F16F32_TT
{
  using MMA_Op = XE_4x16x16_F32F16F16F32_TT;
};

template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_2x16x16_F32F16F32_TT : XE_2x16x16_F32F16F16F32_TT
{
  using MMA_Op = XE_2x16x16_F32F16F16F32_TT;
};

template <class TypeA, class TypeB>
struct XE_MIXED_DTYPE_1x16x16_F32F16F32_TT : XE_1x16x16_F32F16F16F32_TT
{
  using MMA_Op = XE_1x16x16_F32F16F16F32_TT;
};


} //namespace cute
