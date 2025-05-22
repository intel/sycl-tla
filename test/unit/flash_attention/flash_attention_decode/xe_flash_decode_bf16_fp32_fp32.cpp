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

/*! \file
    \brief Tests for Xe flash attention decode bf16
*/

#include "flash_decode_testbed_3x.hpp"

namespace cutlass {

using ShapeQK_h64 = Shape<_8, _512, _64>;
using ShapePV_h64 = Shape<_8, _32, _512>;
using ShapeOutput_h64 = Shape<_8, _64, _512>;

using ShapeQK_h96 = Shape<_8, _1024, _64>;
using ShapePV_h96 = Shape<_8, _32, _1024>;
using ShapeOutput_h96 = Shape<_8, _96, _1024>;

using ShapeQK_h128 = Shape<_8, _1024, _64>;
using ShapePV_h128 = Shape<_8, _32, _1024>;
using ShapeOutput_h128 = Shape<_8, _128, _1024>;

using ShapeQK_h192 = Shape<_8, _1024, _64>;
using ShapePV_h192 = Shape<_8, _32, _1024>;
using ShapeOutput_h192 = Shape<_8, _192, _1024>;

using SubgroupLayout_h64 = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
using SubgroupLayout_h96 = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
using SubgroupLayout_h128 = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
using SubgroupLayout_h192 = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;

using MMAOperationBF16 = XE_8x16x16_F32BF16BF16F32_TT;

#define EXECUTE_TEST_BF16(NAME, NAME_CAUSAL_VARLEN, DTYPE_IN, DTYPE_ACCUM, DTYPE_OUT, MMAOperation, CAUSAL, VARLEN, HEADSIZE) \
TEST(NAME##HEADSIZE, NAME_CAUSAL_VARLEN) { \
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<DTYPE_IN, DTYPE_ACCUM, DTYPE_OUT, ShapeQK_h##HEADSIZE, ShapePV_h##HEADSIZE, \
                                            ShapeOutput_h##HEADSIZE, SubgroupLayout_h##HEADSIZE, MMAOperation, CAUSAL, VARLEN>::Kernel; \
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(HEADSIZE)); \
}

#define EXECUTE_TEST_HEAD_SIZE_BF16(NAME, CAUSAL, VARLEN) \
EXECUTE_TEST_BF16(XE_Flash_Attention_Decode_bf16_fp32_fp32_h, NAME, bfloat16_t, float, float, MMAOperationBF16, CAUSAL, VARLEN, 64) \
EXECUTE_TEST_BF16(XE_Flash_Attention_Decode_bf16_fp32_fp32_h, NAME, bfloat16_t, float, float, MMAOperationBF16, CAUSAL, VARLEN, 96) \
EXECUTE_TEST_BF16(XE_Flash_Attention_Decode_bf16_fp32_fp32_h, NAME, bfloat16_t, float, float, MMAOperationBF16, CAUSAL, VARLEN, 128) \
EXECUTE_TEST_BF16(XE_Flash_Attention_Decode_bf16_fp32_fp32_h, NAME, bfloat16_t, float, float, MMAOperationBF16, CAUSAL, VARLEN, 192)


EXECUTE_TEST_HEAD_SIZE_BF16(causal, true, false)
EXECUTE_TEST_HEAD_SIZE_BF16(noncausal, false, false)
EXECUTE_TEST_HEAD_SIZE_BF16(varlen_causal, true, true)
EXECUTE_TEST_HEAD_SIZE_BF16(varlen_noncausal, false, true)

#undef EXECUTE_TEST_HEAD_SIZE_BF16
#undef EXECUTE_TEST_BF16

} // namespace cutlass
