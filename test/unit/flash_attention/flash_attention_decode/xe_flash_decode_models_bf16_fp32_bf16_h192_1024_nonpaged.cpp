/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation. All rights reserved.
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
    \brief Tests for Xe flash attention decode bf16 h192 1024 nonpaged with model-specific configurations
*/

#include "flash_decode_testbed_3x.hpp"

namespace cutlass {

using MMAOperationBF16 = test::flash_attention::MMAOperationBF16;
using GmemTiledCopyQ = test::flash_attention::GmemTiledCopyQU16;
using GmemTiledCopyK = test::flash_attention::GmemTiledCopyKU16;
using GmemTiledCopyV = test::flash_attention::GmemTiledCopyVU16;
using GmemTiledCopyStore = test::flash_attention::GmemTiledCopyStoreU16;
using Shape_h = test::flash_attention::Shape_h192<1024, 8>;

TEST(XE_Flash_Attention_Decode_Models_bf16_fp32_bf16_h192_1024_nonpaged, Whisper_V3_Large) {
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                                            typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, MMAOperationBF16, false, false,
                                            GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Decode_Models_bf16_fp32_bf16_h192_1024_nonpaged, Llama3_8B) {
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                                            typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, MMAOperationBF16, true, false,
                                            GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Decode_Models_bf16_fp32_bf16_h192_1024_nonpaged, Llama3_405B) {
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                                            typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, MMAOperationBF16, true, false,
                                            GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Decode_Models_bf16_fp32_bf16_h192_1024_nonpaged, Qwen2_5_72B) {
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                                            typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, MMAOperationBF16, true, false,
                                            GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Decode_Models_bf16_fp32_bf16_h192_1024_nonpaged, DeepSeek_R1) {
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                                            typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, MMAOperationBF16, true, false,
                                            GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

} // namespace cutlass
