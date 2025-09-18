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
    \brief Parameterized tests for Xe flash attention decode with model-specific configurations
*/

#include "flash_decode_testbed_3x.hpp"
#include <gtest/gtest.h>
#include <string>
#include <tuple>

namespace cutlass {

using MMAOperationFP16 = test::flash_attention::MMAOperationFP16;
using MMAOperationBF16 = test::flash_attention::MMAOperationBF16;
using GmemTiledCopyQ = test::flash_attention::GmemTiledCopyQU16;
using GmemTiledCopyK = test::flash_attention::GmemTiledCopyKU16;
using GmemTiledCopyV = test::flash_attention::GmemTiledCopyVU16;
using GmemTiledCopyStore = test::flash_attention::GmemTiledCopyStoreU32;

// Test parameters: data_type, head_size, model_name, is_causal
using TestParams = std::tuple<std::string, int, std::string, bool>;

class FlashDecodeModelsTest : public testing::TestWithParam<TestParams> {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

// FP16 Tests
TEST_P(FlashDecodeModelsTest, FP16_Models) {
  auto [data_type, head_size, model_name, is_causal] = GetParam();
  
  if (data_type != "fp16") return;

  if (head_size == 64) {
    using Shape_h = test::flash_attention::Shape_h64<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationFP16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  } else if (head_size == 96) {
    using Shape_h = test::flash_attention::Shape_h96<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationFP16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  } else if (head_size == 128) {
    using Shape_h = test::flash_attention::Shape_h128<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationFP16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  } else if (head_size == 192) {
    using Shape_h = test::flash_attention::Shape_h192<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationFP16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  }
}

// BF16 Tests
TEST_P(FlashDecodeModelsTest, BF16_Models) {
  auto [data_type, head_size, model_name, is_causal] = GetParam();
  
  if (data_type != "bf16") return;

  if (head_size == 64) {
    using Shape_h = test::flash_attention::Shape_h64<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationBF16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  } else if (head_size == 96) {
    using Shape_h = test::flash_attention::Shape_h96<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationBF16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  } else if (head_size == 128) {
    using Shape_h = test::flash_attention::Shape_h128<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationBF16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  } else if (head_size == 192) {
    using Shape_h = test::flash_attention::Shape_h192<1024, 8>;
    using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, bfloat16_t, 
                    typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                    typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                    MMAOperationBF16, is_causal, false,
                    GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
    EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(head_size, model_name));
  }
}

// Parameterize tests for all combinations of data_type, head_size, model_name, is_causal
INSTANTIATE_TEST_SUITE_P(
  ModelConfigurations,
  FlashDecodeModelsTest,
  testing::Values(
    // FP16 tests - Whisper V3 Large (non-causal)
    std::make_tuple("fp16", 64, "whisper_v3_large", false),
    std::make_tuple("fp16", 96, "whisper_v3_large", false),
    std::make_tuple("fp16", 128, "whisper_v3_large", false),
    std::make_tuple("fp16", 192, "whisper_v3_large", false),
    
    // FP16 tests - Llama3 8B (causal)
    std::make_tuple("fp16", 64, "llama3_8b", true),
    std::make_tuple("fp16", 96, "llama3_8b", true),
    std::make_tuple("fp16", 128, "llama3_8b", true),
    std::make_tuple("fp16", 192, "llama3_8b", true),
    
    // FP16 tests - Llama3 405B (causal)
    std::make_tuple("fp16", 64, "llama3_405b", true),
    std::make_tuple("fp16", 96, "llama3_405b", true),
    std::make_tuple("fp16", 128, "llama3_405b", true),
    std::make_tuple("fp16", 192, "llama3_405b", true),
    
    // FP16 tests - Qwen2.5 72B (causal)
    std::make_tuple("fp16", 64, "qwen2_5_72b", true),
    std::make_tuple("fp16", 96, "qwen2_5_72b", true),
    std::make_tuple("fp16", 128, "qwen2_5_72b", true),
    std::make_tuple("fp16", 192, "qwen2_5_72b", true),
    
    // FP16 tests - DeepSeek R1 (causal)
    std::make_tuple("fp16", 64, "deepseek_r1", true),
    std::make_tuple("fp16", 96, "deepseek_r1", true),
    std::make_tuple("fp16", 128, "deepseek_r1", true),
    std::make_tuple("fp16", 192, "deepseek_r1", true),
    
    // BF16 tests - same model coverage
    std::make_tuple("bf16", 64, "whisper_v3_large", false),
    std::make_tuple("bf16", 96, "whisper_v3_large", false),
    std::make_tuple("bf16", 128, "whisper_v3_large", false),
    std::make_tuple("bf16", 192, "whisper_v3_large", false),
    
    std::make_tuple("bf16", 64, "llama3_8b", true),
    std::make_tuple("bf16", 96, "llama3_8b", true),
    std::make_tuple("bf16", 128, "llama3_8b", true),
    std::make_tuple("bf16", 192, "llama3_8b", true),
    
    std::make_tuple("bf16", 64, "llama3_405b", true),
    std::make_tuple("bf16", 96, "llama3_405b", true),
    std::make_tuple("bf16", 128, "llama3_405b", true),
    std::make_tuple("bf16", 192, "llama3_405b", true),
    
    std::make_tuple("bf16", 64, "qwen2_5_72b", true),
    std::make_tuple("bf16", 96, "qwen2_5_72b", true),
    std::make_tuple("bf16", 128, "qwen2_5_72b", true),
    std::make_tuple("bf16", 192, "qwen2_5_72b", true),
    
    std::make_tuple("bf16", 64, "deepseek_r1", true),
    std::make_tuple("bf16", 96, "deepseek_r1", true),
    std::make_tuple("bf16", 128, "deepseek_r1", true),
    std::make_tuple("bf16", 192, "deepseek_r1", true)
  ),
  [](const testing::TestParamInfo<TestParams>& info) {
    auto [data_type, head_size, model_name, is_causal] = info.param;
    return data_type + "_h" + std::to_string(head_size) + "_" + model_name;
  }
);

} // namespace cutlass