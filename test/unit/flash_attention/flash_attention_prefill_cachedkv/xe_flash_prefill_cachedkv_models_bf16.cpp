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
    \brief Xe flash attention prefill cachedKV complete 320-test coverage - BF16
    
    Complete exhaustive coverage: 2×4×5×8 = 320 total tests
    - Data Types: FP16, BF16 (2)
    - Head Dimensions: 64, 96, 128, 192 (4)
    - Models: whisper_v3_large, llama3_8b, llama3_405b, qwen2_5_72b, deepseek_r1 (5)
    - Template Combinations: HasCausalMask×UsePagedKV×isVarLen = 2×2×2 = 8
    
    This file covers 160 BF16 combinations.
*/

#include "flash_prefill_cachedkv_testbed_3x.hpp"

namespace cutlass {

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_noncausal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h64_causal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(64, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_noncausal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h96_causal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_noncausal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h128_causal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(128, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_fixlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_varlen_whisper_v3_large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_fixlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_varlen_llama3_8b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_fixlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_varlen_llama3_405b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_fixlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_varlen_qwen2_5_72b) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_noncausal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_standard_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, false, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_fixlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_BF16_Complete_320, bf16_h192_causal_paged_varlen_deepseek_r1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<bfloat16_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, true, true, true, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(192, "deepseek_r1"));
}
} // namespace cutlass
