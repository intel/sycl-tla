/****************************************************************************
 * Copyright (C) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * COMPLETE FP16 exhaustive tests - ALL 320 combinations non-paged
 * Coverage: 4 heads × 2 KV × 2 causal × 2 varlen × 5 models = 160 FP16 tests
 * Total Matrix: 2×4×2×5×2×2 = 320 combinations (FP16 + BF16)
 ***************************************************************************/

#include "flash_decode_testbed_3x.hpp"

namespace cutlass {

using MMAOperationFP16 = test::flash_attention::MMAOperationFP16;
using GmemTiledCopyQ = test::flash_attention::GmemTiledCopyQU16;
using GmemTiledCopyK = test::flash_attention::GmemTiledCopyKU16;
using GmemTiledCopyV = test::flash_attention::GmemTiledCopyVU16;
using GmemTiledCopyStore = test::flash_attention::GmemTiledCopyStoreU32;

//==============================================================================
// FP16 HEAD SIZE 64 - ALL COMBINATIONS × ALL MODELS (40 tests)
//==============================================================================

// h64 × KV512 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV512 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV512 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV512 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV512 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h64 × KV512 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV512 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV512 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV512 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV512 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h64 × KV512 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV512 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV512 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV512 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV512 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h64 × KV512 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV512 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV512 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV512 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV512 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv512_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h64 × KV1024 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV1024 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV1024 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV1024 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV1024 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h64 × KV1024 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV1024 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV1024 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV1024 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV1024 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h64 × KV1024 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV1024 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV1024 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV1024 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV1024 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h64 × KV1024 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}

// h64 × KV1024 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}

// h64 × KV1024 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}

// h64 × KV1024 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}

// h64 × KV1024 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h64_kv1024_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

//==============================================================================
// FP16 HEAD SIZE 96 - ALL COMBINATIONS × ALL MODELS (40 tests)
//==============================================================================

// h96 × KV512 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV512 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV512 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV512 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV512 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h96 × KV512 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV512 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV512 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV512 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV512 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h96 × KV512 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV512 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV512 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV512 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV512 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h96 × KV512 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV512 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV512 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV512 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV512 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv512_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h96 × KV1024 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV1024 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV1024 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV1024 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV1024 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h96 × KV1024 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV1024 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV1024 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV1024 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV1024 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h96 × KV1024 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV1024 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV1024 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV1024 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV1024 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h96 × KV1024 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}

// h96 × KV1024 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}

// h96 × KV1024 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}

// h96 × KV1024 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}

// h96 × KV1024 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h96_kv1024_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

//==============================================================================
// FP16 HEAD SIZE 128 - ALL COMBINATIONS × ALL MODELS (40 tests)
//==============================================================================

// h128 × KV512 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV512 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV512 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV512 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV512 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h128 × KV512 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV512 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV512 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV512 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV512 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h128 × KV512 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV512 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV512 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV512 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV512 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h128 × KV512 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV512 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV512 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV512 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV512 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv512_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h128 × KV1024 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV1024 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV1024 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV1024 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV1024 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h128 × KV1024 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV1024 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV1024 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV1024 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV1024 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h128 × KV1024 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV1024 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV1024 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV1024 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV1024 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h128 × KV1024 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}

// h128 × KV1024 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}

// h128 × KV1024 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}

// h128 × KV1024 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}

// h128 × KV1024 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h128_kv1024_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

//==============================================================================
// FP16 HEAD SIZE 192 - ALL COMBINATIONS × ALL MODELS (40 tests)
//==============================================================================

// h192 × KV512 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV512 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV512 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV512 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV512 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

// h192 × KV512 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV512 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV512 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV512 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV512 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

// h192 × KV512 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV512 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV512 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV512 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV512 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

// h192 × KV512 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV512 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV512 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV512 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV512 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv512_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

// h192 × KV1024 × Causal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV1024 × Causal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV1024 × Causal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV1024 × Causal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV1024 × Causal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

// h192 × KV1024 × Causal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV1024 × Causal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV1024 × Causal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV1024 × Causal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV1024 × Causal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_causal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, true, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

// h192 × KV1024 × NonCausal × VarLen × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV1024 × NonCausal × VarLen × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV1024 × NonCausal × VarLen × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV1024 × NonCausal × VarLen × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV1024 × NonCausal × VarLen × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

// h192 × KV1024 × NonCausal × Standard × whisper_v3_large
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_standard_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}

// h192 × KV1024 × NonCausal × Standard × llama3_8b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_standard_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}

// h192 × KV1024 × NonCausal × Standard × llama3_405b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_standard_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}

// h192 × KV1024 × NonCausal × Standard × qwen2_5_72b
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_standard_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}

// h192 × KV1024 × NonCausal × Standard × deepseek_r1
TEST(XE_Flash_Attention_Decode_FP16_Complete_320, fp16_h192_kv1024_noncausal_standard_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<1024, 16>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<half_t, float, float, 
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout, 
                  MMAOperationFP16, false, false,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

} // namespace cutlass

// Total FP16 tests generated: 160
