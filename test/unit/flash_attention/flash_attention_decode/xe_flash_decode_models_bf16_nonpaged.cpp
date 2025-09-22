/****************************************************************************
 * Copyright (C) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * COMPLETE BF16 exhaustive tests - ALL 320 combinations non-paged
 * Coverage: 4 heads × 2 KV × 2 causal × 2 varlen × 5 models = 160 BF16 tests
 * Total Matrix: 2×4×2×5×2×2 = 320 combinations (FP16 + BF16)
 ***************************************************************************/

#include "flash_decode_testbed_3x.hpp"

namespace cutlass {

using MMAOperationBF16 = test::flash_attention::MMAOperationBF16;
using GmemTiledCopyQ = test::flash_attention::GmemTiledCopyQU16;
using GmemTiledCopyK = test::flash_attention::GmemTiledCopyKU16;
using GmemTiledCopyV = test::flash_attention::GmemTiledCopyVU16;
using GmemTiledCopyStore = test::flash_attention::GmemTiledCopyStoreU32;

// 20 tests: 5 models × 4 head sizes, KV512, causal, varlen

// h64 × KV512 × Causal × VarLen
TEST(XE_Flash_Attention_Decode_BF16, bf16_h64_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "whisper_v3_large"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h64_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_8b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h64_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "llama3_405b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h64_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "qwen2_5_72b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h64_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h64<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(64, "deepseek_r1"));
}

// h96 × KV512 × Causal × VarLen
TEST(XE_Flash_Attention_Decode_BF16, bf16_h96_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "whisper_v3_large"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h96_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_8b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h96_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "llama3_405b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h96_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "qwen2_5_72b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h96_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h96<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(96, "deepseek_r1"));
}

// h128 × KV512 × Causal × VarLen
TEST(XE_Flash_Attention_Decode_BF16, bf16_h128_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "whisper_v3_large"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h128_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_8b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h128_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "llama3_405b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h128_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "qwen2_5_72b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h128_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h128<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(128, "deepseek_r1"));
}

// h192 × KV512 × Causal × VarLen
TEST(XE_Flash_Attention_Decode_BF16, bf16_h192_kv512_causal_varlen_whisper) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "whisper_v3_large"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h192_kv512_causal_varlen_llama8b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_8b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h192_kv512_causal_varlen_llama405b) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "llama3_405b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h192_kv512_causal_varlen_qwen25) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "qwen2_5_72b"));
}
TEST(XE_Flash_Attention_Decode_BF16, bf16_h192_kv512_causal_varlen_deepseek) {
  using Shape_h = test::flash_attention::Shape_h192<512, 8>;
  using Kernel = test::flash_attention::XE_Flash_Attention_Decode<bfloat16_t, float, float,
                  typename Shape_h::ShapeQK, typename Shape_h::ShapePV,
                  typename Shape_h::ShapeOutput, typename Shape_h::SubgroupLayout,
                  MMAOperationBF16, true, true,
                  GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyStore, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashDecodeAll<Kernel>(192, "deepseek_r1"));
}

} // namespace cutlass
