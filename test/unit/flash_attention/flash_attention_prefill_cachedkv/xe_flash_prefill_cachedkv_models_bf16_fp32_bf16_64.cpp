/*
   Model-specific test configurations for CUTLASS SYCL Flash Attention Prefill CachedKV
   Data types: bfloat16_t, float, float (FIXED: Using float output for maximum compatibility)
   Head dimension: 64
*/

#include "flash_prefill_cachedkv_testbed_3x.hpp"

namespace cutlass {

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_bfloat16_float_float_64, Whisper_V3_Large) {
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

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_bfloat16_float_float_64, Llama3_8B) {
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

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_bfloat16_float_float_64, Llama3_405B) {
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

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_bfloat16_float_float_64, Qwen2_5_72B) {
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

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_bfloat16_float_float_64, DeepSeek_R1) {
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

} // namespace cutlass
