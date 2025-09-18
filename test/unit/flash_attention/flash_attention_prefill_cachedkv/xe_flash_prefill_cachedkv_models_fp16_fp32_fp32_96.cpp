/*
   Model-specific test configurations for CUTLASS SYCL Flash Attention Prefill CachedKV
   Data types: half_t, float, float
   Head dimension: 96
   Fixed: Using H64-compatible kernel patterns
*/

#include "flash_prefill_cachedkv_testbed_3x.hpp"

namespace cutlass {

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_half_t_float_float_96, Whisper_V3_Large) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;     // Use H64 base tile shapes
  using ShapePV = Shape<_128, _32, _64>;     // Use H64 base tile shapes  
  using ShapeOutput = Shape<_128, _64, _64>; // Use H64 base tile shapes
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32F16F16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<half_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "whisper_v3_large"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_half_t_float_float_96, Llama3_8B) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32F16F16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<half_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_8b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_half_t_float_float_96, Llama3_405B) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32F16F16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<half_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "llama3_405b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_half_t_float_float_96, Qwen2_5_72B) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32F16F16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<half_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "qwen2_5_72b"));
}

TEST(XE_Flash_Attention_Prefill_CachedKV_Models_half_t_float_float_96, DeepSeek_R1) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
  using MMAOperation = XE_8x16x16_F32F16F16F32_TT;
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<half_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, false, false, false, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>(96, "deepseek_r1"));
}

} // namespace cutlass
