#!/usr/bin/env python3
"""
Generate comprehensive 320-test coverage for Flash Attention Prefill CachedKV tests.

Coverage Matrix: 2×4×5×8 = 320 total tests
- Data Types: FP16, BF16 (2)
- Head Dimensions: 64, 96, 128, 192 (4) 
- Models: whisper_v3_large, llama3_8b, llama3_405b, qwen2_5_72b, deepseek_r1 (5)
- Template Combinations: HasCausalMask×UsePagedKV×isVarLen = 2×2×2 = 8

Each data type gets 160 tests (4×5×8).
"""

# Template parameters for different head dimensions
head_dim_configs = {
    64: {
        'ShapeQK': 'Shape<_128, _64, _64>',
        'ShapePV': 'Shape<_128, _32, _64>',
        'ShapeOutput': 'Shape<_128, _64, _64>',
        'SubgroupLayout': 'Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>'
    },
    96: {
        'ShapeQK': 'Shape<_128, _64, _32>',
        'ShapePV': 'Shape<_128, _32, _64>',
        'ShapeOutput': 'Shape<_128, _96, _64>',
        'SubgroupLayout': 'Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>'
    },
    128: {
        'ShapeQK': 'Shape<_128, _64, _64>',
        'ShapePV': 'Shape<_128, _32, _64>',
        'ShapeOutput': 'Shape<_128, _128, _64>',
        'SubgroupLayout': 'Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>'
    },
    192: {
        'ShapeQK': 'Shape<_256, _64, _64>',
        'ShapePV': 'Shape<_256, _32, _64>',
        'ShapeOutput': 'Shape<_256, _192, _64>',
        'SubgroupLayout': 'Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>'
    }
}

models = ['whisper_v3_large', 'llama3_8b', 'llama3_405b', 'qwen2_5_72b', 'deepseek_r1']

# Template parameter combinations: HasCausalMask, UsePagedKV, isVarLen
template_combinations = [
    (False, False, False),  # causal_none_standard
    (False, False, True),   # causal_none_varlen
    (False, True, False),   # causal_paged_standard
    (False, True, True),    # causal_paged_varlen
    (True, False, False),   # causal_standard
    (True, False, True),    # causal_varlen
    (True, True, False),    # causal_paged_standard
    (True, True, True),     # causal_paged_varlen
]

def get_test_suffix(has_causal, use_paged, is_varlen):
    """Generate test name suffix based on template parameters."""
    causal_str = 'causal' if has_causal else 'noncausal'
    paged_str = 'paged' if use_paged else 'standard'
    varlen_str = 'varlen' if is_varlen else 'fixlen'
    return f"{causal_str}_{paged_str}_{varlen_str}"

def generate_test_case(data_type, mma_op, head_dim, model, has_causal, use_paged, is_varlen):
    """Generate a single test case."""
    config = head_dim_configs[head_dim]
    suffix = get_test_suffix(has_causal, use_paged, is_varlen)
    
    causal_str = 'true' if has_causal else 'false'
    paged_str = 'true' if use_paged else 'false'
    varlen_str = 'true' if is_varlen else 'false'
    
    return f"""TEST(XE_Flash_Attention_Prefill_CachedKV_{data_type}_Complete_320, {data_type.lower()}_h{head_dim}_{suffix}_{model}) {{
  constexpr int PipelineStages = 2;
  using ShapeQK = {config['ShapeQK']};
  using ShapePV = {config['ShapePV']};
  using ShapeOutput = {config['ShapeOutput']};
  using SubgroupLayout = {config['SubgroupLayout']}; 
  using MMAOperation = {mma_op};
  using Kernel = test::flash_attention::XE_Flash_Attention_Prefill_CachedKV<{data_type.lower()}_t, float, float, ShapeQK, ShapePV, ShapeOutput, 
                                            SubgroupLayout, MMAOperation, {causal_str}, {paged_str}, {varlen_str}, PipelineStages>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestFlashPrefillCachedKVAll<Kernel>({head_dim}, "{model}"));
}}"""

def generate_complete_file(data_type, mma_op):
    """Generate complete test file for a data type."""
    header = f"""/***************************************************************************************************
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

/*! \\file
    \\brief Xe flash attention prefill cachedKV complete 320-test coverage - {data_type}
    
    Complete exhaustive coverage: 2×4×5×8 = 320 total tests
    - Data Types: FP16, BF16 (2)
    - Head Dimensions: 64, 96, 128, 192 (4)
    - Models: whisper_v3_large, llama3_8b, llama3_405b, qwen2_5_72b, deepseek_r1 (5)
    - Template Combinations: HasCausalMask×UsePagedKV×isVarLen = 2×2×2 = 8
    
    This file covers {160} {data_type} combinations.
*/

#include "flash_prefill_cachedkv_testbed_3x.hpp"

namespace cutlass {{

"""

    footer = "\n} // namespace cutlass\n"
    
    tests = []
    for head_dim in [64, 96, 128, 192]:
        for model in models:
            for has_causal, use_paged, is_varlen in template_combinations:
                tests.append(generate_test_case(data_type, mma_op, head_dim, model, has_causal, use_paged, is_varlen))
    
    return header + '\n\n'.join(tests) + footer

# Generate files
fp16_content = generate_complete_file('FP16', 'XE_8x16x16_F32F16F16F32_TT')
bf16_content = generate_complete_file('BF16', 'XE_8x16x16_F32BF16BF16F32_TT')

with open('xe_flash_prefill_cachedkv_models_fp16_complete_320.cpp', 'w') as f:
    f.write(fp16_content)

with open('xe_flash_prefill_cachedkv_models_bf16_complete_320.cpp', 'w') as f:
    f.write(bf16_content)

print("Generated complete 320-test files:")
print("- xe_flash_prefill_cachedkv_models_fp16_complete_320.cpp (160 FP16 tests)")
print("- xe_flash_prefill_cachedkv_models_bf16_complete_320.cpp (160 BF16 tests)")
print("Total: 320 tests covering all combinations")
print()
print("Coverage Matrix:")
print("- Data Types: FP16, BF16 (2)")
print("- Head Dimensions: 64, 96, 128, 192 (4)")
print("- Models: whisper_v3_large, llama3_8b, llama3_405b, qwen2_5_72b, deepseek_r1 (5)")
print("- Template Combinations: HasCausalMask×UsePagedKV×isVarLen = 2×2×2 = 8")
print("Total: 2×4×5×8 = 320 tests")