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

#pragma once

#include "benchmark_runner.hpp"
#include "fmha_prefill_configuration.hpp"

using namespace cutlass;
using namespace cutlass::flash_attention;

struct Shape_h64 {
  static constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutPut = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
};

struct Shape_h96 {
  static constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutPut = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
};

struct Shape_h128 {
  static constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutPut = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>; 
};

struct Shape_h192 {
  static constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutPut = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
};

template<class QKVType, class AccumType, class OutType, bool Causal, bool VarLen, class TileShapeConfig>
struct FMHAPrefillConfigGen {
  using type = cutlass::flash_attention::FMHAPrefillConfig<
      QKVType, AccumType, OutType,
      typename TileShapeConfig::ShapeQK,
      typename TileShapeConfig::ShapePV,
      typename TileShapeConfig::ShapeOutPut,
      typename TileShapeConfig::SubgroupLayout,
      Causal, VarLen, TileShapeConfig::PipelineStages>;
};

/////////////////////////////////////////////////////////////////////////////////////
template <typename FMHAPrefill>
static void inline FMHAPrefillFunc(::benchmark::State& state,
                                cutlass::benchmark::FMHAPrefillOptions const& options,
                                KernelHardwareInfo const& hw_info) {
  auto bench = cutlass::benchmark::BenchmarkRunnerFMHA<FMHAPrefill>();
  bench.run(state, options, hw_info);
}

struct FMHAPrefillBenchGenConfig {
  static constexpr auto get_bool_tuple() {
    return std::make_tuple(true, false);
  }

  template <typename String, typename InT>
  static constexpr String get_input_string() {
    if constexpr (std::is_same_v<InT, bfloat16_t>) {
      return String{"BF16BF16FP32"};
    } else if constexpr (std::is_same_v<InT, half_t>) {
      return String{"FP16FP16FP32"};
    } else if constexpr (std::is_same_v<InT, float_e5m2_t>) {
      return String{"FP8E5M2FP8E5M2FP32"};
    } else {
      return String{"FP8E4M3FP8E4M3FP32"};
    }
  }

  template <typename String, typename OutT>
  static constexpr String get_output_string() {
    if constexpr (std::is_same_v<OutT, bfloat16_t>) {
      return String{"BF16_RCR_"};
    } else if constexpr (std::is_same_v<OutT, half_t>) {
      return String{"FP16_RCR_"};
    } else if constexpr (std::is_same_v<OutT, float_e5m2_t>) {
      return String{"FP8E5M2_RCR_"};
    } else if constexpr (std::is_same_v<OutT, float_e4m3_t>) {
      return String{"FP8E4M3_RCR_"};
    } else {
      return String{"FP32_RCR_"};
    }
  }
};
/////////////////////////////////////////////////////////////////////////////////////

template <typename ConfigTupleGen, typename String, typename InT, typename AccumT, typename OutT, bool Causal, bool VarLen>
static constexpr void generate_benchmarks() {
  using F = typename FMHAPrefillConfigGen<InT, AccumT, OutT, Causal, VarLen, SHAPE_H>::type;

  String str = "FMHAPrefill";
  String input_str = str + ConfigTupleGen::template get_input_string<String, InT>();
  String out_str = input_str + ConfigTupleGen::template get_output_string<String, OutT>();
  String head_dim_str = out_str + String{"h"} + String{std::to_string(HEAD_DIM)} + String{"_"};
  String causal_str = head_dim_str + String{Causal ? "Causal_" : "NonCausal_"};
  String bench_name = causal_str + String{VarLen ? "VarLen" : "FixedLen"};

  cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::FMHAPrefillOptions>::Register(bench_name, FMHAPrefillFunc<F>);
}

template <typename ConfigTupleGen, typename InT, typename AccumT, typename OutT, bool Causal, int varlen_idx = 0>
static constexpr void generate_benchmarks_varlen() {
  if constexpr (varlen_idx < std::tuple_size_v<decltype(ConfigTupleGen::get_bool_tuple())>) {
    generate_benchmarks<ConfigTupleGen, std::string, InT, AccumT, OutT, Causal, get<varlen_idx>(ConfigTupleGen::get_bool_tuple())>();
    generate_benchmarks_varlen<ConfigTupleGen, InT, AccumT, OutT, Causal, varlen_idx + 1>();
  }
}

template <typename ConfigTupleGen, typename InT, typename AccumT, typename OutT, int causal_idx = 0>
static constexpr void generate_benchmarks_causal() {
  if constexpr (causal_idx < std::tuple_size_v<decltype(ConfigTupleGen::get_bool_tuple())>) {
    generate_benchmarks_varlen<ConfigTupleGen, InT, AccumT, OutT, get<causal_idx>(ConfigTupleGen::get_bool_tuple())>();
    generate_benchmarks_causal<ConfigTupleGen, InT, AccumT, OutT, causal_idx + 1>();
  }
}

static constexpr void register_flash_attention_prefill_benchmarks() {
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::bfloat16_t, float, float>();
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::bfloat16_t, float, cutlass::bfloat16_t>();
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::half_t, float, float>();
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::half_t, float, cutlass::half_t>();
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::float_e5m2_t, float, float>();
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::float_e5m2_t, float, cutlass::float_e5m2_t>();
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::float_e4m3_t, float, float>();
  generate_benchmarks_causal<FMHAPrefillBenchGenConfig, cutlass::float_e4m3_t, float, cutlass::float_e4m3_t>();
}
