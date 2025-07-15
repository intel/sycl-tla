/***************************************************************************************************
* Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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
#include "fmha_decode_configuration.hpp"

using namespace cutlass;
using namespace cutlass::flash_attention;


template <typename FMHADecode>
static void inline FMHADecodeFunc(::benchmark::State& state,
                                cutlass::benchmark::FMHADecodeOptions const& options,
                                KernelHardwareInfo const& hw_info) {
  auto bench = cutlass::benchmark::BenchmarkRunnerFMHADecode<FMHADecode>();
  bench.run(state, options, hw_info);
}

struct FMHADecodeBenchGenConfig {
  static constexpr auto get_bool_tuple() {
    return std::make_tuple(true, false);
  }

  static constexpr auto get_kvtile_tuple() {
    return std::make_tuple(512, 1024);
  }

  static constexpr auto get_numsg_tuple() {
    return std::make_tuple(8, 16);
  }
};

template <typename String, typename InT, typename AccumT, typename OutT, bool Causal, bool VarLen, int KVTile, int NumSG, bool PagedKV>
static constexpr void generate_benchmarks() {
  using F = typename FMHADecodeConfigGen<InT, AccumT, OutT, Causal, VarLen, SHAPE_H<KVTile, NumSG>, PagedKV>::type;

  String str = "FMHADecode";
  String input_str = str + String{std::is_same_v<InT, bfloat16_t> ? "BF16BF16FP32" : "FP16FP16FP32"};
  String out_str = input_str + String{std::is_same_v<OutT, bfloat16_t> ? "BF16_RCR_" : std::is_same_v<OutT, half_t> ? "FP16_RCR_" : "FP32_RCR_"};
  String page_str = out_str + String{PagedKV ? "Paged_" : "NonPaged_"};
  String kvtile_str = page_str + String{"KVTile"} + String{std::to_string(KVTile)} + String{"_"};
  String head_dim_str = kvtile_str + String{"h"} + String{std::to_string(HEAD_DIM)} + String{"_"};
  String causal_str = head_dim_str + String{Causal ? "Causal_" : "NonCausal_"};
  String bench_name = causal_str + String{VarLen ? "VarLen" : "FixedLen"};

  cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::FMHADecodeOptions>::Register(bench_name, FMHADecodeFunc<F>);
}

template <typename ConfigTupleGen, typename InT, typename AccumT, typename OutT, bool Causal, bool VarLen, int KVTile, int NumSG, int paged_idx = 0>
static constexpr void generate_benchmarks_paged() {
  if constexpr (paged_idx < std::tuple_size_v<decltype(ConfigTupleGen::get_bool_tuple())>) {
    generate_benchmarks<std::string, InT, AccumT, OutT, Causal, VarLen, KVTile, NumSG, get<paged_idx>(ConfigTupleGen::get_bool_tuple())>();
    generate_benchmarks_paged<ConfigTupleGen, InT, AccumT, OutT, Causal, VarLen, KVTile, NumSG, paged_idx + 1>();
  }
}

template <typename ConfigTupleGen, typename InT, typename AccumT, typename OutT, bool Causal, bool VarLen, int kvtile_idx = 0>
static constexpr void generate_benchmarks_kvtile() {
  if constexpr (kvtile_idx < std::tuple_size_v<decltype(ConfigTupleGen::get_kvtile_tuple())>) {
    generate_benchmarks_paged<ConfigTupleGen, InT, AccumT, OutT, Causal, VarLen, get<kvtile_idx>(ConfigTupleGen::get_kvtile_tuple()), get<kvtile_idx>(ConfigTupleGen::get_numsg_tuple())>();
    generate_benchmarks_kvtile<ConfigTupleGen, InT, AccumT, OutT, Causal, VarLen, kvtile_idx + 1>();
  }
}

template <typename ConfigTupleGen, typename InT, typename AccumT, typename OutT, bool Causal, int varlen_idx = 0>
static constexpr void generate_benchmarks_varlen() {
  if constexpr (varlen_idx < std::tuple_size_v<decltype(ConfigTupleGen::get_bool_tuple())>) {
    generate_benchmarks_kvtile<ConfigTupleGen, InT, AccumT, OutT, Causal, get<varlen_idx>(ConfigTupleGen::get_bool_tuple())>();
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

static constexpr void register_flash_attention_decode_benchmarks() {
  generate_benchmarks_causal<FMHADecodeBenchGenConfig, cutlass::bfloat16_t, float, float>();
  generate_benchmarks_causal<FMHADecodeBenchGenConfig, cutlass::bfloat16_t, float, cutlass::bfloat16_t>();
  generate_benchmarks_causal<FMHADecodeBenchGenConfig, cutlass::half_t, float, float>();
  generate_benchmarks_causal<FMHADecodeBenchGenConfig, cutlass::half_t, float, cutlass::half_t>();
}
