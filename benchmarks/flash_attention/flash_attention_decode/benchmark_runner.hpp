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

#include "cutlass/../../benchmarks/common.hpp"
#include "flash_attention_v2/flash_decode_runner.hpp"

using namespace cute;
using namespace cutlass::flash_attention;

namespace cutlass::benchmark {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class FMHADecodeConfiguration>
struct BenchmarkRunnerFMHADecode : public FlashDecodeRunner<FMHADecodeConfiguration> {

  using FMHADecodeKernel = typename FMHADecodeConfiguration::FMHADecodeKernel;
  using ProblemShapeType = typename FMHADecodeKernel::ProblemShape;
  static constexpr bool Causal = FMHADecodeConfiguration::Causal;
  static constexpr bool isVarLen = FMHADecodeConfiguration::VarLen;
  static constexpr bool PagedKV = FMHADecodeConfiguration::PagedKV;

  int32_t count;
  //
  // Methods
  //

  bool verify(ProblemShapeType problem_size) {
    return FlashDecodeRunner<FMHADecodeConfiguration>::verify(problem_size);
  }

  /// Initialize operands to be used in the Flash Attention
  ProblemShapeType initialize(const FMHADecodeOptions &options) {
    auto [problem_shape, mem_count] = FlashDecodeRunner<FMHADecodeConfiguration>::initialize(options, true);
    count = mem_count;
    return problem_shape;
  }

  void run(::benchmark::State& state, const FMHADecodeOptions &options, const cutlass::KernelHardwareInfo &hw_info) {

    ProblemShapeType problem_size = initialize(options);

    typename FMHADecodeKernel::Arguments arguments = FlashDecodeRunner<FMHADecodeConfiguration>::get_arguments(problem_size, hw_info, options.softmax_scale, 0);

    size_t workspace_size = FMHADecodeKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    FMHADecodeKernel::can_implement(arguments);

    // Initialize the workspace
    auto status = FMHADecodeKernel::initialize_workspace(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      return;
    }

    typename FMHADecodeKernel::Params params = FMHADecodeKernel::to_underlying_arguments(arguments, workspace.get());

    // Run Flash Attention Decode
    FlashDecodeRunner<FMHADecodeConfiguration>::run(params);

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size);
    if(not passed) {
      state.SkipWithError("Disposition Failed.");
    }

    state.counters["batch"] = options.batch;
    state.counters["num_heads_q"] = options.num_heads_q;
    state.counters["num_heads_kv"] = options.num_heads_kv;
    state.counters["seq_len_qo"] = options.seq_len_qo;
    state.counters["seq_len_kv"] = options.seq_len_kv;
    state.counters["seq_len_kv_cache"] = options.seq_len_kv_cache;
    state.counters["head_size_kv"] = options.head_size_qk;
    state.counters["head_size_vo"] = options.head_size_vo;
    state.counters["page_size"] = options.page_size;
    state.counters["scale"] = options.softmax_scale;
    state.counters["causal"] = Causal;
    state.counters["varlen"] = isVarLen;
    state.counters["paged_kv"] = PagedKV;

    std::stringstream extra_label;
    extra_label << "layoutQ=RowMajor ";
    extra_label << "layoutK=ColumnMajor ";
    extra_label << "layoutV=RowMajor ";

    state.SetLabel(extra_label.str());

    double flops_qk, flops_pv, gbps_qk, gbps_pv;
    FlashDecodeRunner<FMHADecodeConfiguration>::calculate_flops_gbps(options, flops_qk, flops_pv, gbps_qk, gbps_pv);

    double gflops = (flops_qk + flops_pv) * 1e-9;
    double mega_bytes_transferred = (gbps_qk + gbps_pv) * (1e-6);

    initialize_counters(state);
    int32_t counter = 1;
    for(auto _ : state) {
      state.PauseTiming();
      int input_num = std::max(int(0), counter % count);

      typename FMHADecodeKernel::Arguments arguments = FlashDecodeRunner<FMHADecodeConfiguration>::get_arguments(problem_size, hw_info, options.softmax_scale, input_num);

      size_t workspace_size = FMHADecodeKernel::get_workspace_size(arguments);
      cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

      FMHADecodeKernel::can_implement(arguments);

      // Initialize the workspace
      auto status = FMHADecodeKernel::initialize_workspace(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        return;
      }

      typename FMHADecodeKernel::Params params = FMHADecodeKernel::to_underlying_arguments(arguments, workspace.get());

      state.ResumeTiming();

      GPU_Clock timer;
      timer.start();
      FlashDecodeRunner<FMHADecodeConfiguration>::run(params);
      auto ms_elapsed = timer.milliseconds();
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
      counter++;
    }
    finalize_counters(state, gflops, mega_bytes_transferred);
  }

private:
  static void initialize_counters(::benchmark::State& state) {
    state.counters["avg_runtime_ms"] = 0;
    state.counters["best_runtime_ms"] = std::numeric_limits<double>::max();
  }

  static void update_counters(::benchmark::State& state, double ms_elapsed) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms_elapsed;
    state.counters["best_runtime_ms"] = std::min<double>(state.counters["best_runtime_ms"], ms_elapsed);
    state.ResumeTiming();
  }

  static void finalize_counters(::benchmark::State& state,  double gflop, double mega_bytes_transferred) {
    state.counters["avg_runtime_ms"] =
      state.counters["total_runtime_ms"] / static_cast<double>(state.iterations());
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["avg_throughput"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }
};

}

#define CUTLASS_FMHA_DECODE_BENCHMARK(F) cutlass::benchmark::BenchmarkRegistry<cutlass::flash_attention::FMHADecodeOptions>::Register(#F, &F##_func)

#define CUTLASS_CREATE_FMHA_DECODE_BENCHMARK(F)                          \
  static void F##_func(                                           \
      ::benchmark::State& state,                                  \
      cutlass::flash_attention::FMHADecodeOptions const& options,                 \
      cutlass::KernelHardwareInfo const& hw_info) {               \
    auto bench = cutlass::benchmark::BenchmarkRunnerFMHADecode<F>();    \
    bench.run(state, options, hw_info);                           \
  }
