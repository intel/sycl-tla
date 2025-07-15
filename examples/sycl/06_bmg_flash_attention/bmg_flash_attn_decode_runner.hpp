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

#include "flash_attention_v2/flash_decode_runner.hpp"
#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;
using namespace cutlass::flash_attention;

// Command line options parsing
struct Options : public FMHADecodeOptions {

  bool is_causal;
  bool varlen;
  bool use_paged_kv;

  int head_size_vo;
  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("is_causal")) {
      is_causal = true;
    }

    if (cmd.check_cmd_line_flag("varlen")) {
      varlen = true;
    }

    if (cmd.check_cmd_line_flag("use_paged_kv")) {
      use_paged_kv = true;
    }

    FMHADecodeOptions::parse(cmd, argc, args);
    head_size_vo = FMHADecodeOptions::head_size_vo;
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {
    return FMHADecodeOptions::print_usage(out);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class FMHADecodeConfiguration>
struct ExampleRunner : public FlashDecodeRunner<FMHADecodeConfiguration> {

  using FMHADecodeKernel = typename FMHADecodeConfiguration::FMHADecodeKernel;
  using ProblemShapeType = typename FMHADecodeKernel::ProblemShape;
  static constexpr bool Causal = FMHADecodeConfiguration::Causal;
  static constexpr bool VarLen = FMHADecodeConfiguration::VarLen;
  static constexpr bool PagedKV = FMHADecodeConfiguration::PagedKV;
  //
  // Methods
  //

  bool verify(ProblemShapeType problem_size) {
    return FlashDecodeRunner<FMHADecodeConfiguration>::verify(problem_size);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShapeType initialize(const FMHADecodeOptions &options) {
    auto [problem_shape, mem_count] = FlashDecodeRunner<FMHADecodeConfiguration>::initialize(options);
    return problem_shape;
  }

  cutlass::Status run(const FMHADecodeOptions &options, const cutlass::KernelHardwareInfo &hw_info) {

    ProblemShapeType problem_size = initialize(options);

    typename FMHADecodeKernel::Arguments arguments = FlashDecodeRunner<FMHADecodeConfiguration>::get_arguments(problem_size, hw_info, options.softmax_scale, 0);

    size_t workspace_size = FMHADecodeKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHADecodeKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << options.batch << 'x' << options.num_heads_q << 'x' <<
        options.seq_len_qo << 'x' << options.seq_len_kv << 'x' << options.head_size_qk << 'x'  << options.head_size_vo 
        << (Causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHADecodeKernel::initialize_workspace(arguments, workspace.get()));

    auto params = FMHADecodeKernel::to_underlying_arguments(arguments, workspace.get());

    // Run Flash Attention Decode
    FlashDecodeRunner<FMHADecodeConfiguration>::run(params);

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if (!passed) {
      return cutlass::Status::kErrorInternal;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        FlashDecodeRunner<FMHADecodeConfiguration>::run(params);
      }
      syclcompat::wait();

      double cute_time = timer.seconds() / options.iterations;

      double flops_qk, flops_pv, gbps_qk, gbps_pv;
      FlashDecodeRunner<FMHADecodeConfiguration>::calculate_flops_gbps(options, flops_qk, flops_pv, gbps_qk, gbps_pv);
      double tflops = ((flops_qk + flops_pv) * 1e-12) / cute_time;
      double gbps = ((gbps_qk + gbps_pv) * 1e-9) / (cute_time);
      std::cout << "Batch: " << options.batch << "\tNumHeads_q: " << options.num_heads_q << "\tNumHeads_kv: " << options.num_heads_kv << "\tSeq Length QO: " << options.seq_len_qo
                << "\tSeq Length KV: " << options.seq_len_kv << "\tSeq Length KV Cache: " << options.seq_len_kv_cache << "\tHead Size QK: " << options.head_size_qk
                << "\tHead Size VO: " << options.head_size_vo << "\tCausal Mask: " << (Causal ? "true" : "false")
                << "\tVariable Sequence Length: " << (VarLen ? "true" : "false") << "\t Scheduler: Individual"
                << "\t Paged KV cache: " << (PagedKV ? "true" : "false");
      printf("\nPerformance:   %4.3f  GB/s,    %4.3f  TFlop/s,   %6.4f  ms\n\n", gbps, tflops, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};

template <bool Causal, 
          bool PagedKV,
          typename TileShapeQK, 
          typename TileShapePV, 
          typename TileShapeOutput, 
          typename TileSubgroupLayout, 
          bool isVarLen,
          typename ElementInputQ = bfloat16_t, 
          typename ElementInputKV = bfloat16_t,
          typename ElementAccumulator = float,
          typename ElementComputeEpilogue = float,
          typename ElementOutput = float> struct FMHAConfig {

  struct TileShapeConfig {
    using ShapeQK = TileShapeQK;
    using ShapePV = TileShapePV;
    using ShapeOutput = TileShapeOutput;
    using SubgroupLayout = TileSubgroupLayout;
  };

  static int run(const FMHADecodeOptions &options) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;

    using FlashDecodeConfig = typename FMHADecodeConfigGen<ElementInputQ, ElementAccumulator, ElementOutput, Causal, isVarLen, TileShapeConfig, PagedKV>::type;
    ExampleRunner<FlashDecodeConfig> runner;

    CUTLASS_CHECK(runner.run(options, hw_info));
    return 0;
  }
};
