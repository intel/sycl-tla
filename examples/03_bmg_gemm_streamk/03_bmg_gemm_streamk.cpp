/***************************************************************************************************
 * Copyright (C) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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
    \brief CUTLASS Intel BMG Gemm with 2 Tile Hybrid Stream-K Scheduling (https://arxiv.org/pdf/2301.03598)

    This example implements a Stream-K scheduled GEMM on Intel BMG hardware. Stream-K avoids tail
    effects on performance where conventional tiling wouldn't evenly divide the work across hardware.
    Whereas the conventional GEMM implementation requires that each work-group handle a 'whole' tile
    (i.e. iterate across the entire range of K), Stream-K permits the scheduler to split tiles along
    the K dimension between work-groups. This requires inter work-group communication to combine
    partial tile results. The hybrid variant of the StreamK scheduling combines the best of vanilla
    GEMM and basic StreamK to maximize performance.

    This example demonstrates 3 scheduling modes (DecompositionModes):
    - DataParallel - conventional GEMM
    - Split-K  - split all the output tiles into fixed size chunks along the K dimension
    - 2 Tile Hybrid Stream-K - apply basic StreamK scheduling to (sk_tiles = num_wgs + total_tiles % num_wgs)
      and split them along the K dimension across multiple workgroups if needed. Apply Data Parallel scheduling
      to the remaining tiles (total_tiles - sk_tiles).

    Verification for this example is a conventional GEMM, since Split/Stream-K is just a performance optimization of GEMM.

    To build & run this example (from your build dir):

      $ ninja 03_bmg_gemm_streamk
      $ ./examples/sycl/03_bmg_gemm_streamk/03_bmg_gemm_streamk

    Call with `--help` for information about available options.

    Note: the code may spill registers once compiled which will result in sub-optimal performance. This is because
    of an issue inside Intel Graphics Compiler (IGC) related to VectorAliasBBThreshold being debugged internally. 
    To avoid register spills, build the example by setting the environment variable:
      $ export IGC_VectorAliasBBThreshold=10000
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"

#include "cutlass/gemm/kernel/xe_persistent_tile_scheduler_params_streamk.hpp"
#include <string>
#include <type_traits>
using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool splitk;
  bool dp;
  bool time_run_only;
  std::string dtype;

  int m, n, k, l, iterations, warmup_iterations, splits, verify;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    splitk(false),
    dp(false),
    time_run_only(false),
    dtype("bf16"),
    m(5120), n(4096), k(4096), l(1), iterations(20), warmup_iterations(0), verify(1), splits(1),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("splitk")) {
      splitk = true;
    }

    if (cmd.check_cmd_line_flag("dp")) {
      dp = true;
    }

    if (cmd.check_cmd_line_flag("time_run_only")) {
      time_run_only = true;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, 0);
    cmd.get_cmd_line_argument("splits", splits, 1);
    cmd.get_cmd_line_argument("verify", verify, 1);
    cmd.get_cmd_line_argument("dtype", dtype, std::string("bf16"));

    if (dtype != "bf16" && dtype != "f16" &&
        dtype != "bf16_bf16" && dtype != "f16_f16" &&
        dtype != "tf32" && dtype != "s8") {
      error = true;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "BMG GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --dp                        If specified, uses Data Parallel decomposition\n"
      << "  --splitk                    If specified, uses SplitK decomposition\n"
      << "  --time_run_only             If specified, excludes per-iteration initialize from timing\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --splits=<int>              Sets the splitting factor for GEMM\n"
      << "  --alpha=<s32>               Epilogue scalar alpha\n"
      << "  --beta=<s32>                Epilogue scalar beta\n\n"
      << "  --dtype=<bf16|f16|bf16_bf16|f16_f16|tf32|s8>  Dtype preset\n"
      << "  --warmup_iterations=<int> Warmup iterations before measurement\n"
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Gemm
>
struct ExampleRunner {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int32_t count;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha,
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          beta,
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = false;
    if constexpr (std::is_same_v<ElementOutput, cutlass::half_t> ||
                  std::is_same_v<ElementOutput, cutlass::bfloat16_t>) {
      passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), block_D.size(), ElementOutput(0.5f), ElementOutput(1.0f));
    } else {
      passed = cutlass::reference::device::BlockCompareEqual(
        block_ref_D.get(), block_D.get(), block_D.size());
    }

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{ElementCompute(options.alpha), ElementCompute(options.beta)}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info,
      {options.splits, // Setting splits > 1 will force SplitK decomposition
      // Set the decomposition mode based on user provided options
      options.dp ? cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode::DataParallel :
      options.splitk ? cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode::SplitK :
                          cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode::StreamK}
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();

    if (options.verify != 0) {
      // Verify that the result is correct
      bool passed = verify(problem_size, ElementCompute(options.alpha), ElementCompute(options.beta));
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      for (int i = 0; i < options.warmup_iterations; ++i) {
        gemm_op.initialize(arguments, workspace.get());
        gemm_op.run();
      }
      compat::wait();

      float elapsed_time_seconds = 0.f;
      for (int i = 0; i < options.iterations; ++i) {
        GPU_Clock timer;
        if (!options.time_run_only) {
          gemm_op.initialize(arguments, workspace.get());
        }
        timer.start();
        gemm_op.run();
        compat::wait();
        elapsed_time_seconds += timer.seconds();
      }

      float cute_time = elapsed_time_seconds / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};

template <class ElementInputA, class ElementInputB, class ElementAccumulator, class ElementOutput>
int run_gemm(Options const& options, cutlass::KernelHardwareInfo const& hw_info) {
  using ElementComputeEpilogue = ElementAccumulator;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  using TileShape = Shape<_256, _256, _32>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementInputA, ElementInputB, ElementAccumulator>>, Layout<TileShape>,
                                     Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages, cutlass::gemm::KernelXeCooperative>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallBacks,
          void,
          void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputA,
          cutlass::gemm::TagToStrideA_t<LayoutA>,
          ElementInputB,
          cutlass::gemm::TagToStrideB_t<LayoutB>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,
          GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler
    >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  ExampleRunner<Gemm> runner;
  CUTLASS_CHECK(runner.run(options, hw_info));
  return 0;
}

int main(int argc, const char** argv)
{
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution. Supported --dtype values: bf16, f16, bf16_bf16, f16_f16, tf32, s8." << std::endl;
    return -1;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  if (options.dtype == "f16") {
    return run_gemm<half_t, half_t, float, float>(options, hw_info);
  }
  if (options.dtype == "bf16_bf16") {
    std::cerr << "Aborting execution. --dtype=bf16_bf16 requires BF16 atomic add in StreamK reduction, which is not supported by SYCL atomic_ref." << std::endl;
    return -1;
  }
  if (options.dtype == "f16_f16") {
    return run_gemm<half_t, half_t, half_t, half_t>(options, hw_info);
  }
  if (options.dtype == "tf32") {
    return run_gemm<tfloat32_t, tfloat32_t, float, float>(options, hw_info);
  }
  if (options.dtype == "s8") {
    return run_gemm<int8_t, int8_t, int32_t, int32_t>(options, hw_info);
  }
  return run_gemm<bfloat16_t, bfloat16_t, float, float>(options, hw_info);
}
