/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "dual_gemm/collective/xe_dual_gemm_epilogue.hpp"
#include "dual_gemm/kernel/xe_dual_gemm.hpp"
#include "dual_gemm/collective/xe_dual_gemm_epilogue_elementwise_activation.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "../common.hpp"
#include "helper.h"
#include "tensor_silu.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations;
  float alpha0, alpha1, beta0, beta1;

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), l(1), iterations(20),
    alpha0(1.f), beta0(0.f), alpha1(1.f), beta1(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha0", alpha0, 1.f);
    cmd.get_cmd_line_argument("beta0", beta0, 0.f);
    cmd.get_cmd_line_argument("alpha1", alpha1, 1.f);
    cmd.get_cmd_line_argument("beta1", beta1, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "PVC Dual GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha0=<s32>               Dual GEMM Epilogue scalar alpha0\n"
      << "  --alpha1=<s32>               Dual GEMM Epilogue scalar alpha1\n"
      << "  --beta0=<s32>                Dual GEMM Epilogue scalar beta0\n\n"
      << "  --beta1=<s32>                Dual GEMM Epilogue scalar beta1\n\n"
      << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

template <
  class GemmKernel
>
struct ExampleRunner {

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementAcc = typename GemmKernel::ElementAccumulator;

  using CollectiveEpilogue0 = typename GemmKernel::CollectiveEpilogue0;
  using ElementC = typename GemmKernel::ElementC;
  using ElementOutput = typename CollectiveEpilogue0::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue0::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue0::ElementAccumulator;

  using ProblemShapeType = typename GemmKernel::ProblemShape;

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
  cutlass::DeviceAllocation<ElementB> block_B0;
  cutlass::DeviceAllocation<ElementB> block_B1;
  cutlass::DeviceAllocation<ElementC> block_C0;
  cutlass::DeviceAllocation<ElementC> block_C1;
  cutlass::DeviceAllocation<ElementOutput> block_D0;
  cutlass::DeviceAllocation<ElementOutput> block_D1;
  cutlass::DeviceAllocation<ElementOutput> block_D2;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D0;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D1;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D2;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha0, ElementCompute alpha1, ElementCompute beta0, ElementCompute beta1) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B0(block_B0.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_B1(block_B1.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C0(block_C0.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_C1(block_C1.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D0(block_ref_D0.get(), LayoutD::packed({M, N}));
    cutlass::TensorRef ref_D1(block_ref_D1.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha0,
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B0,
          cutlass::ComplexTransform::kNone,
          beta0,
          ref_C0,
          ref_D0,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha1,
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B1,
          cutlass::ComplexTransform::kNone,
          beta1,
          ref_C1,
          ref_D1,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    syclcompat::wait();

    using TensorView = cutlass::TensorView<ElementOutput, LayoutD>;
    cutlass::reference::device::TensorSiLu(TensorView(block_ref_D2.get(), LayoutD::packed({M, N}), cutlass::make_Coord(M, N)),
                                          TensorView(block_ref_D0.get(), LayoutD::packed({M, N}), cutlass::make_Coord(M, N)),
                                          TensorView(block_ref_D1.get(), LayoutD::packed({M, N}), cutlass::make_Coord(M, N)));

    syclcompat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed_D0 = cutlass::reference::device::BlockCompareEqual(
      block_ref_D0.get(), block_D0.get(), block_D0.size());

    bool passed_D1 = cutlass::reference::device::BlockCompareEqual(
      block_ref_D1.get(), block_D1.get(), block_D1.size());

    bool passed_D2 = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D2.get(), block_D2.get(), block_D2.size(), 0.5f, 0.5f);

    return passed_D0 && passed_D1 && passed_D2;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(M * K * L);
    block_B0.reset(K * N * L);
    block_B1.reset(K * N * L);
    block_C0.reset(M * N * L);
    block_C1.reset(M * N * L);
    block_D0.reset(M * N * L);
    block_D1.reset(M * N * L);
    block_D2.reset(M * N * L);
    block_ref_D0.reset(M * N * L);
    block_ref_D1.reset(M * N * L);
    block_ref_D2.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B0, seed + 2022);
    initialize_block(block_B1, seed + 2021);
    initialize_block(block_C0, seed + 2020);
    initialize_block(block_C1, seed + 2019);
  }

  static cutlass::Status run(typename GemmKernel::Params params) {
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = GemmKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

    using namespace syclcompat::experimental;
    auto event = launch<cutlass::device_kernel<GemmKernel>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<GemmKernel::DispatchPolicy::SubgroupSize>}},
        params);

    EventManager::getInstance().addEvent(event);
    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B0.get(), stride_B, block_B1.get(), stride_B},
      {{options.alpha0, options.beta0}, block_C0.get(), stride_C, block_D0.get(), stride_D},
      {{options.alpha1, options.beta1}, block_C1.get(), stride_C, block_D1.get(), stride_D},
      {nullptr, stride_C, block_D2.get(), stride_D},
      hw_info
    };

    size_t workspace_size = GemmKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!GemmKernel::can_implement(arguments)){
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(GemmKernel::initialize_workspace(arguments, workspace.get()));

    typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, workspace.get());

    // Run the GEMM
    CUTLASS_CHECK(run(params));

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha0, options.alpha1, options.beta0, options.beta1);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if(!passed) return cutlass::Status::kErrorInternal;

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        run(params);
      }
      syclcompat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = 2 * (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass Dual GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};

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
    std::cerr << "Aborting execution." << std::endl;
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

  bool passed;

  // The code section below describes datatype for input, output matrices and computation between
  // elements in input matrices.
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA = bfloat16_t;                        // <- data type of elements in input matrix A
  using ElementInputB = bfloat16_t;                        // <- data type of elements in input matrix B
  using ElementOutput = float;                        // <- data type of elements in output matrix D

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  // Workgroup-level tile
  using TileShape = Shape<_256, _128, _32>;

  // The Tile of this layout describes how 8x4x1 sub-groups tile the TileShape of <256, 128, 32>. 
  // This permutation (which can be thought of as a scatter operation on the default tiling) 
  // ensures that each sub-group operates on a contiguous 32x32x32 chunk (4x2x2 iterations)
  // See 0t_mma_atom.md#TiledMMAs for more info.
  using TiledMma =
      TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
               Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                    Layout<Shape<_16, _4, _2>, Stride<_1, _32, _16>>, _32>>;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;

  using LinearEpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using LinearFusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, LinearEpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::DualGemmEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          LinearFusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          XE_2D_U32x8x16_ST_N>;

  using CollectiveEpilogueActivation = cutlass::epilogue::collective::DualGemmElemActEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          void,
          XE_2D_U32x8x16_ST_N>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::DualGemmMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputA,
          cutlass::gemm::TagToStrideA_t<LayoutA>,
          ElementInputB,
          cutlass::gemm::TagToStrideB_t<LayoutB>,
          TiledMma,
          GmemTiledCopyA, // A
          GmemTiledCopyB // B
  >;

  using GemmKernel = cutlass::gemm::kernel::DualGemm<
  Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue,
  CollectiveEpilogue,
  CollectiveEpilogueActivation
  >;

  ExampleRunner<GemmKernel> runner;

  CUTLASS_CHECK(runner.run(options, hw_info));

  return 0;
}
