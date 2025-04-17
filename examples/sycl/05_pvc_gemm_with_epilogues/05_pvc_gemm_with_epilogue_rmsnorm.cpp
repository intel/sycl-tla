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
#include "cutlass/util/reference/device/tensor_relu.h"
#include "cutlass/tensor_view.h"
#include "cutlass/coord.h"

#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
inline
bool isclose(T a, T b, float atol, float rtol) {
    return std::abs((float)a - (float)b) <= atol + rtol * std::abs((float)b);
}

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), l(1), iterations(100),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 512);
    cmd.get_cmd_line_argument("n", n, 512);
    cmd.get_cmd_line_argument("k", k, 64);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "PVC GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<s32>               Epilogue scalar alpha\n"
      << "  --beta=<s32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Iterations\n\n";

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
  using LayoutW = cutlass::layout::RowMajor;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;
  using ElementW = typename Gemm::ElementA;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 10;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;
  cutlass::DeviceAllocation<ElementW> block_W;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));
    cutlass::TensorRef ref_W(block_W.get(), LayoutW::packed({1, N}));
    // printf("ref_D:");
    // for (int i = 0; i < 5; ++i) {
    //     printf("%f ", block_A.get()[i]);
    // }
    // printf("\nstride:");
    // auto stride = ref_D.stride();
    // // auto layout = ref_A.layout();
    // for (int i = 0; i < 3; ++i) {
    //     printf("%d ", ref_D.stride(i));
    // }
    // printf("\n");
    // printf("\nshape");
    // for (int i = 0; i < 2; ++i) {
    //     printf("%d ", ref_A.layout(i));
    // }
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

    syclcompat::wait();

    ElementOutput *ptr_refD =
        (ElementOutput *)std::malloc(M * N * L * sizeof(ElementOutput));
    syclcompat::memcpy(ptr_refD, block_ref_D.get(),
                       M * N * L * sizeof(ElementOutput));
    syclcompat::wait();
    ElementW *ptr_wgt =
        (ElementW *)std::malloc(N * L * sizeof(ElementW));
    syclcompat::memcpy(ptr_wgt, block_W.get(),
                       N * L * sizeof(ElementW));
    syclcompat::wait();
    // printf("ptr_ref:\n");

    // for (int m = 0; m < M; ++m) {
    //     for (int nn = 0; nn < N / 16; ++nn) {
    //         printf("%04d:(%03d) ", m, nn * 16);
    //         for (int n = 0; n < 16; ++n) {
    //             printf("%5.1f ", ptr_refD[m * N + nn * 16 + n]);
    //         }
    //         printf("\n");
    //     }
    // }

    constexpr float eps = 1e-5;
    float p2[M * N];
    float p2sum[M];
    for (int l = 0; l < L; l++) {
        for (int m = 0; m < M; m++) {
            float pow2_sum = (float)0;
            for (int n = 0; n < N; n++) {
                p2[m * N + n] = pow(ptr_refD[l * M * N + m * N + n], 2);
                pow2_sum += p2[m * N + n];
            }
            p2sum[m] = pow2_sum;
            float rms = 1.0f / sqrt(pow2_sum / (float)N + eps);

            for (int n = 0; n < N; n++) {
                ptr_refD[l * M * N + m * N + n] = ptr_refD[l * M * N + m * N + n] * rms * (float)ptr_wgt[n];
            }
        }
    }
    // printf("p2:\n");
    // for (int m = 0; m < M; ++m) {
    //     for (int nn = 0; nn < N / 16; ++nn) {
    //         printf("%4d:(%2d) ", m, nn * 16);
    //         for (int n = 0; n < 16; ++n) {
    //             printf("%5.1f ", p2[m * N + nn * 16 + n]);
    //         }
    //         printf("\n");
    //     }
    // }
    // printf("ptr_wgt:\n");
    // for (int nn = 0; nn < 4; ++nn) {
    //     printf("%d: ", nn * 16);
    //     for (int n = 0; n < 16; ++n) {
    //         printf("%5.1f ", (float)ptr_wgt[nn * 16 + n]);
    //     }
    //     printf("\n");
    // }
    // printf("p2sum:\n");
    // for (int mm = 0; mm < M / 16; ++mm) {
    //     for (int m = 0; m < 16; ++m) {
    //         printf("%5.1f ", p2sum[mm * 16 + m]);
    //     }
    //     printf("\n");
    // }

    // printf("\n");
    // syclcompat::memcpy(block_ref_D.get(), ptr,
    //                    M * N * L * sizeof(ElementOutput));
    // syclcompat::wait();

    ElementOutput *ptr_D =
        (ElementOutput *)std::malloc((size_t)M * N * L * sizeof(ElementOutput));
    syclcompat::memcpy(ptr_D, block_D.get(),
                       (size_t)M * N * L * sizeof(ElementOutput));
    syclcompat::wait();

    uint32_t err_cnt = 0;

    float atol = 1e-3;
    float rtol = 1e-3;
    for (int b = 0; b < L; b++) {
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
          int idx = b * M * N + m * N + n;
          auto expect = ptr_refD[idx];
          auto val = ptr_D[idx];

          if (not (std::isinf(ptr_D[idx]) || std::isnan(ptr_D[idx]))) {
            if (not isclose(val, expect, atol, rtol)) {
                printf("(%d,%d,%d): host: %f and device: %f ratio: %f\n", b, n, m, expect, val, val / expect);
              err_cnt++;
            } //  else{
            //     printf("(%d,%d,%d): host: %f and device: %f\n", b, i, m, expect, val);
            // }
          } else {
              printf("(%d,%d,%d): host: %f and device: %f\n", b, n, m, expect, val);
            err_cnt++;
          }
        }
      }
    }

    std::free(ptr_refD);
    std::free(ptr_D);
    std::cout << "err count: " << err_cnt
              << ", pass rate: " << 100 - (100 * err_cnt / (M * N * L)) << "%"
              << std::endl;
    return err_cnt == 0;
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
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);
    block_W.reset(N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
    initialize_block(block_W, seed + 2020);
    // auto a_ptr = block_A.get();
    // for (size_t m = 0; m < M; ++m) {
    //     for (size_t k = 0; k < K; ++k) {
    //         a_ptr[m * K + k] = (bfloat16_t)(float)(m * 1000 + k);
    //     }
    // }
    // auto b_ptr = block_B.get();
    // for (size_t k = 0; k < K; ++k) {
    //     for (size_t n = 0; n < N; ++n) {
    //         if (k == n)
    //             b_ptr[k * N + n] = (bfloat16_t)1.0f;
    //         else
    //             b_ptr[k * N + n] = (bfloat16_t)0.0f;
    //     }
    // }
    // printf("initialize done\n");
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);
    using EpilogueArguments = typename Gemm::GemmKernel::EpilogueArguments;
    EpilogueArguments epilogue_arguments{
      {options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D};
    epilogue_arguments.thread.output_ptr = block_D.get();
    epilogue_arguments.thread.weight_ptr = block_W.get();

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      epilogue_arguments,
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    gemm_op.can_implement(arguments);

    gemm_op.initialize(arguments, workspace.get());

    // Run the GEMM
    gemm_op.run();

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed) return cutlass::Status::kErrorInternal;

    if (passed && options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      syclcompat::wait();
    double io =
        options.l *
        (options.m * options.k * sizeof(ElementA) + options.k * options.n * sizeof(ElementB) +
         options.m * options.n * sizeof(ElementOutput)) *
        1e-9;
      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]GB/s,   [%4.3f]TF/s, [%6.4f]ms\n", io / cute_time, tflops/cute_time,  cute_time*1000);
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
  using ElementWeight = bfloat16_t; // <- data type for elements in rmsnorm weight
  using ElementOutput = float;                        // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x8x16_LD_N;
  using GmemTiledCopyB = XE_2D_U16x16x16_LD_V;

  // Workgroup-level tile
  using TileShape = Shape<_32, _512, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_2, _16, _1>, Stride<_16, _1, _0>>>::TiledMMA;

  using EpilogueTile = Shape<_16, _32>;
  constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;

  using EpilogueOp = cutlass::epilogue::fusion::LinCombRMSNormRow<ElementWeight, ElementOutput,
          ElementComputeEpilogue, XE_2D_U32x8x16_ST_N, ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          EpilogueTile>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          void, void,
          void,
          void, void>;

// Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputA,
          cutlass::gemm::TagToStrideA_t<LayoutA>,
          ElementInputB,
          cutlass::gemm::TagToStrideB_t<LayoutB>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,  // A
          GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;

  CUTLASS_CHECK(runner.run(options, hw_info));

  return 0;
}
