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
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/epilogue/collective/intel_pvc_epilogue_tensor_softmax.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

template <typename T> static void fill_matrix(std::vector<T>& M) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<float> dist((T)0.0,
#ifdef EPILOGUE_SOFTMAX
      (T)0.1);
#else
      (T)1.0);
#endif
  std::generate(std::begin(M), std::end(M), [&] { return static_cast<T>(dist(rng)); });
}

template <typename T>
static void vnni_matrix(T* dst, T const* src, int batch, int numRows, int numCols, int factor) {
  for (int b = 0; b < batch; b++) {
    for (int r = 0; r < numRows / factor; r++) {
      for (int c = 0; c < numCols; c++) {
        for (int k = 0; k < factor; k++) {
          dst[((b * (numRows / factor) + r) * numCols + c) * factor + k] =
              src[((b * (numRows / factor) + r) * factor + k) * numCols + c];
        }
      }
    }
  }
}

using namespace cute;

using ElementAccumulator = float;     // <- data type of accumulator
using ElementComputeEpilogue = float; // <- data type of epilogue operations
using ElementInputA = bfloat16_t;     // <- data type of elements in input matrix A
using ElementInputB = bfloat16_t;     // <- data type of elements in input matrix B
using ElementOutput = float;          // <- data type of elements in output matrix D

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations;
  float alpha, beta;

  Options()
      : help(false), error(false), m(4096), n(4096), k(4096), l(1), iterations(100), alpha(1.f),
        beta(0.f) {}

  // Parses the command line
  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }

  /// Prints the usage statement.
  std::ostream& print_usage(std::ostream& out) const {

    out << "PVC GEMM Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this "
           "usage statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   Sets the L extent (batch count) "
           "of the GEMM\n"
        << "  --alpha=<s32>               Epilogue scalar alpha\n"
        << "  --beta=<s32>                Epilogue scalar beta\n\n"
        << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm, bool cache_clear> struct ExampleRunner {

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

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  // cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  static auto constexpr l3_cache_size = 256 * 1024 * 1024;

  size_t PINGPONG_ITER = 1;
  size_t pingpong_size_a;
  size_t pingpong_size_b;
  size_t pingpong_size_d;

  std::vector<ElementA> a;
  std::vector<ElementB> b;
  std::vector<ElementC> d;
  //
  // Methods
  //

  bool verify(ProblemShapeType const& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C((ElementC*)nullptr /*block_C.get()*/, LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
        {M, N, K}, alpha, ref_A, cutlass::ComplexTransform::kNone, ref_B,
        cutlass::ComplexTransform::kNone, beta, ref_C, ref_D, ElementAccumulator(0),
        L,     // batch_count
        M * K, // batch_stride_A
        K * N, // batch_stride_B
        M * N, // batch_stride_C
        M * N  // batch_stride_D
    );

#ifdef EPILOGUE_SOFTMAX

    ElementOutput* ptr = (ElementOutput*)std::malloc(M * N * L * sizeof(ElementOutput));
    syclcompat::memcpy(ptr, block_ref_D.get(), M * N * L * sizeof(ElementOutput));
    syclcompat::wait();
    for (int l = 0; l < L; l++) {
      for (int i = 0; i < M; i++) {
        auto row_idx = l * M * N + i * N;
        auto row_max = ptr[l * M * N + i * N];

        ElementOutput exp_sum = (ElementOutput)0;
        for (int j = 0; j < N; j++) {
          auto idx = row_idx + j;
          row_max = max(row_max, ptr[idx]);
        }
        for (int j = 0; j < N; j++) {
          auto idx = row_idx + j;
          ptr[idx] = ptr[idx] - row_max;
          ptr[idx] = exp(ptr[idx]);
          exp_sum += ptr[idx];
        }

        for (int j = 0; j < N; j++) {
          auto idx = row_idx + j;
          ptr[idx] = ptr[idx] / exp_sum;
        }
      }
    }

    syclcompat::memcpy(block_ref_D.get(), ptr, M * N * L * sizeof(ElementOutput));
    syclcompat::wait();

    std::free(ptr);

#endif

#if 0
    ElementOutput *ptr =
        (ElementOutput *)std::malloc(M * N * L * sizeof(ElementOutput));

    syclcompat::memcpy(ptr, block_D.get(), M * N * L * sizeof(ElementOutput));

    ElementOutput *ptr_refD =
        (ElementOutput *)std::malloc((size_t)M * N * L * sizeof(ElementOutput));
    syclcompat::memcpy(ptr_refD, block_ref_D.get(),
                       (size_t)M * N * L * sizeof(ElementOutput));
    syclcompat::wait();
    for (int b = 0; b < L; b++) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          int idx = b * M * N + i * N + j;
          if (abs(ptr[idx] - ptr_refD[idx]) / ptr_refD[idx] >= 0.01f)
            std::cout << "(" << b << ", " << i << ", " << j << "): " << "host: " << ptr[idx]
                        << "   and device: " << ptr_refD[idx] << std::endl;
        }
      }
    }
    std::free(ptr);
    std::free(ptr_refD);
#endif
    syclcompat::wait();

    // Check if output from CUTLASS kernel and reference kernel are relatively
    // equal or not need to set a larger error margin for comparison to succeed
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), M * N * L, 0.5f, 0.5f);

    return passed;
  }

  void init_cache_clear(ProblemShapeType const& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    pingpong_size_a = max((size_t)M * K * L, l3_cache_size / sizeof(ElementA));
    pingpong_size_b = max((size_t)K * N * L, l3_cache_size / sizeof(ElementB));
    pingpong_size_d = max((size_t)M * N * L, l3_cache_size / sizeof(ElementOutput));
    auto gmem_size = syclcompat::get_current_device().get_global_mem_size();
    PINGPONG_ITER = std::min((size_t)3,
        std::max((size_t)1, (size_t)gmem_size / ((pingpong_size_a * sizeof(ElementA) +
                                                    pingpong_size_b * sizeof(ElementB) +
                                                    pingpong_size_d * sizeof(ElementOutput))) -
                                1));
    block_A.reset(pingpong_size_a * PINGPONG_ITER);
    block_B.reset(pingpong_size_b * PINGPONG_ITER);
    // block_C.reset(M * N * L * ITER);
    block_D.reset(pingpong_size_d * PINGPONG_ITER);

    for (int i = 0; i < PINGPONG_ITER; i++) {
      syclcompat::memcpy(
          block_A.get() + i * pingpong_size_a, a.data(), a.size() * sizeof(ElementA));
      syclcompat::memcpy(
          block_B.get() + i * pingpong_size_b, b.data(), b.size() * sizeof(ElementB));
      syclcompat::memcpy(
          block_D.get() + i * pingpong_size_d, d.data(), d.size() * sizeof(ElementC));
    }
    // syclcompat::wait();
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(ProblemShapeType const& problem_size) {
    auto [M, N, K, L] = problem_size;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(K, N, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
    block_A.reset((size_t)M * K * L);
    block_B.reset((size_t)K * N * L);
    // block_C.reset(M * N * L);
    block_D.reset((size_t)M * N * L);
    block_ref_D.reset((size_t)max(l3_cache_size / sizeof(ElementOutput), (size_t)M * N * L));

    // TODO: Enable initialization on device directly once RNG is
    // available through SYCL.
    a = std::vector<ElementA>((size_t)M * K * L);
    b = std::vector<ElementB>((size_t)K * N * L);
    d = std::vector<ElementC>((size_t)M * N * L, ElementC{0});
    std::cout << "random generating..." << std::endl;
    fill_matrix(a);
    fill_matrix(b);
    syclcompat::memcpy(block_A.get(), a.data(), a.size() * sizeof(ElementA));
    syclcompat::memcpy(block_B.get(), b.data(), b.size() * sizeof(ElementB));
    // syclcompat::memcpy(block_C.get(), c.data(), c.size() * sizeof(ElementC));
    syclcompat::memcpy(block_D.get(), d.data(), d.size() * sizeof(ElementC));
  }

  template <int wg_tile_m, int wg_tile_n, int sg_tile_m, int sg_tile_n, int sg_tile_k>
  void run(int M, int K, int N, int L, cutlass::KernelHardwareInfo const& hw_info) {
    static auto constexpr warmup = 10;
    static auto constexpr testIterations = 10;
    static auto constexpr total_iterations = warmup + testIterations;
    ProblemShapeType problem_size = ProblemShapeType{M, N, K, L};
    initialize(problem_size);
    // ================ verfy the gemm result first ================
    typename Gemm::GemmKernel::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size, {block_A.get(), stride_A, block_B.get(), stride_B},
        {{1, 0.f}, nullptr /*block_C.get()*/, stride_C, block_D.get(), stride_D}, hw_info};
    Gemm gemm_op_verify;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    gemm_op_verify.can_implement(arguments);

    gemm_op_verify.initialize(arguments, workspace.get());

    // Run the GEMM
    gemm_op_verify.run();
    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size, 1, 0.f);
    if (!passed) {
      printf("PVC GEMM%s%s Example %s, MKNL(%d, %d,%d,%d), Config(%d, "
             "%d,%d,%d,%d)  !!!!!!!!!!!!!\n\n",
#ifdef EPILOGUE_RELU
          "-relu"
#else
          ""
#endif
          ,
#ifdef EPILOGUE_SOFTMAX
          "-softmax"
#else
          ""
#endif
          ,
          (passed ? "Passed" : "Failed"), M, K, N, L, wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n,
          sg_tile_k);
      // return;
    }

    // ================ init cache clear ================
    if constexpr (cache_clear) {
      init_cache_clear(problem_size);
    }

    // ================ run and collect performance data ================
    if (total_iterations > 0) {
      auto total_time = 0.f;
      auto best = 999.f;
      auto worst = 0.f;

      for (int i = 0; i < testIterations + warmup; ++i) {
        typename Gemm::GemmKernel::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            {block_A.get() + (i % PINGPONG_ITER) * pingpong_size_a, stride_A,
                block_B.get() + (i % PINGPONG_ITER) * pingpong_size_b, stride_B},
            {{1, 0.f}, nullptr /*block_C.get() + i * M * N * L*/, stride_C,
                block_D.get() + (i % PINGPONG_ITER) * pingpong_size_d, stride_D},
            hw_info};

        Gemm gemm_op;
        gemm_op.can_implement(arguments);
        gemm_op.initialize(arguments, workspace.get());

        GPU_Clock timer;
        timer.start();
        gemm_op.run();
        syclcompat::wait();

        auto current_time = timer.seconds();
        if (i >= warmup) {
          total_time += current_time;

          best = min(best, current_time);

          worst = max(worst, current_time);
        }
      }

      float average = total_time / testIterations;
      double tflops = (2.0 * M * N * K * L) * 1e-12;

      double hbm = L *
                   (M * K * sizeof(ElementInputA) + K * N * sizeof(ElementInputB) +
                       M * N * sizeof(ElementOutput)) *
                   1e-9;

      printf("Collective pvc gemm%s, MKNL(%d, %d, %d, %d), Config(%d, %d, "
             "%d, %d, %d):\n     max:     (%6.4f)ms, (%4.2f)TFlop/s, "
             "(%4.2f)GB/s\n     min:     (%6.4f)ms, (%4.2f)TFlop/s, "
             "(%4.2f)GB/s\n     average: (%6.4f)ms, (%4.2f)TFlop/s, "
             "(%4.2f)GB/s\n\n\n",
#if defined(EPILOGUE_RELU)
          "-relu"
#elif defined(EPILOGUE_SOFTMAX)
          "softmax"
#else
          ""
#endif
          ,
          M, K, N, L, wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k, best * 1000,
          tflops / best, hbm / best, worst * 1000, tflops / worst, hbm / worst, average * 1000,
          tflops / average, hbm / average);
    }
  }
};

template <int wg_tile_m,
    int wg_tile_n,
    int sg_tile_m,
    int sg_tile_n,
    int sg_tile_k,
    bool wg_order_m_first = false,
    uint32_t snake_n = 0,
    bool cache_clear = true>
void collective_gemm(int M, int K, int N, int L = 1) {
  //
  // Parse options
  //

  Options options;

  // options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x8x16x4x2_LD_N;
  using GmemTiledCopyB = XE_2D_U16x16x16x2x2_V;

  using TileShape =
      Shape<Int<wg_tile_m>, Int<wg_tile_n>, Int<sg_tile_m>, Int<sg_tile_n>, Int<sg_tile_k>>;

  using TiledMma = TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>, Layout<Shape<_8, _16, _1>>>;

  using DispatchPolicy = cutlass::gemm::MainloopIntelPVCUnpredicated;
#ifdef EPILOGUE_RELU
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombinationRelu<ElementOutput, // <- data type of output
                                                                      // matrix
          128 / cutlass::sizeof_bits<ElementOutput>::value,           // <- the number of
          // elements per vectorized
          // memory access. For a byte, it's 16
          // elements. This becomes the vector width of
          // math instructions in the epilogue too
          ElementAccumulator,      // <- data type of accumulator
          ElementComputeEpilogue>; // <- data type for alpha/beta in linear

#else
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<ElementOutput, // <- data type of output matrix
          128 / cutlass::sizeof_bits<ElementOutput>::value,       // <- the number of
          // elements per vectorized
          // memory access. For a byte, it's 16
          // elements. This becomes the vector width of
          // math instructions in the epilogue too
          ElementAccumulator,      // <- data type of accumulator
          ElementComputeEpilogue>; // <- data type for alpha/beta in linear
  // combination function
#endif
  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<DispatchPolicy, TileShape,
      ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>, ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB>, TiledMma, GmemTiledCopyA, void, void,
      cute::identity,                            // A
      GmemTiledCopyB, void, void, cute::identity // B
      >;

#ifdef EPILOGUE_SOFTMAX
  using CollectiveEpilogue = cutlass::epilogue::collective::PvcEpilogueTensorSoftmax<
      cutlass::gemm::TagToStrideC_t<LayoutC>, cutlass::gemm::TagToStrideC_t<LayoutD>, EpilogueOp,
      cutlass::gemm::EpilogueDefault, CollectiveMainloop::sg_tile_m,
      CollectiveMainloop::sg_tile_n / CollectiveMainloop::SubgroupSize>;
#else
  using CollectiveEpilogue =
      cutlass::epilogue::collective::DefaultEpilogue<cutlass::gemm::TagToStrideC_t<LayoutC>,
          cutlass::gemm::TagToStrideC_t<LayoutD>, EpilogueOp, cutlass::gemm::EpilogueDefault>;
#endif

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm, cache_clear> runner;

  runner.template run<wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k>(M, K, N, L, hw_info);
}

int main() {
  auto gmem_size = syclcompat::get_current_device().get_global_mem_size();
#if !defined(EPILOGUE_RELU) && !defined(EPILOGUE_SOFTMAX)
  collective_gemm<256, 256, 32, 64, 32>(4096, 4096, 4096);
  collective_gemm<256, 256, 32, 64, 32>(8192, 8192, 8192);
  collective_gemm<256, 256, 32, 64, 32>(1, 5120, 13824);
  collective_gemm<256, 256, 32, 64, 32>(1024, 28672, 8192);
  collective_gemm<256, 256, 32, 64, 32>(3072, 4096, 3072);
  collective_gemm<256, 256, 32, 64, 32>(4, 4096, 12288);

  // collective shape from habana
  collective_gemm<256, 256, 32, 64, 32>(512, 8192, 8192);
  collective_gemm<256, 256, 32, 64, 32>(512, 8192, 32768);
  collective_gemm<256, 256, 32, 64, 32>(512, 32768, 8192);
  collective_gemm<256, 256, 32, 64, 32>(16384, 8192, 1024);
  collective_gemm<256, 256, 32, 64, 32>(16384, 1024, 8192);
  collective_gemm<256, 256, 32, 64, 32>(16384, 8192, 4096);
  collective_gemm<256, 256, 32, 64, 32>(16384, 4096, 8192);
  collective_gemm<256, 256, 32, 64, 32>(4096, 16384, 8192);
  collective_gemm<256, 256, 32, 64, 32>(8192, 16384, 4096);
  collective_gemm<256, 256, 32, 64, 32>(1024, 16384, 8192);
  collective_gemm<256, 256, 32, 64, 32>(8192, 16384, 1024);

  collective_gemm<256, 256, 32, 64, 32>(8, 128, 16384, 4096);
  collective_gemm<16, 512, 16, 16, 32>(8, 16384, 128, 4096);

  collective_gemm<256, 256, 32, 64, 32>(32768, 128, 4096, 4);
  collective_gemm<256, 256, 32, 64, 32>(32768, 4096, 128, 4);
  collective_gemm<256, 256, 32, 64, 32>(4096, 4096, 128, 32);
#endif

#if defined(EPILOGUE_SOFTMAX)
  // gemm + softmax
  collective_gemm<64, 1024, 16, 64, 32>(1024, 64, 1024, 4);
  collective_gemm<128, 512, 16, 64, 32>(512, 64, 512, 32);
  collective_gemm<64, 1024, 16, 64, 32>(1024, 64, 1024, 16);
  collective_gemm<32, 2048, 16, 64, 16>(2048, 64, 2048, 8);
  collective_gemm<16, 4096, 16, 64, 32>(4096, 64, 4096, 4);
  collective_gemm<8, 8192, 8, 128, 16>(8192, 64, 8192, 2);
#endif

#if defined(EPILOGUE_RELU)
  // gemm + relu
  collective_gemm<256, 256, 32, 64, 32>(4096, 4096, 4096);
#endif
}
