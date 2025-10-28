/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG Fused MoE API example based on cutlass Group GEMM

    This example demonstrates fusing multiple GEMM operations into one kernel.

    Note that the scalar arguments to e.g. the standard 00_bmg_gemm example,
   have been replaced with vector equivalents, as each individual GEMM has its
   own inputs and outputs, which needn't be contiguous in memory. For example,
   where 00_bmg_gemm receives an `ElementA *` defining Matrix A, grouped gemm
   receives a `ElementA **`, i.e. a pointer to pointers, each pointing to a
   distinct Matrix A. Likewise, each individual GEMM operation may have its own
   alpha and beta factors for linear combination. This example demonstrates two
   approaches: the user can provide `options.alpha` and `options.beta`, in which
   case they will apply to all GEMMs; otherwise, random values are generated per
   GEMM.

    While a nullptr can be passed for C, in this example, we don't do that,
   because the reference kernel doesn't accept nullptr for C.

    Group GEMM scheduling (cutlass::gemm::GroupScheduler) is more complex than
   standard GEMM, because each GEMM may have a unique size, only known at
   runtime. Thus, the scheduler will distribute an a priori unknown number of
   tiles to each work-group. See
    include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp for
   implementation.

    Note that for simplicity, this example hard-codes input shapes.

    Verification for this example is a conventional GEMM kernel, executed
   iteratively per group.

    To build & run this example (from your build dir):

      $ ninja 11_bmg_fused_moe_bf16
      $ ./examples/sycl/11_bmg_fused_moe_bf16/11_bmg_fused_moe_bf16

    Note: the code may spill registers once compiled which will result in
   sub-optimal performance. This is because of an issue inside Intel Graphics
   Compiler (IGC) related to VectorAliasBBThreshold being debugged internally.
    To avoid register spills, build the example by setting the environment
   variable: $ export IGC_VectorAliasBBThreshold=10000
*/
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "helper.h"
#include "sycl_common.hpp"

#include <cfloat>

using namespace cute;
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>; // <M,N,K> per group

using ElementAccumulator = float;     // <- data type of accumulator
using ElementComputeEpilogue = float; // <- data type of epilogue operations
using ElementA = bfloat16_t;      // <- data type of elements in input matrix A
using ElementB = bfloat16_t;      // <- data type of elements in input matrix B
using ElementOutput = bfloat16_t; // <- data type of elements in output matrix D

///////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_SYCL_PROFILING_ENABLED

// Command line options parsing
struct GroupGEMMOptions {

  bool error = false;
  bool help = false;

  float alpha = 1.f;
  float beta = 0.f;
  int iterations;
  int m=0, n=0, k=0, groups;
  int *num_rows_per_expert = nullptr;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  GroupGEMMOptions()
      : error(false), help(false), alpha(1.f), beta(0.f), iterations(100) {
  }

  void parse(const int num_experts, const int *num_tokens_per_expert_host,
             int moe_n, int moe_k,
             const int *num_tokens_per_expert_device = nullptr) {
    n = moe_n;
    k = moe_k;
    groups = num_experts;
    iterations = 2;
    num_rows_per_expert = const_cast<int *>(num_tokens_per_expert_device);
    assert(groups > 0);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for (int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({num_tokens_per_expert_host[i], n, k});
    }
  }

  /// Compute performance in GFLOP/s
  std::tuple<double, double, double> gflops(double runtime_s,
                std::vector<typename ProblemShape::UnderlyingProblemShape>
                    problem_sizes_host) const {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();
    uint64_t bytes_loaded = 0;

    for (auto const &problem : problem_sizes_host) {
      auto M = static_cast<uint64_t>(get<0>(problem));
      auto N = static_cast<uint64_t>(get<1>(problem));
      auto K = static_cast<uint64_t>(get<2>(problem));
      fmas +=  M * N * K;
      bytes_loaded += /* sizeof(cutlass::bfloat16_t) */ 2 * (2 * M * N + N * K + M * K);
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    double arithmetic_intensity = double(flop) / double(bytes_loaded);
    double peak_mwm_bw = 456.0;
    double gflops_attainable = std::min<double>(117 * double(1.0e12), arithmetic_intensity * (peak_mwm_bw * 1024 * 1024 * 1024));
    double projected_time = flop/gflops_attainable;
    return std::make_tuple(gflop / runtime_s, double(bytes_loaded) / 1024 / 1024 / 1024 / runtime_s, projected_time * 1000);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Gemm> struct ExampleRunner {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  // Host-side allocations
  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  std::vector<ElementAccumulator> alpha_host;
  std::vector<ElementAccumulator> beta_host;

  // Device-side allocations
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
      problem_sizes;

  // This example defines all matrices in a single allocation (e.g. block_A),
  // but this is not a requirement. Matrix base pointers are read from device
  // allocation (e.g. ptr_A)
  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementOutput *> ptr_D;
  cutlass::DeviceAllocation<ElementOutput *> ptr_ref_D;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  // Note, this is an array of pointers to alpha and beta scaling values per
  // group
  cutlass::DeviceAllocation<ElementAccumulator *> alpha_device;
  cutlass::DeviceAllocation<ElementAccumulator *> beta_device;
  cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
  cutlass::DeviceAllocation<ElementAccumulator> block_beta;
  int* cumsum_host;
  cutlass::DeviceAllocation<int32_t> cumsum_device;

  uint64_t seed = 0;

  //
  // Methods
  //

  bool verify(const GroupGEMMOptions &options) {
    bool passed = true;
    // Verify against individual reference GEMMs
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i),
                               LayoutA::packed({M, K}));
      cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i),
                               LayoutB::packed({K, N}));
      cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i),
                               LayoutC::packed({M, N}));
      cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i),
                               LayoutD::packed({M, N}));

      //
      // Compute reference output
      //
      cutlass::reference::device::GemmComplex(
          {M, N, K}, alpha_host.at(i), ref_A, cutlass::ComplexTransform::kNone,
          ref_B, cutlass::ComplexTransform::kNone, beta_host.at(i), ref_C,
          ref_D, ElementAccumulator(0),
          1,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
      );

      // Wait for kernel to finish
      compat::wait();

      // Check if output from CUTLASS kernel and reference kernel are equal or
      // not
      passed &= cutlass::reference::device::BlockCompareEqual(
          block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i),
          M * N);
      if (!passed)
        break;
    }
    return passed;
  }

  /// Allocates device-side data for reference GEMM
  void allocate_for_ref_gemm(const GroupGEMMOptions &options, const ElementA *block_A_ptr,
                const ElementA *block_B_ptr, ElementOutput*block_C_ptr,
                int block_A_size, int block_B_size, int block_C_size) {
    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;
    cumsum_device.reset(options.groups + 1);
    cumsum_host = (int32_t*)(malloc((options.groups + 1) * sizeof(int32_t)));
    cumsum_host[0] = 0;
    // Compute total allocation sizes across group
    for (int32_t i = 0; i < options.groups; ++i) {

      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      cumsum_host[i + 1] += M + cumsum_host[i];
      // Offset into block allocation of each matrix base pointer
      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      int64_t elements_A = M * K;
      int64_t elements_B = K * N;
      int64_t elements_C = M * N;
      int64_t elements_D = M * N;

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;
      total_elements_D += elements_D;

      stride_A_host.push_back(
          cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
      stride_B_host.push_back(
          cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
      stride_C_host.push_back(
          cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
      stride_D_host.push_back(
          cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    }
    assert(total_elements_A == block_A_size);
    // change this assert to static assert because it's known at compile-time
    assert(total_elements_B == block_B_size);
    assert(total_elements_C == block_C_size);
    block_A.reset(const_cast<bfloat16_t *>(block_A_ptr), block_A_size);
    block_B.reset(const_cast<bfloat16_t *>(block_B_ptr), block_B_size);
    block_C.reset(total_elements_D);
    block_D.reset(block_C_ptr, block_C_size);
    block_ref_D.reset(total_elements_D);
    block_alpha.reset(options.groups);
    block_beta.reset(options.groups);
    cumsum_device.copy_from_host(cumsum_host);
  }

  /// Initialize operands to be used in the reference GEMM
  void initialize(const GroupGEMMOptions &options) {

    uint64_t seed = 2020;

    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    //
    // Assign pointers
    //

    std::vector<ElementA *> ptr_A_host(options.groups);
    std::vector<ElementB *> ptr_B_host(options.groups);
    std::vector<ElementC *> ptr_C_host(options.groups);
    std::vector<ElementOutput *> ptr_D_host(options.groups);
    std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
    std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

    // Compute offsets, alpha & beta over group on host
    for (int32_t i = 0; i < options.groups; ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
      // Fill host vector of alpha & beta with random values if using per-group values
      alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
      beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
      // Fill host ptr vectors with offset addresses into device alpha/beta blocks
      ptr_alpha_host.at(i) = block_alpha.get() + i;
      ptr_beta_host.at(i) = block_beta.get() + i;
    }

    // Allocate device memory & copy from host
    ptr_A.reset(options.groups);
    // Per-group alpha and beta
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(options.groups);
    ptr_B.copy_from_host(ptr_B_host.data());

    ptr_C.reset(options.groups);
    ptr_C.copy_from_host(ptr_C_host.data());

    ptr_D.reset(options.groups);
    ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(options.groups);
    stride_A.copy_from_host(stride_A_host.data());

    stride_B.reset(options.groups);
    stride_B.copy_from_host(stride_B_host.data());

    stride_C.reset(options.groups);
    stride_C.copy_from_host(stride_C_host.data());

    stride_D.reset(options.groups);
    stride_D.copy_from_host(stride_D_host.data());

    // Per-group alpha and beta ptrs
    alpha_device.reset(options.groups);
    alpha_device.copy_from_host(ptr_alpha_host.data());
    beta_device.reset(options.groups);
    beta_device.copy_from_host(ptr_beta_host.data());
    // Per-group alpha and beta values - note these are not directly passed to kernel - the pointers
    // (alpha_device/beta_device) are passed instead
    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());
  }

  /// Populates a Gemm::Arguments structure from the given commandline options
  typename Gemm::Arguments
  args_from_options(const GroupGEMMOptions &options,
                    const cutlass::KernelHardwareInfo &hw_info,
                    const ElementA* A_ptr,
                    const ElementB* B_ptr,
                    ElementOutput* D_ptr,
                    const int gemm_N,
                    const int gemm_K) {
    typename Gemm::Arguments arguments;
    decltype(arguments.fusion_args) fusion_args;
    bool host_problem_shapes_available = false;
    if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
      // If both alpha/beta are provided (via cmd line args) and are scalar,
      // i.e., same alpha/beta applies to all batches.
      fusion_args.alpha = 1;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      // Single alpha and beta for all groups
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    } else {
      // If pointers to alpha/beta are provided, i.e., alpha/beta can differ
      // between batches/groups.
      fusion_args.alpha = 1;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      // One alpha and beta per each group
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    }
    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<
            ProblemShape>::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    if (host_problem_shapes_available) {
      arguments = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped, // this just means grouped GEMM
          static_cast<const ElementA**>((void*)A_ptr),
          static_cast<const ElementB**>((void*)B_ptr),
          nullptr,//static_cast<const ElementC**>((void*)D_ptr), // we could also pass nullptr
          static_cast<ElementOutput**>((void*)D_ptr),
          fusion_args,
          hw_info,
          {1, RasterOrderOptions::AlongN},
          options.num_rows_per_expert,
          options.groups,
          gemm_N,
          gemm_K};
    } else {
      arguments = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          static_cast<const ElementA**>((void*)A_ptr),
          static_cast<const ElementB**>((void*)B_ptr),
          nullptr, // static_cast<const ElementC**>((void*)D_ptr),
          static_cast<ElementOutput**>((void*)D_ptr),
          fusion_args,
          hw_info,
          {1, RasterOrderOptions::AlongN},
          options.num_rows_per_expert,
          options.groups,
          gemm_N,
          gemm_K};
    }

    return arguments;
  }

  cutlass::Status run(const GroupGEMMOptions &options,
                      const cutlass::KernelHardwareInfo &hw_info,
                      const ElementA *A_ptr, const ElementB *B_ptr,
                      ElementOutput *D_ptr, int A_size, int B_size, int D_size, const int gemm_n, const int gemm_k) {
    allocate_for_ref_gemm(options, A_ptr, B_ptr, D_ptr, A_size, B_size, D_size);

    Gemm gemm_op;
    auto arguments = args_from_options(options, hw_info, A_ptr, B_ptr, D_ptr, gemm_n, gemm_k);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();
    initialize(options);
    // Verify that the result is correct
    bool passed = verify(options);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed)
      return cutlass::Status::kErrorInternal;
    compat::wait();
    arguments = args_from_options(options, hw_info, A_ptr, B_ptr, D_ptr, gemm_n, gemm_k);
    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm_op.run());
      }
      compat::wait();

      float cute_time = timer.seconds() * 1000;
      double cute_average_time = double(cute_time) / double(options.iterations);
      auto [gflops, mem_bw_util, projected_time] = options.gflops(cute_average_time / 1000.0,
                                     options.problem_sizes_host);

      std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
      for (int32_t i = 0; i < options.groups; ++i) {
        std::cout << "    " << options.problem_sizes_host.at(i);
        std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i)
                  << std::endl;
      }
      std::cout << "  Groups      : " << options.groups << std::endl;
      std::cout << "  Avg runtime : " << cute_average_time << " ms"
                << std::endl;
      std::cout << "  GFLOPS      : " << gflops << std::endl;
      std::cout << "  Memory BW utilization : " << mem_bw_util << "  GBPs" << std::endl;
    }

    return cutlass::Status::kSuccess;
  }
};

void MoEGEMM(const bfloat16_t *activations, const bfloat16_t *weights,
             bfloat16_t *outputs, const int gemm_n, const int gemm_k,
             const int *num_rows_per_expert_device, const int num_experts) {
  GroupGEMMOptions options;

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;
  int num_tokens_incl_duplicated = 0;
  int total_rows_for_each_expert[num_experts];
  cutlass::DeviceAllocation<int32_t> num_rows_per_expert_obj;
  num_rows_per_expert_obj.reset(
      const_cast<int32_t *>(num_rows_per_expert_device), num_experts);
  num_rows_per_expert_obj.copy_to_host(total_rows_for_each_expert);
  options.parse(num_experts, total_rows_for_each_expert, gemm_n, gemm_k,
                num_rows_per_expert_device);

  for (int i = 0; i < num_experts; i++) {
    num_tokens_incl_duplicated += total_rows_for_each_expert[i];
  }
  size_t A_size = num_tokens_incl_duplicated * gemm_k;
  size_t B_size = num_experts * gemm_n * gemm_k;
  size_t D_size = num_tokens_incl_duplicated * gemm_n;
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =                    // M=8,N=16,K=16, D=f32,A=bf16,B=bf16,C=f32
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  // Dispatch to grouped gemm algorithm
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16Group<PipelineStages,
                                               cutlass::gemm::KernelXeMoEGEMM>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  // ScaledAcc needs to be supported in xe_builder.inl and xe_callbacks.cpp
  // This is a workaround
  using EpilogueOp =
      cutlass::epilogue::fusion::LinearCombination<float_t, float_t, float_t, float_t, cutlass::FloatRoundStyle::round_to_nearest, false>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp, TileShape,
          Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto,
          float, float, float, LayoutC, 1, ElementOutput, LayoutC, 1,
          EpilogueDispatchPolicy, EpilogueOp>::CollectiveOp;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape, ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA *>, ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB *>, TiledMma, GmemTiledCopyA, void,
      void, cute::identity,                      // A
      GmemTiledCopyB, void, void, cute::identity // B
      >;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::GroupScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  ExampleRunner<Gemm> runner;

  runner.run(options, hw_info, activations, weights, outputs, A_size, B_size,
             D_size, gemm_n, gemm_k);
  num_rows_per_expert_obj.release();
}

void launcher(int* M_per_expert, int N, int K, const int& num_experts) {
  int n_moe = N;
  int k_moe = K;
  int num_tokens_incl_duplicated = 0;
  for(int i=0; i < num_experts; i++) {
    num_tokens_incl_duplicated += M_per_expert[i];
  }

  float M_occupancy = 0.f;
  float actual_num_units = 0.f;
  int total_num_M_tiles = 0;
  for (int i=0; i < num_experts; i++) {
    total_num_M_tiles += (M_per_expert[i] + 63)/64;
    actual_num_units += M_per_expert[i]/64.0;
  }
  M_occupancy = actual_num_units / total_num_M_tiles;
  std::cout << "\n\n M-occupancy is " << M_occupancy << std::endl;
  cutlass::DeviceAllocation<int32_t> num_rows_per_expert_device;
  cutlass::DeviceAllocation<bfloat16_t> activations_data;
  cutlass::DeviceAllocation<bfloat16_t> weights_data;
  cutlass::DeviceAllocation<bfloat16_t> output_data;
  size_t A_size = num_tokens_incl_duplicated * k_moe;
  size_t B_size = num_experts * n_moe * k_moe;
  size_t D_size = num_tokens_incl_duplicated * n_moe;
  num_rows_per_expert_device.reset(num_experts);
  num_rows_per_expert_device.copy_from_host(M_per_expert);
  activations_data.reset(A_size);
  weights_data.reset(B_size);
  output_data.reset(D_size);
  uint64_t seed = 2023;
  initialize_block(activations_data, seed + 2023);
  initialize_block(weights_data, seed + 2022);
  initialize_block(output_data, seed + 2021);
  MoEGEMM(activations_data.get(), weights_data.get(), output_data.get(), n_moe,
          k_moe, num_rows_per_expert_device.get(), num_experts);
  activations_data.release();
  weights_data.release();
  output_data.release();
  num_rows_per_expert_device.release();
}


int main(int argc, const char **argv) {
  constexpr int num_experts = 32;
  constexpr int num_layers = 24;

 int total_rows_for_each_expert[num_layers][num_experts] = {
  {148, 231, 404, 180, 127, 244, 224, 244, 110, 617, 289, 845, 191, 424, 30, 97, 57, 324, 62, 77, 75, 144, 250, 287, 629, 370, 161, 101, 215, 113, 224, 35},
  {666, 214, 448, 87, 4, 28, 48, 13, 74, 40, 546, 397, 487, 350, 26, 95, 517, 487, 295, 58, 637, 97, 139, 33, 126, 15, 352, 311, 995, 193, 135, 135},
  {1016, 30, 36, 452, 469, 473, 232, 0, 493, 14, 954, 6, 4, 6, 279, 3, 94, 106, 96, 48, 49, 113, 142, 169, 75, 99, 25, 220, 249, 289, 4, 1803},
  {350, 229, 703, 154, 8, 64, 80, 339, 2, 56, 5, 312, 1005, 29, 9, 11, 23, 0, 23, 431, 48, 129, 496, 476, 8, 1234, 7, 130, 34, 58, 41, 1554},
  {39, 10, 6, 2, 110, 1, 894, 8, 53, 0, 275, 6, 506, 421, 700, 178, 0, 530, 1623, 15, 231, 74, 6, 222, 1246, 116, 35, 20, 0, 6, 381, 334},
  {399, 5, 201, 6, 134, 93, 1748, 1, 51, 4, 38, 336, 53, 88, 328, 724, 15, 388, 706, 52, 19, 55, 52, 33, 623, 1, 222, 215, 69, 45, 308, 1036},
  {11, 8, 407, 571, 458, 275, 197, 211, 13, 564, 462, 114, 15, 13, 132, 24, 514, 2, 71, 13, 694, 47, 16, 203, 610, 40, 0, 1587, 66, 23, 196, 491},
  {0, 230, 116, 136, 315, 643, 6, 183, 37, 26, 960, 1, 8, 258, 21, 1602, 213, 198, 6, 196, 455, 557, 47, 282, 493, 18, 101, 11, 616, 45, 268, 0},
  {392, 305, 179, 14, 227, 98, 114, 39, 64, 1456, 465, 0, 18, 372, 0, 0, 189, 257, 25, 290, 486, 0, 12, 1534, 468, 4, 555, 35, 146, 0, 161, 143},
  {4, 107, 20, 125, 236, 898, 0, 0, 375, 2, 125, 0, 0, 1429, 36, 195, 1660, 0, 127, 454, 73, 358, 47, 79, 32, 20, 1465, 0, 0, 6, 109, 66},
  {19, 0, 0, 0, 2, 1638, 75, 135, 392, 2, 1494, 3, 23, 5, 4, 58, 0, 0, 71, 1285, 8, 441, 0, 145, 209, 408, 450, 2, 824, 13, 326, 16},
  {4, 2, 14, 0, 30, 206, 41, 131, 0, 429, 16, 895, 35, 21, 44, 128, 12, 0, 417, 0, 838, 917, 42, 115, 109, 1759, 0, 36, 17, 0, 1790, 0},
  {6, 483, 241, 1327, 17, 11, 480, 9, 880, 58, 4, 0, 61, 30, 16, 176, 9, 309, 26, 0, 0, 1882, 4, 281, 475, 783, 197, 0, 19, 15, 6, 243}, 
  {370, 1222, 0, 6, 108, 929, 2, 7, 157, 348, 149, 106, 2, 5, 25, 33, 1569, 8, 6, 106, 69, 1298, 0, 2, 529, 520, 0, 421, 0, 25, 26, 0},
  {59, 89, 0, 26, 25, 40, 1873, 141, 527, 371, 262, 62, 16, 0, 127, 234, 1637, 64, 132, 8, 0, 7, 161, 1005, 22, 1, 49, 6, 83, 925, 80, 16},
  {269, 617, 30, 4, 90, 26, 0, 16, 154, 212, 21, 269, 379, 174, 129, 32, 8, 121, 344, 15, 0, 591, 1494, 6, 737, 50, 112, 856, 483, 25, 454, 330},
  {0, 98, 1488, 22, 73, 0, 0, 343, 77, 4, 0, 612, 165, 268, 4, 10, 43, 0, 598, 271, 2, 73, 185, 0, 112, 779, 24, 1626, 0, 0, 0, 1171},
  {0, 0, 0, 189, 266, 1743, 0, 462, 20, 7, 668, 310, 40, 0, 10, 236, 423, 18, 0, 0, 0, 999, 0, 139, 1754, 8, 619, 3, 23, 0, 102, 9},
  {131, 1753, 0, 113, 24, 94, 2, 12, 108, 0, 0, 252, 97, 0, 1319, 233, 93, 1254, 195, 152, 14, 413, 4, 2, 220, 67, 20, 4, 34, 559, 837, 42},
  {55, 76, 0, 8, 0, 3, 1557, 975, 135, 271, 4, 0, 0, 666, 207, 152, 5, 2, 97, 364, 0, 13, 1423, 771, 159, 31, 223, 0, 431, 7, 409, 4},
  {4, 1026, 1799, 166, 694, 753, 0, 16, 0, 240, 1119, 19, 6, 0, 46, 659, 10, 0, 112, 808, 181, 0, 28, 22, 90, 0, 176, 0, 37, 5, 10, 22},
  {44, 0, 4, 153, 299, 1357, 6, 23, 0, 12, 4, 419, 73, 24, 16, 24, 1, 4, 4, 102, 16, 4, 0, 1953, 1850, 0, 908, 4, 0, 13, 708, 23},
  {6, 13, 123, 28, 197, 0, 202, 69, 0, 6, 0, 21, 1434, 1582, 11, 0, 6, 0, 7, 190, 4, 1700, 6, 434, 1886, 0, 14, 28, 8, 30, 25, 18},
  {5, 27, 1442, 18, 0, 6, 0, 73, 6, 781, 0, 1915, 291,  649, 98, 4, 33, 77, 6, 22, 73, 9, 8, 587, 1486, 32, 10, 244, 37, 0, 100, 9}
  };

  for (int i = 0; i < num_layers; i++) {
    launcher(total_rows_for_each_expert[i], 5760, 2880, num_experts);
    launcher(total_rows_for_each_expert[i], 2880, 2880, num_experts);    
  }  

  return 0;
}

