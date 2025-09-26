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
  double gflops(double runtime_s,
                std::vector<typename ProblemShape::UnderlyingProblemShape>
                    problem_sizes_host) const {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const &problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
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
      syclcompat::wait();

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

  /// Allocates device-side data
  void allocate(const GroupGEMMOptions &options, const ElementA *block_A_ptr,
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

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize_for_moe_gemm(const GroupGEMMOptions &options) {

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

      ptr_A_host.at(0) = block_A.get();
      ptr_B_host.at(0) = block_B.get();
      ptr_C_host.at(0) = block_C.get();
      ptr_D_host.at(0) = block_D.get();
    for (int32_t i = 0; i < options.groups; ++i) {
      // Fill host vector of alpha & beta with random values if using per-group
      // values
      alpha_host.push_back(
          (options.alpha == FLT_MAX)
              ? static_cast<ElementAccumulator>((rand() % 5) + 1)
              : options.alpha);
      beta_host.push_back((options.beta == FLT_MAX)
                              ? static_cast<ElementAccumulator>(rand() % 5)
                              : options.beta);
      // Fill host ptr vectors with offset addresses into device alpha/beta
      // blocks
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

    // Per-group alpha and beta values - note these are not directly passed to
    // kernel - the pointers (alpha_device/beta_device) are passed instead
    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize_for_ref_gemm(const GroupGEMMOptions &options) {

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
      // Fill host ptr vectors with offset addresses into device alpha/beta
      // blocks
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

    // Per-group alpha and beta values - note these are not directly passed to
    // kernel - the pointers (alpha_device/beta_device) are passed instead
    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());
  }

  /// Populates a Gemm::Arguments structure from the given commandline options
  typename Gemm::Arguments
  args_from_options(const GroupGEMMOptions &options,
                    const cutlass::KernelHardwareInfo &hw_info,
                    const int gemm_N,
                    const int gemm_K) {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    bool host_problem_shapes_available = false;
    if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
      // If both alpha/beta are provided (via cmd line args) and are scalar,
      // i.e., same alpha/beta applies to all batches.
      fusion_args.alpha = options.alpha;
      fusion_args.beta = options.beta;
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
      fusion_args.alpha = 0;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = alpha_device.get();
      fusion_args.beta_ptr_array = beta_device.get();
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
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
          {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(),
           stride_D.get()},
          hw_info,
          {1, RasterOrderOptions::AlongN},
          options.num_rows_per_expert,
          options.groups,
          gemm_N,
          gemm_K};
    } else {
      arguments = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
          {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(),
           stride_D.get()},
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
                      ElementOutput *C_ptr, int A_size, int B_size, int D_size, const int gemm_n, const int gemm_k) {
    allocate(options, A_ptr, B_ptr, C_ptr, A_size, B_size, D_size);
    initialize_for_moe_gemm(options);

    Gemm gemm_op;
    auto arguments = args_from_options(options, hw_info, gemm_n, gemm_k);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();
    initialize_for_ref_gemm(options);
    // Verify that the result is correct
    bool passed = verify(options);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed)
      return cutlass::Status::kErrorInternal;
    initialize_for_moe_gemm(options);
    syclcompat::wait();
    arguments = args_from_options(options, hw_info, gemm_n, gemm_k);
    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm_op.run());
      }
      syclcompat::wait();

      float cute_time = timer.seconds() * 1000;
      double cute_average_time = double(cute_time) / double(options.iterations);
      double gflops = options.gflops(cute_average_time / 1000.0,
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
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x16x16_LD_T;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;
/*
  using TiledMma =
      TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>;
*/

  using TiledMma =                    // M=8,N=16,K=16, D=f32,A=bf16,B=bf16,C=f32
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  // Dispatch to grouped gemm algorithm
  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16Group<PipelineStages,
                                               cutlass::gemm::KernelXeMoEGEMM>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp =
      cutlass::epilogue::fusion::LinearCombination<float_t, float_t>;

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


int main(int argc, const char **argv) {
  const int num_experts = 32;

  int total_rows_for_each_expert[num_experts] = {
    148, 231, 404, 180, 127, 244, 224, 244, 110, 617, 289, 845, 191, 424, 30, 97, 57, 324,
  62, 77, 75, 144, 250, 287, 629, 370, 161, 101, 215, 113, 224, 35};

  int num_tokens_incl_duplicated = 0;
  for (int i = 0; i < num_experts; i++) {
    num_tokens_incl_duplicated += total_rows_for_each_expert[i];
  }
  int n_moe = 3072;
  int k_moe = 4096;

  cutlass::DeviceAllocation<int32_t> num_rows_per_expert_device;
  cutlass::DeviceAllocation<bfloat16_t> activations_data;
  cutlass::DeviceAllocation<bfloat16_t> weights_data;
  cutlass::DeviceAllocation<bfloat16_t> output_data;
  size_t A_size = num_tokens_incl_duplicated * k_moe;
  size_t B_size = num_experts * n_moe * k_moe;
  size_t D_size = num_tokens_incl_duplicated * n_moe;
  num_rows_per_expert_device.reset(num_experts);
  num_rows_per_expert_device.copy_from_host(total_rows_for_each_expert);
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
  return 0;
}
