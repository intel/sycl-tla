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
/*! \file
    \brief Tests for device-wide Flash Attention interface
*/

#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/kernel/xe_flash_attn_gemm.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_epilogue.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_softmax_epilogue.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/sycl_tensor_fill.h"

#include "../gemm/device/testbed_utils.h"
#include "../common/cutlass_unit_test.h"

namespace test {
namespace flash_attention {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
        cutlass::DeviceAllocation<Element>& block,
        uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
       block.get(), block.size(), seed, scope_max, scope_min, 0);

  syclcompat::wait();
  return true;
}

template <typename FlashAttention>
struct TestbedImpl {
  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using StrideQ = typename FlashAttention::StrideQ;
  using StrideK = typename FlashAttention::StrideK;
  using StrideV = typename FlashAttention::StrideV;
  using StrideO = typename FlashAttention::StrideO;

  using ElementQ = typename FlashAttention::ElementQ;
  using ElementK = typename FlashAttention::ElementK;
  using ElementV = typename FlashAttention::ElementV;
  using ElementAcc = typename FlashAttention::ElementAccumulator;

  using CollectiveMainloop = typename FlashAttention::CollectiveMainloop;
  using CollectiveEpilogue = typename FlashAttention::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename FlashAttention::ProblemShape;
  static constexpr bool HasCausalMask = CollectiveMainloop::CausalMask;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementOutput> block_O;
  cutlass::DeviceAllocation<ElementOutput> block_ref_O;

  //
  // Methods
  //

  /// Initializes data structures
  bool initialize(ProblemShapeType problem_size) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::initialize(problem_size)");
#endif
    auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = problem_size;

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(seq_len_qo, head_size_qk, batch * num_heads));
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, head_size_qk, batch * num_heads));
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size_vo, seq_len_kv, batch * num_heads));
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(seq_len_qo, head_size_vo, batch * num_heads));

    block_Q.reset(batch * num_heads * seq_len_qo * head_size_qk);
    block_K.reset(batch * num_heads * seq_len_kv * head_size_qk);
    block_V.reset(batch * num_heads * seq_len_kv * head_size_vo);
    block_O.reset(batch * num_heads * seq_len_qo * head_size_vo);
    block_ref_O.reset(batch * num_heads * seq_len_qo * head_size_vo);

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022); // assume K is already transposed
    initialize_block(block_V, seed + 2021);

    return true;
  }

  /// Verifies the result
  bool verify(ProblemShapeType problem_size, float softmax_scale)
  {
    auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = problem_size;

    int batch_size = batch * num_heads;

    // loop over the batch dimension to compute the output
    // to avoid the risk of running out of device memory
    for (int b = 0; b < batch_size; b++) {

      cutlass::DeviceAllocation<ElementOutput> block_S;
      block_S.reset(seq_len_qo * seq_len_kv);
      int offset_q = b * seq_len_qo * head_size_qk;
      int offset_k = b * seq_len_kv * head_size_qk;
      int offset_v = b * seq_len_kv * head_size_vo;
      int offset_o = b * seq_len_qo * head_size_vo;

      cutlass::TensorRef ref_Q(block_Q.get() + offset_q, LayoutQ::packed({seq_len_qo, head_size_qk}));
      cutlass::TensorRef ref_K(block_K.get() + offset_k, LayoutK::packed({head_size_qk, seq_len_kv}));
      cutlass::TensorRef ref_V(block_V.get() + offset_v, LayoutV::packed({seq_len_kv, head_size_vo}));
      cutlass::TensorRef ref_S(block_S.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));
      cutlass::TensorRef ref_O(block_ref_O.get() + offset_o, LayoutO::packed({seq_len_qo, head_size_vo}));

      cutlass::reference::device::GemmComplex({seq_len_qo, seq_len_kv, head_size_qk}, 1.f, ref_Q,
                                              cutlass::ComplexTransform::kNone, ref_K, cutlass::ComplexTransform::kNone,
                                              0.f, ref_S, ref_S, ElementAccumulator(0),
                                              1,                   // batch_count
                                              seq_len_qo * head_size_qk, // batch_stride_Q
                                              seq_len_kv * head_size_qk, // batch_stride_K
                                              seq_len_qo * seq_len_kv,   // batch_stride_S
                                              seq_len_qo * seq_len_kv    // batch_stride_S
      );

      syclcompat::wait();

      std::vector<ElementOutput> host_S(block_S.size());
      syclcompat::memcpy<ElementOutput>(host_S.data(), block_S.get(), host_S.size());
      syclcompat::wait();

      // delete this memory as it is no longer needed
      block_S.reset();

      if constexpr (HasCausalMask) {
        // apply mask to S
        for (int row = 0; row < seq_len_qo; row++) {
          for (int col = 0; col < seq_len_kv; col++) {
            if (col > row)
              host_S[col + row * seq_len_kv] = -INFINITY;
          }
        }
      }

      // compute max element per row of S
      std::vector<ElementOutput> max_vec(seq_len_qo, -INFINITY);
      for (int row = 0; row < seq_len_qo; row++) {
        int idx = row * seq_len_kv;
        int max_idx = row;
        max_vec[max_idx] = host_S[idx++];
        for (int col = 1; col < seq_len_kv; col++, idx++) {
          if (max_vec[max_idx] < host_S[idx])
            max_vec[max_idx] = host_S[idx];
        }
      }

      // compute exp of S
      for (int row = 0; row < seq_len_qo; row++) {
        int idx = row * seq_len_kv;
        int max_idx = row;
        for (int col = 0; col < seq_len_kv; col++, idx++) {
          host_S[idx] = expf((host_S[idx] - max_vec[max_idx]) / sqrt(static_cast<ElementOutput>((head_size_qk))));
        }
      }

      // compute sum per row of S
      std::vector<ElementOutput> sum_vec(seq_len_qo, ElementOutput{0});
      for (int row = 0; row < seq_len_qo; row++) {
        int idx = row * seq_len_kv;
        int sum_idx = row;
        for (int col = 0; col < seq_len_kv; col++, idx++) {
          sum_vec[sum_idx] += host_S[idx];
        }

        // scale each row with the sum to compute softmax
        idx = row * seq_len_kv;
        sum_idx = row;
        for (int col = 0; col < seq_len_kv; col++, idx++) {
          host_S[idx] /= sum_vec[sum_idx];
        }
      }

      std::vector<ElementV> host_P(host_S.size());
      for (int p = 0; p < host_P.size(); p++)
        host_P[p] = static_cast<ElementV>(host_S[p]);

      cutlass::DeviceAllocation<ElementV> block_P;
      block_P.reset(host_P.size());

      syclcompat::memcpy<ElementV>(block_P.get(), host_P.data(), host_P.size());
      syclcompat::wait();

      cutlass::TensorRef ref_P(block_P.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));

      cutlass::reference::device::GemmComplex({seq_len_qo, head_size_vo, seq_len_kv}, 1.f, ref_P,
                                              cutlass::ComplexTransform::kNone, ref_V, cutlass::ComplexTransform::kNone,
                                              0.f, ref_O, ref_O, ElementAccumulator(0),
                                              1,                   // batch_count
                                              seq_len_qo * seq_len_kv,   // batch_stride_P
                                              seq_len_kv * head_size_vo, // batch_stride_V
                                              seq_len_qo * head_size_vo, // batch_stride_O
                                              seq_len_qo * head_size_vo  // batch_stride_O
      );

      syclcompat::wait();
      // delete this memory as it is no longer needed
      block_P.reset();
    }

    syclcompat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_O.get(), block_O.get(),
                                                                          block_O.size(), 0.5f, 0.5f);
    return passed;
  }

  bool sufficient() {
    return true;
  }

  /// Executes one test
  bool run(ProblemShapeType problem_size, float softmax_scale)
  {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run"); 
#endif

    // Fail test if insufficient device
    if (!sufficient()) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: Test failed due to insufficient device");
      std::cout << "Test failed due to insufficient device." << std::endl;
      return false;
    }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    else {
      CUTLASS_TRACE_HOST("TestbedImpl::run: sufficient() returned true");
    }
#endif

    try {
      const bool initialized = this->initialize(problem_size);
      if (not initialized) {
        CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize returned false");
        std::cerr << "Initialization failed \n";
        return false;
      }
    }
    catch ([[maybe_unused]] std::exception const& e) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize threw an exception: " << e.what());
      throw;
    }
    catch (...) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize threw an unknown exception");
      throw;
    }

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize() returned true");
#endif

    //
    // Initialize the Flash attention operator
    //
    cutlass::KernelHardwareInfo hw_info;
    typename FlashAttention::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_Q.get(), stride_Q, block_K.get(), stride_K, block_V.get(), stride_V},
      {softmax_scale},
      {block_O.get(), stride_O},
      hw_info};

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling FlashAttention::get_workspace_size");
#endif
    size_t workspace_size = FlashAttention::get_workspace_size(arguments);
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Allocating workspace of size " << workspace_size);
#endif
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling FlashAttention::can_implement");
#endif
    auto can_implement = FlashAttention::can_implement(arguments);

    if (!can_implement) {
      std::cerr << "This test is not supported." << "\n";
    }

    //
    // Run Flash attention
    //

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling to_underlying_arguments");
#endif
    auto params = FlashAttention::to_underlying_arguments(arguments, workspace.get());

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling run");
#endif
    auto const block = FlashAttention::get_block_shape();
    auto const grid = FlashAttention::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FlashAttention::SharedStorageSize;

    const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace syclcompat::experimental;
    auto event = launch<cutlass::device_kernel<FlashAttention>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<FlashAttention::DispatchPolicy::SubgroupSize>}},
        params);
#else
    syclcompat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    syclcompat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<FlashAttention::DispatchPolicy::SubgroupSize>
    };
    syclcompat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = syclcompat::experimental::launch<cutlass::device_kernel<FlashAttention>>(policy, params);
#endif
    EventManager::getInstance().addEvent(event);

    try {
      syclcompat::wait_and_throw();
    } catch (std::exception const &e) {
      ADD_FAILURE() << "Error at Kernel Sync.";
      return false;
    }

    //
    // Verify
    //
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling this->verify");
#endif
    bool passed = this->verify(problem_size, softmax_scale);
    if (!passed) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->verify FAILED");
      std::cout << "Error : Failed \n";
    }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    else {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->verify passed");
    }
#endif

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Reached end");
#endif
    return passed;
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename FlashAttention
>
struct Testbed3x {
  using TestBedImpl = typename detail::TestbedImpl<FlashAttention>;
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3x() : impl_() {}

  /// Executes one test
  bool run(
   typename TestBedImpl::ProblemShapeType problem_size,
   float softmax_scale
    )
  {
    return impl_.run(problem_size, softmax_scale);
  }
};

template <typename FlashAttention>
bool TestAll(int head_size) {
  Testbed3x<FlashAttention> testbed;

  using ProblemShapeType = typename FlashAttention::ProblemShape;

  std::vector<int> problem_size_batch{32};
  std::vector<int> problem_size_num_heads{16};
  std::vector<int> problem_size_seq_len{512};
  std::vector<float> problem_size_softmax_scale{ 1.f / sqrt(static_cast<float>(head_size)) };
  bool passed = true;

  for (int batch : problem_size_batch) {
    for (int num_heads : problem_size_num_heads) {
      for (int seq_len : problem_size_seq_len) {
        for (float softmax_scale : problem_size_softmax_scale) {
          auto seq_len_qo = seq_len;
          auto seq_len_kv = seq_len;
          auto head_size_qk = head_size;
          auto head_size_vo = head_size;

          ProblemShapeType problem_size{batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo};
          try {
            passed = testbed.run(problem_size, softmax_scale);
          }
          catch (std::exception const& e) {
            EXPECT_TRUE(false) << "TestAll: testbed.run {"
              << "batch: " << batch << ", num_heads: " << num_heads
              << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv
              << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
              << ", scale: " << softmax_scale
              << "} threw an exception: " << e.what();
            throw;
          }
          catch (...) {
            EXPECT_TRUE(false) << "TestAll: testbed.run {"
              << "batch: " << batch << ", num_heads: " << num_heads
              << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv
              << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
              << ", scale: " << softmax_scale
              << "} threw an exception (unknown)";
            throw;
          }

          EXPECT_TRUE(passed) << "TestAll: testbed.run {"
            << "batch: " << batch << ", num_heads: " << num_heads
            << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv
            << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
            << ", scale: " << softmax_scale
            << "} failed";

          if (!passed) {
            std::cout << __FILE__ << ':' << __LINE__ << " : Flash attention FAILED.\n";
            return false;
          }
        } // softmax_scale
      } // seq_len
    } // num_heads
  }  // batch
  return passed;
}

} // namespace flash_attention
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
