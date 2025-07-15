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
/*! \file
    \brief Tests for device-wide Flash Attention Decode interface
*/

#pragma once

#include "flash_attention_v2/flash_decode_runner.hpp"

#include "../gemm/device/testbed_utils.h"
#include "../common/cutlass_unit_test.h"

namespace test {
namespace flash_attention {

using namespace cute;
using namespace cutlass::flash_attention;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename FlashDecode>
struct TestbedImpl : public FlashDecodeRunner<FlashDecode> {
  using FMHADecodeKernel = typename FlashDecode::FMHADecodeKernel;
  using ProblemShapeType = typename FMHADecodeKernel::ProblemShape;

  //
  // Methods
  //

  /// Initializes data structures
  template <class ProblemShape>
  ProblemShapeType initialize(ProblemShape problem_shape_in, int page_size) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::initialize(problem_size)");
#endif
    auto [problem_shape, mem_count] = FlashDecodeRunner<FlashDecode>::initialize(problem_shape_in, page_size);
    return problem_shape;
  }

  /// Verifies the result
  bool verify(ProblemShapeType problem_size, float softmax_scale) {
    return FlashDecodeRunner<FlashDecode>::verify(problem_size, softmax_scale);
  }

  bool sufficient() {
    return true;
  }

  /// Executes one test
  template<class ProblemShape>
  bool run(ProblemShape problem_size_init, float softmax_scale, int page_size)
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

    ProblemShapeType problem_size = this->initialize(problem_size_init, page_size);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize() returned true");
#endif

    //
    // Initialize the Flash attention operator
    //
    cutlass::KernelHardwareInfo hw_info;
    typename FMHADecodeKernel::Arguments arguments = FlashDecodeRunner<FlashDecode>::get_arguments(problem_size, hw_info, softmax_scale, 0);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling FMHADecodeKernel::get_workspace_size");
#endif
    size_t workspace_size = FMHADecodeKernel::get_workspace_size(arguments);
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Allocating workspace of size " << workspace_size);
#endif
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling FMHADecodeKernel::can_implement");
#endif
    auto can_implement = FMHADecodeKernel::can_implement(arguments);

    if (!can_implement) {
      std::cerr << "This test is not supported." << "\n";
    }

    //
    // Run Flash attention
    //

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling to_underlying_arguments");
#endif
    auto params = FMHADecodeKernel::to_underlying_arguments(arguments, workspace.get());

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling run");
#endif
    FlashDecodeRunner<FlashDecode>::run(params);

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
  typename FlashDecode
>
struct Testbed3x {
  using TestBedImpl = typename detail::TestbedImpl<FlashDecode>;
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3x() : impl_() {}

  /// Executes one test
  template <class ProblemShape>
  bool run(
   ProblemShape problem_size,
   float softmax_scale,
   int page_size
    )
  {
    return impl_.run(problem_size, softmax_scale, page_size);
  }
};

template <typename FlashDecode>
bool TestFlashDecodeAll(int head_size) {
  Testbed3x<FlashDecode> testbed;

  std::vector<int> problem_size_batch{16};
  std::vector<int> problem_size_num_heads{32};
  std::vector<int> problem_size_seq_len{1024};
  std::vector<int> problem_size_seq_len_cache{0, 1024};
  std::vector<int> cache_page_size{64, 128};
  std::vector<float> problem_size_softmax_scale{ 1.f / sqrt(static_cast<float>(head_size)) };
  bool passed = true;

  for (int batch : problem_size_batch) {
    for (int num_heads : problem_size_num_heads) {
      for (int seq_len : problem_size_seq_len) {
        for (int seq_len_cache : problem_size_seq_len_cache) {
          for (int page_size : cache_page_size) {
            for (float softmax_scale : problem_size_softmax_scale) {
              auto num_heads_q = num_heads;
              auto num_heads_kv = num_heads;
              auto seq_len_qo = 1;
              auto seq_len_kv = seq_len;
              auto seq_len_kv_cache = seq_len_cache;
              auto head_size_qk = head_size;
              auto head_size_vo = head_size;

              auto problem_size = cute::make_tuple(
                batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo);
              try {
                passed = testbed.run(problem_size, softmax_scale, page_size);
              }
              catch (std::exception const& e) {
                EXPECT_TRUE(false) << "TestFlashDecodeAll: testbed.run {"
                  << "batch: " << batch << ", num_heads_q: " << num_heads_q << ", num_heads_kv: " << num_heads_kv
                  << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv << ", seq_len_kv_cache: "
                  << seq_len_cache << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
                  << ", scale: " << softmax_scale << ", page_size: " << page_size
                  << "} threw an exception: " << e.what();
                throw;
              }
              catch (...) {
                EXPECT_TRUE(false) << "TestFlashDecodeAll: testbed.run {"
                  << "batch: " << batch << ", num_heads_q: " << num_heads_q << ", num_heads_kv: " << num_heads_kv
                  << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv << ", seq_len_kv_cache: "
                  << seq_len_cache << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
                  << ", scale: " << softmax_scale << ", page_size: " << page_size
                  << "} threw an exception (unknown)";
                throw;
              }

              EXPECT_TRUE(passed) << "TestFlashDecodeAll: testbed.run {"
                << "batch: " << batch << ", num_heads_q: " << num_heads_q << ", num_heads_kv: " << num_heads_kv
                << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv << ", seq_len_kv_cache: "
                << seq_len_cache << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
                << ", scale: " << softmax_scale << ", page_size: " << page_size
                << "} failed";

              if (!passed) {
                std::cout << __FILE__ << ':' << __LINE__ << " : Flash Decode FAILED.\n";
                return false;
              }
            } // softmax_scale
          } // page_size
        } // seq_len_cache
      } // seq_len
    } // num_heads
  }  // batch
  return passed;
}

} // namespace flash_attention
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
