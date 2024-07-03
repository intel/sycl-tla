/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include <cute/util/sycl_vec.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class StrideC_, class StrideD_, class ThreadEpilogueOp_,
          class EpilogueSchedule_, uint32_t m, uint32_t n>
class PvcEpilogueTensorSoftmax {
public:
  using EpilogueSchedule = EpilogueSchedule_;
  using DispatchPolicy = EpilogueSchedule_;

  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const *ptr_C = nullptr;
    StrideC dC{};
    ElementD *ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments([[maybe_unused]] ProblemShape const &_,
                          Arguments const &args,
                          [[maybe_unused]] void *workspace) {
    return args;
  }

  template <typename T> CUTLASS_DEVICE void operator()(T &t) {
    static_assert(cute::is_same_v<typename T::value_type, float> && m <= 32);

    auto const &group =
        sycl::ext::oneapi::experimental::this_nd_item<3>().get_group();

    static constexpr auto vec_size = 4;

    static_assert((m % vec_size) == 0 && vec_size <= 16);
    static constexpr auto loop_cnt = m / vec_size;

    sycl::vec<float, vec_size> local_max;
    sycl::vec<float, vec_size> local_plus;

    for (int loop = 0; loop < loop_cnt; loop++) {

      auto base_row = loop * vec_size;
      // init local max
      for (int i = 0; i < vec_size; i++) {
        local_max[i] = t[(base_row + i) * n];
      }

      for (int i = 0; i < vec_size; i++) {
        for (int j = 0; j < n; j++) {
          local_max[i] = max(local_max[i], t((base_row + i) * n + j));
        }
      }

      // get group max
      auto group_max = reduce_over_group(group, local_max, sycl::maximum<>());

      // -max, exp, and get local plus
      for (int i = 0; i < vec_size; i++) {
        for (int j = 0; j < n; j++) {
          auto offset = (base_row + i) * n + j;
          t[offset] -= group_max[i];
          t[offset] = sycl::exp(t[offset]);

          local_plus[i] += t[offset];
        }
      }

      // get group plus
      auto group_plus = reduce_over_group(group, local_plus, sycl::plus<>());

      // last div
      for (int i = 0; i < vec_size; i++) {
        for (int j = 0; j < n; j++) {
          auto offset = (base_row + i) * n + j;
          t[offset] = t[offset] / group_plus[i];
          // local_sum += t[i * n + j];
        }
      }
    }

    // printf("verify softmax, local_sum: %f, group_sum: %f\n", local_sum,
    // reduce_over_group(group, local_sum, sycl::plus<>()));
    //  }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
