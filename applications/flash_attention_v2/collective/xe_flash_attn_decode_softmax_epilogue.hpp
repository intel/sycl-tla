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
  \brief Functor performing online softmax.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace flash_attention {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <bool CausalMask_, class DispatchPolicy, class... Args> class FlashDecodeSoftmaxEpilogue {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Could not find an epilogue specialization.");
};


template <bool CausalMask_, class Element_>
class FlashDecodeSoftmaxEpilogue<CausalMask_, epilogue::IntelPVCEpilogue, Element_> {
public:

  //
  // Type Aliases
  //
  using DispatchPolicy = epilogue::IntelPVCEpilogue;
  using Element = Element_;

  static constexpr bool CausalMask = CausalMask_;

  using GmemTiledCopyOut = void;

  // Host side epilogue arguments
  struct Arguments {
    Element const scale;
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  static constexpr Params to_underlying_arguments(Arguments const &args) {
    constexpr double kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
    Element val = args.scale * static_cast<Element>(kLog2e);
    return Params{val};
  }

  template <class ProblemShape>
  static size_t get_workspace_size() {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace() {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement() {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FlashDecodeSoftmaxEpilogue(Params const &params_) : params(params_) {}

  template <int Vec, int FragsM, int FragsN, class FragAcc, class STensorMax, class STensorSum>
  CUTLASS_DEVICE void scale_exp_log2(FragAcc &frag_s, STensorMax const &stensor_max, STensorSum &stensor_sum) {
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto group = syclcompat::get_nd_item<1>().get_group();
    int sg_local_id = sg.get_local_id()[0];

    CUTLASS_PRAGMA_UNROLL
    for (int indx = 0; indx < Vec * FragsM; indx++) {
      const Element max_scale = stensor_max(indx) * params.scale;
      Element sum_val = Element{0};
      CUTLASS_PRAGMA_UNROLL
      for (int z = 0; z < FragsN; z++) {
        auto base_indx = indx + (z * Vec * FragsM);
        Element eq = frag_s(base_indx) - max_scale;
        frag_s(base_indx) = sycl::native::exp2(eq);
        sum_val += frag_s(base_indx);
      }
      sum_val = reduce_over_group(sg, sum_val, sycl::plus<>());
      if(sg_local_id == 0) {
        stensor_sum(indx + sg.get_group_id()[0] * Vec * FragsM) = sum_val;
      }
    }

    sycl::group_barrier(group);
    int local_id = group.get_local_id()[0];
    const int NumSG = sg.get_group_range()[0];

    CUTLASS_PRAGMA_UNROLL
    for(int id = (Vec * FragsM * NumSG) >> 1; id > Vec * FragsM; id >>= 1) {
      if(local_id < id) {
        auto left_val = stensor_sum(local_id);
        auto right_val = stensor_sum(local_id + id);
        stensor_sum(local_id) = left_val + right_val;
      }
      sycl::group_barrier(group);
    }
  }

  template <int Vec, int FragsM, int FragsN, class FragSrc, class STensor>
  CUTLASS_DEVICE void reduce_max(FragSrc &src, STensor &stensor_max) {
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto group = syclcompat::get_nd_item<1>().get_group();
    int sg_local_id = sg.get_local_id()[0];

    CUTLASS_PRAGMA_UNROLL
    for (int indx = 0; indx < Vec * FragsM; indx++) {
      Element max_val = src(indx);
      src(indx) *= params.scale;
      CUTLASS_PRAGMA_UNROLL
      for (int z = 1; z < FragsN; z++) {
        auto base_indx = indx + (z * Vec * FragsM);
        max_val = sycl::max(max_val, src(base_indx));
        src(base_indx) *= params.scale;
      }
      max_val = reduce_over_group(sg, max_val, sycl::maximum<>());
      if (sg_local_id == 0) {
        stensor_max(indx + sg.get_group_id()[0] * Vec * FragsM) = max_val;
      }
    }
    sycl::group_barrier(group);
    int local_id = group.get_local_id()[0];
    const int NumSG = sg.get_group_range()[0];

    CUTLASS_PRAGMA_UNROLL
    for(int id = (Vec * FragsM * NumSG) >> 1; id > Vec * FragsM; id >>= 1) {
      if(local_id < id) {
        auto left_val = stensor_max(local_id);
        auto right_val = stensor_max(local_id + id);
        stensor_max(local_id) = sycl::max(left_val, right_val);
      }
      sycl::group_barrier(group);
    }
  }

  template <class FragAcc, class STensorMax, class STensorSum>
  CUTLASS_DEVICE void operator()(FragAcc &frag_s, STensorMax &stensor_max, STensorSum &stensor_sum) {
    using FragAccLayout = typename FragAcc::layout_type;
    constexpr int Vec = get<0>(FragAccLayout{}.shape());
    constexpr int FragsM = get<1>(FragAccLayout{}.shape());
    constexpr int FragsN = get<2>(FragAccLayout{}.shape());
    reduce_max<Vec, FragsM, FragsN>(frag_s, stensor_max);
    scale_exp_log2<Vec, FragsM, FragsN>(frag_s, stensor_max, stensor_sum);
  }

  Params params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace flash_attention
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
