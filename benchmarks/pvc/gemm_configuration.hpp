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

#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

using namespace cute;

namespace cutlass {
namespace gemm {
namespace device {

enum class Scheduler { Gemm, GemmSplitK, GemmStreamK };

template<
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator,
  class TileShape,
  Scheduler TileScheduler>
struct GemmConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid GemmConfiguration configuration exists.");
};

/////////////////////////////////////////////////////////////////////////

// bfloat16

template<typename LayoutA, typename LayoutB, typename LayoutC,
  class TileShape, Scheduler TileScheduler>
struct GemmConfiguration<
      arch::IntelPVC,
      bfloat16_t, LayoutA,
      bfloat16_t, LayoutB,
      float, LayoutC,
      float, TileShape,
      TileScheduler> {
  using KernelScheduleType =std::conditional_t<TileScheduler == Scheduler::Gemm, cutlass::gemm::KernelPVC, cutlass::gemm::KernelPVCCooperative>;


  static_assert(std::is_same_v<LayoutC, cutlass::layout::RowMajor>, "LayoutC unsupported in collective builder");
  using LayoutD = LayoutC;
  using ClusterShape = Shape<_1, _1, _1>;
  // Mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelPVC, cutlass::arch::OpClassTensorOp,
    bfloat16_t, LayoutA, sizeof(bfloat16_t),
    bfloat16_t, LayoutB, sizeof(bfloat16_t),
    float,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    KernelScheduleType
  >::CollectiveOp;

  // Epilogue
  using EpilogueOp = epilogue::fusion::LinearCombination<float, float, float, float, FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::IntelPVC, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float,
    float,
    float, LayoutC, sizeof(float),
    float, LayoutD, sizeof(float),
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp
  >::CollectiveOp;

  using GemmKernel = kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    std::conditional_t<TileScheduler == Scheduler::Gemm, void, cutlass::gemm::StreamKScheduler>
  >;

  using Gemm = GemmUniversalAdapter<GemmKernel>;

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    using StreamKMode =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode;
    if constexpr (TileScheduler == Scheduler::Gemm) {
      return {};
    } else if constexpr (TileScheduler == Scheduler::GemmStreamK) {
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {1, StreamKMode::StreamK};
      return arguments;
    } else {
      static_assert(TileScheduler == Scheduler::GemmSplitK);
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {1, StreamKMode::SplitK};
      return arguments;
    }
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass
