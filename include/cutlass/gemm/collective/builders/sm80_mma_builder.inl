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

#include <cutlass/arch/arch.h>
#include "cutlass/gemm/collective/sm80_mma_multistage.hpp"

#include "sm80_common.inl"

using namespace cute;

namespace cutlass::gemm::collective {

template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder <
  arch::Sm80,
  arch::OpClassTensorOp,
  ElementA,
  GmemLayoutA,
  AlignmentA,
  ElementB,
  GmemLayoutB,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  Shape<_1, _1, _1>,
  StageCountType,
  KernelScheduleType,
  cute::enable_if_t<
    (cute::is_same_v<KernelScheduleType, KernelScheduleAuto> ||
    cute::is_same_v<KernelScheduleType, KernelMultistage>)
  >
> {
      using DispatchPolicy = MainloopSm80CpAsync<5>;
      using GmemTiledCopyA =  typename detail::getMemoryAtomsOperandA<GmemLayoutA, ElementA, ElementAccumulator>::GmemTiledCopy;
      using GmemTiledCopyB =  typename detail::getMemoryAtomsOperandB<GmemLayoutB, ElementB, ElementAccumulator>::GmemTiledCopy;

      using SmemLayoutAtomA =  typename detail::getMemoryAtomsOperandA<GmemLayoutA, ElementA, ElementAccumulator>::SmemLayoutAtom;
      using SmemLayoutAtomB =  typename detail::getMemoryAtomsOperandA<GmemLayoutB, ElementB, ElementAccumulator>::SmemLayoutAtom;

      using SmemCopyAtomA = typename detail::getMemoryAtomsOperandA<GmemLayoutA, ElementA, ElementAccumulator>::SmemCopyAtom;
      using SmemCopyAtomB = typename detail::getMemoryAtomsOperandB<GmemLayoutB, ElementB, ElementAccumulator>::SmemCopyAtom;

      using TiledMMA = typename detail::getMMAType<ElementA, ElementB, ElementAccumulator>::TiledMMA;

      using CollectiveOp = collective::CollectiveMma<
            DispatchPolicy,
            TileShape_MNK,
            ElementA,
            cutlass::gemm::TagToStrideA_t<GmemLayoutA>,
            ElementB,
            cutlass::gemm::TagToStrideA_t<GmemLayoutB>,
            TiledMMA,
            GmemTiledCopyA,
            SmemLayoutAtomA,
            SmemCopyAtomA,
            cute::identity,
            GmemTiledCopyB,
            SmemLayoutAtomB,
            SmemCopyAtomB,
            cute::identity
          >;
  };
}
