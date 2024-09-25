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
#include <cute/arch/copy.hpp>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/sm80_epilogue.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"


namespace cutlass::epilogue::collective {
  namespace detail {
    template<typename ElementAccumulator>
    struct GetMMATileShape;

    template<>
    struct GetMMATileShape<float> {
      using TileShape = Tile<_32, _32, _8>;
    };

    template<>
    struct GetMMATileShape<double> {
      using TileShape = Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,
                             Layout<Shape<_16,_2>,Stride<_2,_1>>,
                             Underscore>;
    };

    template<>
    struct GetMMATileShape<half_t> {
      using TileShape = Tile<_32, _32, _16>;
    };

    template<>
    struct GetMMATileShape<int32_t> {
      using TileShape = Tile<_64, _64, _16>;
    };
  }


  template<
  class TileShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOpOrCallbacks
  >
    struct CollectiveBuilder<
      arch::Sm80,
      arch::OpClassTensorOp, 
      TileShape_MNK,
      Shape<_1, _1, _1>,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      EpilogueScheduleAuto, 
      FusionOpOrCallbacks,
      cute::enable_if_t<
        (cute::is_same_v<FusionOpOrCallbacks, 
                        cutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>> ||
        cute::is_same_v<FusionOpOrCallbacks,
                        cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::ReLu,
                        ElementD,ElementCompute,ElementC,ElementCompute>>)
      >
    >
    {
      
      using DispatchPolicy = cutlass::epilogue::Sm80EpilogueElementwise<1, 1, 1, false>; // StagesC_, StagesD_, FragmentSize_, ReuseSmemC_
      
      using CopyOpG2R = cute::DefaultCopy;
      using CopyOpR2G = cute::DefaultCopy;

      using SmemLayoutAtomC = void;
      using SmemLayoutAtomD = void;

      using CopyOpS2R = void;
      using CopyOpR2S = void;

      using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
          DispatchPolicy, FusionOpOrCallbacks, TileShape_MNK,
          typename detail::GetMMATileShape<ElementAccumulator>::TileShape
      >;

      using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
        Sm80EpilogueElementwise<1, 1, 1, false>, // StagesC_, StagesD_, FragmentSize_, ReuseSmemC_
        TileShape_MNK,
        ElementAccumulator,
        cutlass::gemm::TagToStrideC_t<GmemLayoutTagC>,
        ElementD,
        cutlass::gemm::TagToStrideC_t<GmemLayoutTagD>,
        FusionCallBacks,
        CopyOpG2R,
        SmemLayoutAtomC,
        CopyOpS2R,
        CopyOpR2G,
        SmemLayoutAtomD,
        CopyOpR2S
      >;
    };
}
