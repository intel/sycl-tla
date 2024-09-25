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
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/fusion/sm80_callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/tensor.hpp"


namespace cutlass::epilogue::collective {

template <
  class CtaTileMNK_,
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpG2R_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpR2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_
>
class CollectiveEpilogue<
    Sm80EpilogueElementwise<1, 1, 1, false>, // StagesC_, StagesD_, FragmentSize_, ReuseSmemC_
    CtaTileMNK_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpG2R_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpR2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_
> 
{
public:
  using DispatchPolicy = Sm80EpilogueElementwise<1, 1, 1, false>; // StagesC_, StagesD_, FragmentSize_, ReuseSmemC_
  using CtaTileMNK = CtaTileMNK_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpG2R = CopyOpG2R_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpR2G = CopyOpR2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;

  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyC = CopyOpG2R;
  using GmemTiledCopyD = CopyOpR2G;
  using ElementOutput = typename FusionCallbacks::ElementOutput;
  using ElementCompute = typename FusionCallbacks::ElementCompute;

  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  public:
    // This SM80 Epilogue type does not use shared memory
    struct TensorStorageImpl: cute::tuple<cute::tuple<>, cute::tuple<>> {
      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    };

    struct SharedStorage {
      using TensorStorage = TensorStorageImpl;

      TensorStorage tensors;
    };

  using TensorStorage = typename SharedStorage::TensorStorage;
    struct Arguments {
      typename FusionCallbacks::Arguments thread{};
      ElementC const* ptr_C;
      StrideC dC;
      ElementD const* ptr_D;
      StrideD dD;
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(
        [[maybe_unused]] ProblemShape const& _,
        Arguments const& args,
        [[maybe_unused]] void* workspace) {
      return args;
    }

    // No workspace as this is linear combination only
    template <class ProblemShape>
    static size_t
    get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
      return 0;
    }

    template <class ProblemShape>
    static cutlass::Status
    initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
      CudaHostAdapter* cuda_adapter = nullptr) {
      return cutlass::Status::kSuccess;
    }

    template<class ProblemShape>
    static bool
    can_implement(
        [[maybe_unused]] ProblemShape const& problem_shape,
        [[maybe_unused]] Arguments const& args) {
      return true;
    }

    CUTLASS_HOST_DEVICE
    CollectiveEpilogue(Params const& params_, TensorStorage const& shared_storage_)
        : params(params_), fusion_callbacks(params_.thread, shared_storage_.thread) {}

    CUTLASS_DEVICE
    bool
    is_producer_load_needed() const {
      return fusion_callbacks.is_producer_load_needed();
    }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class Accumulator,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
    ProblemShapeMNKL problem_shape_mnkl,
    TileShapeMNK tile_shape_MNK,
    TileCoordMNKL tile_coord_mnkl,
    Accumulator accumulators, 
    TiledMma tiled_mma,
    ResidueMNK residue_mnk,
    int thread_idx,
    [[maybe_unused]] char* smem) 
    {
      using namespace cute;

      static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
      static_assert(is_static<TileShapeMNK>::value, "ThreadBlock tile shape must be static");
      static_assert(cute::rank(TileShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
      static_assert(cute::rank(TileCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

      auto M = get<0>(problem_shape_mnkl);
      auto N = get<1>(problem_shape_mnkl);
      auto L = get<3>(problem_shape_mnkl);

      auto stride_c = params.dC;
      auto stride_d = params.dD;

      Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), stride_c);
      Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), stride_d);
      Tensor gC_mnl = local_tile(mC_mnl, tile_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});
      Tensor gD_mnl = local_tile(mD_mnl, tile_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});
      
      auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
      Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);
      Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);

      auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
      Tensor tCgD = thr_mma.partition_C(gD);
      Tensor tCgC = thr_mma.partition_C(gC);

      auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
      Tensor tCcD = thr_mma.partition_C(cD);

      if (fusion_callbacks.is_C_load_needed()) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accumulators); ++i) {
          if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
            tCgD(i) = epilogue_op(accumulators(i), tCgC(i));
          }
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accumulators); ++i) {
          if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
            tCgD(i) = epilogue_op(accumulators(i));
          }
        }
      }
    }

    private:
      Params const& params;
      FusionCallbacks fusion_callbacks;
  };
}
