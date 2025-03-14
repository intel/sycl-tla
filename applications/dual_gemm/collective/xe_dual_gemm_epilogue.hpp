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

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

template <
  class DispatchPolicy,
  class CtaTileMNK_,
  class ElementC0_,
  class StrideC0_,
  class ElementC1_,
  class StrideC1_,
  class ElementD0_,
  class StrideD0_,
  class ElementD1_,
  class StrideD1_,
  class FusionCallbacks0_,
  class FusionCallbacks1_,
  class CopyOpG2R_,
  class CopyOpR2G_
>
class DualGemmEpilogue;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class CtaTileMNK_,
  class ElementC0_,
  class StrideC0_,
  class ElementC1_,
  class StrideC1_,
  class ElementD0_,
  class StrideD0_,
  class ElementD1_,
  class StrideD1_,
  class FusionCallbacks0_,
  class FusionCallbacks1_,
  class CopyOpG2R_,
  class CopyOpR2G_
>
class DualGemmEpilogue<
    IntelPVCEpilogue,
    CtaTileMNK_,
    ElementC0_,
    StrideC0_,
    ElementC1_,
    StrideC1_,
    ElementD0_,
    StrideD0_,
    ElementD1_,
    StrideD1_,
    FusionCallbacks0_,
    FusionCallbacks1_,
    CopyOpG2R_,
    CopyOpR2G_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = IntelPVCEpilogue;
  using CtaTileMNK = CtaTileMNK_;
  using FusionCallbacks0 = FusionCallbacks0_;
  using FusionCallbacks1 = FusionCallbacks1_;
  using ElementC0 = ElementC0_;
  using ElementC1 = ElementC1_;
  using ElementAccumulator = ElementC0_;
  using StrideC0 = StrideC0_;
  using StrideC1 = StrideC1_;
  using ElementD0 = ElementD0_;
  using StrideD0 = StrideD0_;
  using ElementD1 = ElementD1_;
  using StrideD1 = StrideD1_;
  using CopyOpG2R = CopyOpG2R_;
  using CopyOpR2G = CopyOpR2G_;

  using ElementC = ElementC0_;
  using StrideC = StrideC0_;
  using ElementD = ElementD0_;
  using StrideD = StrideD0_;

  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks0>::Operation;
  using GmemTiledCopyC = CopyOpG2R;

private:
  constexpr static bool is_source_c0_supported = not cute::is_void_v<ElementC0>;
  constexpr static bool is_source_c1_supported = not cute::is_void_v<ElementC1>;
  constexpr static bool is_destination_d0_supported = not cute::is_void_v<ElementD0> && not cute::is_void_v<CopyOpR2G>;
  constexpr static bool is_destination_d1_supported = not cute::is_void_v<ElementD1> && not cute::is_void_v<CopyOpR2G>;

public:
  using GmemTiledCopyD = cute::conditional_t<is_destination_d0_supported, CopyOpR2G, XE_2D_U32x8x16_ST_N>;
  using ElementOutput = typename FusionCallbacks0::ElementOutput;
  using ElementCompute = typename FusionCallbacks0::ElementCompute;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(StrideC0{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD0{}) == 3, "StrideD must be rank-3: [M, N, L]");

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  using Trait_C = Copy_Traits<GmemTiledCopyC, StrideC0>;
  using XE_Copy_C = decltype(make_tiled_copy(Copy_Atom<Trait_C, ElementC0>{},
                                             Layout<CopyThreadShape>{},
                                             make_layout(shape_div(typename Trait_C::BlockShape{}, CopyThreadShape{}))));
  using Trait_D = Copy_Traits<GmemTiledCopyD, StrideD0>;
  using XE_Copy_D = decltype(make_tiled_copy(Copy_Atom<Trait_D, ElementD0>{},
                                             Layout<CopyThreadShape>{},
                                             make_layout(shape_div(typename Trait_D::BlockShape{}, CopyThreadShape{}))));

  using EmptyType = cute::tuple<>;
  using SmemCStorage = EmptyType;
  using SmemDStorage = EmptyType;

  struct TensorStorageImpl: cute::tuple<SmemCStorage, SmemDStorage> {
    using FusionStorage = typename FusionCallbacks0::SharedStorage;
    FusionStorage thread;
  };

  struct SharedStorage {
    using TensorStorage = TensorStorageImpl;

    TensorStorage tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  using TensorC = decltype(make_tensor(make_gmem_ptr(static_cast<ElementC1 const*>(nullptr)), make_shape(0,0,0), StrideC1{}));   //(m, n)
  using TensorD = decltype(make_tensor(make_gmem_ptr(static_cast<ElementD1 const*>(nullptr)), make_shape(0,0,0), StrideD1{}));   //(m, n)

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks0::Arguments thread0{};
    typename FusionCallbacks1::Arguments thread1{};
    ElementC0 const* ptr_C0;
    StrideC0 dC0;
    ElementC1 const* ptr_C1;
    StrideC1 dC1;
    ElementD0* ptr_D0;
    StrideD0 dD0;
    ElementD1* ptr_D1;
    StrideD1 dD1;
  };

  // Device side epilogue params
  struct Params {
    typename FusionCallbacks0::Params thread0{};
    typename FusionCallbacks1::Params thread1{};
    XE_Copy_C xe_load_c;
    XE_Copy_D xe_store_d;
    TensorC mC1;
    TensorD mD1;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    XE_Copy_C xe_load_c = {};
    if constexpr (is_source_c0_supported) {
      auto mC0 = make_tensor(make_gmem_ptr(static_cast<ElementC0 const*>(args.ptr_C0)),
                            make_layout(make_shape(M, N, L), args.dC0));
      xe_load_c = make_tiled_copy(Copy_Atom<Trait_C, ElementC0>{}.with(mC0),
                                  Layout<CopyThreadShape>{},
                                  make_layout(shape_div(typename Trait_C::BlockShape{}, CopyThreadShape{})));
    }

    TensorC mC1 = {};
    if constexpr (is_source_c1_supported) {
      mC1 = make_tensor(make_gmem_ptr(static_cast<ElementC1 const*>(args.ptr_C1)),
                            make_layout(make_shape(M, N, L), args.dC1));
    }

    XE_Copy_D xe_store_d = {};
    if constexpr (is_destination_d0_supported) {
      auto mD0 = make_tensor(make_gmem_ptr(static_cast<ElementD0 const*>(args.ptr_D0)),
                            make_layout(make_shape(M, N, L), args.dD0));
      xe_store_d = make_tiled_copy(Copy_Atom<Trait_D, ElementD0>{}.with(mD0),
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename Trait_D::BlockShape{}, CopyThreadShape{})));
    }

    TensorD mD1 = {};
    if constexpr (is_destination_d1_supported) {
      mD1 = make_tensor(make_gmem_ptr(static_cast<ElementD1 const*>(args.ptr_D1)),
                      make_layout(make_shape(M, N, L), args.dD1));
    }

    return {
      FusionCallbacks0::to_underlying_arguments(problem_shape, args.thread0, workspace),
      FusionCallbacks1::to_underlying_arguments(problem_shape, args.thread1, workspace),
      xe_load_c,
      xe_store_d,
      mC1,
      mD1
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, 
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  DualGemmEpilogue(Params const& params_, TensorStorage const& shared_storage_)
      : params(params_), fusion_callbacks0(params_.thread0, shared_storage_.thread), 
        fusion_callbacks1(params_.thread1, shared_storage_.thread) {}

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return fusion_callbacks0.is_producer_load_needed();
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class Accumulator0,
    class Accumulator1,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      Accumulator0& accumulators0,
      Accumulator1& accumulators1,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem) {
    
    (void) tiled_mma;
    (void) residue_mnk;
    (void) smem;
    using namespace cute;

    static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
    static_assert(cute::rank(StrideC0{}) == 3, "StrideC must be rank-3: [M, N, L]");
    static_assert(cute::rank(StrideD0{}) == 3, "StrideD must be rank-3: [M, N, L]");

    using MmaAtomShape = typename TiledMma::AtomShape_MNK;
    static constexpr auto BLK_M = get<0>(CtaTileMNK{});
    static constexpr auto BLK_N = get<1>(CtaTileMNK{});
    static constexpr auto BLK_K = get<2>(CtaTileMNK{});
    // static_assert(is_same_v<typename TiledMma::ThrLayoutVMNK, int>, "assertation fail");
    static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
    
    static_assert(
      BLK_M % ATOM_M == 0 &&
      BLK_N % ATOM_N == 0 &&
      BLK_K % ATOM_K == 0,
      "expected CTATileMNK to be evenly divided by TiledMma::ThrLayoutVMNK");
    static constexpr auto SG_M = BLK_M / ATOM_M;
    static constexpr auto SG_N = BLK_N / ATOM_N;
    static constexpr auto SG_K = BLK_K / ATOM_K;
    using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

    static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape()); // A frags per sub_group
    static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape()); // B frags per sub_group

    static constexpr int FragmentSize = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    auto m_sg = get_sub_group_id() / ATOM_N;
    auto n_sg = get_sub_group_id() % ATOM_N;

    using EpilogueTile = decltype(get<0>(params.xe_store_d.get_layoutS_MN()).shape());

    auto sg_local_m_coord = get_sub_group_id() / ATOM_N;
    auto sg_local_n_coord = get_sub_group_id() % ATOM_N;

    auto sg_m_coord = m_coord * ATOM_M + sg_local_m_coord;
    auto sg_n_coord = n_coord * ATOM_N + sg_local_n_coord;
    auto sg_coord = make_coord(sg_m_coord, sg_n_coord, k_coord, l_coord);

    bool is_C0_load_needed = is_source_c0_supported && fusion_callbacks0.is_C_load_needed();
    bool is_C1_load_needed = is_source_c1_supported && fusion_callbacks1.is_C_load_needed();
    
    // Represent the full output tensor
    Tensor mD_mnl = params.xe_store_d.get_pvc_tensor(make_shape(M,N,L));

    // Tile the output tensor per WG and select the tile for current WG
    Tensor g_wg_D = local_tile(mD_mnl, take<0,2>(CtaTileMNK{}), make_coord(m_coord,n_coord,l_coord));  // (BLK_M,BLK_N)
    
    // Tile the output tensor per SG and select tile for the current SG
    Tensor gD = local_tile(g_wg_D, take<0,2>(SubgroupTileShape{}), make_coord(m_sg,n_sg));            // (SG_M,SG_N)

    auto thread_xe_store_d = params.xe_store_d.get_thread_slice(thread_idx);
    Tensor tCgD = thread_xe_store_d.partition_D(gD);

    Tensor trC0 = make_tensor<typename TiledMma::ValTypeC>(Shape<Int<FragmentSize>>{});
    Tensor trC1 = make_tensor<typename TiledMma::ValTypeC>(Shape<Int<FragmentSize>>{});
    Tensor trD0 = make_tensor<typename TiledMma::ValTypeD>(Shape<Int<FragmentSize>>{});
    Tensor trD1 = make_tensor<typename TiledMma::ValTypeD>(Shape<Int<FragmentSize>>{});

    // Because Sm90 uses shared memory, they are not tied to using the same accumulator values
    // for MMA and Epilogue. But because we are operating directly in the accumulators, we need to be
    // sure that we are operating on the same values.
    ThrCopy thread_g2r = params.xe_load_c.get_slice(thread_idx);

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD = local_tile(mD_crd, take<0,2>(SubgroupTileShape{}), make_coord(sg_m_coord, sg_n_coord));
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    Tensor tRS_cD_mn = thread_g2r.partition_S(flat_divide(cD_mn, EpilogueTile{}));     // (G2R,G2R_M,G2R_N,EPI_M,EPI_N)

    Tensor tRS_cD = make_counting_tensor(tRS_cD_mn.layout());                          // (G2R,G2R_M,G2R_N,EPI_M,EPI_N)

    // Get the fusion callbacks
    // Arguments passed here relate to sub-group tiles, rather than CTA (work-group) tiles
    constexpr bool RefSrc = true;
    auto residue_mn = make_coord(M, N); //TODO(Codeplay): this is not correct
    auto cst_args0 = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      SubgroupTileShape{},
                      sg_coord,
                      tiled_mma,
                      EpilogueTile{},
                      params.xe_store_d,
                      cD,
                      residue_mn,
                      tRS_cD,
                      residue_mn,
                      trC0,
                      thread_idx,
                    };
    auto cst_args1 = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      SubgroupTileShape{},
                      sg_coord,
                      tiled_mma,
                      EpilogueTile{},
                      params.xe_store_d,
                      cD,
                      residue_mn,
                      tRS_cD,
                      residue_mn,
                      trC1,
                      thread_idx,
                    };
    auto cst_callbacks0 = fusion_callbacks0.template get_consumer_store_callbacks<RefSrc>(cst_args0);
    auto cst_callbacks1 = fusion_callbacks1.template get_consumer_store_callbacks<RefSrc>(cst_args1);

    cst_callbacks0.begin();
    cst_callbacks1.begin();

    auto acc_frag0 = recast<Array<ElementOutput, FragmentSize>>(accumulators0);
    auto acc_frag1 = recast<Array<ElementOutput, FragmentSize>>(accumulators1);
    auto trD_frag0 = recast<Array<ElementOutput, FragmentSize>>(trD0);
    auto trD_frag1 = recast<Array<ElementOutput, FragmentSize>>(trD1);

    constexpr int ValuesLoaded =
      FragsM * FragsN * FragmentSize * SubgroupSize * ATOM_M * ATOM_N * ATOM_K;
    constexpr int MN = get<0>(CtaTileMNK{}) * get<1>(CtaTileMNK{});
    static_assert(ValuesLoaded == MN, "the total elements loaded by all threads should be the same as MxN" );
    
    auto synchronize = [&] () {};
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < FragsN; epi_n++) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < FragsM; epi_m++) {

        if (is_C0_load_needed) {
          //cordinates for C and D are the same
          copy(params.xe_load_c, tCgD(_, epi_m, epi_n), trC0);
        }

        if (is_C1_load_needed) {
          //cordinates for C and D are the same
          copy(params.xe_load_c.with(params.mC1), tCgD(_, epi_m, epi_n), trC1);
        }

        cst_callbacks0.previsit(epi_m, epi_n, 0, is_C0_load_needed);
        cst_callbacks1.previsit(epi_m, epi_n, 0, is_C1_load_needed);

        auto acc_frag_mn0 = acc_frag0(_, epi_m, epi_n);
        auto acc_frag_mn1 = acc_frag1(_, epi_m, epi_n);

        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size<0>(trD_frag0); ++epi_v) {
          trD_frag0(epi_v) = cst_callbacks0.visit(acc_frag_mn0(epi_v), epi_v, epi_m, epi_n);
          trD_frag1(epi_v) = cst_callbacks1.visit(acc_frag_mn1(epi_v), epi_v, epi_m, epi_n);
        }
        
        if constexpr (is_destination_d0_supported) {
          copy(params.xe_store_d, trD0, tCgD(_, epi_m, epi_n));
        }

        if constexpr (is_destination_d1_supported) {
          copy(params.xe_store_d.with(params.mD1), trD1, tCgD(_, epi_m, epi_n));
        }
      }
    }

    cst_callbacks0.end();
  }

private:
  Params const& params;
  FusionCallbacks0 fusion_callbacks0;
  FusionCallbacks1 fusion_callbacks1;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
