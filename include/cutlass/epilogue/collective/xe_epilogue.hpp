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

/////////////////////////////////////////////////////////////////////////////////////////////////

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
    IntelXeL1Staged,
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
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = IntelXeL1Staged;
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
  using ElementOutput = ElementD;
  using ElementCompute = ElementAccumulator;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  static_assert(std::is_same_v<CopyOpS2R, void>, "Copy operation to shared memory is not supported");
  static_assert(std::is_same_v<CopyOpR2S, void>, "Copy operation to shared memory is not supported");
  static_assert(std::is_same_v<SmemLayoutAtomC, void>, "Copy operation to shared memory is not supported");
  static_assert(std::is_same_v<SmemLayoutAtomD, void>, "Copy operation to shared memory is not supported");

//remember this PR https://github.com/intel/sycl-tla/pull/565/files
private:
  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;

  constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();

public:

  using EmptyType = cute::tuple<>;
  using SmemCStorage = EmptyType;
  using SmemDStorage = EmptyType;

  struct TensorStorageImpl: cute::tuple<SmemCStorage, SmemDStorage> {
    using FusionStorage = typename FusionCallbacks::SharedStorage;
    FusionStorage thread;
  };

  struct SharedStorage {
    using TensorStorage = TensorStorageImpl;

    TensorStorage tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C;
    StrideC dC;
    ElementD* ptr_D;
    StrideD dD;
  };

  // Device side epilogue params
  struct Params {
    typename FusionCallbacks::Params thread{};
    ElementC const* ptr_C;
    ElementD* ptr_D;
    int M, N, K, L; 
    StrideC dC;
    StrideD dD;
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

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      args.ptr_C,
      args.ptr_D,
      M, N, K, L,
      args.dC,
      args.dD
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
      ProblemShape const& problem_shapes,
      Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    bool fusion_implementable = true;

    if constexpr (is_destination_supported) {
      constexpr int min_aligned_elements_D = copy_alignment_bits / sizeof_bits<ElementD>::value;
      implementable &= cutlass::detail::check_alignment<min_aligned_elements_D>(cute::make_shape(M,N,L), args.dD);
      if (L > 1) {
        constexpr int min_batch_aligned_elements_D = batch_alignment_bits / sizeof_bits<ElementD>::value;
        implementable &= get<2>(args.dD) % min_batch_aligned_elements_D == 0;
      }
    }

    if constexpr (is_source_supported) {
      constexpr int min_aligned_elements_C = copy_alignment_bits / sizeof_bits<ElementC>::value;
      implementable &= cutlass::detail::check_alignment<min_aligned_elements_C>(cute::make_shape(M,N,L), args.dC);
      if (L > 1) {
        constexpr int min_batch_aligned_elements_C = batch_alignment_bits / sizeof_bits<ElementC>::value;
        implementable &= get<2>(args.dC) % min_batch_aligned_elements_C == 0;
      }
    }

    fusion_implementable = fusion_implementable && FusionCallbacks::can_implement(problem_shape_MNKL, args.thread);

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for XE 2D copy.\n");
    }

    if (!fusion_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for FusionCallbacks.\n");
    }

    return implementable && fusion_implementable;
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
    class TiledMma
  >
  CUTLASS_DEVICE void
  operator() (
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      Accumulator accumulators, 
      TiledMma tiled_mma,
      int thread_idx) {

    using namespace cute;

    static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

    using MmaAtomShape = typename TiledMma::AtomShape_MNK;
    static constexpr auto BLK_M = get<0>(CtaTileMNK{});
    static constexpr auto BLK_N = get<1>(CtaTileMNK{});
    static constexpr auto BLK_K = get<2>(CtaTileMNK{});
  
    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    
    // Workgroup coordinates (no subgroup indexing needed)
    auto wg_coord = make_coord(m_coord, n_coord, k_coord, l_coord);
    auto batch_idx = get<3>(wg_coord);

    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();
    auto mC = make_tensor(make_gmem_ptr(params.ptr_C), make_layout(make_shape(params.M, params.N, params.L),  params.dC));
    auto mD = make_tensor(make_gmem_ptr(params.ptr_D), make_layout(make_shape(params.M, params.N, params.L),  params.dD));

    auto copy_c = [&]() {
      if constexpr (!std::is_void_v<CopyOpG2R>) {
        return make_block_2d_copy_CD(CopyOpG2R{}, tiled_mma, mC(_,_,batch_idx));
      } else {
        return make_block_2d_copy_C(tiled_mma, mC(_,_,batch_idx));
      }
    }();
    auto copy_d = [&]() {
      if constexpr (!std::is_void_v<CopyOpR2G>) {
        return make_block_2d_copy_CD(CopyOpR2G{}, tiled_mma, mD(_,_,batch_idx));
      } else {
        return make_block_2d_copy_D(tiled_mma, mD(_,_,batch_idx));
      }
    }();


    // Represent the full output tensor
    Tensor mD_mnl = cute::get_xe_tensor(make_shape(M,N,L));

    // Tile the output tensor for the current workgroup
    Tensor gD = local_tile(mD_mnl, take<0,2>(CtaTileMNK{}), remove<2>(wg_coord));  // (BLK_M,BLK_N) // change made

    // Get thread-level partitioning across the entire workgroup tile
    auto thread_xe_load_c = copy_c.get_thread_slice(thread_idx);
    Tensor tCgC = thread_xe_load_c.partition_S(gD);

    auto thread_xe_store_d = copy_d.get_thread_slice(thread_idx);
    Tensor tCgD = thread_xe_store_d.partition_D(gD);

    // Create tensors sized for workgroup-level operation
    Tensor trC = make_tensor<typename TiledMma::ValTypeC>(tCgC.shape());
    Tensor trD_compute = make_tensor<ElementCompute>(tCgD.shape());
    Tensor trD = make_tensor<ElementOutput>(tCgD.shape());

    ThrCopy thread_g2r = copy_c.get_slice(thread_idx);

    auto mn_shape = shape(typename decltype(copy_d)::Tiler_MN{});

    // OOB predication for tile quantization "residue"
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));
    Tensor cD = local_tile(mD_crd, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord));
    Tensor tRS_cD = thread_g2r.partition_S(flat_divide(cD, mn_shape));

    Tensor tRS_cD_coord = make_coord_tensor(tRS_cD.layout());

    // Get fusion callbacks at workgroup level
    constexpr bool RefSrc = true;
    auto residue_mn = make_coord(M, N);
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      CtaTileMNK{},  // Use workgroup tile shape
                      wg_coord,      // Use workgroup coordinates
                      tiled_mma,
                      mn_shape,
                      copy_d,
                      cD,
                      residue_mn,
                      tRS_cD_coord,
                      residue_mn,
                      trC,
                      thread_idx,
                    };
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    auto synchronize = [&] () {};

    cst_callbacks.begin();
    // Load C tile if needed (distributed across all threads in workgroup)
    if (is_C_load_needed) {
      copy(copy_c, tCgC, trC);
    }

    // Single previsit for entire workgroup tile
    cst_callbacks.previsit(0, 0, 0, is_C_load_needed);   
    
    static constexpr int FragmentSize = get<0>(MmaAtomShape()) * get<1>(MmaAtomShape());
    constexpr int num_fragments = size(accumulators) / FragmentSize;

    CUTLASS_PRAGMA_UNROLL
    for (int epi_v = 0; epi_v < num_fragments; ++epi_v) {
      // Extract fragment
      Array<ElementAccumulator, FragmentSize> frg_acc;
      CUTLASS_PRAGMA_UNROLL
      for (int f = 0; f < FragmentSize; ++f) {
        frg_acc[f] = accumulators(epi_v * FragmentSize + f);
      }
      
      // Process fragment
      auto result_frg = cst_callbacks.visit(frg_acc, epi_v, 0, 0);
      
      // Store results
      CUTLASS_PRAGMA_UNROLL
      for (int f = 0; f < FragmentSize; ++f) {
        trD_compute(epi_v * FragmentSize + f) = result_frg[f];
      }
    }

    cst_callbacks.reduce(nullptr, synchronize, 0, 0, true, trD_compute);

    if constexpr (is_destination_supported) {
      // Convert fragments using NumericArrayConverter
      constexpr int num_fragments_trD_compute = size(trD_compute) / FragmentSize;
      using Converter = cutlass::NumericArrayConverter<ElementOutput, ElementCompute, FragmentSize>;
      Converter converter{};
      
      CUTLASS_PRAGMA_UNROLL
      for (int epi_v = 0; epi_v < num_fragments_trD_compute; ++epi_v) {
        // Extract compute fragment
        Array<ElementCompute, FragmentSize> trD_compute_frag;
        CUTLASS_PRAGMA_UNROLL
        for (int f = 0; f < FragmentSize; ++f) {
          trD_compute_frag[f] = trD_compute(epi_v * FragmentSize + f);
        }
        
        // Convert fragment
        auto trD_frag = converter(trD_compute_frag);
        
        // Store converted fragment
        CUTLASS_PRAGMA_UNROLL
        for (int f = 0; f < FragmentSize; ++f) {
          trD(epi_v * FragmentSize + f) = trD_frag[f];
          
        }
      }
      
      copy(copy_d, trD, tCgD);
 }

  cst_callbacks.end();  
    
}

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////