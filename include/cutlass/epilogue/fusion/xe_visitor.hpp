/***************************************************************************************************

 * Copyright (c) 2026 Intel Corporation, All rights reserved.
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
  \brief Visitor tree operations for the Xe epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/arch/copy_xe_2d.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// XeAuxLoad - Loads auxiliary tensor in Xe epilogue using direct G2R (global-to-register) path
//
// Design:
//   - Auto-deduces XE_LOAD_2D copy operation from Element type if CopyOpG2R_ is void
//   - Stores full aux tensor in Params (rank-3: M×N×L)
//   - Creates block 2D copy at runtime from batch slice
//   - Follows xe_epilogue pattern: coordinate tensors with epilogue tile structure
//
// Key differences from NVIDIA TMA:
//   - No descriptors (Intel Xe uses direct memory access)
//   - No shared memory staging (direct G2R path)
//   - Copy operation created per-batch-slice, not stored in Params
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Element,
  class StrideMNL,
  class CopyOpG2R_ = void,
  bool EnableNullptr = true
>
struct XeAuxLoad {
  using SharedStorage = Element;

  struct Arguments {
    Element const* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
  };

  static constexpr int CopyBits = cute::min(sizeof_bits_v<Element>, 64);

  // Define 3D tensor type for aux tensor (M, N, L)
  using TensorAux = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                         Layout<Shape<int,int,int>, StrideMNL>{}));

  struct Params {
    TensorAux mAux;            // 3D tensor (M, N, L)
    Element null_default = Element(0);
    bool use_default = false;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;

    // Create 3D tensor with shape (M, N, L) and stride (stride_M, stride_N, stride_L)
    auto shape_MNL = make_shape(int(M), int(N), int(L));
    auto mAux = make_tensor(make_gmem_ptr(args.ptr_aux),
                           make_layout(shape_MNL, args.dAux));

    bool use_default = false;
    if constexpr (EnableNullptr) {
      use_default = args.ptr_aux == nullptr;
    }

    return Params{mAux, args.null_default, use_default};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
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
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XeAuxLoad() { }

  CUTLASS_HOST_DEVICE
  XeAuxLoad(Params const& params, SharedStorage const&) : params_ptr(&params) { }

  Params const* params_ptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return (params_ptr->use_default && params_ptr->null_default == Element(0));
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  // Callback for epilogue visitor - loads aux data and returns it via visit()
  template <class CTensor, class RTensor, class AuxCopy>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CTensor rw_coord;         // Coordinate tensor (atom_v, atom_m, atom_n, epi_m, epi_n)
    AuxCopy xe_copy_aux;      // Block 2D copy operation
    RTensor tC_rAux;          // Register fragment (SubgroupTensor)
    Params const* params_ptr;

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CTensor rw_coord, AuxCopy xe_copy_aux, RTensor&& tC_rAux, Params const* params_ptr)
      : rw_coord(cute::forward<CTensor>(rw_coord)), xe_copy_aux(xe_copy_aux), tC_rAux(cute::forward<RTensor>(tC_rAux)), params_ptr(params_ptr) { }

    // Load aux data for epilogue tile (epi_m, epi_n)
    CUTLASS_DEVICE void
    previsit(int epi_m, int epi_n, int load_iteration, bool is_producer_load_needed) {
       if constexpr (EnableNullptr) {
         if (params_ptr->use_default) {
           fill(tC_rAux, params_ptr->null_default);
           return;
         }
       }

       // Copy using 5-mode coordinates, slice to 3 modes like epilogue does
       copy(xe_copy_aux, rw_coord(_,_,_,epi_m,epi_n), tC_rAux);
    }

    // Return loaded aux values for epilogue computation
    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const&, int epi_v, int epi_m, int epi_n) {
       Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux.tensor()));  // (EPI_V)
       return tC_rAux_frg(epi_v);
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = args.tile_coord_mnkl;

    auto mAux_batch = params_ptr->mAux(_,_,l_coord);

    // use TiledMMA to properly partition coordinates
    // This ensures we get the correct number of epilogue tiles for the actual problem size
    auto thr_mma = args.tiled_mma.get_slice(args.thread_idx);
    auto tCDgCD = thr_mma.partition_C(args.cD);  // Use workgroup coord tensor from args

    // Calculate MMA tiles per epilogue tile
    using MMATile = decltype(take<0,2>(typename cute::remove_cvref_t<decltype(args.tiled_mma)>::AtomShape_MNK{}));
    
    // Deduce copy operation tile dimensions to match xe_epilogue's logic:
    // - Preferred: 8 rows, 512 bits per row (adjusts to element size)
    // - Use gcd with MMATile to ensure alignment with MMA tile boundaries
    // - Example for fp16 with 16x16 MMA: gcd(8,16)=8, gcd(32,16)=16 -> 8x16 tile
    // - Unless user explicitly provided CopyOpG2R_, then use that instead
    using ActualCopyOpG2R = cute::conditional_t<
      cute::is_void_v<CopyOpG2R_>,
      XE_LOAD_2D<CopyBits, 
                 cute::gcd(8, get<0>(MMATile{})), 
                 cute::gcd(512 / CopyBits, get<1>(MMATile{}))>,
      CopyOpG2R_
    >;
    
    auto mma_per_epi = shape_div(args.epi_tile, MMATile{});

    // Create epilogue-tiled coordinate structure matching xe_epilogue
    auto sg_v_coord = prepend(flat_divide(remove<0>(tCDgCD.layout()), mma_per_epi),
                              get<0>(tCDgCD.layout()));

    auto gAux_epi_layout = append(append(make_identity_layout(args.epi_tile),
                                        get<3>(sg_v_coord)), get<4>(sg_v_coord));
    auto gAux_epi = make_tensor(tCDgCD.data(), gAux_epi_layout);  // (epi_m, epi_n, num_epi_m, num_epi_n)

    // Create copy operation from aux tensor using computed tile dimensions
    auto xe_copy_aux = make_block_2d_copy(ActualCopyOpG2R{}, mAux_batch);
    auto thr_copy_aux = xe_copy_aux.get_slice(args.thread_idx % intel::sg_size);

    // Partition coordinates for epilogue iteration (now with correct tile counts)
    auto tCgAux = thr_copy_aux.partition_S(gAux_epi);  // (atom_v,atom_m,atom_n,epi_m,epi_n)

    // Create register fragment
    auto trAux = thr_copy_aux.partition_sg_fragment_D(gAux_epi(_,_,0,0));  // (atom_v,atom_m,atom_n)


    return ConsumerStoreCallbacks(tCgAux, xe_copy_aux, cute::move(trAux), params_ptr);
  }
};


} // namespace cutlass::epilogue::fusion
