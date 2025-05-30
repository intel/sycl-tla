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
  \brief Visitor tree RMSNorm fusion operation for the Intel PVC epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "xe_visitor_softmax.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementWeight,
  class ElementOutput,
  class ElementCompute,
  class CopyOpR2G,
  FloatRoundStyle RoundStyle
>
struct XeRMSNormRowReduction
{
public:
  static constexpr int FragmentSize = 8;
  static constexpr auto Tile_M = get<0>(CtaTileShapeMNK{});
  static constexpr auto Tile_N = get<1>(CtaTileShapeMNK{});
  static constexpr auto Epi_M = get<0>(EpilogueTile{});
  static constexpr auto Epi_N = get<1>(EpilogueTile{});
  static constexpr auto Sg_M = Tile_M / Epi_M;
  static constexpr auto Sg_N = Tile_N / Epi_N;
  static constexpr auto Sg_Nums = Sg_M * Sg_N;

  using Trait_Output = Copy_Traits<CopyOpR2G>;
  using XE_Copy_output = decltype(make_tiled_copy(Copy_Atom<Trait_Output, ElementOutput>{}
                                             .with(static_cast<ElementOutput const*>(nullptr),int32_t(0), int32_t(0)),
                                             Layout<Shape<_1, Int<IntelXeXMX16::SubgroupSize>>>{},
                                             make_layout(make_shape(get<0>(typename Trait_Output::BlockShape{}),
                                                                    get<1>(typename Trait_Output::BlockShape{}) / Int<IntelXeXMX16::SubgroupSize>{}))));

  struct SharedStorage { };

  struct Arguments {
    ElementOutput* ptr_output;
    ElementWeight const*ptr_weight;
    const float eps;
    // StrideOutput dOutput;
  };

  struct Params {
    XE_Copy_output xe_store_output;
    ElementWeight const *weight;
    float eps;
    int inner_dim;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    XE_Copy_output output = make_tiled_copy(Copy_Atom<Copy_Traits<CopyOpR2G>, ElementOutput>{}.with(
                            args.ptr_output, M, N),
                            Layout<Shape<_1, Int<IntelXeXMX16::SubgroupSize>>>{},
                            make_layout(make_shape(get<0>(typename XE_Copy_output::BlockShape{}),
                                                   get<1>(typename XE_Copy_output::BlockShape{}) / Int<IntelXeXMX16::SubgroupSize>{})));
    return {output, args.ptr_weight, args.eps, N};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = problem_shape;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
    // Cross CTA reduction is not possible because there is no guarantee that all CTAs run
    // concurrently.
    // Cross epilogue tile reduction is possible, but re-visiting and applying reduction
    // to accumulators is only possible for the current epilogue tile.
    auto [epi_M, epi_N] = EpilogueTile{};
    return N <= tile_N;
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

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  XeRMSNormRowReduction() { }

  CUTLASS_HOST_DEVICE
  XeRMSNormRowReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class RTensor, class CoordTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(RTensor&& res_tensor, CoordTensor&& coord, Params const& params)
      : res_tensor(cute::forward<RTensor>(res_tensor)),
        coord(cute::forward<CoordTensor>(coord)),
        params(params) {}

    RTensor res_tensor;
    CoordTensor coord;
    Params const& params;
    template <typename ElementInput, typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {

      return frg_acc;
    }

    template<class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
        const float eps = params.eps;
        auto sg = syclcompat::get_nd_item<1>().get_sub_group();
        auto group = syclcompat::get_nd_item<1>().get_group()[0];
        auto group_id = group;
        auto sg_group_id = sg.get_group_id();
        auto sg_local_id = sg.get_local_id()[0];
        if(is_last_iteration) {
            for(int epi_v = 0; epi_v < visit_results(0).size(); epi_v++) {
                res_tensor(epi_v, epi_m, epi_n) = visit_results(0)[epi_v];
            }

            constexpr auto vec_size = min(Epi_M, Sg_N);
            constexpr auto vec_folds = Epi_M / vec_size;
            auto smem = syclcompat::local_mem<float[Sg_Nums * vec_size]>();
            Tensor stensor = make_tensor(make_smem_ptr(smem), make_shape(Int<vec_size>{}, Int<Sg_N>{}, Int<Sg_M>{}));
            auto wgt_ptr=params.weight;
            Tensor res =
                make_tensor(static_cast<decltype(res_tensor) &&>(res_tensor).data(),
                            make_shape(Int<vec_size>{}, Int<vec_folds>{}, Int<Epi_N / IntelXeXMX16::SubgroupSize>{}));
            // square
            Tensor pow2_buff = make_tensor_like<float>(res);
            CUTLASS_PRAGMA_UNROLL
            for (int loop = 0; loop < vec_folds; loop++) {
                auto loop_t = res(_, loop, _);
                auto pow2_t = pow2_buff(_, loop, _);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < Epi_N / IntelXeXMX16::SubgroupSize; i++) {
                    auto x_vec = loop_t(_, i);
                    auto p2_vec = pow2_t(_, i);
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < vec_size; j++) {
                        p2_vec(j) = x_vec(j) * x_vec(j);
                    }
                }
            }
            int gx = syclcompat::global_id::x() % 256;
            auto gid = gx / 16 * 32 + gx % 16;
            const float rev_dim = 1 / (float)params.inner_dim;
            CUTLASS_PRAGMA_UNROLL
            for (int loop = 0; loop < vec_folds; loop++) {
                auto loop_t = res(_, loop, _);
                auto pow2_t = pow2_buff(_, loop, _);
                Tensor group_sum = make_tensor<float>(make_shape(Int<vec_size>{}));
                group_reduce_sum<Sg_N>(stensor, pow2_t, group_sum);
                Tensor rms = make_tensor<float>(make_shape(Int<vec_size>{}));
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < vec_size; ++i) {
                    rms(i) = pow(group_sum(i) * rev_dim + eps, -0.5);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < Epi_N / IntelXeXMX16::SubgroupSize; i++) {
                    const float wgt_per_col = (float)wgt_ptr[gid + i * IntelXeXMX16::SubgroupSize];
                    auto rmsnorm_vec = loop_t(_, i);
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < vec_size; j++) {
                        rmsnorm_vec(j) = rmsnorm_vec(j) * rms(j) * wgt_per_col;
                    }
                }
            }

            copy(params.xe_store_output, res_tensor, coord);
        }
        else {
            for(int epi_v = 0; epi_v < visit_results(0).size(); epi_v++) {
                res_tensor(epi_v, epi_m, epi_n) = visit_results(0)[epi_v];
            }
        }
    }
  };

  template <
  bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
  class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    using MmaAtomShape = typename decltype(args.tiled_mma)::AtomShape_MNK;
    static constexpr int FragsM = get<0>(EpilogueTile{}) / get<0>(MmaAtomShape()); // A frags per sub_group
    static constexpr int FragsN = get<1>(EpilogueTile{}) / get<1>(MmaAtomShape()); // B frags per sub_group
    Tensor res = make_tensor<ElementOutput>(Shape<Int<FragmentSize>, Int<FragsM>, Int<FragsN>>{});

    auto [sg_m_coord, sg_n_coord, k_coord, l_offset] = args.tile_coord_mnkl;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    Tensor mAux_mnl = cute::get_xe_tensor(make_shape(M,N,L));
    // Tiling is done differently than in epilogue as we get in coordinates of subgroup in kernel
    Tensor gAux = local_tile(mAux_mnl, select<0,1>(EpilogueTile{}), make_coord(sg_m_coord,sg_n_coord,l_offset));
    Tensor tCgAux = args.tiled_copy.get_thread_slice(args.thread_idx).partition_D(gAux);
    return ConsumerStoreCallbacks<decltype(res),decltype(tCgAux)>(
      cute::move(res),
      cute::move(tCgAux),
      params);
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
