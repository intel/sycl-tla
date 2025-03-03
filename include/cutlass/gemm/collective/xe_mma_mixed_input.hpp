/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopIntelPVCMixedPrecision<Stages>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelPVCMixedPrecision<Stages>;
  using WorkgroupTileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});
  
  static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
  using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

  static constexpr size_t cacheline_bytes = 64;
  static constexpr auto block_size_w_a = cute::min(SG_K, cacheline_bytes * sizeof_bits_v<ElementA>/ sizeof_bits_v<int8_t>);
  static constexpr auto block_size_w_b = cute::min(SG_N, cacheline_bytes * sizeof_bits_v<ElementB>/ sizeof_bits_v<int8_t>);
  static constexpr auto nums_block_w_a = ceil_div(SG_K, block_size_w_a);
  static constexpr auto nums_block_w_b = ceil_div(SG_N, block_size_w_b);
  using PrefetchAThrShape = Shape<Int<ATOM_N /cute::gcd(ATOM_N, nums_block_w_a)>, Int<cute::gcd(ATOM_N, nums_block_w_a)>>;
  using PrefetchBThrShape = Shape<Int<ATOM_M /cute::gcd(ATOM_M, nums_block_w_b)>, Int<cute::gcd(ATOM_M, nums_block_w_b)>>;
  using PrefetchATileSize = decltype(ceil_div(Shape<Int<SG_M>, Int<SG_K>>{},PrefetchAThrShape{}));
  using PrefetchBTileSize = decltype(ceil_div(Shape<Int<SG_K>, Int<SG_N>>{},PrefetchBThrShape{}));
  
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});
  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;

  using XE_Prefetch_A = decltype(cute::detail::prefetch_selector<PrefetchATileSize, ElementA>());
  using XE_Prefetch_B = decltype(cute::detail::prefetch_selector<PrefetchBTileSize, ElementB>());

  using  TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), make_shape(0,0,0), StrideA{}));   //(m, k)
  using  TensorNKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), make_shape(0,0,0), StrideB{}));   //(n, k)
 
  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  struct Params {
    TensorMKL mA;
    TensorNKL mB;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    auto [M,N,K,L] = problem_shape;

    auto mA_mkl = make_tensor(make_gmem_ptr(static_cast<ElementA const*>(args.ptr_A)),
                              make_layout(make_shape(M, K, L), args.dA));

    auto mB_nkl = make_tensor(make_gmem_ptr(static_cast<ElementB const*>(args.ptr_B)),
                              make_layout(make_shape(N, K, L), args.dB));

    return Params{mA_mkl, mB_nkl};
  }

  // Helper functions to select packing for conversion
  template <class SrcType,
            class DstType,
            int Cosize>
  struct select_packing { // Naive packing policy
    static constexpr auto value() {
      return Int<cute::gcd(Cosize, 32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>))>{};
    }
  };

  /// Utilities to transform A.
  template <class DstType,
            class EngineIn,
            class LayoutIn>
  CUTLASS_DEVICE
  auto transform_if_needed(Tensor<EngineIn, LayoutIn> const& in) {

    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);

    using SrcType = typename EngineIn::value_type;

    if constexpr (std::is_same_v<SrcType, DstType>) {
      return in;
    } else if constexpr (sizeof_bits_v<SrcType> < 8) {
      auto out = make_fragment_like<DstType>(in);

      // TODO: hard code for test
      for (int i = 0; i < out.size(); i++) {
        out[i] = static_cast<DstType>(in[i].get());
      }
      return out;
    } else {
      auto out = make_fragment_like<DstType>(in);

      auto const& src = in(_, _, _);
      auto const& dst = out(_, _, _);
      auto pSrc = const_cast<SrcType*>(raw_pointer_cast(src.data()));
      auto pDst = const_cast<DstType*>(raw_pointer_cast(dst.data()));
      constexpr int num_elements = decltype(size(src))::value;

      constexpr int pack = decltype(select_packing<SrcType, DstType, num_elements>::value())::value;
      using Converter = cutlass::NumericArrayConverter<DstType, SrcType, pack, cutlass::FloatRoundStyle::round_to_nearest>;
      using SrcArray = cutlass::Array<SrcType, pack>;
      using DstArray = cutlass::Array<DstType, pack>;
      constexpr int iters = num_elements / pack;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < iters; ++i) {
        SrcArray const* pSrcArr = reinterpret_cast<SrcArray const*>(pSrc) + i;
        DstArray* pDstArr = reinterpret_cast<DstArray*>(pDst) + i;
        *pDstArr = Converter::convert(*pSrcArr);
      }
      return out;
    }
  }

  template <class EngineIn,
            class LayoutIn>
  static auto shuffle_A_if_needed(Tensor<EngineIn, LayoutIn> const& tensor) {
    static_assert(rank(LayoutIn{}) == 3);
    static_assert(is_rmem<EngineIn>::value, "Input tensor for shuffle must come from registers");

    using tensor_t = Tensor<EngineIn, LayoutIn>;

    if constexpr (sizeof_bits_v<typename tensor_t::value_type> >= 8) {
      return tensor;
    } else {
      // H  -->  height
      // W  -->  width
      static constexpr auto DPAS = decltype(size<0>(tensor))::value;
      static constexpr auto H = decltype(size<1>(tensor))::value;
      static constexpr auto W = decltype(size<2>(tensor))::value;
      static constexpr auto copy_H = decltype(size<0>(get<1>(tensor.layout())))::value;
      static constexpr auto copy_W = decltype(size<0>(get<2>(tensor.layout())))::value;

      auto sg = syclcompat::get_nd_item<1>().get_sub_group();

      int id = int(ThreadIdxX()) % 16;

      auto shuffled_tensor = make_fragment_like(tensor);

      for (int cw = 0; cw < copy_W; cw++) {
        auto remote_id = (id + cw * SubgroupSize) / copy_W;

        auto remote_tensor = select_from_group(sg, tensor, remote_id);

        auto reshape_tensor = make_tensor(remote_tensor.data(), Shape<Int<copy_W>,
                                          Int<DPAS>, Int<copy_H>, Int<H / copy_H>, Int<W / copy_W>>{});

        for (int d = 0; d < DPAS; d++) {
          for (int ch = 0; ch < copy_H; ch++) {
            for (int w = 0; w < W / copy_W; w++) {
              for (int h = 0; h < H / copy_H; h++) {
                shuffled_tensor(d, h * copy_H + ch, w * copy_W + cw) = reshape_tensor(id % copy_W, d, ch, h, w);
              }
            }
          }
        }
      }

      return shuffled_tensor;
    }
  }

  template <class EngineIn,
            class LayoutIn>
  static auto shuffle_B_if_needed(Tensor<EngineIn, LayoutIn> const& tensor) {
    static_assert(rank(LayoutIn{}) == 3);
    static_assert(is_rmem<EngineIn>::value, "Input tensor for shuffle must come from registers");

    using tensor_t = Tensor<EngineIn, LayoutIn>;

    if constexpr (sizeof_bits_v<typename tensor_t::value_type> >= 8) {
      return tensor;
    } else {
      static constexpr auto DPAS = decltype(size<0>(tensor))::value;
      static constexpr auto W = decltype(size<1>(tensor))::value;
      static constexpr auto H = decltype(size<2>(tensor))::value;
      static constexpr auto copy_W = decltype(size<0>(get<1>(tensor.layout())))::value;
      static constexpr auto copy_H = decltype(size<0>(get<2>(tensor.layout())))::value;

      auto sg = syclcompat::get_nd_item<1>().get_sub_group();

      int id = int(ThreadIdxX()) % 16;

      auto shuffled_tensor = make_fragment_like(tensor);

      for (int cw = 0; cw < copy_W; cw++) {
        auto remote_id = (id + cw * SubgroupSize) / copy_W;

        auto remote_tensor = select_from_group(sg, tensor, remote_id);

        auto reshape_tensor = make_tensor(remote_tensor.data(), Shape<Int<copy_W>,
                                          Int<DPAS>, Int<copy_H>, Int<W / copy_W>, Int<H / copy_H>>{});

        for (int d = 0; d < DPAS; d++) {
          for (int ch = 0; ch < copy_H; ch++) {
            for (int w = 0; w < W / copy_W; w++) {
              for (int h = 0; h < H / copy_H; h++) {
                shuffled_tensor(d, w * copy_W + cw, h * copy_H + ch) = reshape_tensor(id % copy_W, d, ch, w, h);
              }
            }
          }
        }
      }

      return shuffled_tensor;
    }
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK,
    class BlkCoord
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      BlkCoord const &blk_coord,
      int const &K_start,
      int thread_idx,
      char *smem_buf,
      Params const& mainloop) 
  {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    (void)residue_mnk;
    (void)thread_idx;
    (void)smem_buf;

    auto tiled_copy_a = make_xe_2d_copy(atom_load_A{}.with(mainloop.mA),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{});
    auto tiled_copy_b = make_xe_2d_copy(atom_load_B{}.with(mainloop.mB),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{});

    // Partition the copying of A and B tiles across the threads
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    // Partition fragment
    Tensor fragment_A = thr_mma.partition_fragment_A(gA(_, _, 0));
    Tensor fragment_B = thr_mma.partition_fragment_B(gB(_, _, 0));

    // Retile for copy
    Tensor copy_tCrA = thr_copy_A.retile_D(fragment_A);
    Tensor copy_tCrB = thr_copy_B.retile_D(fragment_B);

    // Retile for cute::gemm
    Tensor mma_tCrA = thr_copy_A.retile_MMA(thr_mma, fragment_A);
    Tensor mma_tCrB = thr_copy_B.retile_MMA(thr_mma, fragment_B);

  #if CUTLASS_ENABLE_DEBUG_PRINTS
    if (cutlass::thread(LOG_THREAD, LOG_GROUP)) {
        print("======================= A: \n");
        print("gA         : "); print(gA); print("\n");
        print("fragment_A : "); print(fragment_A); print("\n");
        print("copy_tCrA  : "); print(copy_tCrA); print("\n");
        print("mma_tCrA   : "); print(mma_tCrA); print("\n");

        print("=======================  B: \n");
        print("gB         : "); print(gB); print("\n");
        print("fragment_B : "); print(fragment_B); print("\n");
        print("copy_tCrB  : "); print(copy_tCrB); print("\n");
        print("mma_tCrB   : "); print(mma_tCrB); print("\n");

        print("=======================  Config: \n");
        print("threads per workgroup : "); print(MaxThreadsPerBlock); print("\n");
        print("SubgroupTileShape     : "); print(SubgroupTileShape{}); print("\n");

        print("=======================  Prefetch: \n");
        print(" PrefetchAThrShape : ");print(PrefetchAThrShape{});print("\n");
        print(" PrefetchBThrShape : ");print(PrefetchBThrShape{});print("\n");
        print(" PrefetchATileSize : ");print(PrefetchATileSize{});print("\n");
        print(" PrefetchBTileSize : ");print(PrefetchBTileSize{});print("\n");
      }
  #endif

    //
    // Mainloop
    //
    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
  #ifdef CUTLASS_SYCL_SWITCH_WG
    const int m_coord = n_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = m_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
  #else
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
  #endif
    const int l_coord = l_idx;
    Tensor block2d_copy_iter_a = tiled_copy_a.get_pvc_tensor(make_coord(m_coord, 0, l_coord), copy_tCrA.shape());
    auto copy_iter_a = append_pvc_tensor<1>(block2d_copy_iter_a, k_tile_count, BLK_K);

    Tensor block2d_copy_iter_b = tiled_copy_b.get_pvc_tensor(make_coord(n_coord, 0, l_coord), copy_tCrB.shape());
    auto copy_iter_b = append_pvc_tensor<1>(block2d_copy_iter_b, k_tile_count, BLK_K);

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    int prefetch_k = 0;

    Tensor block2d_prefetch_iter_a = XE_Prefetch_A{}.get_pvc_tensor(
                               make_coord(m_coord + (get_sub_group_id() % ATOM_N) / get<1>(PrefetchAThrShape{}) * get<0>(PrefetchATileSize{}),
                                          (k_start_idx + (get_sub_group_id() % ATOM_N) % get<1>(PrefetchAThrShape{})) * get<1>(PrefetchATileSize{}),
                                          l_coord),
                               make_shape(_1{}, _1{}, _1{}));
    auto prefetch_iter_a = append_pvc_tensor<1>(block2d_prefetch_iter_a, k_tile_count, BLK_K);

    Tensor block2d_prefetch_iter_b = XE_Prefetch_B{}.get_pvc_tensor(
                               make_coord((get_sub_group_id() / ATOM_N / get<1>(PrefetchBThrShape{}) + k_start_idx) * get<0>(PrefetchBTileSize{}),
                                           n_coord + (get_sub_group_id() / ATOM_N) % get<1>(PrefetchBThrShape{}) * get<1>(PrefetchBTileSize{}),
                                           l_coord),
                               make_shape(_1{}, _1{}, _1{}));
    auto prefetch_iter_b = append_pvc_tensor<0>(block2d_prefetch_iter_b, k_tile_count, BLK_K);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      if constexpr(cute::detail::has_prefetch<GmemTiledCopyA>) {
        prefetch(tiled_copy_a, prefetch_iter_a(_,_,_,prefetch_k));
      }
      if constexpr(cute::detail::has_prefetch<GmemTiledCopyB>) {
        prefetch(tiled_copy_b, prefetch_iter_b(_,_,_,prefetch_k));
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0, k = k_start_idx; k_tile < k_tile_count; ++k_tile, ++k, ++prefetch_k) {
      // Copy gmem to rmem for the first k_tile
      copy(tiled_copy_a, copy_iter_a(_,_,_,k), copy_tCrA);
      copy(tiled_copy_b, copy_iter_b(_,_,_,k), copy_tCrB);

      auto shuffled_A = shuffle_A_if_needed(mma_tCrA);
      auto shuffled_B = shuffle_B_if_needed(mma_tCrB);

      using mma_type = typename TiledMma::MMA_Type;
      auto mma_A = transform_if_needed<mma_type>(shuffled_A);
      auto mma_B = transform_if_needed<mma_type>(shuffled_B);

      if(prefetch_k < k_tile_count) {
        if constexpr(cute::detail::has_prefetch<GmemTiledCopyA>) {
          prefetch(tiled_copy_a, prefetch_iter_a(_,_,_,prefetch_k));
        }
        if constexpr(cute::detail::has_prefetch<GmemTiledCopyB>) {
          prefetch(tiled_copy_b, prefetch_iter_b(_,_,_,prefetch_k));
        } 
      }

      cute::gemm(tiled_mma, mma_A, mma_B, accum);
    }
  }
};


} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
