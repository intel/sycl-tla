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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fp8_to_fp16.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
#define CUTLASS_ENABLE_DEBUG_PRINTS (1)
#define LOG_GROUP (0)
#define LOG_THREAD (0)

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages, class Schedule, class TileShape_, class ElementA_, class StrideA_, class ElementB_, class StrideB_,
          class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_, class TransformA_,
          class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_, class TransformB_>
struct CollectiveMma<MainloopIntelW8A8<Stages, Schedule>, TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_,
                     GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_, SmemLayoutAtomB_,
                     SmemCopyAtomB_, TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelW8A8<Stages, Schedule>;
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

  static_assert(platform::is_same<ElementA, ElementB>::value, "MainloopIntelW8A8 requires that A and B have same type.");
  static_assert(std::is_same_v<ElementA, cutlass::float_e4m3_t> || std::is_same_v<ElementA, cutlass::float_e5m2_t>, "ElementA must be FP8 (E4M3 or E5M2)");
  static_assert(std::is_same_v<ElementB, cutlass::float_e4m3_t> || std::is_same_v<ElementB, cutlass::float_e5m2_t>, "ElementB must be FP8 (E4M3 or E5M2)");

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
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, uint8_t>;

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, uint8_t>;

  using TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<uint8_t const*>(nullptr)), make_shape(0,0,0), StrideA{})); // UINT8
  using TensorNKL = decltype(make_tensor(make_gmem_ptr(static_cast<uint8_t const*>(nullptr)), make_shape(0,0,0), StrideB{})); // UINT8

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  using Copy_A = decltype(make_tiled_copy(atom_load_A{},
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{}))));

  using Copy_B = decltype(make_tiled_copy(atom_load_B{},
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{}))));
  // Host side kernel arguments
  struct Arguments {
    ElementA  const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  struct Params {
    Copy_A copy_A;
    Copy_B copy_B;
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
    
    auto mA_mkl = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t const*>(args.ptr_A)),
                              make_layout(make_shape(M, K, L), args.dA));
    auto mB_nkl = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t const*>(args.ptr_B)),
                              make_layout(make_shape(N, K, L), args.dB));

    auto tiled_copy_a = make_tiled_copy(atom_load_A{}.with(mA_mkl),
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
    auto tiled_copy_b = make_tiled_copy(atom_load_B{}.with(mB_nkl),
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));

    return Params{tiled_copy_a, tiled_copy_b, mA_mkl, mB_nkl};
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

  template <class EngineIn,
      class EngineOut,
      class LayoutIn,
      class LayoutOut,
      class... Ts>
  CUTLASS_DEVICE
      void convert_E4M3_to_FP16(
          Tensor<EngineIn, LayoutIn> const& tCrA_load,
          Tensor<EngineOut, LayoutOut>& tCrA_mma) {

      static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
      static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
      static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
      static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
      static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
      
      using SrcType = typename EngineIn::value_type;
      using DstType = typename EngineOut::value_type;

      static_assert(std::is_same_v<SrcType, uint8_t>, "Expected fp8 (E4M3) input as uint8_t");
      static_assert(std::is_same_v<DstType, half_t>, "Expected fp16 output as half_t");

      auto const& src = tCrA_load(_, _, _);
      auto const& dst = tCrA_mma(_, _, _);

      SrcType const* pSrc = src.data();
      DstType* pDst = dst.data();

      constexpr int num_elements = decltype(size(src))::value;
      constexpr int vec_size = 16;
      static_assert(num_elements % vec_size == 0, "Number of elements must be divisible by vector size (16)");

      CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < num_elements / vec_size; ++i) {
            cute::intel::uchar16 src_vec;
            for (int j = 0; j < vec_size; ++j) {
                src_vec[j] = pSrc[i * vec_size + j];
            }
            cute::intel::ushort16 dst_vec = E4M3_to_FP16_vec16(src_vec);
            for (int j = 0; j < vec_size; ++j) {
                reinterpret_cast<uint16_t*>(pDst)[i * vec_size + j] = dst_vec[j];

            }
        }
  }
  
  template <class FrgTensorD, class TensorA, class TensorB, class FrgTensorC, class KTileIterator, class ResidueMNK,
            class BlkCoord>
  CUTLASS_DEVICE void operator()(FrgTensorD &accum, TensorA gA, TensorB gB, FrgTensorC const &src_accum,
                                 KTileIterator k_tile_iter, int k_tile_count, ResidueMNK residue_mnk,
                                 BlkCoord const &blk_coord, int const &K_start, int thread_idx, char *smem_buf,
                                 Params const &mainloop) {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    (void)residue_mnk;
    (void)thread_idx;
    (void)smem_buf;

    auto tiled_copy_a = make_tiled_copy(atom_load_A{}.with(mainloop.mA),
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
    auto tiled_copy_b = make_tiled_copy(atom_load_B{}.with(mainloop.mB),
                                   Layout<CopyThreadShape>{},
                                   make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    // TODO(Codeplay): see if we can make this nicer
    // To make all work items in a subgroup have the same global tensors pass in the index of work item 0 in each subgroup
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    // Partition
    Tensor tCgA = thr_mma.partition_A(gA); // uint8
    Tensor tCgB = thr_mma.partition_B(gB); // uint8

    Tensor tCrA = make_tensor<uint8_t>(make_fragment_layout(tiled_copy_a, tCgA(_,_,_,0).shape())); // uint8
    Tensor tCrB = make_tensor<uint8_t>(make_fragment_layout(tiled_copy_b, tCgB(_,_,_,0).shape())); // uint8

    // Retile registers for copies
    Tensor tArA = thr_copy_A.retile_D(tCrA); // uint8
    Tensor tBrB = thr_copy_B.retile_D(tCrB); // uint8
    
    // Retile global tile for copies
    Tensor tAgA = thr_copy_A.retile_S(tCgA); // uint8
    Tensor tBgB = thr_copy_B.retile_S(tCgB); // uint8

    Tensor tCrA_fp16 = make_fragment_like<half_t>(tCrA); // fp16
    Tensor tCrB_fp16 = make_fragment_like<half_t>(tCrB); // fp16

    //
    // Mainloop
    //
    const auto k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    constexpr int barrier_scope = 2;

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++) {
      barrier_arrive(barrier_scope);

      // copy fp8 using fake int8
      copy(tiled_copy_a, tAgA(_,_,_,k_tile), tArA); // uint8
      copy(tiled_copy_b, tBgB(_,_,_,k_tile), tBrB); // uint8
      
      // convert fp8 to fp16
      // TODO: register pressure
      convert_E4M3_to_FP16(tCrA, tCrA_fp16); // fp8 (stored as fake uint8) -> fp16
      convert_E4M3_to_FP16(tCrB, tCrB_fp16); // fp8 (stored as fake uint8) -> fp16

      // compute using fp16
      cute::gemm(tiled_mma, tCrA_fp16, tCrB_fp16, accum);

      barrier_wait(barrier_scope);
    }
    
  }
  
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
