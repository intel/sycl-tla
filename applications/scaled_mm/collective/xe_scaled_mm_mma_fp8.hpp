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
 /*
 * This implements the scaled_mm for W8A8 GEMM (FP8 weights and activations) using FP16 compute as a workaround,
 * since current Intel GPUs (e.g., PVC, BMG) lack native FP8 support.
 * The kernel converts FP8 inputs to FP16 on-the-fly and performs GEMM using FP16 MMA.
 */
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fp8_to_fp16.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include <typeinfo> 

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////


template <int Stages, class Schedule, class TileShape_, class ElementA_, class StrideA_, class ElementB_, class StrideB_,
          class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_, class TransformA_,
          class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_, class TransformB_>
struct CollectiveMma<MainloopIntelScaledMMW8A8<Stages, Schedule>, TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_,
                     GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_, SmemLayoutAtomB_,
                     SmemCopyAtomB_, TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = cutlass::gemm::MainloopIntelScaledMMW8A8<Stages, Schedule>;
  using WorkgroupTileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_; 
  using GmemTiledCopyScaleA = XE_2D_U16x32x32_LD_N; // Shape of the copy atom for scales A must match shape of the copy atom for A in the number of elements
  using GmemTiledCopyScaleB = XE_2D_U16x32x32_LD_N; // Shape of the copy atom for scales A must match shape of the copy atom for A in the number of elements

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ElementScaleA = half_t;
  using ElementScaleB = half_t;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(platform::is_same<ElementA, ElementB>::value, "MainloopIntelW8A8 requires that A and B have same type.");
  static_assert(std::is_same_v<TransformA, cute::identity>, "Transformation for A is not currently supported on Intel PVC");
  static_assert(std::is_same_v<TransformB, cute::identity>, "Transformation for B is not currently supported on Intel PVC");

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

  // 32
  static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;

  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
  using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
  using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
  using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
  using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));
  
  using traits_load_scaleA = Copy_Traits<GmemTiledCopyScaleA, StrideA>;
  using atom_load_scaleA = Copy_Atom<traits_load_scaleA, ElementScaleA>;
  using val_layout_load_scaleA = decltype(make_layout(shape_div(typename traits_load_scaleA::BlockShape{}, CopyThreadShape{})));
  using Copy_scaleA = decltype(make_tiled_copy(atom_load_scaleA{}, Layout<CopyThreadShape>{}, val_layout_load_scaleA{}));

  using traits_load_scaleB = Copy_Traits<GmemTiledCopyScaleB, StrideB>;
  using atom_load_scaleB = Copy_Atom<traits_load_scaleB, ElementScaleB>;
  using val_layout_load_scaleB = decltype(make_layout(shape_div(typename traits_load_scaleB::BlockShape{}, CopyThreadShape{})));
  using Copy_scaleB = decltype(make_tiled_copy(atom_load_scaleB{}, Layout<CopyThreadShape>{}, val_layout_load_scaleB{}));

  // Host side kernel arguments
  struct Arguments {
    ElementA  const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScaleA const* ptr_scaleA;
    StrideA dscaleA;
    ElementScaleA const* ptr_scaleB;
    StrideB dscaleB;
  };

  struct Params {
    Copy_A tiled_copy_a;
    Copy_B tiled_copy_b;
    Copy_scaleA tiled_copy_scale_a;
    Copy_scaleB tiled_copy_scale_b;
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
    auto mScaleA_mkl = make_tensor(make_gmem_ptr(static_cast<ElementScaleA const*>(args.ptr_scaleA)),
                              make_layout(make_shape(M, K, L), args.dscaleA));
    auto mScaleB_nkl = make_tensor(make_gmem_ptr(static_cast<ElementScaleB const*>(args.ptr_scaleB)),
                              make_layout(make_shape(N, K, L), args.dscaleB));

    Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl)};
    Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl)};
    Copy_scaleA tiled_copy_scale_a{Copy_scaleA{}.with(mScaleA_mkl)};
    Copy_scaleB tiled_copy_scale_b{Copy_scaleB{}.with(mScaleB_nkl)};

    return Params{tiled_copy_a, tiled_copy_b, tiled_copy_scale_a, tiled_copy_scale_b};
  }

  template <
  class EngineA,
  class EngineB,
  class EnginescaleA,
  class EnginescaleB,
  class LayoutA,
  class LayoutB,
  class LayoutscaleA,
  class LayoutscaleB,
  class... Ts>
  CUTLASS_DEVICE
  void dequantize(
          Tensor<EngineA, LayoutA>& a_tensor,
          Tensor<EngineB, LayoutB>& b_tensor, 
          Tensor<EnginescaleA, LayoutscaleA>& scale_a_tensor, 
          Tensor<EnginescaleB, LayoutscaleB>& scale_b_tensor) {

      using aType = typename EngineA::value_type;
      using bType = typename EngineB::value_type;
      using saType = typename EnginescaleA::value_type;
      using sbType = typename EnginescaleB::value_type;

      aType* pA = a_tensor.data();
      bType* pB = b_tensor.data();
      saType* psA = scale_a_tensor.data();
      sbType* psB = scale_b_tensor.data();

      constexpr int a_num_elements = decltype(size(a_tensor))::value;
      constexpr int b_num_elements = decltype(size(b_tensor))::value;

      for (int i = 0; i < a_num_elements; i++){
        pA[i] *= psA[i];
      }
      for (int i = 0; i < b_num_elements; i++){
        pB[i] *= psB[i];
      }
  }   

  //
  // Methods
  //
  template <class EngineIn,
      class EngineOut,
      class LayoutIn,
      class LayoutOut,
      class... Ts>
  CUTLASS_DEVICE
  void convert_FP8_to_FP16(
          Tensor<EngineIn, LayoutIn> const& in,
          Tensor<EngineOut, LayoutOut>& out) {

    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    static_assert(std::is_same_v<SrcType, uint8_t>, "Expected fp8 (E4M3) input as uint8_t");
    static_assert(std::is_same_v<DstType, half_t>, "Expected fp16 output as half_t");

    SrcType const* pSrc = in.data();
    DstType* pDst = out.data();

    constexpr int num_elements = decltype(size(in))::value;
    // TODO(Codeplay): Move conversion to NumericArrayConverter
    if constexpr (std::is_same_v<ElementA, float_e5m2_t>) {
      // Using something as simple as the following code surprisingly
      // leads to poor performance.
      // CUTLASS_PRAGMA_UNROLL
      // for (int i = 0; i < num_elements; i++) {
      //   reinterpret_cast<uint16_t*>(pDst)[i] = (static_cast<uint16_t>((pSrc[i]))) << 8;
      // }
      // The root-cause is unknown, but private memory use is seen in this case.
      using SrcArray = cutlass::Array<uint8_t, num_elements>;
      using DstArray = cutlass::Array<uint16_t, num_elements>;
      SrcArray const* pSrcArr = reinterpret_cast<SrcArray const*>(pSrc);
      DstArray* pDstArr = reinterpret_cast<DstArray*>(pDst);
      E5M2_to_FP16<num_elements>(*pSrcArr, *pDstArr);
    } else {
      // E4M3 -> FP16 conversion
      constexpr int chunk_size = 16;
      constexpr int iters = num_elements / chunk_size;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < iters; ++i) {
        cute::intel::uchar16 src_vec;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < chunk_size; ++j) {
          src_vec[j] = pSrc[i * chunk_size + j];
        }
        cute::intel::ushort16 dst_vec;
        dst_vec = E4M3_to_FP16_chunk16(src_vec);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < chunk_size; ++j) {
          reinterpret_cast<uint16_t*>(pDst)[i * chunk_size + j] = dst_vec[j];
        }
      }
    }
  }
  
  // Perform a subgroup-scoped matrix multiply-accumulate
  template <class FrgTensorD, class TensorA, class TensorB, class TensorsA, class TensorsB, class FrgTensorC, class KTileIterator, class BlkCoord>
  CUTLASS_DEVICE void operator()(FrgTensorD &accum, TensorA gA, TensorB gB, TensorsA gscaleA, TensorsB gscaleB, FrgTensorC const &src_accum,
                                 KTileIterator k_tile_iter, int k_tile_count, BlkCoord const &blk_coord, int const &K_start, int thread_idx,
                                 Params const &mainloop) {
    (void)blk_coord;
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    // Just getting a subview.
    auto thr_copy_A = mainloop.tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = mainloop.tiled_copy_b.get_slice(thread_idx);
    auto thr_copy_scaleA = mainloop.tiled_copy_scale_a.get_slice(thread_idx);
    auto thr_copy_scaleB = mainloop.tiled_copy_scale_b.get_slice(thread_idx); 
    
    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;

    // Same settings as the tiled_mma
    using TileShape = Shape<_256, _256, _32>;
    using TiledMmaForScalingFactors =
          typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>, // Have to use same MMA_Atom shape used in the example.
          Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

    TiledMmaForScalingFactors tiled_scalingfactors_mma;
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);
    auto thr_scaledfactors_mma = tiled_scalingfactors_mma.get_slice(first_thread_in_sg_idx);

    // Partition
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgscaleA = thr_scaledfactors_mma.partition_A(gscaleA);
    Tensor tCgscaleB = thr_scaledfactors_mma.partition_B(gscaleB);

    Tensor tCrA = make_tensor<uint8_t>(make_fragment_layout(mainloop.tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<uint8_t>(make_fragment_layout(mainloop.tiled_copy_b, tCgB(_,_,_,0).shape()));
    Tensor tCrscaleA = make_tensor<ElementScaleA>(make_fragment_layout(mainloop.tiled_copy_scale_a, tCgscaleA(_,_,_,0).shape()));
    Tensor tCrscaleB = make_tensor<ElementScaleB>(make_fragment_layout(mainloop.tiled_copy_scale_b, tCgscaleB(_,_,_,0).shape()));

    Tensor tCrA_fp16 = make_fragment_like<half_t>(tCrA);
    Tensor tCrB_fp16 = make_fragment_like<half_t>(tCrB);

    // Retile registers for copies
    Tensor tArA = thr_copy_A.retile_D(tCrA);
    Tensor tBrB = thr_copy_B.retile_D(tCrB);
    Tensor tArsA = thr_copy_scaleA.retile_D(tCrscaleA);
    Tensor tBrsB = thr_copy_scaleB.retile_D(tCrscaleB);

    // Retile global tile for copies
    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);
    Tensor tAgsA = thr_copy_scaleA.retile_S(tCgscaleA);
    Tensor tBgsB = thr_copy_scaleB.retile_S(tCgscaleB);  

    //
    // Mainloop
    //
    const auto k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    constexpr int barrier_scope = 2;

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++) {
      barrier_arrive(barrier_scope);

      // Copy gmem to rmem for the first k tile
      copy(mainloop.tiled_copy_a, tAgA(_,_,_,k_tile), tArA);
      copy(mainloop.tiled_copy_b, tBgB(_,_,_,k_tile), tBrB);

      copy(mainloop.tiled_copy_scale_a, tAgsA(_,_,_,k_tile), tArsA);
      copy(mainloop.tiled_copy_scale_b, tBgsB(_,_,_,k_tile), tBrsB);

      // TODO: register pressure
      convert_FP8_to_FP16(tCrA, tCrA_fp16);
      convert_FP8_to_FP16(tCrB, tCrB_fp16);

      // Apply dequantization
      dequantize(tCrA_fp16, tCrB_fp16, tCrscaleA, tCrscaleB);

      // compute using fp16
      cute::gemm(tiled_mma, tCrA_fp16, tCrB_fp16, accum);

      barrier_wait(barrier_scope);
    }
    
  }
  
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
