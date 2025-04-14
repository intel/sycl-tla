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
#include <cutlass/gemm/dispatch_policy.hpp>

#include "cutlass/gemm/collective/collective_mma.hpp"

namespace cutlass::gemm::collective {

namespace {
template <typename LayoutA, class TileShape_MNK, int sgs_M>
inline auto pick_load_atom_for_A() {
  if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
    constexpr int tile_M = get<0>(TileShape_MNK{});
    constexpr int tile_K = get<2>(TileShape_MNK{});
    static_assert(tile_M % sgs_M == 0);
    constexpr int atoms_in_M_dim = tile_M / sgs_M;
    if constexpr (atoms_in_M_dim >= 32 && tile_K >= 32) {
      return XE_2D_U16x32x32_LD_N{};
    } else if constexpr (atoms_in_M_dim >= 32) {
      return XE_2D_U16x32x16_LD_N{};
    } else if constexpr (atoms_in_M_dim == 8 && tile_K >= 32) {
      return XE_2D_U16x8x32_LD_N{};
    } else {
      return XE_2D_U16x16x16_LD_N{};
    }
  } else {
    return XE_2D_U16x16x16_LD_T{};
  }
}

template <typename LayoutB, class TileShape_MNK, int sgs_N>
inline auto pick_load_atom_for_B() {
  if constexpr (cute::is_same_v<LayoutB, cutlass::layout::RowMajor>) {
    constexpr int tile_N = get<1>(TileShape_MNK{});
    constexpr int tile_K = get<2>(TileShape_MNK{});
    constexpr int atoms_in_N_dim = tile_N / sgs_N;
    if constexpr (atoms_in_N_dim >= 32 && tile_K >= 32) {
      return XE_2D_U16x32x32_LD_V{};
    } else {
      return XE_2D_U16x16x16_LD_V{};
    }
  } else {
    return XE_2D_U16x16x16_LD_T{};
  }
}

template <typename LayoutA, class TileShape_MNK>
constexpr inline int calculate_sgs_in_M() {
  constexpr int tile_M = get<0>(TileShape_MNK{});
  if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
    // Non-transpose load can be size 1, 2, 4, 8, 16, or 32 in the M dim (for bf16),
    // but we are only supporting 8, 16 and 32 so far.
    for (auto atom_m : {32,16,8}) {
      auto atoms_in_m = tile_M / atom_m;
      for (auto atoms : {8,4,2}) {
        if (atoms_in_m >= atoms) {
          return atoms;
        }
      }
    }
    return 1;
  } else {
    // Transpose loads are always size 16 in the M dim (for bf16).
    static_assert(tile_M / 16 > 0 and tile_M % 16 == 0, "Invalid Tile size in M dim");
    return tile_M / 16;
  }
}

// Lookup table for subgroup layout
// This is the default case
template <typename TileShape, typename LayoutA, typename LayoutB>
struct SubgroupTilingMap {
  private:
      static constexpr auto tile_M = get<0>(TileShape{});
      static constexpr auto tile_N = get<1>(TileShape{});
      static constexpr bool tall_and_skinny = tile_M >= tile_N;
      static constexpr int sgs_total = 32;
      static constexpr int atom_N = 32; // size of the copy atom in N
      static_assert(tile_N >= atom_N, "Tile N dim must be greater than or equal to 32");

  public:
      using sgs_M = Int<calculate_sgs_in_M<LayoutA, TileShape>()>;
      using sgs_N = Int<std::min(tile_N/atom_N, sgs_total/sgs_M::value)>;
      using GmemTiledCopyA = decltype(pick_load_atom_for_A<LayoutA, TileShape, sgs_M{}>());
      using GmemTiledCopyB = decltype(pick_load_atom_for_B<LayoutB, TileShape, sgs_N{}>());

};

template <>
struct SubgroupTilingMap<Shape<_256,_256,_32>, cutlass::layout::RowMajor, cutlass::layout::RowMajor> {
      using sgs_M = Int<8>;
      using sgs_N = Int<4>;
      using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
      using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
};
template <>
struct SubgroupTilingMap<Shape<_256,_256,_32>, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor> {
      using sgs_M = Int<8>;
      using sgs_N = Int<4>;
      using GmemTiledCopyA = XE_2D_U16x8x32_LD_N;
      using GmemTiledCopyB = XE_2D_U16x16x16_LD_T;
};
template <typename LayoutB>
struct SubgroupTilingMap<Shape<_256,_256,_32>, cutlass::layout::ColumnMajor, LayoutB> {
      using sgs_M = Int<8>;
      using sgs_N = Int<4>;
      using GmemTiledCopyA = XE_2D_U16x16x16_LD_T;
      using GmemTiledCopyB = std::conditional_t<
        std::is_same_v<LayoutB, cutlass::layout::RowMajor>,
        XE_2D_U16x32x32_LD_V,
        XE_2D_U16x16x16_LD_T>;
};

template <>
struct SubgroupTilingMap<Shape<_256,_128,_32>, cutlass::layout::RowMajor, cutlass::layout::RowMajor> {
      using sgs_M = Int<8>;
      using sgs_N = Int<4>;
      using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
      using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
};

template <>
struct SubgroupTilingMap<Shape<_128,_512,_32>, cutlass::layout::RowMajor, cutlass::layout::RowMajor> {
      using sgs_M = Int<4>;
      using sgs_N = Int<8>;
      using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
      using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
};

template <>
struct SubgroupTilingMap<Shape<_128,_256,_16>, cutlass::layout::RowMajor, cutlass::layout::RowMajor> {
      using sgs_M = Int<4>;
      using sgs_N = Int<8>;
      using GmemTiledCopyA = XE_2D_U16x32x16_LD_N;
      using GmemTiledCopyB = XE_2D_U16x16x16_LD_V;
};

template <>
struct SubgroupTilingMap<Shape<_8,_128,_16>, cutlass::layout::RowMajor, cutlass::layout::RowMajor> {
      using sgs_M = Int<1>;
      using sgs_N = Int<4>;
      using GmemTiledCopyA = XE_2D_U16x8x32_LD_N;
      using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
};

} // namespace

// Intel PVC 3 stage pipeline, using prefetch
// Also the auto builder

template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class KernelScheduleType
  > 
struct CollectiveBuilder<
  arch::IntelPVC,
  arch::OpClassTensorOp,   // Reusing opClassTensorOp for Intel devices
  ElementA,
  GmemLayoutATag,
  AlignmentA,
  ElementB,
  GmemLayoutBTag,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  Shape<_1, _1, _1>,    // Cluster Shape
  cutlass::gemm::collective::StageCountAuto, 
  KernelScheduleType,
  cute::enable_if_t<
    cute::is_any_of_v<KernelScheduleType, KernelScheduleAuto, KernelPVC, KernelPVCCooperative, KernelPVCPtrArrayCooperative> &&
    cute::is_same_v<ElementA, ElementB> &&
    cute::is_any_of_v<ElementA, bfloat16_t, half_t> &&
    cute::is_any_of_v<ElementB, bfloat16_t, half_t>
  >>{
      #ifdef SYCL_NVIDIA_TARGET
        static_assert(cutlass::detail::dependent_false<arch::IntelPVC>, 
          "Trying to use Intel pipeline on Non Intel hardware");
      #endif
      static_assert(is_static<TileShape_MNK>::value);
      static_assert(cute::is_same_v<ElementAccumulator, float>, "Intel multi-stage pipeline requires ElementC to be of type float");

      using MMAAtom = MMA_Atom<std::conditional_t<cute::is_same_v<ElementA, bfloat16_t>,
                                                  XE_8x16x16_F32BF16BF16F32_TT,
                                                  XE_8x16x16_F32F16F16F32_TT>>;

      // We have too many subgroups, we can have at most 32, but only 8 are needed for 8x128 values (8x16 mma)
      // Prepare Template arguments required of CollectiveMainLoop
      static constexpr auto tile_M = get<0>(TileShape_MNK{});
      static constexpr auto tile_N = get<1>(TileShape_MNK{});
      static constexpr auto tile_K = get<2>(TileShape_MNK{});

      // number of subgroups in a dim is at most (values in a dim)/(atom size in a dim)
      using atom_mnk = typename MMAAtom::Shape_MNK;
      using max_subgroups = decltype(take<0,2>(shape_div(TileShape_MNK{}, atom_mnk{}))); // M, N

      using SgTilingMap = SubgroupTilingMap<TileShape_MNK, GmemLayoutATag, GmemLayoutBTag>;
      using sgs_M = typename SgTilingMap::sgs_M;
      using sgs_N = typename SgTilingMap::sgs_N;
      using GmemTiledCopyA = typename SgTilingMap::GmemTiledCopyA;
      using GmemTiledCopyB = typename SgTilingMap::GmemTiledCopyB;
      using TiledMma =
          typename TiledMMAHelper<MMAAtom,
                                  Layout<TileShape_MNK>,
                                  Layout<Shape<sgs_M, sgs_N, _1>, Stride<sgs_N, _1, _0>>>::TiledMMA;

      static constexpr bool IsGroup = cute::is_same_v<KernelScheduleType, KernelPVCPtrArrayCooperative>;

      using KernelSchedule = std::conditional_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>, KernelPVC, KernelScheduleType>;
      static constexpr int PipelineStages = IsGroup ? 2 : 3;
      using DispatchPolicy = std::conditional_t<IsGroup, 
                                                cutlass::gemm::MainloopIntelPVCGroup<PipelineStages, KernelSchedule>,
                                                cutlass::gemm::MainloopIntelPVC<PipelineStages, KernelSchedule>>;

      // PVC pipeline does not use shared memory
      using SmemLayoutAtomA = void; 
      using SmemLayoutAtomB = void; 
      using SmemCopyAtomA = void;
      using SmemCopyAtomB = void;

      using TransformA = cute::identity;
      using TransformB = cute::identity;

      using StrideA = cutlass::gemm::TagToStrideA_t<std::conditional_t<IsGroup, GmemLayoutATag*, GmemLayoutATag>>;
      using StrideB = cutlass::gemm::TagToStrideB_t<std::conditional_t<IsGroup, GmemLayoutBTag*, GmemLayoutBTag>>;


      using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
              DispatchPolicy,
              TileShape_MNK,
              ElementA,
              StrideA,
              ElementB,
              StrideB,
              TiledMma,
              GmemTiledCopyA,
              SmemLayoutAtomA,
              SmemCopyAtomA,
              TransformA,
              GmemTiledCopyB,
              SmemLayoutAtomB,
              SmemCopyAtomB,
              TransformB
          >;
    };
}
