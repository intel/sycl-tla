/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include "../../../common/sycl_cute_common.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

template <class T> struct is_16_bit_fp : std::false_type {};

template <> struct is_16_bit_fp<cutlass::half_t> : std::true_type {};
template <> struct is_16_bit_fp<cutlass::bfloat16_t> : std::true_type {};

template <class T>
inline constexpr bool is_16_bit_fp_v =
    is_16_bit_fp<std::remove_cv_t<std::remove_reference_t<T>>>::value;

// Making sure I got this right
static_assert(is_16_bit_fp_v<cutlass::bfloat16_t>);
static_assert(is_16_bit_fp_v<cutlass::half_t>);

namespace MoE {

using namespace cute;

template <
    class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
    class ATensor, class BTensor, class DTensor, class TiledMMA,
    class = std::enable_if_t<is_16_bit_fp_v<typename ATensor::element_type> &&
                             is_16_bit_fp_v<typename BTensor::element_type>>>
CUTE_DEVICE void moe_gemm(ATensor const &A, // (M,K)
                          BTensor const &B, // (N,K)
                          DTensor &D,       // (M,N)
                          Coord<int, int, cute::Underscore, int> blk_coord,
                          TiledMMA const &mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  auto local_id = int(item.get_local_id(0));

  Tensor cA = make_identity_tensor(A.shape()); // (M,K)
  Tensor cB = make_identity_tensor(B.shape()); // (N,K)
  Tensor cD = make_identity_tensor(D.shape()); // (M,N)

  auto wg_coord = make_coord(wg_m, wg_n, 0);
  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  auto tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto tiled_copy_d = make_block_2d_copy_CD(GmemTiledCopyD{}, mma, D);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_sg_fragment_C(gD);
  auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  auto tCgD = thr_copy_d.partition_D(gD);

  auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
  auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr int barrier_scope = 2;
  int k_start_idx = 0;
  int prefetch_k = k_start_idx;
  const int prefetch_dist = 3;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

  CUTE_UNROLL
  for (; prefetch_k < prefetch_dist; prefetch_k++) {
    prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
    prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
  }

  for (int k_tile = k_start_idx; k_tile < k_tile_count;
       k_tile++, prefetch_k++) {
    barrier_arrive(barrier_scope);

    copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (prefetch_k < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrD);
    barrier_wait(barrier_scope);
  }
  reorder(tCrD, tCrD_final);
  copy(tiled_copy_d, tCrD_final, tCgD);
}

// Assumes n-major SG layout
// TODO: This one is only for MXFP4. Add template specialization for dtype
template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          class ATensor, class BTensor, class STensor, class DTensor,
          class TiledMMA,
          class = std::enable_if_t<
              !cute::is_void_v<typename STensor::element_type> &&
              is_same_v<typename BTensor::element_type, float_e2m1_t> &&
              is_same_v<typename STensor::element_type, float_ue8m0_t>>>
CUTE_DEVICE void moe_gemm(ATensor const &A, // (M,K)
                          BTensor const &B, // (N,K)
                          STensor const &S, // (K/32, N)
                          DTensor &D,       // (M,N)
                          Coord<int, int, cute::Underscore, int> blk_coord,
                          TiledMMA const &mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto local_id = int(item.get_local_id(0));
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  uint32_t sg_id = sg.get_group_linear_id();
  uint32_t lane = sg.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape()); // (M,K)
  Tensor cB = make_identity_tensor(B.shape()); // (N,K)
  Tensor cD = make_identity_tensor(D.shape()); // (M,N)

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = make_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = make_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_d = make_block_2d_copy_CD(GmemTiledCopyD{}, mma, D);

  auto S_tile = coalesce(local_tile(S, make_shape(Int<1>{}, get<1>(wg_tile)),
                                    make_coord(_, wg_n)));

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_d = copy_d.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_sg_fragment_C(gD);

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
  auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  auto tCgD = thr_copy_d.partition_D(gD);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrD);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    float_ue8m0_t frag[8]; // per-thread registers (compiler will keep in regs)
    float frag_fp32[8];
    Tensor scales_e8m0 =
        make_tensor(make_rmem_ptr(frag), make_layout(make_shape(Int<8>{})));
    Tensor scales_float = make_tensor(make_rmem_ptr(frag_fp32),
                                      make_layout(make_shape(Int<8>{})));

    auto TV_row = make_layout(make_shape(Int<16>{}, Int<8>{}),
                              make_stride(Int<8>{}, Int<1>{}));
    auto scales_e8m0_sg_tensor = make_subgroup_tensor(scales_e8m0, TV_row);
    auto scales_float_sg_tensor = make_subgroup_tensor(scales_float, TV_row);

    // TODO: Update with broadcast loads recently added. It seems it's okay to
    // create a new TiledCopy object in each mainloop iteration.

    // Pair threads as (0,1), (2,3), ..., (14,15)
    bool is_pair_leader = (local_id % 2) == 0;
    uint32_t pair_leader_lane = lane & ~1u; // even lane id
    uint32_t lane_idx = lane >> 1;          // 0..7

    // Only the even lane loads elements
    if (is_pair_leader) {
      CUTE_UNROLL
      for (int i = 0; i < 8; i++) {
        // TODO: Remove magic numbers
        // Assumes scales are [K/32, N], as the reads are coalesced in that
        // case (although shuffles can't be prevented later). Since loaded
        // scales would be read columnwise when scales would be shaped [N,
        // K/32], use UniversalCopy instead.
        scales_e8m0[i] = *(S_tile.data() + (64 * (sg_id % 4)) + k_tile * 256 +
                           lane_idx + i * 8);
      }
    }

    CUTE_UNROLL
    for (int i = 0; i < 8; i++) {
      // Broadcast from the pair leader to both lanes in the pair
      scales_e8m0[i] =
          sycl::select_from_group(sg, scales_e8m0[i], pair_leader_lane);
    }

    reorder(scales_e8m0_sg_tensor, scales_float_sg_tensor);

    // hardcoded for work-item B subgroup fragment sized 128
    // I created 4 loops just so that I can easily copy-paste some of them to
    // the case of 32 or 64 elements per B thread fragment with compile-time
    // expression evaluation. Othwerwise, the loops below can be combined.
    // However, instead of hardcoding, figure out CuTe algebra based
    // transformations that can lead to generic code.
    CUTE_UNROLL
    for (int i = 0, j = 0; i < 15; i += 2) {
      tCrB[i] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                        static_cast<float>(tCrB[i]));
      tCrB[i + 1] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                            static_cast<float>(tCrB[i + 1]));
      tCrB[i + 64] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                             static_cast<float>(tCrB[i + 64]));
      tCrB[i + 65] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                             static_cast<float>(tCrB[i + 65]));
    }
    CUTE_UNROLL
    for (int i = 16, j = 2; i < 31; i += 2) {
      tCrB[i] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                        static_cast<float>(tCrB[i]));
      tCrB[i + 1] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                            static_cast<float>(tCrB[i + 1]));
      tCrB[i + 64] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                             static_cast<float>(tCrB[i + 64]));
      tCrB[i + 65] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                             static_cast<float>(tCrB[i + 65]));
    }
    CUTE_UNROLL
    for (int i = 32, j = 4; i < 47; i += 2) {
      tCrB[i] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                        static_cast<float>(tCrB[i]));
      tCrB[i + 1] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                            static_cast<float>(tCrB[i + 1]));
      tCrB[i + 64] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                             static_cast<float>(tCrB[i + 64]));
      tCrB[i + 65] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                             static_cast<float>(tCrB[i + 65]));
    }
    CUTE_UNROLL
    for (int i = 48, j = 6; i < 63; i += 2) {
      tCrB[i] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                        static_cast<float>(tCrB[i]));
      tCrB[i + 1] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                            static_cast<float>(tCrB[i + 1]));
      tCrB[i + 64] = static_cast<bfloat16_t>(scales_float_sg_tensor[j] *
                                             static_cast<float>(tCrB[i + 64]));
      tCrB[i + 65] = static_cast<bfloat16_t>(scales_float_sg_tensor[j + 1] *
                                             static_cast<float>(tCrB[i + 65]));
    }

    gemm(mma, tCrA, tCrB, tCrD);

    barrier_wait(barrier_scope);
  }

  reorder(tCrD, tCrD_final);
  copy(copy_d, tCrD_final, tCgD);
}
} // namespace MoE
