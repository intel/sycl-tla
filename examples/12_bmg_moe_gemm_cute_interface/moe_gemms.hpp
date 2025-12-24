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

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

template <class T> struct is_16_bit_fp : std::false_type {};

template <> struct is_16_bit_fp<cutlass::half_t> : std::true_type {};
template <> struct is_16_bit_fp<cutlass::bfloat16_t> : std::true_type {};

template <class T>
inline constexpr bool is_16_bit_fp_v =
    is_16_bit_fp<std::remove_cv_t<std::remove_reference_t<T>>>::value;

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
                          Coord<int, int, cute::Underscore, int> &blk_coord,
                          TiledMMA const &mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = item.get_local_linear_id();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);

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
  auto tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);

  auto tDrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tDrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tDrD = thr_mma.partition_sg_fragment_C(gD);
  auto tDrD_final = thr_copy_d.partition_sg_fragment_S(gD);

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

    reorder(tArA, tDrA);
    reorder(tBrB, tDrB);

    cute::gemm(mma, tDrA, tDrB, tDrD);
    barrier_wait(barrier_scope);
  }
  reorder(tDrD, tDrD_final);
  copy(tiled_copy_d, tDrD_final, tCgD);
}

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          int SG_N, int WG_N, int q_group_size, class ATensor, class BTensor,
          class STensor, class DTensor, class TiledMMA,
          class = std::enable_if_t<
              !cute::is_void_v<typename STensor::element_type> &&
              is_any_of_v<typename BTensor::element_type, float_e2m1_t,
                          float_e4m3_t, float_e5m2_t, int4_t> &&
              is_any_of_v<typename STensor::element_type, float_ue8m0_t, half_t,
                          bfloat16_t> &&
              is_any_of_v<typename ATensor::element_type, bfloat16_t, half_t>>>
CUTE_DEVICE void moe_gemm(ATensor const &A, // (M,K)
                          BTensor const &B, // (N,K)
                          STensor const &S, // (K/q_group_size, N)
                          DTensor &D,       // (M,N)
                          Coord<int, int, cute::Underscore, int> &blk_coord,
                          TiledMMA const &mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  uint32_t sg_id = sg.get_group_linear_id();
  uint32_t lane = sg.get_local_linear_id();

  auto total_N = get<0>(B.shape());

  Tensor cA = make_identity_tensor(A.shape()); // (M,K)
  Tensor cB = make_identity_tensor(B.shape()); // (N,K)
  Tensor cD = make_identity_tensor(D.shape()); // (M,N)
  Tensor cS = make_identity_tensor(S.shape()); // (K/q_group_size,N)
  Tensor cScales_per_sg =
      make_identity_tensor(make_shape(Int<1>{}, Int<SG_N>{}));
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile),
                         make_coord(wg_m, _)); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile),
                         make_coord(wg_n, _)); // (BLK_N,BLK_K,k)
  Tensor gD =
      local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  constexpr int num_N_SG_tiles = WG_N / SG_N;
  constexpr int num_scales_per_col = (SG_N == 32) ? 4 : 2;

  // When we use E8M0, the compiler behaves differently & loads more data than
  // needed. The rest is discarded.
  // The scales might be FP16 or BF16 in case of int4 weights
  using scaleLoadType =
      conditional_t<is_same_v<typename STensor::element_type, float_ue8m0_t>,
                    int8_t, int16_t>;

  auto S_tile = coalesce(local_tile(S, make_shape(Int<1>{}, get<1>(wg_tile)),
                                    make_coord(_, wg_n)));

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_d = copy_d.get_slice(local_id);

  auto tDrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tDrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
  auto tDrD = thr_mma.partition_sg_fragment_C(gD);

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tDgD = thr_copy_d.partition_D(gD);

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
  constexpr int num_threads_per_sg = 16;

  typename STensor::element_type
      frag[num_scales_per_col / 2]; // per-thread registers (compiler
                                    // will keep in regs)
  float frag_fp32[num_scales_per_col];
  // assuming SG_K = WG_K
  constexpr int frequency_scale_change = q_group_size / get<2>(wg_tile);
  Tensor scales_e8m0 =
      make_tensor(make_rmem_ptr(frag),
                  make_layout(make_shape(Int<num_scales_per_col / 2>{})));
  Tensor scales_float =
      make_tensor(make_rmem_ptr(frag_fp32),
                  make_layout(make_shape(Int<num_scales_per_col>{})));

  auto srcTVLayout = make_layout(
      make_shape(Int<num_threads_per_sg>{}, Int<num_scales_per_col / 2>{}),
      make_stride(Int<1>{}, Int<num_threads_per_sg>{}));
  auto dstTVLayout = make_layout(
      make_shape(make_shape(Int<2>{}, Int<num_threads_per_sg / 2>{}),
                 make_shape(Int<num_scales_per_col / 2>{})),
      make_stride(make_stride(Int<0>{}, Int<1>{}), make_stride(Int<8>{})));
  auto scales_e8m0_sg_tensor = make_subgroup_tensor(scales_e8m0, srcTVLayout);
  auto scales_float_sg_tensor = make_subgroup_tensor(scales_float, dstTVLayout);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }
  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    reorder(tBrB, tDrB);

    if (k_tile % frequency_scale_change == 0) {
      auto scales_tensor = make_tensor(
          make_gmem_ptr(reinterpret_cast<scaleLoadType *>(
              static_cast<void *>(cute::raw_pointer_cast(
                  S_tile.data() + (SG_N * (sg_id % num_N_SG_tiles)) +
                  (k_tile / frequency_scale_change) * total_N)))),
          make_layout(make_shape(Int<1>{}, Int<SG_N>{})));
      auto copy_scales = make_block_2d_copy(
          XE_LOAD_2D<sizeof_bits_v<typename STensor::element_type>, 1, SG_N,
                     SG_N>{},
          scales_tensor);
      auto thr_copy_scales = copy_scales.get_slice(lane);
      auto scales_per_thread = thr_copy_scales.partition_S(cScales_per_sg);
      copy(copy_scales, scales_per_thread(_, 0, 0), scales_e8m0);
      reorder(scales_e8m0_sg_tensor, scales_float_sg_tensor);
      if (k_tile != (k_tile_count - frequency_scale_change)) {
        auto next_scales_tensor = make_tensor(
            make_gmem_ptr(reinterpret_cast<scaleLoadType *>(
                static_cast<void *>(cute::raw_pointer_cast(
                    S_tile.data() + (SG_N * (sg_id % num_N_SG_tiles)) +
                    ((k_tile / frequency_scale_change) + 1) * total_N)))),
            make_layout(make_shape(Int<1>{}, Int<SG_N>{})));
        auto prefetch_scales = make_block_2d_prefetch<1>(
            make_shape(Int<1>{}, Int<SG_N>{}), next_scales_tensor);
        auto thr_prefetch_scales = prefetch_scales.get_slice(lane);
        auto pSgS = thr_prefetch_scales.partition_S(cScales_per_sg);
        prefetch(prefetch_scales, pSgS(_, 0, 0));
      }
    }
    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    reorder(tArA, tDrA);
    // Instead of hardcoding, figure out CuTe algebra based
    // transformations that can lead to generic code.
    auto scale0 = scales_float_sg_tensor[0];
    auto scale1 = scales_float_sg_tensor[1];
    if (num_scales_per_col == 4) {
      auto scale2 = scales_float_sg_tensor[2];
      auto scale3 = scales_float_sg_tensor[3];
      CUTE_UNROLL
      for (int i = 0; i < 16; i += 2) {
        tDrB[i] = static_cast<typename ATensor::element_type>(
            scale0 * static_cast<float>(tDrB[i]));
        tDrB[i + 1] = static_cast<typename ATensor::element_type>(
            scale1 * static_cast<float>(tDrB[i + 1]));
      }
      CUTE_UNROLL
      for (int i = 16; i < 32; i += 2) {
        tDrB[i] = static_cast<typename ATensor::element_type>(
            scale2 * static_cast<float>(tDrB[i]));
        tDrB[i + 1] = static_cast<typename ATensor::element_type>(
            scale3 * static_cast<float>(tDrB[i + 1]));
      }
      CUTE_UNROLL
      for (int i = 32; i < 48; i += 2) {
        tDrB[i] = static_cast<typename ATensor::element_type>(
            scale0 * static_cast<float>(tDrB[i]));
        tDrB[i + 1] = static_cast<typename ATensor::element_type>(
            scale1 * static_cast<float>(tDrB[i + 1]));
      }
      CUTE_UNROLL
      for (int i = 48; i < 64; i += 2) {
        tDrB[i] = static_cast<typename ATensor::element_type>(
            scale2 * static_cast<float>(tDrB[i]));
        tDrB[i + 1] = static_cast<typename ATensor::element_type>(
            scale3 * static_cast<float>(tDrB[i + 1]));
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < 32; i += 2) {
        tDrB[i] = static_cast<typename ATensor::element_type>(
            scale0 * static_cast<float>(tDrB[i]));
        tDrB[i + 1] = static_cast<typename ATensor::element_type>(
            scale1 * static_cast<float>(tDrB[i + 1]));
      }
    }

    gemm(mma, tDrA, tDrB, tDrD);
    barrier_wait(barrier_scope);
  }
  auto tDrD_final = thr_copy_d.partition_sg_fragment_S(gD);
  reorder(tDrD, tDrD_final);
  copy(copy_d, tDrD_final, tDgD);
}

} // namespace MoE
