/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

///////////////////////////////////////////////////////////////////////////////
//
// Dual-GEMM with SLM (Shared Local Memory) intermediate buffer example.
//
// This example demonstrates:
//   1. GEMM1:  C1[M,N] = A1[M,K1] x B1[N,K1]   (global mem -> accum fp32 -> bf16/fp16 -> SLM)
//   2. GEMM2:  C2[M,P] = C1_slm[M,N] x B2[P,N]  (SLM read -> accum fp32 -> bf16/fp16 -> global)
//
// C1 produced by GEMM1 is stored in SLM as bf16/fp16.
// C1 can serve as GEMM2's A operand in row-major (no transpose) or
// column-major (transposed) layout via a template parameter InvertNM.
//
// The key SLM patterns are adapted from sdpa_backward-slm.cpp:
//   - save_slm:           scatter fp32 accumulator to SLM as half
//   - gemm_kernel_Aslm:   read A from SLM element-by-element using coord map
//
///////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <random>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "../common/sycl_cute_common.hpp"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Trait: bundles tile shapes, subgroup layout, TiledMMA types for both GEMMs
///////////////////////////////////////////////////////////////////////////////

template <typename DType_,
          int kBlockM_,     // M tile for GEMM1 output / GEMM2 A
          int kBlockN_,     // N tile for GEMM1 output (K dim of GEMM2)
          int kBlockP_,     // P tile for GEMM2 output N dim
          int kK1_,         // K tile for GEMM1
          int kNSGs_,       // number of subgroups per workgroup
          int AtomLayoutM1_,  // SG tiling M for GEMM1
          int AtomLayoutM2_>  // SG tiling M for GEMM2
struct DualGemmTrait {
    using DType = DType_;
    using VType = float;  // accumulation type

    static constexpr int kBlockM  = kBlockM_;
    static constexpr int kBlockN  = kBlockN_;
    static constexpr int kBlockP  = kBlockP_;
    static constexpr int kK1      = kK1_;
    static constexpr int kNSGs    = kNSGs_;
    static constexpr int SubgroupSize = 16;

    // SLM size in elements for the intermediate C1 buffer [kBlockM x kBlockN]
    static constexpr int slm_size = kBlockM * kBlockN;

    // MMA atom: XE_DPAS_TT<systolic_depth=8, accum, input>
    using MMA_Atom_ARCH = XE_DPAS_TT<8, VType, DType>;
    using _K = Int<MMA_Atom_ARCH::K>;

    // ---------- GEMM1: C1[kBlockM, kBlockN] = A1[kBlockM, kK1] x B1[kBlockN, kK1] ----------
    using SubgroupLayout1 = Layout<Shape<Int<AtomLayoutM1_>, Int<kNSGs / AtomLayoutM1_>, _1>>;
    using TileShape1 = Layout<Shape<Int<kBlockM>, Int<kBlockN>, _K>>;
    using TiledMma1 = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                              TileShape1,
                                              SubgroupLayout1>::TiledMMA;

    // ---------- GEMM2: C2[kBlockM, kBlockP] = C1[kBlockM, kBlockN] x B2[kBlockP, kBlockN] ----
    using SubgroupLayout2 = Layout<Shape<Int<AtomLayoutM2_>, Int<kNSGs / AtomLayoutM2_>, _1>>;
    using TileShape2 = Layout<Shape<Int<kBlockM>, Int<kBlockP>, _K>>;
    using TiledMma2 = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                              TileShape2,
                                              SubgroupLayout2>::TiledMMA;

    DualGemmTrait() {}
};

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel: reads A,B from global memory, accumulates into fp32 register
///////////////////////////////////////////////////////////////////////////////

template <bool clear_acc, class Trait,
          class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2, class TVLayout2,
          class TiledMMA>
CUTLASS_DEVICE void
gemm_kernel(Trait &trait,
            Tensor<Engine0, Layout0> const &A,
            Tensor<Engine1, Layout1> const &B,
            SubgroupTensor<Engine2, Layout2, TVLayout2> &acc,
            TiledMMA const &mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0, 2>(tile_mnk), make_coord(0, _));
    Tensor gB = local_tile(cB, select<1, 2>(tile_mnk), make_coord(0, _));

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    auto thr_mma    = mma.get_slice(first_thread_in_sg_idx);
    auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
    auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_prefetch_A = prefetch_a.get_slice(first_thread_in_sg_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(first_thread_in_sg_idx);
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    constexpr int prefetch_dist = 3;
    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_prefetch = 0;

    if constexpr (clear_acc) clear(acc);

    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count; ++k_tile_prefetch) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
        barrier_arrive(barrier_scope);
        copy(copy_a, tAgA(_, _, _, k_tile), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
        if (k_tile_prefetch < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
            prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
        }
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);
        gemm(mma, tCrA, tCrB, acc);
        barrier_wait(barrier_scope);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Save fp32 accumulator to SLM as DType (bf16/fp16).
// Uses per-lane partition_C coordinates.
///////////////////////////////////////////////////////////////////////////////

template <class Trait, class TiledMma,
          class Engine0, class Layout0, class TVLayout0,
          class Engine1, class Layout1>
CUTLASS_DEVICE void
save_slm(Trait &trait, TiledMma &tiled_mma,
         SubgroupTensor<Engine0, Layout0, TVLayout0> &acc,
         Tensor<Engine1, Layout1> &s) {
    using DType = typename Trait::DType;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    int local_id = sg.get_local_id();
    auto tile_mnk = tiled_mma.tile_mnk();

    Tensor cC = make_identity_tensor(s.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx + local_id);
    Tensor tCgC = thr_mma.partition_C(gC);

    CUTE_UNROLL
    for (int i = 0; i < size(tCgC); ++i) {
        auto [mi, ni] = tCgC(i);
        s(mi, ni) = static_cast<DType>(acc(i));
    }
}

///////////////////////////////////////////////////////////////////////////////
// Packed SLM store: pack two adjacent bf16/fp16 values into one uint32,
// write to SLM using inline ASM lsc_store.slm d32.
//
// Adjacent partition_C values (v_i, v_{i+1}) are consecutive columns
// in the same row. Packing stores two bf16 in one d32 scatter write,
// halving the number of SLM store instructions vs d16u32.
///////////////////////////////////////////////////////////////////////////////

template <class Trait, class TiledMma,
          class Engine0, class Layout0, class TVLayout0,
          class Engine1, class Layout1>
CUTLASS_DEVICE void
save_slm_packed(Trait &trait, TiledMma &tiled_mma,
                SubgroupTensor<Engine0, Layout0, TVLayout0> &acc,
                Tensor<Engine1, Layout1> &s) {
    using DType = typename Trait::DType;
    static_assert(sizeof(DType) == sizeof(uint16_t));
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    int local_id = sg.get_local_id();
    auto tile_mnk = tiled_mma.tile_mnk();

    Tensor cC = make_identity_tensor(s.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx + local_id);
    Tensor tCgC = thr_mma.partition_C(gC);

    CUTE_UNROLL
    for (int i = 0; i < size(tCgC); ++i) {
        auto [mi, ni] = tCgC(i);
        // Convert fp32 accumulator -> DType
        DType v = static_cast<DType>(acc(i));
        uint16_t bits = *reinterpret_cast<uint16_t*>(&v);

        // Shuffle: get adjacent lane's value (XOR 1 swaps even<->odd lanes).
        // Identity-tensor is col-major, so lane T owns column T.
        // Adjacent lanes (T, T^1) have adjacent columns — adjacent in SLM.
        uint16_t adj = sycl::select_from_group(sg, bits, local_id ^ 1);

        if (local_id % 2 == 0) {
            // Even lane packs own value (lo) + odd lane's value (hi)
            // and writes uint32 covering columns ni and ni+1
            uint32_t packed = uint32_t(bits) | (uint32_t(adj) << 16);
            *reinterpret_cast<uint32_t*>(&s(mi, ni)) = packed;
        }
        // Odd lanes skip — their data is in the even lane's d32 write
    }
}

///////////////////////////////////////////////////////////////////////////////
// Load from a 2D tensor (SLM or gmem) into MMA register fragment.
// Uses per-lane MMA partition_A coordinates.
///////////////////////////////////////////////////////////////////////////////

template <class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2, class TVLayout2>
CUTLASS_DEVICE void
load_slm(Tensor<Engine0, Layout0> const &s,
         Tensor<Engine1, Layout1> const &g,
         SubgroupTensor<Engine2, Layout2, TVLayout2> &r) {
    CUTE_UNROLL
    for (int i = 0; i < size(g); ++i) {
        auto [mi, ni] = g(i);
        r(i) = s(mi, ni);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Packed SLM load: d32 scatter load from SLM via inline ASM,
// unpack uint32 into two bf16/fp16 values.
// Uses partition_C coordinates (same pairing as save_slm_packed).
// Results are written element-wise to a 2D output tensor.
///////////////////////////////////////////////////////////////////////////////

template <class Trait, class TiledMma,
          class Engine0, class Layout0,
          class Engine1, class Layout1>
CUTLASS_DEVICE void
load_slm_packed(Trait &trait, TiledMma &tiled_mma,
                Tensor<Engine0, Layout0> const &s,
                Tensor<Engine1, Layout1> &gOut) {
    using DType = typename Trait::DType;
    static_assert(sizeof(DType) == sizeof(uint16_t));
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    int local_id = sg.get_local_id();
    auto tile_mnk = tiled_mma.tile_mnk();

    Tensor cC = make_identity_tensor(s.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx + local_id);
    Tensor tCgC = thr_mma.partition_C(gC);

    CUTE_UNROLL
    for (int i = 0; i < size(tCgC); ++i) {
        auto [mi, ni] = tCgC(i);
        // d32 load from even-aligned SLM address
        // Even lane ni is even, odd lane ni is odd.
        // Round ni down to even to get 4-byte aligned address.
        int ni_even = ni & ~1;
        uint32_t loaded = *reinterpret_cast<uint32_t const*>(&s(mi, ni_even));

        // Even lane takes lo 16 bits, odd lane takes hi 16 bits
        DType v;
        if (local_id % 2 == 0) {
            uint16_t lo = loaded & 0xFFFF;
            *reinterpret_cast<uint16_t*>(&v) = lo;
        } else {
            uint16_t hi = loaded >> 16;
            *reinterpret_cast<uint16_t*>(&v) = hi;
        }
        gOut(mi, ni) = v;
    }
}

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel that reads A from a 2D tensor (SLM or gmem), B from global memory.
// Uses MMA partition_A with per-lane offset for scatter read coordinates,
// filling tCrA directly (skips copy-A tArA + reorder for A path).
//
// The A tensor's layout encodes any transpose: for transposed reads, pass
// a transposed view of the SLM buffer so the caller controls the mapping.
///////////////////////////////////////////////////////////////////////////////

template <class Trait,
          class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2, class TVLayout2,
          class TiledMMA>
CUTLASS_DEVICE void
gemm_kernel_Aslm(Trait &trait,
                 Tensor<Engine0, Layout0> const &Aslm,
                 Tensor<Engine1, Layout1> const &B,
                 SubgroupTensor<Engine2, Layout2, TVLayout2> &acc,
                 TiledMMA const &mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    int local_id = sg.get_local_id();

    Tensor cA = make_identity_tensor(Aslm.shape());
    Tensor cB = make_identity_tensor(B.shape());
    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0, 2>(tile_mnk), make_coord(0, _));
    Tensor gB = local_tile(cB, select<1, 2>(tile_mnk), make_coord(0, _));

    auto copy_b = make_block_2d_copy_B(mma, B);
    auto thr_mma    = mma.get_slice(first_thread_in_sg_idx);
    auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);

    // Per-lane MMA partition for A scatter-read coordinates
    auto thr_mma_lane = mma.get_slice(first_thread_in_sg_idx + local_id);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    // Per-lane coordinate partition for A (using MMA partition_A, not copy partition_S)
    Tensor tCcA = thr_mma_lane.partition_A(gA);

    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_prefetch_B = prefetch_b.get_slice(first_thread_in_sg_idx);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    constexpr int barrier_scope = 2;
    int k_tile_count = ceil_div(shape<1>(Aslm), get<2>(tile_mnk));
    int k_tile_prefetch = 0;

    clear(acc);

    CUTE_UNROLL
    for (; k_tile_prefetch < 3 && k_tile_prefetch < k_tile_count; ++k_tile_prefetch)
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
        barrier_arrive(barrier_scope);

        // Read A from SLM/gmem using per-lane MMA partition_A coordinates
        load_slm(Aslm, tCcA(_, _, _, k_tile), tCrA);

        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
        if (k_tile_prefetch < k_tile_count)
            prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

        // Only reorder B (A is already in MMA layout via tCrA)
        reorder(tBrB, tCrB);
        gemm(mma, tCrA, tCrB, acc);
        barrier_wait(barrier_scope);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Reorder fp32 accumulator to 16-bit and copy to global memory
///////////////////////////////////////////////////////////////////////////////

template <class Trait, class TiledMma,
          class Engine0, class Layout0, class TVLayout0,
          class Engine1, class Layout1>
CUTLASS_DEVICE void
reorder_copy(Trait &trait, TiledMma &tiled_mma,
             SubgroupTensor<Engine0, Layout0, TVLayout0> &acc,
             Tensor<Engine1, Layout1> &mOut) {
    using DType = typename Trait::DType;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto tile_mnk = tiled_mma.tile_mnk();

    // Create 16-bit register fragment matching the copy-D layout
    Tensor cC = make_identity_tensor(mOut.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto copy_c = make_block_2d_copy_D(tiled_mma, mOut);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto r16 = thr_copy_c.partition_sg_fragment_S(gC);
    Tensor tCgC = thr_copy_c.partition_D(gC);

    // Reorder fp32 acc -> 16-bit layout, then write to global
    reorder(acc, r16);
    copy(copy_c, r16, tCgC);
}

///////////////////////////////////////////////////////////////////////////////
// create_acc_reg: allocate fp32 accumulator register fragment
///////////////////////////////////////////////////////////////////////////////

template <class Trait, class MTensor, class TiledMMA>
auto
create_acc_reg(Trait const &trait,
               MTensor const &C,
               TiledMMA const &tiled_mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);
    auto tile_mnk = tiled_mma.tile_mnk();
    auto r32 = thr_mma.partition_sg_fragment_C(
        make_identity_tensor(select<0, 1>(tile_mnk)));
    return r32;
}

///////////////////////////////////////////////////////////////////////////////
// Test kernel 1: Single GEMM → write to global memory via block 2D copy
// Verifies that gemm_kernel + reorder_copy produce correct C1.
///////////////////////////////////////////////////////////////////////////////

template <class Trait>
CUTLASS_DEVICE void
single_gemm_device(Trait trait,
                   typename Trait::DType const *A_ptr,
                   typename Trait::DType const *B_ptr,
                   typename Trait::DType *C_ptr,
                   int K) {
    using DType = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;

    Tensor mA = make_tensor(make_gmem_ptr(A_ptr),
                            make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr),
                            make_layout(make_shape(Int<kBlockN>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));
    Tensor mC = make_tensor(make_gmem_ptr(C_ptr),
                            make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                        make_stride(Int<kBlockN>{}, _1{})));

    typename Trait::TiledMma1 tiled_mma;
    auto acc = create_acc_reg(trait, mC, tiled_mma);
    gemm_kernel<true>(trait, mA, mB, acc, tiled_mma);
    reorder_copy(trait, tiled_mma, acc, mC);
}

///////////////////////////////////////////////////////////////////////////////
// Test kernel 2: GEMM → SLM (scatter write) → dump SLM to global (plain copy)
// Also writes GEMM result to a second gmem buffer via block 2D copy for comparison.
// Also scatter-writes to a gmem buffer using the same partition_C coordinates.
///////////////////////////////////////////////////////////////////////////////

template <class Trait>
CUTLASS_DEVICE void
gemm_slm_roundtrip_device(Trait trait,
                          typename Trait::DType const *A_ptr,
                          typename Trait::DType const *B_ptr,
                          typename Trait::DType *C1_2dcopy_ptr,  // GEMM1 via 2D store
                          typename Trait::DType *slm_dump_ptr,   // SLM content dump
                          typename Trait::DType *scatter_gmem_ptr, // scatter to gmem (same coords as SLM)
                          float *coord_dump_ptr,  // debug: dump (sg_id, i, m, n, flat_idx, val)
                          int K) {
    using DType = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();

    Tensor mA = make_tensor(make_gmem_ptr(A_ptr),
                            make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr),
                            make_layout(make_shape(Int<kBlockN>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));
    Tensor mC1_gmem = make_tensor(make_gmem_ptr(C1_2dcopy_ptr),
                                  make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                              make_stride(Int<kBlockN>{}, _1{})));

    auto *slm_buf = compat::local_mem<DType[kBlockM * kBlockN]>();
    // 2D SLM tensor for C1 [M, N] row-major
    auto sC1 = make_tensor(make_smem_ptr(slm_buf),
                           make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                       make_stride(Int<kBlockN>{}, _1{})));
    // 2D gmem tensor for scatter comparison
    auto gScatter = make_tensor(make_gmem_ptr(scatter_gmem_ptr),
                                make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                            make_stride(Int<kBlockN>{}, _1{})));

    // Zero-init SLM
    {
        int tid = sg.get_group_linear_id() * Trait::SubgroupSize + sg.get_local_id();
        int total_threads = Trait::kNSGs * Trait::SubgroupSize;
        for (int e = tid; e < kBlockM * kBlockN; e += total_threads) {
            slm_buf[e] = DType(0.f);
        }
        sycl::group_barrier(group);
    }

    // GEMM1
    typename Trait::TiledMma1 tiled_mma;
    auto acc = create_acc_reg(trait, sC1, tiled_mma);
    gemm_kernel<true>(trait, mA, mB, acc, tiled_mma);

    // Write to gmem via 2D block store (known-good path)
    reorder_copy(trait, tiled_mma, acc, mC1_gmem);

    // ---- Scatter write using partition_C coordinates ----
    // Dump coordinates to debug buffer
    {
        auto tile_mnk = tiled_mma.tile_mnk();
        auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
        int local_id = sg.get_local_id();
        Tensor cC = make_identity_tensor(sC1.shape());
        Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
        auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx + local_id);
        Tensor tCgC = thr_mma.partition_C(gC);

        int sg_id = sg.get_group_linear_id();
        // Only lane 0 of each subgroup dumps coordinates
        if (local_id == 0) {
            int base = sg_id * size(tCgC) * 5;  // 5 floats per entry
            for (int i = 0; i < size(tCgC); ++i) {
                auto [m, n] = tCgC(i);
                float val = acc(i);
                coord_dump_ptr[base + i * 5 + 0] = float(sg_id);
                coord_dump_ptr[base + i * 5 + 1] = float(i);
                coord_dump_ptr[base + i * 5 + 2] = float(m);
                coord_dump_ptr[base + i * 5 + 3] = float(n);
                coord_dump_ptr[base + i * 5 + 4] = val;
            }
        }

        CUTE_UNROLL
        for (int i = 0; i < size(tCgC); ++i) {
            auto [m, n] = tCgC(i);
            DType val = static_cast<DType>(acc(i));
            sC1(m, n) = val;
            gScatter(m, n) = val;
        }
    }

    // Barrier
    sycl::group_barrier(group);

    // Dump SLM to gmem
    {
        int tid = sg.get_group_linear_id() * Trait::SubgroupSize + sg.get_local_id();
        int total_threads = Trait::kNSGs * Trait::SubgroupSize;
        for (int e = tid; e < kBlockM * kBlockN; e += total_threads) {
            slm_dump_ptr[e] = slm_buf[e];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Test kernel 3: GEMM → SLM → use gmem as A source for GEMM2 (bypass SLM read)
// To isolate SLM read vs SLM write issues
///////////////////////////////////////////////////////////////////////////////

template <class Trait>
CUTLASS_DEVICE void
dual_gemm_gmem_intermediate_device(Trait trait,
                                   typename Trait::DType const *A1_ptr,
                                   typename Trait::DType const *B1_ptr,
                                   typename Trait::DType const *B2_ptr,
                                   typename Trait::DType *C1_ptr,   // intermediate C1 in gmem
                                   typename Trait::DType *C2_ptr,
                                   int K1, int N, int P) {
    using DType = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kBlockP = Trait::kBlockP;

    // GEMM1 operands
    Tensor mA1 = make_tensor(make_gmem_ptr(A1_ptr),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                         make_stride(K1, _1{})));
    Tensor mB1 = make_tensor(make_gmem_ptr(B1_ptr),
                             make_layout(make_shape(Int<kBlockN>{}, Int<kBlockN>{}),
                                         make_stride(K1, _1{})));
    // C1 in gmem (row-major)
    Tensor mC1 = make_tensor(make_gmem_ptr(C1_ptr),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                         make_stride(Int<kBlockN>{}, _1{})));
    // B2
    Tensor mB2 = make_tensor(make_gmem_ptr(B2_ptr),
                             make_layout(make_shape(Int<kBlockP>{}, Int<kBlockN>{}),
                                         make_stride(N, _1{})));
    // C2 output
    Tensor mC2 = make_tensor(make_gmem_ptr(C2_ptr),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockP>{}),
                                         make_stride(P, _1{})));

    // GEMM1
    typename Trait::TiledMma1 tiled_mma1;
    auto acc1 = create_acc_reg(trait, mC1, tiled_mma1);
    gemm_kernel<true>(trait, mA1, mB1, acc1, tiled_mma1);
    reorder_copy(trait, tiled_mma1, acc1, mC1);

    // Fence/barrier not needed because C1 is in gmem and we just wrote it
    // but need to ensure writes are visible before reads
    sycl::group_barrier(compat::get_nd_item<1>().get_group());

    // GEMM2: read A from gmem (the C1 we just wrote)
    // A2 = C1[kBlockM, kBlockN], B2 = B2[kBlockP, kBlockN]
    Tensor mA2 = make_tensor(make_gmem_ptr(static_cast<DType const *>(C1_ptr)),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                         make_stride(Int<kBlockN>{}, _1{})));
    typename Trait::TiledMma2 tiled_mma2;
    auto acc2 = create_acc_reg(trait, mC2, tiled_mma2);
    gemm_kernel<true>(trait, mA2, mB2, acc2, tiled_mma2);
    reorder_copy(trait, tiled_mma2, acc2, mC2);
}

///////////////////////////////////////////////////////////////////////////////
// Test kernel: Scatter-read from GMEM (not SLM) using gemm_kernel_Aslm coords
// This tests whether partition_A coordinates are correct.
///////////////////////////////////////////////////////////////////////////////

template <class Trait>
CUTLASS_DEVICE void
scatter_read_test_device(Trait trait,
                         typename Trait::DType const *A2_gmem_ptr,  // correct C1 data in gmem
                         typename Trait::DType const *B2_ptr,
                         typename Trait::DType *C2_ptr,
                         int N, int P) {
    using DType = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kBlockP = Trait::kBlockP;

    // A2 as 2D gmem tensor (row-major, same layout as C1)
    Tensor gA2 = make_tensor(make_gmem_ptr(A2_gmem_ptr),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                         make_stride(Int<kBlockN>{}, _1{})));
    // B2
    Tensor mB2 = make_tensor(make_gmem_ptr(B2_ptr),
                             make_layout(make_shape(Int<kBlockP>{}, Int<kBlockN>{}),
                                         make_stride(N, _1{})));
    // C2 output
    Tensor mC2 = make_tensor(make_gmem_ptr(C2_ptr),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockP>{}),
                                         make_stride(P, _1{})));

    typename Trait::TiledMma2 tiled_mma2;
    auto acc2 = create_acc_reg(trait, mC2, tiled_mma2);

    // Use gemm_kernel_Aslm with gmem 2D tensor (same scatter-read logic, no SLM)
    gemm_kernel_Aslm(trait, gA2, mB2, acc2, tiled_mma2);

    reorder_copy(trait, tiled_mma2, acc2, mC2);
}

///////////////////////////////////////////////////////////////////////////////
// Test kernel 8: Packed SLM roundtrip (plain store + plain load)
// GEMM -> save_slm_packed (d32 store) -> barrier -> flat dump + load_slm_packed
///////////////////////////////////////////////////////////////////////////////

template <class Trait>
CUTLASS_DEVICE void
packed_slm_roundtrip_device(Trait trait,
                           typename Trait::DType const *A_ptr,
                           typename Trait::DType const *B_ptr,
                           typename Trait::DType *slm_dump_ptr,
                           typename Trait::DType *packed_load_ptr,
                           int K) {
    using DType = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();

    Tensor mA = make_tensor(make_gmem_ptr(A_ptr),
                            make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr),
                            make_layout(make_shape(Int<kBlockN>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));

    auto *slm_buf = compat::local_mem<DType[kBlockM * kBlockN]>();
    auto sC1 = make_tensor(make_smem_ptr(slm_buf),
                           make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                       make_stride(Int<kBlockN>{}, _1{})));

    // Gmem output for packed-load results
    auto gLoad = make_tensor(make_gmem_ptr(packed_load_ptr),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                         make_stride(Int<kBlockN>{}, _1{})));

    // Zero-init SLM
    {
        int tid = sg.get_group_linear_id() * Trait::SubgroupSize + sg.get_local_id();
        int total_threads = Trait::kNSGs * Trait::SubgroupSize;
        for (int e = tid; e < kBlockM * kBlockN; e += total_threads)
            slm_buf[e] = DType(0.f);
        sycl::group_barrier(group);
    }

    // GEMM
    typename Trait::TiledMma1 tiled_mma;
    auto acc = create_acc_reg(trait, sC1, tiled_mma);
    gemm_kernel<true>(trait, mA, mB, acc, tiled_mma);

    // Packed store to SLM (d32)
    save_slm_packed(trait, tiled_mma, acc, sC1);
    sycl::group_barrier(group);

    // 1) Flat dump SLM to gmem (verify packed store wrote correct bf16)
    {
        int tid = sg.get_group_linear_id() * Trait::SubgroupSize + sg.get_local_id();
        int total_threads = Trait::kNSGs * Trait::SubgroupSize;
        for (int e = tid; e < kBlockM * kBlockN; e += total_threads)
            slm_dump_ptr[e] = slm_buf[e];
    }

    // 2) Packed load from SLM (d32) -> unpack -> write to gmem
    load_slm_packed(trait, tiled_mma, sC1, gLoad);
}

///////////////////////////////////////////////////////////////////////////////
// Test kernel 9: Packed SLM store -> transpose read
// GEMM -> save_slm_packed (d32 store) -> barrier -> flat transpose dump
///////////////////////////////////////////////////////////////////////////////

template <class Trait>
CUTLASS_DEVICE void
packed_slm_transpose_test_device(Trait trait,
                                typename Trait::DType const *A_ptr,
                                typename Trait::DType const *B_ptr,
                                typename Trait::DType *transpose_dump_ptr,
                                int K) {
    using DType = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();

    Tensor mA = make_tensor(make_gmem_ptr(A_ptr),
                            make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr),
                            make_layout(make_shape(Int<kBlockN>{}, Int<kBlockN>{}),
                                        make_stride(K, _1{})));

    auto *slm_buf = compat::local_mem<DType[kBlockM * kBlockN]>();
    auto sC1 = make_tensor(make_smem_ptr(slm_buf),
                           make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                       make_stride(Int<kBlockN>{}, _1{})));

    // Zero-init SLM
    {
        int tid = sg.get_group_linear_id() * Trait::SubgroupSize + sg.get_local_id();
        int total_threads = Trait::kNSGs * Trait::SubgroupSize;
        for (int e = tid; e < kBlockM * kBlockN; e += total_threads)
            slm_buf[e] = DType(0.f);
        sycl::group_barrier(group);
    }

    // GEMM
    typename Trait::TiledMma1 tiled_mma;
    auto acc = create_acc_reg(trait, sC1, tiled_mma);
    gemm_kernel<true>(trait, mA, mB, acc, tiled_mma);

    // Packed store to SLM (d32)
    save_slm_packed(trait, tiled_mma, acc, sC1);
    sycl::group_barrier(group);

    // Flat transpose dump: read C1[m,n] from SLM, write as C1_T[n,m] to gmem
    {
        int tid = sg.get_group_linear_id() * Trait::SubgroupSize + sg.get_local_id();
        int total_threads = Trait::kNSGs * Trait::SubgroupSize;
        for (int e = tid; e < kBlockM * kBlockN; e += total_threads) {
            int m = e / kBlockN;
            int n = e % kBlockN;
            transpose_dump_ptr[n * kBlockM + m] = slm_buf[m * kBlockN + n];
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Device kernel: dual GEMM with SLM intermediate
//
// Template parameter InvertNM controls whether C1 is read transposed for GEMM2:
//   false => GEMM2 A = C1[M,N] (row-major, no transpose)
//   true  => GEMM2 A = C1^T[N,M] (column-major, transposed)
//
// Shapes:
//   GEMM1: A1[kBlockM, kK1] x B1[kBlockN, kK1] -> C1[kBlockM, kBlockN]
//   GEMM2: C1[kBlockM, kBlockN] x B2[kBlockP, kBlockN] -> C2[kBlockM, kBlockP]
//       or C1^T[kBlockN, kBlockM] x B2[kBlockP, kBlockM] -> C2[kBlockN, kBlockP]
///////////////////////////////////////////////////////////////////////////////

template <bool InvertNM, class Trait>
CUTLASS_DEVICE void
dual_gemm_slm_device(Trait trait,
                     typename Trait::DType const *A1_ptr,
                     typename Trait::DType const *B1_ptr,
                     typename Trait::DType const *B2_ptr,
                     typename Trait::DType       *C2_ptr,
                     int M, int K1, int N, int P) {
    using DType = typename Trait::DType;
    using VType = typename Trait::VType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kBlockP = Trait::kBlockP;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();

    // ---- Create global memory tensor views ----

    // GEMM1 operands
    Tensor mA1 = make_tensor(make_gmem_ptr(A1_ptr),
                             make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                         make_stride(K1, _1{})));
    Tensor mB1 = make_tensor(make_gmem_ptr(B1_ptr),
                             make_layout(make_shape(Int<kBlockN>{}, Int<kBlockN>{}),
                                         make_stride(K1, _1{})));

    // GEMM2 B operand
    Tensor mB2 = make_tensor(make_gmem_ptr(B2_ptr),
                             make_layout(make_shape(Int<kBlockP>{}, Int<kBlockN>{}),
                                         make_stride(N, _1{})));

    // Output C2
    auto C2_M = InvertNM ? Int<kBlockN>{} : Int<kBlockM>{};
    Tensor mC2 = make_tensor(make_gmem_ptr(C2_ptr),
                             make_layout(make_shape(C2_M, Int<kBlockP>{}),
                                         make_stride(P, _1{})));

    // ---- Allocate SLM for intermediate C1 ----
    auto *slm_buf = compat::local_mem<DType[kBlockM * kBlockN]>();

    // 2D SLM tensor: C1 stored as [M, N] row-major
    auto sC1 = make_tensor(make_smem_ptr(slm_buf),
                           make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                       make_stride(Int<kBlockN>{}, _1{})));

    // ---- GEMM1: C1 = A1 x B1, accum in fp32 ----
    typename Trait::TiledMma1 tiled_mma1;
    auto acc1 = create_acc_reg(trait, sC1, tiled_mma1);
    gemm_kernel<true>(trait, mA1, mB1, acc1, tiled_mma1);

    // Convert fp32 accumulator to bf16/fp16 and scatter-write into SLM
    save_slm(trait, tiled_mma1, acc1, sC1);

    // Barrier: ensure all subgroups have written their C1 tiles to SLM
    sycl::group_barrier(group);

    // ---- GEMM2: C2 = C1_slm x B2, accum in fp32 ----
    // Create SLM read view: transpose is encoded in the tensor layout
    //   no transpose: A2[M,N] = C1[M,N] → same layout as sC1
    //   transpose:    A2[N,M] = C1^T    → transposed view with stride [1, N]
    auto sA2 = [&]() {
        if constexpr (!InvertNM) {
            return sC1;
        } else {
            return make_tensor(make_smem_ptr(slm_buf),
                               make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                                           make_stride(_1{}, Int<kBlockN>{})));
        }
    }();

    typename Trait::TiledMma2 tiled_mma2;
    auto acc2 = create_acc_reg(trait, mC2, tiled_mma2);

    // Read A from SLM (layout handles transpose), B2 from global
    gemm_kernel_Aslm(trait, sA2, mB2, acc2, tiled_mma2);

    // Convert fp32 accumulator to bf16/fp16 and write to global memory
    reorder_copy(trait, tiled_mma2, acc2, mC2);
}

///////////////////////////////////////////////////////////////////////////////
// Kernel entry point name tags
///////////////////////////////////////////////////////////////////////////////

template <class...> class SingleGemmKernelName;
template <class...> class SlmRoundtripKernelName;
template <class...> class DualGemmGmemKernelName;
template <class...> class ScatterReadTestKernelName;
template <class...> class DualGemmSlmKernelName;
template <class...> class PackedSlmRoundtripKernelName;
template <class...> class PackedSlmTransposeKernelName;

///////////////////////////////////////////////////////////////////////////////
// Host-side launch
///////////////////////////////////////////////////////////////////////////////

// Helper macro for launch boilerplate
#define LAUNCH_KERNEL(KernelNameTag, TraitType, body) \
    do { \
        namespace syclex = sycl::ext::oneapi::experimental; \
        namespace intelex = sycl::ext::intel::experimental; \
        sycl::range<1> _local{static_cast<size_t>(TraitType::kNSGs * TraitType::SubgroupSize)}; \
        sycl::range<1> _global{_local}; \
        syclex::properties _kprops{syclex::sub_group_size<TraitType::SubgroupSize>, intelex::grf_size<256>}; \
        auto event = Q.parallel_for<KernelNameTag>( \
            sycl::nd_range<1>(_global, _local), _kprops, \
            [=](auto) { body; }); \
        EventManager::getInstance().addEvent(event); \
    } while(0)

template <typename DType,
          int kBlockM, int kBlockN, int kBlockP, int kK1,
          int kNSGs, int AtomLayoutM1, int AtomLayoutM2>
void launch_single_gemm(sycl::queue &Q,
                        DType const *A, DType const *B, DType *C, int K) {
    using Trait = DualGemmTrait<DType, kBlockM, kBlockN, kBlockP, kK1,
                                kNSGs, AtomLayoutM1, AtomLayoutM2>;
    Trait trait{};
    LAUNCH_KERNEL(SingleGemmKernelName<Trait>, Trait,
                  single_gemm_device<Trait>(trait, A, B, C, K));
}

template <typename DType,
          int kBlockM, int kBlockN, int kBlockP, int kK1,
          int kNSGs, int AtomLayoutM1, int AtomLayoutM2>
void launch_slm_roundtrip(sycl::queue &Q,
                          DType const *A, DType const *B,
                          DType *C1_2dcopy, DType *slm_dump,
                          DType *scatter_gmem, float *coord_dump, int K) {
    using Trait = DualGemmTrait<DType, kBlockM, kBlockN, kBlockP, kK1,
                                kNSGs, AtomLayoutM1, AtomLayoutM2>;
    Trait trait{};
    LAUNCH_KERNEL(SlmRoundtripKernelName<Trait>, Trait,
                  gemm_slm_roundtrip_device<Trait>(trait, A, B, C1_2dcopy, slm_dump, scatter_gmem, coord_dump, K));
}

template <typename DType,
          int kBlockM, int kBlockN, int kBlockP, int kK1,
          int kNSGs, int AtomLayoutM1, int AtomLayoutM2>
void launch_dual_gemm_gmem(sycl::queue &Q,
                           DType const *A1, DType const *B1,
                           DType const *B2, DType *C1, DType *C2,
                           int K1, int N, int P) {
    using Trait = DualGemmTrait<DType, kBlockM, kBlockN, kBlockP, kK1,
                                kNSGs, AtomLayoutM1, AtomLayoutM2>;
    Trait trait{};
    LAUNCH_KERNEL(DualGemmGmemKernelName<Trait>, Trait,
                  dual_gemm_gmem_intermediate_device<Trait>(trait, A1, B1, B2, C1, C2, K1, N, P));
}

template <typename DType,
          int kBlockM, int kBlockN, int kBlockP, int kK1,
          int kNSGs, int AtomLayoutM1, int AtomLayoutM2>
void launch_scatter_read_test(sycl::queue &Q,
                              DType const *A2_gmem, DType const *B2,
                              DType *C2, int N, int P) {
    using Trait = DualGemmTrait<DType, kBlockM, kBlockN, kBlockP, kK1,
                                kNSGs, AtomLayoutM1, AtomLayoutM2>;
    Trait trait{};
    LAUNCH_KERNEL(ScatterReadTestKernelName<Trait>, Trait,
                  scatter_read_test_device<Trait>(trait, A2_gmem, B2, C2, N, P));
}

template <typename DType,
          int kBlockM, int kBlockN, int kBlockP, int kK1,
          int kNSGs, int AtomLayoutM1, int AtomLayoutM2>
void launch_packed_slm_roundtrip(sycl::queue &Q,
                                DType const *A, DType const *B,
                                DType *slm_dump, DType *packed_load, int K) {
    using Trait = DualGemmTrait<DType, kBlockM, kBlockN, kBlockP, kK1,
                                kNSGs, AtomLayoutM1, AtomLayoutM2>;
    Trait trait{};
    LAUNCH_KERNEL(PackedSlmRoundtripKernelName<Trait>, Trait,
                  packed_slm_roundtrip_device<Trait>(trait, A, B, slm_dump, packed_load, K));
}

template <typename DType,
          int kBlockM, int kBlockN, int kBlockP, int kK1,
          int kNSGs, int AtomLayoutM1, int AtomLayoutM2>
void launch_packed_slm_transpose_test(sycl::queue &Q,
                                     DType const *A, DType const *B,
                                     DType *transpose_dump, int K) {
    using Trait = DualGemmTrait<DType, kBlockM, kBlockN, kBlockP, kK1,
                                kNSGs, AtomLayoutM1, AtomLayoutM2>;
    Trait trait{};
    LAUNCH_KERNEL(PackedSlmTransposeKernelName<Trait>, Trait,
                  packed_slm_transpose_test_device<Trait>(trait, A, B, transpose_dump, K));
}

template <bool InvertNM, typename DType,
          int kBlockM, int kBlockN, int kBlockP, int kK1,
          int kNSGs, int AtomLayoutM1, int AtomLayoutM2>
void launch_dual_gemm_slm(sycl::queue &Q,
                          DType const *A1, DType const *B1,
                          DType const *B2, DType *C2,
                          int M, int K1, int N, int P) {
    using Trait = DualGemmTrait<DType, kBlockM, kBlockN, kBlockP, kK1,
                                kNSGs, AtomLayoutM1, AtomLayoutM2>;
    Trait trait{};
    using KN = DualGemmSlmKernelName<Trait, std::integral_constant<bool, InvertNM>>;

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    sycl::range<1> local{static_cast<size_t>(kNSGs * Trait::SubgroupSize)};
    sycl::range<1> global{local};
    syclex::properties kprops{syclex::sub_group_size<Trait::SubgroupSize>, intelex::grf_size<256>};

    auto event = Q.parallel_for<KN>(
        sycl::nd_range<1>(global, local), kprops,
        [=](auto) {
            dual_gemm_slm_device<InvertNM, Trait>(trait, A1, B1, B2, C2, M, K1, N, P);
        });
    EventManager::getInstance().addEvent(event);
}

///////////////////////////////////////////////////////////////////////////////
// CPU references
///////////////////////////////////////////////////////////////////////////////

// Single GEMM reference: C[M,N] = A[M,K] x B[N,K]^T (TT layout)
template <typename DType>
void reference_single_gemm(DType const *A, int lda,
                           DType const *B, int ldb,
                           DType *C, int ldc,
                           int M, int K, int N) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += float(A[m * lda + k]) * float(B[n * ldb + k]);
            C[m * ldc + n] = static_cast<DType>(acc);
        }
}

// Dual GEMM reference
template <typename DType>
void reference_dual_gemm(bool transpose_c1,
                         DType const *A1, int lda1,
                         DType const *B1, int ldb1,
                         DType const *B2, int ldb2,
                         DType       *C2, int ldc2,
                         int M, int K1, int N, int P) {
    // Intermediate C1[M, N]
    std::vector<float> C1(M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K1; ++k)
                acc += float(A1[m * lda1 + k]) * float(B1[n * ldb1 + k]);
            C1[m * N + n] = acc;
        }

    // Truncate to DType precision to match GPU path
    std::vector<DType> C1_trunc(M * N);
    for (int i = 0; i < M * N; ++i)
        C1_trunc[i] = static_cast<DType>(C1[i]);

    // GEMM2
    int outM = transpose_c1 ? N : M;
    int K2   = transpose_c1 ? M : N;
    for (int m = 0; m < outM; ++m)
        for (int p = 0; p < P; ++p) {
            float acc = 0.f;
            for (int k = 0; k < K2; ++k) {
                float a_val;
                if (!transpose_c1)
                    a_val = float(C1_trunc[m * N + k]);
                else
                    a_val = float(C1_trunc[k * N + m]);
                float b_val = float(B2[p * N + k]);
                acc += a_val * b_val;
            }
            C2[m * ldc2 + p] = static_cast<DType>(acc);
        }
}

///////////////////////////////////////////////////////////////////////////////
// Verification
///////////////////////////////////////////////////////////////////////////////

template <typename DType>
bool verify_results(DType const *ref, DType const *test, int rows, int cols,
                    float atol = 5e-2f, float rtol = 5e-2f) {
    int mismatches = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = float(ref[i * cols + j]);
            float t = float(test[i * cols + j]);
            float diff = std::abs(r - t);
            if (diff > atol + rtol * std::abs(r)) {
                if (mismatches < 10)
                    printf("  Mismatch at (%d,%d): ref=%f test=%f diff=%f\n",
                           i, j, r, t, diff);
                mismatches++;
            }
        }
    }
    if (mismatches > 0)
        printf("  Total mismatches: %d / %d\n", mismatches, rows * cols);
    return mismatches == 0;
}

///////////////////////////////////////////////////////////////////////////////
// Utility: uniform random initialization (matches sdpa_backward_bench style)
///////////////////////////////////////////////////////////////////////////////

template<typename T>
void
uniform_init(int seed, T *dst, size_t N, float a = -1.0f, float b = 1.0f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(a, b);
    for (size_t i = 0; i < N; ++i) {
        dst[i] = static_cast<T>(dis(gen));
    }
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kBlockP = 64;
    constexpr int kK1     = 64;
    constexpr int kNSGs   = 8;
    constexpr int AtomLayoutM1 = 2;
    constexpr int AtomLayoutM2 = 2;

    int M  = kBlockM;
    int K1 = kK1;
    int N  = kBlockN;
    int P  = kBlockP;

    sycl::queue Q = compat::get_default_queue();
    bool all_pass = true;

    printf("=== Incremental Dual-GEMM with SLM Tests ===\n");
    printf("M=%d K1=%d N=%d P=%d  SubgroupSize=16 kNSGs=%d\n\n", M, K1, N, P, kNSGs);

    // Shared input data (bf16 for all incremental tests)
    using DType = cute::bfloat16_t;
    DType *A1 = sycl::malloc_shared<DType>(M * K1, Q);
    DType *B1 = sycl::malloc_shared<DType>(N * K1, Q);
    DType *B2 = sycl::malloc_shared<DType>(P * N,  Q);

    srand(42);
    uniform_init(42, A1, M * K1);
    uniform_init(43, B1, N * K1);
    uniform_init(44, B2, P * N);

    // CPU reference for GEMM1
    std::vector<DType> C1_ref(M * N);
    reference_single_gemm<DType>(A1, K1, B1, K1, C1_ref.data(), N, M, K1, N);

    // ================================================================
    // Test 1: Single GEMM → global memory (block 2D store)
    // ================================================================
    {
        printf("--- Test 1: Single GEMM -> global (block 2D store) ---\n");
        DType *C1_gpu = sycl::malloc_shared<DType>(M * N, Q);
        for (int i = 0; i < M * N; ++i) C1_gpu[i] = DType(0.f);

        launch_single_gemm<DType, kBlockM, kBlockN, kBlockP, kK1,
                           kNSGs, AtomLayoutM1, AtomLayoutM2>(Q, A1, B1, C1_gpu, K1);
        Q.wait_and_throw();

        bool ok = verify_results(C1_ref.data(), C1_gpu, M, N);
        printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");
        all_pass &= ok;
        sycl::free(C1_gpu, Q);
    }

    // ================================================================
    // Test 2: GEMM → SLM (scatter write) → dump SLM to global
    //         Also writes GEMM result via block 2D store for reference
    // ================================================================
    {
        printf("--- Test 2: GEMM -> SLM -> dump SLM to global ---\n");
        DType *C1_2dcopy  = sycl::malloc_shared<DType>(M * N, Q);
        DType *slm_dump   = sycl::malloc_shared<DType>(M * N, Q);
        DType *scatter_gm = sycl::malloc_shared<DType>(M * N, Q);
        // coord dump: 8 SGs * ~32 elements each * 5 floats = 1280 floats max
        float *coord_dump = sycl::malloc_shared<float>(8 * 64 * 5, Q);
        for (int i = 0; i < M * N; ++i) {
            C1_2dcopy[i] = DType(0.f); slm_dump[i] = DType(0.f);
            scatter_gm[i] = DType(0.f);
        }
        for (int i = 0; i < 8 * 64 * 5; ++i) coord_dump[i] = -999.f;

        launch_slm_roundtrip<DType, kBlockM, kBlockN, kBlockP, kK1,
                             kNSGs, AtomLayoutM1, AtomLayoutM2>(
            Q, A1, B1, C1_2dcopy, slm_dump, scatter_gm, coord_dump, K1);
        Q.wait_and_throw();

        printf("  2a: block 2D store vs CPU ref: ");
        bool ok_2d = verify_results(C1_ref.data(), C1_2dcopy, M, N);
        printf("  %s\n", ok_2d ? "PASSED" : "FAILED");

        printf("  2b: SLM dump vs CPU ref: ");
        bool ok_slm = verify_results(C1_ref.data(), slm_dump, M, N);
        printf("  %s\n", ok_slm ? "PASSED" : "FAILED");

        printf("  2c: gmem scatter vs CPU ref: ");
        bool ok_scatter = verify_results(C1_ref.data(), scatter_gm, M, N);
        printf("  %s\n", ok_scatter ? "PASSED" : "FAILED");

        // Print coordinate dump for SG 0 (first 16 elements)
        printf("  Coord dump (SG 0, first 16 elems): sg_id, i, m, n, val\n");
        for (int i = 0; i < 16; ++i) {
            int base = 0 * 64 * 5;  // SG 0, assuming max 64 elements per SG
            float sg_id_f = coord_dump[base + i * 5 + 0];
            float idx_f   = coord_dump[base + i * 5 + 1];
            float m_f     = coord_dump[base + i * 5 + 2];
            float n_f     = coord_dump[base + i * 5 + 3];
            float val_f   = coord_dump[base + i * 5 + 4];
            printf("    sg=%g i=%g m=%g n=%g val=%g\n", sg_id_f, idx_f, m_f, n_f, val_f);
        }

        // Also print SG 1
        printf("  Coord dump (SG 1, first 8 elems):\n");
        int sg1_base = 1 * 64 * 5;
        for (int i = 0; i < 8; ++i) {
            printf("    sg=%g i=%g m=%g n=%g val=%g\n",
                   coord_dump[sg1_base + i * 5 + 0],
                   coord_dump[sg1_base + i * 5 + 1],
                   coord_dump[sg1_base + i * 5 + 2],
                   coord_dump[sg1_base + i * 5 + 3],
                   coord_dump[sg1_base + i * 5 + 4]);
        }

        printf("\n");
        all_pass &= ok_2d && ok_slm && ok_scatter;
        sycl::free(C1_2dcopy, Q);
        sycl::free(slm_dump, Q);
        sycl::free(scatter_gm, Q);
        sycl::free(coord_dump, Q);
    }

    // ================================================================
    // Test 3: Dual GEMM via gmem intermediate (no SLM)
    //         GEMM1 → gmem → GEMM2 → gmem
    // ================================================================
    {
        printf("--- Test 3: Dual GEMM via gmem intermediate ---\n");
        DType *C1_tmp = sycl::malloc_shared<DType>(M * N, Q);
        DType *C2_gpu = sycl::malloc_shared<DType>(M * P, Q);
        for (int i = 0; i < M * N; ++i) C1_tmp[i] = DType(0.f);
        for (int i = 0; i < M * P; ++i) C2_gpu[i] = DType(0.f);

        launch_dual_gemm_gmem<DType, kBlockM, kBlockN, kBlockP, kK1,
                              kNSGs, AtomLayoutM1, AtomLayoutM2>(
            Q, A1, B1, B2, C1_tmp, C2_gpu, K1, N, P);
        Q.wait_and_throw();

        std::vector<DType> C2_ref(M * P, DType(0.f));
        reference_dual_gemm<DType>(false, A1, K1, B1, K1, B2, N,
                                   C2_ref.data(), P, M, K1, N, P);

        printf("  3a: intermediate C1 vs CPU ref: ");
        bool ok_c1 = verify_results(C1_ref.data(), C1_tmp, M, N);
        printf("  %s\n", ok_c1 ? "PASSED" : "FAILED");

        printf("  3b: final C2 vs CPU ref: ");
        bool ok_c2 = verify_results(C2_ref.data(), C2_gpu, M, P);
        printf("  %s\n\n", ok_c2 ? "PASSED" : "FAILED");

        all_pass &= ok_c1 && ok_c2;
        sycl::free(C1_tmp, Q);
        sycl::free(C2_gpu, Q);
    }

    // ================================================================
    // Test 3.5: Scatter-read from GMEM using gemm_kernel_Aslm coords
    //           Uses known-correct C1 data in gmem, reads with same
    //           partition_S coord logic as SLM path. Isolates read coords.
    // ================================================================
    {
        printf("--- Test 3.5: Scatter-read from gmem (partition_S coords test) ---\n");
        // First, compute correct C1 into gmem
        DType *C1_correct = sycl::malloc_shared<DType>(M * N, Q);
        for (int i = 0; i < M * N; ++i) C1_correct[i] = C1_ref[i];  // use CPU ref

        DType *C2_gpu = sycl::malloc_shared<DType>(M * P, Q);
        for (int i = 0; i < M * P; ++i) C2_gpu[i] = DType(0.f);

        launch_scatter_read_test<DType, kBlockM, kBlockN, kBlockP, kK1,
                                 kNSGs, AtomLayoutM1, AtomLayoutM2>(
            Q, C1_correct, B2, C2_gpu, N, P);
        Q.wait_and_throw();

        std::vector<DType> C2_ref(M * P, DType(0.f));
        reference_dual_gemm<DType>(false, A1, K1, B1, K1, B2, N,
                                   C2_ref.data(), P, M, K1, N, P);

        bool ok = verify_results(C2_ref.data(), C2_gpu, M, P);
        printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");
        all_pass &= ok;
        sycl::free(C1_correct, Q);
        sycl::free(C2_gpu, Q);
    }

    // ================================================================
    // Test 4: Dual GEMM via SLM intermediate (no transpose) — bf16
    // ================================================================
    {
        printf("--- Test 4: Dual GEMM via SLM (no transpose, bf16) ---\n");
        DType *C2_gpu = sycl::malloc_shared<DType>(M * P, Q);
        for (int i = 0; i < M * P; ++i) C2_gpu[i] = DType(0.f);

        launch_dual_gemm_slm<false, DType, kBlockM, kBlockN, kBlockP, kK1,
                              kNSGs, AtomLayoutM1, AtomLayoutM2>(
            Q, A1, B1, B2, C2_gpu, M, K1, N, P);
        Q.wait_and_throw();

        std::vector<DType> C2_ref(M * P, DType(0.f));
        reference_dual_gemm<DType>(false, A1, K1, B1, K1, B2, N,
                                   C2_ref.data(), P, M, K1, N, P);

        bool ok = verify_results(C2_ref.data(), C2_gpu, M, P);
        printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");
        all_pass &= ok;
        sycl::free(C2_gpu, Q);
    }

    // ================================================================
    // Test 5: Dual GEMM via SLM with transpose (InvertNM=true) — bf16
    //         C1 in SLM is read transposed: A2 = C1^T[N,M]
    //         C2[N,P] = C1^T[N,M] x B2[P,M]
    // ================================================================
    {
        printf("--- Test 5: Dual GEMM via SLM (with transpose, bf16) ---\n");
        DType *C2_gpu = sycl::malloc_shared<DType>(N * P, Q);
        for (int i = 0; i < N * P; ++i) C2_gpu[i] = DType(0.f);

        launch_dual_gemm_slm<true, DType, kBlockM, kBlockN, kBlockP, kK1,
                              kNSGs, AtomLayoutM1, AtomLayoutM2>(
            Q, A1, B1, B2, C2_gpu, M, K1, N, P);
        Q.wait_and_throw();

        std::vector<DType> C2_ref(N * P, DType(0.f));
        reference_dual_gemm<DType>(true, A1, K1, B1, K1, B2, N,
                                   C2_ref.data(), P, M, K1, N, P);

        bool ok = verify_results(C2_ref.data(), C2_gpu, N, P);
        printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");
        all_pass &= ok;
        sycl::free(C2_gpu, Q);
    }

    // ================================================================
    // Test 8: Packed SLM roundtrip — inline ASM d32 store + d32 load
    //         GEMM -> save_slm_packed -> flat dump (verify store)
    //                                 -> load_slm_packed -> gmem (verify load)
    // ================================================================
    {
        printf("--- Test 8: Packed SLM roundtrip (d32 store + d32 load) ---\n");
        DType *slm_dump    = sycl::malloc_shared<DType>(M * N, Q);
        DType *packed_load = sycl::malloc_shared<DType>(M * N, Q);
        for (int i = 0; i < M * N; ++i) {
            slm_dump[i] = DType(0.f);
            packed_load[i] = DType(0.f);
        }

        launch_packed_slm_roundtrip<DType, kBlockM, kBlockN, kBlockP, kK1,
                                    kNSGs, AtomLayoutM1, AtomLayoutM2>(
            Q, A1, B1, slm_dump, packed_load, K1);
        Q.wait_and_throw();

        printf("  8a: packed d32 store -> flat dump vs CPU ref: ");
        bool ok_store = verify_results(C1_ref.data(), slm_dump, M, N);
        printf("  %s\n", ok_store ? "PASSED" : "FAILED");

        printf("  8b: packed d32 load -> unpack vs CPU ref: ");
        bool ok_load = verify_results(C1_ref.data(), packed_load, M, N);
        printf("  %s\n\n", ok_load ? "PASSED" : "FAILED");

        all_pass &= ok_store && ok_load;
        sycl::free(slm_dump, Q);
        sycl::free(packed_load, Q);
    }

    // ================================================================
    // Test 9: Packed SLM store -> transpose read
    //         GEMM -> save_slm_packed -> flat transpose dump -> verify C1^T
    // ================================================================
    {
        printf("--- Test 9: Packed SLM store -> transpose read ---\n");
        DType *transpose_dump = sycl::malloc_shared<DType>(N * M, Q);
        for (int i = 0; i < N * M; ++i)
            transpose_dump[i] = DType(0.f);

        launch_packed_slm_transpose_test<DType, kBlockM, kBlockN, kBlockP, kK1,
                                         kNSGs, AtomLayoutM1, AtomLayoutM2>(
            Q, A1, B1, transpose_dump, K1);
        Q.wait_and_throw();

        // Build transposed CPU reference: C1_T[n, m] = C1_ref[m * N + n]
        std::vector<DType> C1_T_ref(N * M);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                C1_T_ref[n * M + m] = C1_ref[m * N + n];

        bool ok = verify_results(C1_T_ref.data(), transpose_dump, N, M);
        printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");
        all_pass &= ok;
        sycl::free(transpose_dump, Q);
    }

    sycl::free(A1, Q);
    sycl::free(B1, Q);
    sycl::free(B2, Q);

    // ================================================================
    // Test 6 & 7: Dual GEMM via SLM with fp16 (no transpose + transpose)
    // ================================================================
    {
        using FP16 = cute::half_t;
        FP16 *A1_f16 = sycl::malloc_shared<FP16>(M * K1, Q);
        FP16 *B1_f16 = sycl::malloc_shared<FP16>(N * K1, Q);
        FP16 *B2_f16 = sycl::malloc_shared<FP16>(P * N,  Q);

        srand(42);
        uniform_init(42, A1_f16, M * K1);
        uniform_init(43, B1_f16, N * K1);
        uniform_init(44, B2_f16, P * N);

        // Test 6: fp16, no transpose
        {
            printf("--- Test 6: Dual GEMM via SLM (no transpose, fp16) ---\n");
            FP16 *C2_gpu = sycl::malloc_shared<FP16>(M * P, Q);
            for (int i = 0; i < M * P; ++i) C2_gpu[i] = FP16(0.f);

            launch_dual_gemm_slm<false, FP16, kBlockM, kBlockN, kBlockP, kK1,
                                  kNSGs, AtomLayoutM1, AtomLayoutM2>(
                Q, A1_f16, B1_f16, B2_f16, C2_gpu, M, K1, N, P);
            Q.wait_and_throw();

            std::vector<FP16> C2_ref(M * P, FP16(0.f));
            reference_dual_gemm<FP16>(false, A1_f16, K1, B1_f16, K1, B2_f16, N,
                                      C2_ref.data(), P, M, K1, N, P);

            bool ok = verify_results(C2_ref.data(), C2_gpu, M, P);
            printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");
            all_pass &= ok;
            sycl::free(C2_gpu, Q);
        }

        // Test 7: fp16, with transpose
        {
            printf("--- Test 7: Dual GEMM via SLM (with transpose, fp16) ---\n");
            FP16 *C2_gpu = sycl::malloc_shared<FP16>(N * P, Q);
            for (int i = 0; i < N * P; ++i) C2_gpu[i] = FP16(0.f);

            launch_dual_gemm_slm<true, FP16, kBlockM, kBlockN, kBlockP, kK1,
                                  kNSGs, AtomLayoutM1, AtomLayoutM2>(
                Q, A1_f16, B1_f16, B2_f16, C2_gpu, M, K1, N, P);
            Q.wait_and_throw();

            std::vector<FP16> C2_ref(N * P, FP16(0.f));
            reference_dual_gemm<FP16>(true, A1_f16, K1, B1_f16, K1, B2_f16, N,
                                      C2_ref.data(), P, M, K1, N, P);

            bool ok = verify_results(C2_ref.data(), C2_gpu, N, P);
            printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");
            all_pass &= ok;
            sycl::free(C2_gpu, Q);
        }

        sycl::free(A1_f16, Q);
        sycl::free(B1_f16, Q);
        sycl::free(B2_f16, Q);
    }

    printf("=== %s ===\n", all_pass ? "ALL PASSED" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
