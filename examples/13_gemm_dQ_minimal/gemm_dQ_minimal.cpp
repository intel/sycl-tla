/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Minimal standalone example of gemm_dQ extracted from sdpa_bwd.

    gemm_dQ computes the gradient of Q in Flash-Attention backward pass:

        dQ (M x N) += dP (M x K)  *  K^T (K x N)

    with the same thread/tile layout used in examples/sdpa_bwd/sdpa_backward.hpp.
    Concretely, for the headdim=128 configuration:

        M = kBlockM  = 64   (query block)
        N = kHeadDim = 128  (head dimension)
        K = kBlockN  = 64   (key block, contracted dimension)

    Memory layout (following sdpa_bwd conventions):
        A = dP  : (M, K) = (64,  64) col-major  stride (  1, M=64)
        B = K   : (N, K) = (128, 64) col-major  stride (  1, N=128)
        C = dQaccum : (M, N) = (64, 128) row-major stride (N=128, 1) -- atomically accumulated

    Thread hierarchy:
        kNSGs         = 8  subgroups per workgroup
        SubgroupSize  = 16 work-items per subgroup
        Total threads = 128 per workgroup
        SubgroupLayout for dQ = Shape<4, 2, 1>  (4 SGs along M, 2 SGs along N)
        DPAS atom     = XE_DPAS_TT<8, float, half_t>  (M=8, acc=f32, inputs=f16)
        DPAS K        = 16  (inner contraction tile)
        k-tile count  = K / DPAS_K = 64 / 16 = 4

    To build and run (from your build directory):
        ninja 13_gemm_dQ_minimal
        ./examples/sycl/13_gemm_dQ_minimal/13_gemm_dQ_minimal
*/

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/algorithm/reorder.hpp>
#include <cute/util/xe_split_barrier.hpp>
#include <cute/util/compat.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace cute;

// ---------------------------------------------------------------------------
// Trait struct — mirrors FAKernel from sdpa_bwd/params.hpp but stripped to
// only what gemm_dQ needs.
// ---------------------------------------------------------------------------

template <typename T_,                  // input data type  (e.g. half_t)
          int kBlockM_,                 // M tile = query block size
          int kBlockN_,                 // K tile = key   block size (contraction)
          int kHeadDim_,                // N tile = head dimension
          int kNSGs_,                   // number of subgroups per workgroup
          int AtomLayoutMdQ_ = 4>       // SG layout along M (kNSGs/AtomLayoutMdQ along N)
struct GemmdQTrait {
    using DType = T_;
    using VType = float;  // accumulator type (always f32)

    static constexpr int kBlockM    = kBlockM_;
    static constexpr int kBlockN    = kBlockN_;
    static constexpr int kHeadDim   = kHeadDim_;
    static constexpr int kNSGs      = kNSGs_;
    static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
    static constexpr int SubgroupSize  = 16;

    // DPAS atom: M=8, accumulator=float, A/B=DType
    using MMA_Atom_ARCH = XE_DPAS_TT<8, VType, DType>;
    using _K = Int<MMA_Atom_ARCH::K>;  // = 16 for f16/bf16

    // Subgroup tiling for dQ: AtomLayoutMdQ SGs along M, rest along N
    using SubgroupLayoutdQ = Layout<Shape<Int<AtomLayoutMdQ>,
                                          Int<kNSGs / AtomLayoutMdQ>,
                                          _1>>;
    // CTA tile: (M, N, K_dpas)
    using TileShapedQ = Layout<Shape<Int<kBlockM>, Int<kHeadDim>, _K>>;

    using TiledMmadQ = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                               TileShapedQ,
                                               SubgroupLayoutdQ>::TiledMMA;
};

// ---------------------------------------------------------------------------
// gemm_kernel — generic prefetch+DPAS loop.
// Identical to the one in sdpa_bwd/sdpa_backward.hpp.
//
//   A : (M, K)  col-major global tensor
//   B : (N, K)  col-major global tensor
//   acc: per-subgroup register fragment for C
//   mma: TiledMMA instance
//
// When clear_acc=true the accumulator is zeroed before the first DPAS;
// when false it is left as-is so the caller can accumulate across iterations.
// ---------------------------------------------------------------------------

template <bool clear_acc,
          class Trait,
          class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2, class TVLayout2,
          class TiledMMA>
CUTLASS_DEVICE void
gemm_kernel(Trait &,
            Tensor<Engine0, Layout0> const& A,
            Tensor<Engine1, Layout1> const& B,
            SubgroupTensor<Engine2, Layout2, TVLayout2>& acc,
            TiledMMA const& mma)
{
    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));

    // Coordinate proxy tensors (identity layout, shape only, no data)
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());

    auto tile_mnk = mma.tile_mnk();

    // Tile the proxy tensors to match the DPAS tile footprint
    Tensor gA = local_tile(cA, select<0, 2>(tile_mnk), make_coord(0, _));  // (BLK_M, BLK_K, k)
    Tensor gB = local_tile(cB, select<1, 2>(tile_mnk), make_coord(0, _));  // (BLK_N, BLK_K, k)

    // Block-2D TiledCopy instances derived from TiledMMA + global tensor strides
    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    // Per-work-item slices
    auto thr_mma      = mma.get_slice(local_id);
    auto thr_copy_a   = copy_a.get_slice(local_id);
    auto thr_copy_b   = copy_b.get_slice(local_id);

    // Register fragments for the copy destination
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    // Global (proxy) partitions for copy
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    // Prefetch TiledCopies (issue hardware prefetch-to-L1 for upcoming tiles)
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_prefetch_A = prefetch_a.get_slice(local_id);
    auto thr_prefetch_B = prefetch_b.get_slice(local_id);
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    constexpr int prefetch_dist  = 3;
    constexpr int barrier_scope  = 2;  // workgroup-level split barrier

    const int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));

    if constexpr (clear_acc)
        clear(acc);

    // Warm-up: pre-issue prefetch for the first `prefetch_dist` k-tiles
    int k_tile_prefetch = 0;
    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; ++k_tile_prefetch) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    // Main loop: copy → reorder → DPAS
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
        barrier_arrive(barrier_scope);

        copy(copy_a, tAgA(_, _, _, k_tile), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

        if (k_tile_prefetch < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
            prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
        }

        // Shuffle copy-fragment register layout to MMA-fragment layout
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        // Accumulate: acc += A_tile * B_tile
        gemm(mma, tCrA, tCrB, acc);

        barrier_wait(barrier_scope);
    }
}

// ---------------------------------------------------------------------------
// Batched inline-asm float atomic add: 8 lsc_atomic_fadd.ugm in one asm block.
//
// Exploits the coordinate pattern of tCgC (from the coordinate dump):
//   i=0..7:   n=0,  m=0..7   → addr stride = N*sizeof(float) = 512 bytes
//   i=8..15:  n=0,  m=8..15  → same stride
//   i=16..23: n=16, m=0..7   → same stride
//   ...  (4 groups × 2 batches of 8)
//
// A single "+rw"(addr) register is stepped by the constant stride in-asm via
// VISA `add`, amortising the address computation over 8 fadds per block.
// The scalar v0..v7 copies are front-loaded before the asm block.
// ---------------------------------------------------------------------------
CUTLASS_DEVICE void dq_atomic_fadd_8(uint64_t addr, uint64_t stride,
                                      float v0, float v1, float v2, float v3,
                                      float v4, float v5, float v6, float v7) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__ volatile (
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %2 %%null\n"
        "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %3 %%null\n"
        "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %4 %%null\n"
        "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %5 %%null\n"
        "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %6 %%null\n"
        "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %7 %%null\n"
        "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %8 %%null\n"
        "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
        "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %9 %%null\n"
        : "+rw"(addr)
        : "rw"(stride), "rw"(v0), "rw"(v1), "rw"(v2), "rw"(v3),
                        "rw"(v4), "rw"(v5), "rw"(v6), "rw"(v7)
        : "memory"
    );
#else
    CUTE_INVALID_CONTROL_PATH("dq_atomic_fadd_8 requires Intel Xe GPU (__SYCL_DEVICE_ONLY__)");
#endif
}

// ---------------------------------------------------------------------------
// gemm_dQ kernel function — matches sdpa_bwd/sdpa_backward.hpp exactly.
//
// Computes:  C (M x N, row-major) += A (M x K, col-major) * B (N x K, col-major)^T
//            via an atomic-add scatter into C.
//
// In the SDPA backward context:
//     A = dP        (kBlockM x kBlockN)
//     B = K         (kHeadDim x kBlockN)
//     C = dQaccum   (kBlockM x kHeadDim)
// ---------------------------------------------------------------------------

template <class Trait,
          class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2,
          class TiledMMA>
CUTLASS_DEVICE void
gemm_dQ(Trait& trait,
        Tensor<Engine0, Layout0> const& A,   // dP  : (M, K) col-major
        Tensor<Engine1, Layout1> const& B,   // K   : (N, K) col-major
        Tensor<Engine2, Layout2> const& C,   // dQaccum : (M, N) row-major
        TiledMMA const& mma)
{
    auto local_id  = int(compat::get_nd_item<1>().get_local_id(0));
    auto tile_mnk  = mma.tile_mnk();

    // Coordinate proxy tensor for C (used to map accumulator indices to (m,n))
    Tensor cC  = make_identity_tensor(C.shape());
    Tensor gC  = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));

    auto thr_mma = mma.get_slice(local_id);

    // Allocate accumulator fragment and get its (m, n) coordinate mapping
    auto tCrC = thr_mma.partition_sg_fragment_C(
                    make_identity_tensor(select<0, 1>(tile_mnk)));
    Tensor tCgC = thr_mma.partition_C(gC);

    // Run GEMM (clears accumulator first)
    gemm_kernel<true>(trait, A, B, tCrC, mma);

    // Scatter-accumulate: batch 8 consecutive fadds per asm block.
    // tCgC coordinate pattern: n is constant within each group of 8 (m steps
    // by 1), giving a fixed address stride = N*sizeof(float) = 512 bytes.
    // A single "+rw"(addr) register is stepped in-asm, amortising 8 fadds
    // over 7 VISA add instructions.  (Matches cute_util.new.hpp LscAtomicFadd<Int<8>>.)
    static_assert(decltype(size(tCgC))::value % 8 == 0,
                  "accumulator size must be divisible by 8");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCgC); i += 8) {
        auto [m0, n0]   = tCgC(i);
        auto [m1, n1]   = tCgC(i + 1);
        uint64_t base   = reinterpret_cast<uint64_t>(&C(m0, n0));
        uint64_t stride = reinterpret_cast<uint64_t>(&C(m1, n1)) - base;
        dq_atomic_fadd_8(base, stride,
                         tCrC(i+0), tCrC(i+1), tCrC(i+2), tCrC(i+3),
                         tCrC(i+4), tCrC(i+5), tCrC(i+6), tCrC(i+7));
    }
}

// ---------------------------------------------------------------------------
// Device kernel entry point
// ---------------------------------------------------------------------------

template <class Trait>
class GemmdQKernel;  // unique name for each instantiation

template <class Trait>
void gemm_dQ_kernel(Trait trait,
                    const typename Trait::DType* __restrict__ A_ptr,
                    const typename Trait::DType* __restrict__ B_ptr,
                    float*                                     C_ptr)
{
    using T = typename Trait::DType;
    constexpr int M = Trait::kBlockM;
    constexpr int N = Trait::kHeadDim;
    constexpr int K = Trait::kBlockN;

    //
    // Build global tensors from raw pointers, mirroring sdpa_bwd layouts.
    //
    // A = dP  : (M, K) col-major  → stride (1, M)
    // B = K   : (N, K) col-major  → stride (1, N)
    // C = dQaccum : (M, N) row-major → stride (N, 1)
    //
    Tensor mA = make_tensor(make_gmem_ptr(A_ptr),
                            make_layout(Shape<Int<M>, Int<K>>{},
                                        Stride<_1, Int<M>>{}));

    Tensor mB = make_tensor(make_gmem_ptr(B_ptr),
                            make_layout(Shape<Int<N>, Int<K>>{},
                                        Stride<_1, Int<N>>{}));

    Tensor mC = make_tensor(make_gmem_ptr(C_ptr),
                            make_layout(Shape<Int<M>, Int<N>>{},
                                        Stride<Int<N>, _1>{}));

    typename Trait::TiledMmadQ tiled_mma_dq;
    gemm_dQ(trait, mA, mB, mC, tiled_mma_dq);
}

// ---------------------------------------------------------------------------
// Host reference: C_ref(m, n) += sum_k A_col(m,k) * B_col(n,k)
// (i.e. C = A * B^T  in standard matrix notation)
// ---------------------------------------------------------------------------

template <class T>
void reference_gemm(const std::vector<T>& A,   // (M, K) col-major
                    const std::vector<T>& B,   // (N, K) col-major
                    std::vector<float>&   C,   // (M, N) row-major, accumulate
                    int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                // col-major A: element (m,k) is at A[m + k*M]
                // col-major B: element (n,k) is at B[n + k*N]
                acc += float(A[m + k * M]) * float(B[n + k * N]);
            // row-major C: element (m,n) is at C[m*N + n]
            C[m * N + n] += acc;
        }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    // ---- Configuration identical to sdpa_bwd headdim=128 path ----
    using T = cute::half_t;

    // kBlockM=64, kHeadDim=128, kBlockN=64, kNSGs=8, AtomLayoutMdQ=4
    using Trait = GemmdQTrait<T, 64, 64, 128, 8, 4>;

    constexpr int M = Trait::kBlockM;   // 64
    constexpr int N = Trait::kHeadDim;  // 128
    constexpr int K = Trait::kBlockN;   // 64

    constexpr int kNSGs        = Trait::kNSGs;        // 8
    constexpr int SubgroupSize = Trait::SubgroupSize;  // 16
    const     int total_threads = kNSGs * SubgroupSize; // 128

    printf("gemm_dQ minimal example\n");
    printf("  M=%d  N=%d  K=%d  kNSGs=%d  SubgroupSize=%d  threads=%d\n",
           M, N, K, kNSGs, SubgroupSize, total_threads);
    printf("  A (dP)      : (%d x %d) col-major  stride (1, %d)\n", M, K, M);
    printf("  B (K-matrix): (%d x %d) col-major  stride (1, %d)\n", N, K, N);
    printf("  C (dQaccum) : (%d x %d) row-major  stride (%d, 1)  [atomic-add]\n", M, N, N);
    printf("  DPAS atom   : XE_DPAS_TT<8, float, half_t>  K_dpas=16\n");
    printf("  k-tile count: %d\n\n", K / Trait::MMA_Atom_ARCH::K);

    // ---- Allocate and initialise host data ----
    std::vector<T>     h_A(M * K), h_B(N * K);
    std::vector<float> h_C(M * N, 0.f), h_C_ref(M * N, 0.f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : h_A) v = T(dist(rng));
    for (auto& v : h_B) v = T(dist(rng));

    // ---- CPU reference ----
    reference_gemm(h_A, h_B, h_C_ref, M, N, K);

    // ---- Device allocation ----
    sycl::queue q{sycl::gpu_selector_v};
    T*     d_A = sycl::malloc_device<T>    (M * K, q);
    T*     d_B = sycl::malloc_device<T>    (N * K, q);
    float* d_C = sycl::malloc_device<float>(M * N, q);

    q.memcpy(d_A, h_A.data(), M * K * sizeof(T)).wait();
    q.memcpy(d_B, h_B.data(), N * K * sizeof(T)).wait();
    q.memset(d_C, 0, M * N * sizeof(float)).wait();

    // ---- Launch kernel ----
    //
    // One workgroup covers exactly the (M x N) output tile in a single pass.
    // Grid = (1, 1, 1) — single workgroup.
    // Block = (kNSGs * SubgroupSize, 1, 1) = (128, 1, 1).
    //
    auto trait = Trait{};
    auto dimGrid  = compat::dim3(1, 1, 1);
    auto dimBlock = compat::dim3(total_threads, 1, 1);

    compat::experimental::launch_properties  launch_props{};
    compat::experimental::kernel_properties  kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<Trait::SubgroupSize>};
    compat::experimental::launch_policy policy{dimGrid, dimBlock,
                                               launch_props, kernel_props};

    auto ev = compat::experimental::launch<
                  gemm_dQ_kernel<Trait>,
                  GemmdQKernel<Trait>>(policy, trait, d_A, d_B, d_C);
    ev.wait_and_throw();

    // ---- Copy result back ----
    q.memcpy(h_C.data(), d_C, M * N * sizeof(float)).wait();

    // ---- Verify ----
    float max_abs_err = 0.f, max_rel_err = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float abs_err = std::abs(h_C[i] - h_C_ref[i]);
        float rel_err = abs_err / (std::abs(h_C_ref[i]) + 1e-6f);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
    printf("Verification: max_abs_err=%.6f  max_rel_err=%.6f\n",
           max_abs_err, max_rel_err);

    const float threshold = 1e-2f;
    bool passed = (max_abs_err < threshold);
    printf("Result: %s\n", passed ? "PASSED" : "FAILED");

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    return passed ? 0 : 1;
}
