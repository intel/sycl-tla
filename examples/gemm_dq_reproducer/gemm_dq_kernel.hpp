#pragma once
#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cute/arch/copy_xe_2d.hpp>

using namespace cute;

// SLM atomic add helper
CUTLASS_DEVICE void slm_atomic_add(float *slm_ptr, float val) {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::atomic_ref<float,
        sycl::memory_order::relaxed,
        sycl::memory_scope::work_group,
        sycl::access::address_space::local_space> ref(*slm_ptr);
    ref.fetch_add(val);
#endif
}

// Standalone gemm_dQ kernel.
// Computes dQ[M,D] += dP[M,N] * K[D,N]^T  (accumulated across N-blocks via atomicAdd)
//
// This mirrors the dQ GEMM from sdpa_backward.hpp but isolated for benchmarking.
// A = dP [M, N_block] column-major (stride 1 on M, kBlockM on N)
// B = K  [D, N_block] column-major (stride 1 on D, k_stride on N)
// C = dQ [M, D]       row-major f32 (stride dq_stride on M, 1 on D)

template<typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout), get<1>(layout)), get<2>(layout));
    return l;
}

// The core GEMM engine with prefetch (extracted from sdpa_backward.hpp gemm_kernel)
template<bool clear_acc, class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
gemm_engine(Trait &trait,
            Tensor<Engine0, Layout0> const& A,
            Tensor<Engine1, Layout1> const& B,
            SubgroupTensor<Engine2, Layout2, TVLayout2> & acc,
            TiledMMA const & mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());

    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0,2>(tile_mnk), make_coord(0,_));
    Tensor gB = local_tile(cB, select<1,2>(tile_mnk), make_coord(0,_));

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    auto thr_mma    =    mma.get_slice(first_thread_in_sg_idx);
    auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
    auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    auto thr_prefetch_A = prefetch_a.get_slice(first_thread_in_sg_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(first_thread_in_sg_idx);

    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    const int prefetch_dist = 3;
    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_prefetch = 0;

    if constexpr(clear_acc)
        clear(acc);

    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        barrier_arrive(barrier_scope);

        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        gemm(mma, tCrA, tCrB, acc);

        barrier_wait(barrier_scope);
    }
}

// gemm_dQ: GEMM + atomicAdd to f32 accumulator
// A = dP [M, K], B = K_mat [D, K], C = dQaccum [M, D] (f32)
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TiledMMA>
void
gemm_dQ_atomic(Trait &trait,
               Tensor<Engine0, Layout0> const& A,
               Tensor<Engine1, Layout1> const& B,
               Tensor<Engine2, Layout2> const& C,
               TiledMMA const & mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto tile_mnk = mma.tile_mnk();

    Tensor cC = make_identity_tensor(C.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto thr_mma = mma.get_slice(first_thread_in_sg_idx);
    auto tCrC = thr_mma.partition_sg_fragment_C(
        make_identity_tensor(select<0,1>(tile_mnk)));
    Tensor tCgC = thr_mma.partition_C(gC);

    gemm_engine<true>(trait, A, B, tCrC, mma);

    int local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCgC); ++i) {
        auto [m, n] = tCgC(i);
        cutlass::atomicAdd(&C(m, n + local_id), tCrC(i));
    }
}

// SIMD32 atomic_fadd: packs two SIMD16 atomics into one SIMD32 instruction.
// ext_vector_type ensures contiguous GRF allocation for the paired operands.
using __long2  __attribute__((ext_vector_type(2))) = long;
using __float2 __attribute__((ext_vector_type(2))) = float;

CUTLASS_DEVICE void atomic_fadd_simd32(float *addr0, float *addr1,
                                        float val0, float val1) {
#ifdef __SYCL_DEVICE_ONLY__
    __long2  addrs = {(long)addr0, (long)addr1};
    __float2 vals  = {val0, val1};
    asm volatile(
        "lsc_atomic_fadd.ugm (M1_NM, 32)  %%null:d32  flat[%0]:a64  %1  %%null"
        :: "rw"(addrs), "rw"(vals)
    );
#endif
}

// gemm_dQ with SIMD32 atomicAdd — halves atomic instruction count
// Pairs consecutive accumulator elements into single SIMD32 atomic_fadd.
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TiledMMA>
void
gemm_dQ_atomic_simd32(Trait &trait,
                       Tensor<Engine0, Layout0> const& A,
                       Tensor<Engine1, Layout1> const& B,
                       Tensor<Engine2, Layout2> const& C,
                       TiledMMA const & mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto tile_mnk = mma.tile_mnk();

    Tensor cC = make_identity_tensor(C.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto thr_mma = mma.get_slice(first_thread_in_sg_idx);
    auto tCrC = thr_mma.partition_sg_fragment_C(
        make_identity_tensor(select<0,1>(tile_mnk)));
    Tensor tCgC = thr_mma.partition_C(gC);

    gemm_engine<true>(trait, A, B, tCrC, mma);

    int local_id = sg.get_local_id();

    // Pair consecutive elements into SIMD32 atomics
    static_assert(decltype(size(tCgC))::value % 2 == 0,
                  "Accumulator size must be even for SIMD32 pairing");

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCgC); i += 2) {
        auto [m0, n0] = tCgC(i);
        auto [m1, n1] = tCgC(i + 1);
        atomic_fadd_simd32(&C(m0, n0 + local_id),
                           &C(m1, n1 + local_id),
                           tCrC(i), tCrC(i + 1));
    }
}

// ---------------------------------------------------------------------------
// K-sliced gemm_dQ with SLM reduction.
//
// K-slicing splits kBlockN (the GEMM reduction dim) within a single tile.
// WG has kNSGs * K_SLICES SGs total, organized as K_SLICES groups of kNSGs.
// Each group handles kBlockN/K_SLICES of the K dimension.
// Groups produce partial dQ, atomicAdd to SLM, then one global atomicAdd.
// ---------------------------------------------------------------------------

// GEMM engine without split barriers — for K-sliced path where groups run independently.
// local_thr_idx is the thread index within the K-slice group (0..kNSGs*SubgroupSize-1).
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
gemm_engine_no_barrier(Trait &trait,
                       Tensor<Engine0, Layout0> const& A,
                       Tensor<Engine1, Layout1> const& B,
                       SubgroupTensor<Engine2, Layout2, TVLayout2> & acc,
                       TiledMMA const & mma,
                       int local_thr_idx) {
    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0,2>(tile_mnk), make_coord(0,_));
    Tensor gB = local_tile(cB, select<1,2>(tile_mnk), make_coord(0,_));

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    auto thr_mma    =    mma.get_slice(local_thr_idx);
    auto thr_copy_a = copy_a.get_slice(local_thr_idx);
    auto thr_copy_b = copy_b.get_slice(local_thr_idx);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_pf_A = prefetch_a.get_slice(local_thr_idx);
    auto thr_pf_B = prefetch_b.get_slice(local_thr_idx);
    auto pAgA = thr_pf_A.partition_S(gA);
    auto pBgB = thr_pf_B.partition_S(gB);

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_pf = 0;
    const int pf_dist = 3;

    CUTE_UNROLL
    for (; k_tile_pf < pf_dist; k_tile_pf++) {
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_pf));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_pf));
    }

    for (int kt = 0; kt < k_tile_count; kt++, k_tile_pf++) {
        copy(copy_a, tAgA(_,_,_,kt), tArA);
        copy(copy_b, tBgB(_,_,_,kt), tBrB);
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_pf));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_pf));
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);
        gemm(mma, tCrA, tCrB, acc);
    }
}
