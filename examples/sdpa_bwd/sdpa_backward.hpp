#pragma once
#include "cute_util.hpp"
template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout),
                                     get<1>(layout)),
                         get<2>(layout));
    return l;
}

constexpr int tid = 1;
constexpr int bid = 1;

const bool
is_cur_thread() {
    return cute::thread(tid, bid);
}

template <typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
CUTLASS_DEVICE void
apply_mask_causal(Tensor<Engine0, Layout0> &tensor,
                  Tensor<Engine1, Layout1> &rC,
                  int m_offset, int n_offset, int diagonal_offset = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    int sg_local_id = sg.get_local_id();
    int sg_group_id = sg.get_group_id();
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<1>(tensor); ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(tensor); ++m) {
            int y = m_offset + get<1>(rC_2d(m, n)) + sg_local_id + diagonal_offset;
            int x = n_offset + get<0>(rC_2d(m, n));
            if (x > y) {
                tensor(m, n) = -INFINITY;
            }
        }
    }
    return;
}

template<typename T, class Trait, class MTensor, class TiledMMA>
auto
create_reg(Trait const &trait,
           MTensor const &C,
           TiledMMA const &tiled_mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    Tensor cC = make_identity_tensor(C.shape());   // (M,N)
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));  // (BLK_M,BLK_N)
    auto copy_c = make_block_2d_copy_D(tiled_mma, C);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    if constexpr(is_same_v<T, float>) {
        auto r32 = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_mnk))); // allocate C fragment storage
        return r32;
    } else {
        auto r16 = thr_copy_c.partition_sg_fragment_S(gC);
        return r16;
    }
}

template<bool clear_acc, class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
gemm_kernel(Trait &trait,
            Tensor<Engine0, Layout0> const& A,         // (M,K)
            Tensor<Engine1, Layout1> const& B,         // (N,K)
            SubgroupTensor<Engine2, Layout2, TVLayout2> & acc,
            TiledMMA const & mma) {
    // -----
    // Setup
    // -----

    /* Get workgroup and local IDs */
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;

    /* Create proxy coordinate tensors for each global tensor */
    Tensor cA = make_identity_tensor(A.shape());   // (M,K)
    Tensor cB = make_identity_tensor(B.shape());   // (N,K)

    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0,2>(tile_mnk), make_coord(0,_));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(cB, select<1,2>(tile_mnk), make_coord(0,_));  // (BLK_N,BLK_K,k)

    /* Create block 2D TiledCopies */
    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
    auto thr_mma    =    mma.get_slice(first_thread_in_sg_idx);
    auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
    auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);

    /* Register fragments for MMA */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

    /* Register fragments for copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    /* Partition global tensor (proxies) for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    /* Partition C */
    // Tensor tCrC = partition_fragment_C(mma, select<0,1>(tile_mnk));

    /* Create prefetch TiledCopy instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    auto thr_prefetch_A = prefetch_a.get_slice(first_thread_in_sg_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(first_thread_in_sg_idx);

    /* Partition global tensor (proxies) for prefetch */
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    /* Prefetch distance, in units of k tiles */
    const int prefetch_dist = 3;

    // ------
    // Kernel
    // ------

    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_prefetch = 0;
    /* Clear the accumulators */
    if constexpr(clear_acc)
        clear(acc);
    /* Warm up loops with prefetch to L1 */
    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
    }
    /* Main loop */
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        /* Split barrier keeping threads loosely together */
        barrier_arrive(barrier_scope);

        /* Copy A/B from global memory (ideally L1 cache) to registers */
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        /* Prefetch A/B tiles to L1 */
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

        /* Shuffle data from copy fragments to MMA fragments */
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        /* Accumulate C += A * B */
        gemm(mma, tCrA, tCrB, acc);

        /* Other half of split barrier */
        barrier_wait(barrier_scope);
    }
}

template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TVLayout2, class TiledMMA>
void
gemm_SdP(Trait &trait,
         Tensor<Engine0, Layout0> const& A,         // (M,K)
         Tensor<Engine1, Layout1> const& B,         // (N,K)
         SubgroupTensor<Engine2, Layout2, TVLayout2> & rSdP,
         TiledMMA const & mma) {
    gemm_kernel<true>(trait, A, B, rSdP, mma);
}

template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TVLayout2, class TiledMMA>
void
gemm_dKV(Trait &trait,
         Tensor<Engine0, Layout0> const& A,         // (M,K)
         Tensor<Engine1, Layout1> const& B,         // (N,K)
         SubgroupTensor<Engine2, Layout2, TVLayout2> & rdKV,
         TiledMMA const & mma) {
    gemm_kernel<false>(trait, A, B, rdKV, mma);
}

template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TiledMMA>
void
gemm_dQ(Trait &trait,
        Tensor<Engine0, Layout0> const& A,         // (M,K)
        Tensor<Engine1, Layout1> const& B,         // (N,K)
        Tensor<Engine2, Layout2> const& C,         // (M,N)
        TiledMMA const & mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto tile_mnk = mma.tile_mnk();
    Tensor cC = make_identity_tensor(C.shape());   // (M,N)
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));  // (BLK_M,BLK_N)
    auto thr_mma = mma.get_slice(first_thread_in_sg_idx);
    auto tCrC = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_mnk))); // allocate C fragment storage
    Tensor tCgC = thr_mma.partition_C(gC);
    gemm_kernel<true>(trait, A, B, tCrC, mma);

    int local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i <  size(tCgC); ++i) {
        auto [m, n] = tCgC(i);
        cutlass::atomicAdd(&C(m, n + local_id), tCrC(i));
    }
}

template<class Trait, class TiledMma,
         class Engine0, class Layout0, class TVLayout0,
         class Engine1, class Layout1>
void
mha_copy(Trait & trait, TiledMma &tiled_mma,
         SubgroupTensor<Engine0, Layout0, TVLayout0> &r,
         Tensor<Engine1, Layout1> &m,
         int m_block = 0, int n_block = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto copy_c = make_block_2d_copy_D(tiled_mma, m);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor cC = make_identity_tensor(m.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(m_block, n_block));
    Tensor tCgC = thr_copy_c.partition_D(gC);
    copy(copy_c, r, tCgC);
}

template<class Trait, class TiledMma,
         class Engine0, class Layout0, class TVLayout0,
         class Engine1, class Layout1>
void
mha_reorder_copy(Trait & trait, TiledMma &tiled_mma,
                 SubgroupTensor<Engine0, Layout0, TVLayout0> &r,
                 Tensor<Engine1, Layout1> &m){
    auto r16 = create_reg<typename Trait::DType>(trait, m, tiled_mma);
    reorder(r, r16);
    mha_copy(trait, tiled_mma, r16, m);
}

template <class Engine0, class Layout0,
          class Engine1, class Layout1,
          class Engine2, class Layout2>
CUTLASS_DEVICE void
mha_atomic_add(Tensor<Engine0, Layout0>& m_tile,
               Tensor<Engine1, Layout1>& g_tile,
               Tensor<Engine2, Layout2>& r_tile,
               const int local_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<3>(g_tile); ++ni) {
        auto g = g_tile(_, _, _, ni);
        auto r = r_tile(_, _, _, ni);
        CUTLASS_PRAGMA_UNROLL
        for (int ki = 0; ki < size(g); ++ki) {
            auto [m, n, l] = g(ki);
            cutlass::atomicAdd(&m_tile(m, n + local_id, 0), r(ki));
        }
    }
}
template<bool Is_even_M, class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void
load_1colvec(Tensor0 &reg, Tensor1 &mT, Tensor2 &coord_row,
             int tail_m = 0) {
    if constexpr(Is_even_M) {
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(reg); ++mi) {
            reg(mi) = mT(get<0>(coord_row(mi)));
        }
    } else {
        for (int mi = 0; mi < size(reg); ++mi) {
            int row = get<0>(coord_row(mi));
            if (row < tail_m) {
                reg(mi) = mT(row);
            }
        }
    }
}

template<typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_layout(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_1>{});  // ((2, 2), MMA_M, MMA_N, Tile_M, M, N)
    auto l2 = make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                          make_layout(get<2>(l)));
    return l2;
}

template<bool Is_even_M,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2>
CUTLASS_DEVICE void
scale_apply_exp2(Tensor<Engine0, Layout0> &tensor,
                 Tensor<Engine1, Layout1> &max,
                 Tensor<Engine2, Layout2> &rC,
                 const float scale,
                 const int tail_m = 0) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    if constexpr(Is_even_M) {
#if defined(__SYCL_DEVICE_ONLY__)
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            int n = get<1>(rC_2d(0, ni)) + sg_local_id;
            const float max_scaled = max(n) == -INFINITY ? 0.f : max(n) * M_LOG2E;
            const float neg_max_scaled = -max_scaled;
            ScaleExpHelper<decltype(size<0>(tensor))>::apply(tensor, ni, scale, neg_max_scaled);
        }
#else
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            int n = get<1>(rC_2d(0, ni)) + sg_local_id;
            const float max_scaled = max(n) == -INFINITY ? 0.f : max(n) * M_LOG2E;
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            }
        }
#endif
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            int n = get<1>(rC_2d(0, ni)) + sg_local_id;
            const float max_scaled = ((max(n) == -INFINITY) or (n >= tail_m)) ? 0.f : max(n) * M_LOG2E;
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            }
        }
    }
}

template<bool Is_even_M,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class Engine3, class Layout3>
CUTLASS_DEVICE void
softmax_backward(Tensor<Engine0, Layout0> &P,
                 Tensor<Engine1, Layout1> &dP_sum,
                 Tensor<Engine2, Layout2> &dP,
                 Tensor<Engine3, Layout3> &rC,
                 const float scale,
                 const int tail_m = 0) {
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    if constexpr(Is_even_M) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(dP); ++ni) {
            int n = get<1>(rC_2d(0, ni)) + sg_local_id;
            const float dpsum = dP_sum(n);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(dP); ++mi) {
                dP(mi, ni) = P(mi, ni) * (dP(mi, ni) - dpsum) * scale;
            }
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(dP); ++ni) {
            int n = get<1>(rC_2d(0, ni)) + sg_local_id;
            if (n < tail_m) {
                const float dpsum = dP_sum(n);
                CUTLASS_PRAGMA_UNROLL
                for (int mi = 0; mi < size<0>(dP); ++mi) {
                    dP(mi, ni) = P(mi, ni) * (dP(mi, ni) - dpsum) * scale;
                }
            }
        }
    }
}

template<bool is_causal, bool Is_even_N,
         typename T, typename V,
         int kBlockM, int kBlockN,
         class Tensor0, class Tensor1,
         class Tensor2, class Tensor3,
         class Tensor4, class Tensor5,
         class TilesaveP, class Param>
CUTLASS_DEVICE auto
softmax_forward(Tensor0 &scores, Tensor1 &tSrS,
                Tensor2 &taccScS, Tensor3 &mLSE,
                Tensor4 & mdPsum, Tensor5 &tPgP,
                TilesaveP &tilesaveP, Param &param,
                int m_block, int n_block,
                bool Is_even_M, int tail_m = 0) {
    if constexpr(is_causal) {
        Tensor taccScS_rc = logical_divide(taccScS, Shape<_1>{});
        apply_mask_causal(scores, taccScS_rc,
                          m_block * kBlockM, n_block * kBlockN,
                          param.seq_len_kv - param.seq_len_q);
    }
    Tensor taccScS_row = logical_divide(taccScS, Shape<_1>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<V>(Shape<Int<decltype(size(taccScS_row))::value>>{});

    if (Is_even_M) {
        load_1colvec<true>(lse, mLSE, taccScS_row);
    } else {
        load_1colvec<false>(lse, mLSE, taccScS_row, tail_m);
    }

    // P=softmax(S,lse)
    scale_apply_exp2(scores, lse, param.scale_softmax_log2);
    if (Is_even_M)
        load_1colvec<true>(lse, mdPsum, taccScS_row);
    else
        load_1colvec<false>(lse, mdPsum, taccScS_row, tail_m);
    return lse;
}

template<bool Is_even_N, bool Seq_parallel, class Trait>
void
dq_dk_dv_1colblock(Trait &trait, Param<typename Trait::DType> &param,
                   const int bidb, const int bidh, const int bidhkv, const int n_block,
                   const int tail_n = 0) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kNSGs = Trait::kNSGs;
    constexpr int SubgroupSize = Trait::SubgroupSize;
    constexpr int AtomLayoutMdQ = Trait::AtomLayoutMdQ;
    constexpr bool is_causal = Trait::is_causal;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    const int local_id = sg.get_local_id();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t v_offset = bofst.v_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t dk_offset = bofst.dk_offset(bidb, bidh, n_block * kBlockN);
    const index_t dv_offset = bofst.dv_offset(bidb, bidh, n_block * kBlockN);
    const index_t o_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, 0);
    const index_t lse_offset = bofst.lse_offset(bidb, bidh, 0);
    // buff offset
    const index_t pb_offset = (bidb * param.num_head_q * param.seq_len_kv_pad * kBlockM
                               + bidh * param.seq_len_kv_pad * kBlockM
                               + n_block * kBlockN * kBlockM) * 2;
    const index_t dsb_offset = pb_offset + kBlockN * kBlockM;

    const auto block_n_dim = tail_n == 0 ? Int<kBlockN>{} : ((tail_n + 1) & ~1);
    auto shapeO = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapeQtOt = make_shape(Int<kHeadDim>{}, kBlockM);
    auto shapeSPt = make_shape(Int<kBlockN>{}, kBlockM);
    auto shapeSP = make_shape(kBlockM, block_n_dim);

    using Shape1 = Shape<
        std::conditional_t<Is_even_N, Int<kBlockN>, int>, Int<kHeadDim>>;
    using Shape2 = Shape<
        Int <kHeadDim>,
        std::conditional_t<Is_even_N, Int<kBlockN>, int>>;
    auto shapeQ = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapedQ = Shape<Int<kBlockM>, Int<kHeadDim>>{};
    Shape1 shapeKtVt;
    Shape2 shapeKV;
    if constexpr(Is_even_N) {
        shapeKtVt = make_shape(Int<kBlockN>{}, Int<kHeadDim>{});
        shapeKV = make_shape(Int<kHeadDim>{}, Int<kBlockN>{});
    } else {
        shapeKtVt = make_shape(tail_n, Int<kHeadDim>{});
        shapeKV = make_shape(Int<kHeadDim>{}, tail_n);
    }
    Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{})));
    Tensor mKt = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                             make_layout(
                                 shapeKtVt,
                                 make_stride(param.k_r_stride, _1{})));
    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                             make_layout(
                                 shapeO,
                                 make_stride(param.o_r_stride, _1{})));
    Tensor mVt = make_tensor(make_gmem_ptr(param.v_ptr + v_offset),
                             make_layout(
                                 shapeKtVt,
                                 make_stride(param.v_r_stride, _1{})));
   // intermediate buffer
    Tensor mPt = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                             make_layout(
                                 shapeSPt,
                                 make_stride(Int<kBlockM>{}, _1{})));
    Tensor mdOt = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                              make_layout(
                                  shapeQtOt,
                                  make_stride(_1{}, param.o_r_stride)));
    Tensor mK = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                            make_layout(
                                shapeKV,
                                make_stride(_1{}, param.k_r_stride)));
    Tensor mdPt = make_tensor(make_gmem_ptr(param.pb_ptr + dsb_offset),
                              make_layout(
                                  shapeSPt,
                                  make_stride(Int<kBlockM>{}, _1{})));
    Tensor mQt = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                              make_layout(
                                  shapeQtOt,
                                  make_stride(_1{}, param.q_r_stride)));

    Tensor mLSE = make_tensor(make_gmem_ptr(param.lse_ptr + lse_offset),
                              make_layout(
                                  Shape<Int<kBlockM>>{},
                                  Stride<_1>{}));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + lse_offset),
                                make_layout(
                                    Shape<Int<kBlockM>>{},
                                    Stride<_1>{}));

    Tensor mdV = make_tensor(make_gmem_ptr(param.dv_ptr + dv_offset),
                             make_layout(
                                 shapeKtVt,
                                 make_stride(param.dv_r_stride, _1{})));
    Tensor mdP = make_tensor(make_gmem_ptr(param.pb_ptr + dsb_offset),
                             make_layout(
                                 shapeSP,
                                 make_stride(_1{}, Int<kBlockM>{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      shapedQ,
                                      make_stride(param.dq_r_stride, _1{})));
    Tensor mdK = make_tensor(make_gmem_ptr(param.dk_ptr + dk_offset),
                              make_layout(
                                  shapeKtVt,
                                  make_stride(param.dk_r_stride, _1{})));

    typename Trait::TiledMmaSdP tiled_mma_sdp;
    typename Trait::TiledMmadKV tiled_mma_dkv;
    typename Trait::TiledMmadQ tiled_mma_dq;

    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);

    Tensor caccSt = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{}); // same buffer as accSt
    Tensor taccScSt = thr_mma_sdp.partition_C(caccSt);
    Tensor taccScS_rt = logical_divide(taccScSt, Shape<_1>{});
// static_assert(size<0>(tSrS) * size<1>(tSrS) == size<0>(lse) && "row of acc and lse not match");
    // misc

    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    const int tail_m = param.seq_len_q % kBlockM;

    auto rdV = create_reg<V>(trait,
                             mdV,
                             tiled_mma_dkv);
    auto rdK = create_reg<V>(trait,
                             mdK,
                             tiled_mma_dkv);
    clear(rdV);
    clear(rdK);
    // clear accumulator
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        const bool Is_even_M = not ((m_block == max_m_block - 1) and (tail_m != 0));
        if (not Is_even_M) {
            const int block_m_dim = ((tail_m + 1) & ~1);
            mQ = make_tensor(make_gmem_ptr(mQ.data()),
                             make_layout(
                                 make_shape(tail_m, Int<kHeadDim>{}),
                                 make_stride(param.q_r_stride, _1{})));
            mdO = make_tensor(make_gmem_ptr(mdO.data()),
                             make_layout(
                                 make_shape(tail_m, Int<kHeadDim>{}),
                                 make_stride(param.o_r_stride, _1{})));
            mPt = make_tensor(make_gmem_ptr(mPt.data()),
                              make_layout(
                                  make_shape(Int<kBlockN>{}, block_m_dim),
                                  make_stride(Int<kBlockM>{}, _1{})));
            mdOt = make_tensor(make_gmem_ptr(mdOt.data()),
                               make_layout(
                                   make_shape(Int<kHeadDim>{}, tail_m),
                                   make_stride(_1{}, param.o_r_stride)));
            mdPt = make_tensor(make_gmem_ptr(mdPt.data()),
                               make_layout(
                                   make_shape(Int<kBlockN>{}, block_m_dim),
                                   make_stride(Int<kBlockM>{}, _1{})));
            mdP = make_tensor(make_gmem_ptr(mdP.data()),
                             make_layout(
                                 make_shape(block_m_dim, block_n_dim),
                                 make_stride(_1{}, Int<kBlockM>{})));
            mdQaccum = make_tensor(make_gmem_ptr(mdQaccum.data()),
                                   make_layout(
                                       shapedQ,
                                       make_stride(param.dq_r_stride, _1{})));
            mQt = make_tensor(make_gmem_ptr(mQt.data()),
                              make_layout(
                                  make_shape(Int<kHeadDim>{}, tail_m),
                                  make_stride(_1{}, param.q_r_stride)));
        }
        {
        auto rS = create_reg<V>(trait,
                                mPt,
                                tiled_mma_sdp);
        // S=QKt
        gemm_SdP(trait, mKt, mQ, rS, tiled_mma_sdp);
        Tensor scores = make_tensor(rS.data(), convert_layout_acc_layout(rS.layout()));
        if constexpr(is_causal) {
            apply_mask_causal(scores, taccScS_rt, m_block * kBlockM, n_block * kBlockN, param.seq_len_kv - param.seq_len_q);
        }
        // P=softmax(S,lse)
        if (Is_even_M) {
            scale_apply_exp2<true>(scores, mLSE, taccScS_rt,
                             param.scale_softmax_log2);
        } else {
            scale_apply_exp2<false>(scores, mLSE, taccScS_rt,
                                    param.scale_softmax_log2, tail_m);
        }
        auto rdP = create_reg<V>(trait,
                                 mdPt,
                                 tiled_mma_sdp);
        // dP=dO*Vt
        gemm_SdP(trait, mVt, mdO, rdP, tiled_mma_sdp);
        Tensor dS = make_tensor(rdP.data(), scores.layout());
        // dS=P(dP-sum_row(P))*scale
        if (Is_even_M) {
            softmax_backward<true>(scores, mdPsum, dS, taccScS_rt, param.scale_softmax);
        } else {
            softmax_backward<false>(scores, mdPsum, dS, taccScS_rt, param.scale_softmax, tail_m);
        }
        mha_reorder_copy(trait, tiled_mma_sdp, rS, mPt);
        mha_reorder_copy(trait, tiled_mma_sdp, rdP, mdPt); // copy dP to internal buff
        }
        sycl::group_barrier(group);
        // dV=Pt*dO
        gemm_dKV(trait, mPt, mdOt, rdV, tiled_mma_dkv);
        // dK=dPt*Q
        gemm_dKV(trait, mdPt, mQt, rdK, tiled_mma_dkv);
        // dQ=dP*K
        gemm_dQ(trait, mdP, mK, mdQaccum, tiled_mma_dq);
        // update ptr/atom copy
        mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
        mdO.data() = mdO.data() + int(kBlockM * param.o_r_stride);
        mdOt.data() = mdOt.data() + int(kBlockM * param.o_r_stride);
        mdQaccum.data() = mdQaccum.data() + int(kBlockM * param.dq_r_stride);
        mQt.data() = mQt.data() + int(kBlockM * param.q_r_stride);
        mLSE.data() = mLSE.data() + int(kBlockM);
        mdPsum.data() = mdPsum.data() + int(kBlockM);

    }
    mha_reorder_copy(trait, tiled_mma_dkv, rdV, mdV);
    mha_reorder_copy(trait, tiled_mma_dkv, rdK, mdK);
}

template<class T>
void
mha_backward_seq(T trait,
                 Param<typename T::DType> param) {
    const int bidb = BlockIdxZ();
    const int bidhq = BlockIdxY();
    const int bidnblk = BlockIdxX();
    const int bidhkv = bidhq / param.num_qh_per_kvh;
    for (int n_block = bidnblk; n_block < param.n_block; n_block += GridDimX()) {
        if (param.tail_n > 0 and n_block == param.n_block - 1)
            dq_dk_dv_1colblock<false, false>(trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
        else
            dq_dk_dv_1colblock<true, false>(trait, param, bidb, bidhq, bidhkv, n_block);
    }
}

template<class T>
void
mha_backward_parallel(T trait,
                      Param<typename T::DType> param) {
    const int bidb = BlockIdxZ();
    const int bidhq = BlockIdxY();
    const int n_block = BlockIdxX();
    const int bidhkv = bidhq / param.num_qh_per_kvh;
    if (param.tail_n > 0 and n_block == param.n_block - 1)
        dq_dk_dv_1colblock<false, true>(trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
    else
        dq_dk_dv_1colblock<true, true>(trait, param, bidb, bidhq, bidhkv, n_block);
}
