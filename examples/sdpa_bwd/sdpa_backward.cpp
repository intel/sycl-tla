#include <sycl/sycl.hpp>
#include <cassert>
#include <cute/util/compat.hpp>
#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cnpy.h"
#include "sdpa_util.hpp"
#include "params.hpp"


void read_args(int argc, char**argv, int n, int64_t *p) {
    if (argc >= n + 1)
        sscanf(argv[n], "%ld", p);
}

void
debug_info() {
    print("block idx (%d,%d,%d) dim (%d,%d,%d) thread idx (%d,%d,%d) dim (%d,%d,%d)\n",
          BlockIdxX(), BlockIdxY(), BlockIdxZ(),
          GridDimX(), GridDimY(), GridDimZ(),
          ThreadIdxX(), ThreadIdxY(), ThreadIdxZ(),
          BlockDimX(), BlockDimY(), BlockDimZ());
}

template<class T>
void print_t(T t) {
    print(t);
    for (int i = 0; i < size(t); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10.7f ", (float)t(i));
    }
    print("\n");
}

template<class T>
void print_t_2d(T t) {
    static_assert(rank(t) == 2, "Only support 2D Tensor");
    print(t);
    for (int i = 0; i <  size < 0>(t); ++i) {
        print("\n(%03d): ", i);
        for (int j = 0; j < size<1>(t); ++j) {
            print("%10.7f ", (float)t(i,j));
        }
    }
    print("\n");
}

template<class T>
void print_d(T t) {
    print(t);
    for (int i = 0; i < size(t); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10u ", t(i));
    }
    print("\n");
}

using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>; // batch, num_head_q,num_head_kv,seq_len_qo,seq_len_kv,head_size_qk,head_size_vo

template <typename T>
struct OPS_tobf16{
    template <class Tensor>
    auto operator()(Tensor &src){
        cutlass::NumericConverter<
            T, float, cutlass::FloatRoundStyle::round_toward_zero> converter;
        auto dst = make_tensor_like<T>(src);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src); ++i) {
            dst(i) = converter(src(i));
        }
        return dst;
    }
};

constexpr int tid = 0;
constexpr int bid = 0;


template<int k_tile = 0, typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename Tensor4,
         typename Tensor5, typename Tensor6,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
CUTLASS_DEVICE void gemm_ker(Tensor0 &tCrC, Tensor1 &tCrA, Tensor2 &tCrB,
              Tensor3 &tAgA, Tensor4 &tArA,
              Tensor5 &tBgB, Tensor6 &tBrB, TiledMma &tiled_mma,
              TiledCopyA &copy_a, TiledCopyB &copy_b,
              ThrCopyA &thr_copy_a, ThrCopyB &thr_copy_b) {
    constexpr int barrier_scope = 2;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<3>(tAgA); ++k) {
        barrier_arrive(barrier_scope);
        cute::copy(copy_a, tAgA(_, _, _, k), tArA);
        cute::copy(copy_b, tBgB(_, _, _, k), tBrB);
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);
    }
}

template<class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void load_1colvec(Tensor0 &reg, Tensor1 &mT, Tensor2 &coord_row) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size(reg); ++mi) {
        reg(mi) = mT(get<0>(coord_row(mi)));
    }
}
template<typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_layout(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_1>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
}

template<class Engine0, class Layout0, class Engine1, class Layout1>
CUTLASS_DEVICE void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * M_LOG2E;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

template<class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void softmax_backward(Tensor0 &P, Tensor1 &dP_sum, Tensor2 &dP, const float scale) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(dP); ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int mj = 0; mj < size<1>(dP); ++mj) {
            dP(mi, mj) = P(mi, mj) * (dP(mi, mj) - dP_sum(mi)) * scale;
        }
    }
}

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <class CVT, class T0, class T1>
CUTLASS_DEVICE auto convert_type(CVT &cvt, T0 &src, T1 &dst) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(src); ++i) {
        dst(i) = cvt(src(i));
    }
}

template<bool parallel_kv, bool Is_even_MN, class Trait>
void
dq_dk_dv_1colblock(Trait &trait, Param<typename Trait::DType> &param,
                   const int bidb, const int bidh, const int n_block,
                   const int tail_n = 0) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kBlockK = Trait::kBlockK;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidh, n_block * kBlockN);
    const index_t v_offset = bofst.v_offset(bidb, bidh, n_block * kBlockN);
    const index_t o_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t do_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t dv_offset = bofst.v_offset(bidb, bidh, n_block * kBlockN);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, 0);
    const index_t lse_offset = bofst.lse_offset(bidb, bidh, 0);
    const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, 0);
    // buff offset
    const index_t pb_offset = bidb * param.num_head_q * param.seq_len_kv_pad * kBlockM
        + bidh * param.seq_len_kv_pad * kBlockM + n_block * kBlockN * kBlockM;

    const index_t s_offset = bofst.ps_offset(bidb, bidh, 0, n_block * kBlockN);
    const index_t dp_offset = bofst.ps_offset(bidb, bidh, 0, n_block * kBlockN);

    const auto block_n_dim = tail_n == 0 ? Int<kBlockN>{} : tail_n;
    using Shape1 = Shape<
        std::conditional_t<Is_even_MN, Int<kBlockN>, int>,
        Int <kHeadDim>, Int<1>>;
    using Shape2 = Shape<
        Int <kHeadDim>,
        std::conditional_t<Is_even_MN, Int<kBlockN>, int>,
        Int<1>>;
    Shape shapeQ = make_shape(kBlockM, Int<kHeadDim>{}, _1{});
    Shape1 shapeKt, shapeV;
    Shape2 shapeK;
    if constexpr(Is_even_MN) {
        shapeKt = make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, _1{});
        shapeV = make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, _1{});
        shapeK = make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, _1{});
    } else {
        shapeKt = make_shape(tail_n, Int<kHeadDim>{}, _1{});
        shapeV = make_shape(tail_n, Int<kHeadDim>{}, _1{});
        shapeK = make_shape(Int<kHeadDim>{}, tail_n, _1{});
    }
    Shape shapeO = make_shape(kBlockM, Int<kHeadDim>{}, _1{});
    Shape shapeOt = make_shape(Int<kHeadDim>{}, kBlockM, _1{});


    Shape shapeSP = make_shape(kBlockM, block_n_dim, _1{});
    Shape shapedP = make_shape(kBlockM, block_n_dim, _1{});

    Shape shapePt = make_shape(block_n_dim, kBlockM, _1{});
    Shape shapeQt = make_shape(Int<kHeadDim>{}, kBlockM, _1{});

    Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{}, _1{})));
    Tensor mKt = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                            make_layout(
                                shapeKt,
                                make_stride(param.k_r_stride, _1{}, _1{})));
    Tensor mV = make_tensor(make_gmem_ptr(param.v_ptr + v_offset),
                            make_layout(
                                shapeV,
                                make_stride(param.v_r_stride, _1{}, _1{})));
    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + do_offset),
                             make_layout(
                                 shapeO,
                                 make_stride(param.o_r_stride, _1{}, _1{})));
    // intermediate buffer
    Tensor mP = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                            make_layout(
                                shapeSP,
                                make_stride(block_n_dim, _1{}, _1{})));
    Tensor mPt = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                             make_layout(
                                 shapePt,
                                 make_stride(_1{}, block_n_dim, _1{})));
    Tensor mdOt = make_tensor(make_gmem_ptr(param.do_ptr + do_offset),
                              make_layout(
                                  shapeOt,
                                  make_stride(_1{}, param.o_r_stride, _1{})));
    Tensor mK = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                            make_layout(
                                shapeK,
                                make_stride(_1{}, param.k_r_stride, _1{})));
    Tensor mdPt = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                              make_layout(
                                  shapePt,
                                  make_stride(_1{}, block_n_dim, _1{})));
    Tensor mQt = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                             make_layout(
                                 shapeQt,
                                 make_stride(_1{}, param.q_r_stride, _1{})));

    Tensor mLSE = make_tensor(make_gmem_ptr(param.lse_ptr + lse_offset),
                              make_layout(
                                  Shape<Int<kBlockM>>{},
                                  Stride<_1>{}));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + dpsum_offset),
                                make_layout(
                                    Shape<Int<kBlockM>>{},
                                    Stride<_1>{}));

    Tensor mdV = make_tensor(make_gmem_ptr(param.dv_ptr + dv_offset),
                             make_layout(
                                 shapeV,
                                 make_stride(param.v_r_stride, _1{}, _1{})));
    Tensor mdP = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                             make_layout(
                                 shapedP,
                                 make_stride(block_n_dim, _1{}, _1{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      shapeQ,
                                      make_stride(param.dq_r_stride, _1{}, _1{})));
    Tensor mdK = make_tensor(make_gmem_ptr(param.dk_ptr+k_offset),
                             make_layout(
                                 shapeKt,
                                 make_stride(param.k_r_stride, _1{}, _1{})));

    Tensor mS = make_tensor(make_gmem_ptr(param.s_ptr + s_offset), make_layout(
                                shapeSP,
                                make_stride(param.s_r_stride, _1{}, _1{})));
    Tensor mdPd = make_tensor(make_gmem_ptr(param.dp_ptr + s_offset), make_layout(
                                shapeSP,
                                make_stride(param.s_r_stride, _1{}, _1{})));

    Shape tile_sdp = typename Trait::TileShapeSdP{};
    Shape tile_dkv = typename Trait::TileShapedKV{};
    Shape tile_dq = typename Trait::TileShapedQ{};

    auto tileloadQ = typename Trait::TiledLoadQ{mQ};
    auto tileloadKt = typename Trait::TiledLoadKt{mKt};
    auto tileloaddO = typename Trait::TiledLoaddO{mdO};
    auto tileloadV = typename Trait::TiledLoadV{mV};
    auto tileloadPt = typename Trait::TiledLoadPt{mPt};
    auto tileloaddOt = typename Trait::TiledLoaddOt{mdOt}; // load dO as operand B for dV=Pt*dO
    auto tileloaddP = typename Trait::TiledLoaddP{mdP};
    auto tileloadK = typename Trait::TiledLoadK{mK};
    auto tileloaddQ = typename Trait::TiledLoaddQ{mdQaccum};
    auto tileloaddPt = typename Trait::TiledLoaddPt{mdPt};
    auto tileloadQt = typename Trait::TiledLoadQt{mQt};

    auto tilesaveP = typename Trait::TiledSaveS{mP}; // to internal buffer
    auto tilesavedV = typename Trait::TiledSavedV{mdV};
    auto tilesavedP = typename Trait::TiledSavedP{mdP};
    auto tilesavedQ = typename Trait::TiledSavedQ{mdQaccum};
    auto tilesavedK = typename Trait::TiledSavedK{mdK};


    Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
    Tensor mKt_coord = cute::get_xe_tensor(shapeKt);
    Tensor mV_coord = cute::get_xe_tensor(shapeV);
    Tensor mdO_coord = cute::get_xe_tensor(shapeO);
    Tensor mdOt_coord = cute::get_xe_tensor(shapeOt);
    Tensor mdV_coord = cute::get_xe_tensor(shapeV);
    Tensor mK_coord = cute::get_xe_tensor(shapeK);
    Tensor mQt_coord = cute::get_xe_tensor(shapeQt);

    Tensor mS_coord = cute::get_xe_tensor(shapeSP);
    Tensor mPt_coord = cute::get_xe_tensor(shapePt);
    Tensor mdP_coord = cute::get_xe_tensor(shapedP);

    typename Trait::TiledMmaSdP tiled_mma_sdp;
    typename Trait::TiledMmadKV tiled_mma_dkv;
    typename Trait::TiledMmadQ tiled_mma_dq;

    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);
    auto thr_mma_dkv = tiled_mma_dkv.get_slice(first_thread_in_sg_idx);
    auto thr_mma_dq = tiled_mma_dq.get_slice(first_thread_in_sg_idx);

    Tensor gQ = local_tile(mQ_coord, select<0, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gKt = local_tile(mKt_coord, select<1, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gdO = local_tile(mdO_coord, select<0, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gV = local_tile(mV_coord, select<1, 2>(tile_sdp), make_coord(0, _, 0));
    Tensor gPt = local_tile(mPt_coord, select<0, 2>(tile_dkv), make_coord(0, _, 0)); // load Pt
    Tensor gdOt = local_tile(mdOt_coord, select<1, 2>(tile_dkv), make_coord(0, _, 0));
    Tensor gdPa = local_tile(mdP_coord, select<0, 2>(tile_dq), make_coord(0, _, 0)); // operand A dQ
    Tensor gK = local_tile(mK_coord, select<1, 2>(tile_dq), make_coord(0, _, 0)); // operand B dQ
    Tensor gdPt = local_tile(mPt_coord, select<0, 2>(tile_dkv), make_coord(0, _, 0)); // load dpt
    Tensor gQt = local_tile(mQt_coord, select<1, 2>(tile_dkv), make_coord(0, _, 0)); // load Q as operand B

    Tensor gP = local_tile(mS_coord, select<0, 1>(tile_sdp), make_coord(0, 0, 0)); // dump P
    Tensor gdP = local_tile(mdP_coord, select<0, 1>(tile_sdp), make_coord(0, 0, 0)); // dump dP
    Tensor gdV = local_tile(mdV_coord, select<0, 1>(tile_dkv), make_coord(0, 0, 0)); // dump dV
    Tensor gdQ = local_tile(mQ_coord, select<0, 1>(tile_dq), make_coord(0, 0, 0)); // dump dQ
    Tensor gdK = local_tile(mK_coord, select<0, 1>(tile_dkv), make_coord(0, 0, 0)); // dump dK


    Tensor tSgQ = thr_mma_sdp.partition_A(gQ);
    Tensor tSgKt = thr_mma_sdp.partition_B(gKt);
    Tensor tdPgdO = thr_mma_sdp.partition_A(gdO);
    Tensor tdPgV = thr_mma_sdp.partition_B(gV);
    Tensor tdVgPt = thr_mma_dkv.partition_A(gPt);
    Tensor tdVgdOt = thr_mma_dkv.partition_B(gdOt);
    Tensor tdQgdP = thr_mma_dq.partition_A(gdPa);
    Tensor tdQgK = thr_mma_dq.partition_B(gK);
    Tensor tdKgdPt = thr_mma_dkv.partition_A(gdPt);
    Tensor tdKgQt = thr_mma_dkv.partition_B(gQt);

    Tensor tPgP = thr_mma_sdp.partition_C(gP); // save P to internal buffer
    Tensor tdPgdP = thr_mma_sdp.partition_C(gdP); // save dP to internal buffer
    Tensor tdVgdV = thr_mma_dkv.partition_C(gdV); // save to dv
    Tensor tdQgdQ = thr_mma_dq.partition_C(gdQ); // save to dq
    Tensor tdKgdK = thr_mma_dkv.partition_C(gdK); // save to dk


    Tensor tSrQ = make_tensor<T>(make_fragment_layout(tileloadQ, tSgQ(_,_,_,0).shape()));
    Tensor tSrKt = make_tensor<T>(make_fragment_layout(tileloadKt, tSgKt(_,_,_,0).shape()));
    Tensor tdPrdO = make_tensor<T>(make_fragment_layout(tileloaddO, tdPgdO(_,_,_,0).shape()));
    Tensor tdPrV = make_tensor<T>(make_fragment_layout(tileloadV, tdPgV(_,_,_,0).shape()));
    Tensor tdVrPt = make_tensor<T>(make_fragment_layout(tileloadPt, tdVgPt(_,_,_,0).shape()));
    Tensor tdVrdOt = make_tensor<T>(make_fragment_layout(tileloaddOt, tdVgdOt(_,_,_,0).shape()));
    Tensor tdQrdP = make_tensor<T>(make_fragment_layout(tileloaddP, tdQgdP(_,_,_,0).shape()));
    Tensor tdQrK = make_tensor<T>(make_fragment_layout(tileloadK, tdQgK(_,_,_,0).shape()));
    Tensor tdKrdPt = make_tensor<T>(make_fragment_layout(tileloaddPt, tdKgdPt(_,_,_,0).shape()));
    Tensor tdKrQt = make_tensor<T>(make_fragment_layout(tileloadQt, tdKgQt(_,_,_,0).shape()));

    ThrCopy thr_copy_q = tileloadQ.get_slice(compat::local_id::x());
    ThrCopy thr_copy_kt = tileloadKt.get_slice(compat::local_id::x());
    ThrCopy thr_copy_do = tileloaddO.get_slice(compat::local_id::x());
    ThrCopy thr_copy_v = tileloadV.get_slice(compat::local_id::x());
    ThrCopy thr_copy_pt = tileloadPt.get_slice(compat::local_id::x());
    ThrCopy thr_copy_dot = tileloaddOt.get_slice(compat::local_id::x());
    ThrCopy thr_copy_dp = tileloaddP.get_slice(compat::local_id::x());
    ThrCopy thr_copy_k = tileloadK.get_slice(compat::local_id::x());
    ThrCopy thr_copy_dpt = tileloaddPt.get_slice(compat::local_id::x());
    ThrCopy thr_copy_qt = tileloadQt.get_slice(compat::local_id::x());

    // Retile registers for copies
    Tensor tQrQ = thr_copy_q.retile_D(tSrQ);
    Tensor tKtrKt = thr_copy_kt.retile_D(tSrKt);
    Tensor tdOrdO = thr_copy_do.retile_D(tdPrdO);
    Tensor tVrV = thr_copy_v.retile_D(tdPrV);
    Tensor tPtrPt = thr_copy_pt.retile_D(tdVrPt);
    Tensor tdOtrdOt = thr_copy_dot.retile_D(tdVrdOt);
    Tensor tdPrdPa = thr_copy_dp.retile_D(tdQrdP);
    Tensor tKrK = thr_copy_k.retile_D(tdQrK);
    Tensor tdPtrdPt = thr_copy_dpt.retile_D(tdKrdPt);
    Tensor tQtrQt = thr_copy_qt.retile_D(tdKrQt);

    // Retile global counting tensors for copies
    Tensor tQgQ = thr_copy_q.retile_S(tSgQ);
    Tensor tKtgKt = thr_copy_kt.retile_S(tSgKt);
    Tensor tdOgdO = thr_copy_do.retile_S(tdPgdO);
    Tensor tVgV = thr_copy_v.retile_S(tdPgV);
    Tensor tPtgPt = thr_copy_pt.retile_S(tdVgPt);
    Tensor tdOtgdOt = thr_copy_dot.retile_S(tdVgdOt);
    Tensor tdPgdPa = thr_copy_dp.retile_S(tdQgdP);
    Tensor tKgK = thr_copy_k.retile_S(tdQgK);
    Tensor tdPtgdPt = thr_copy_dpt.retile_S(tdKgdPt);
    Tensor tQtgQt = thr_copy_qt.retile_S(tdKgQt);

    Tensor tSrS = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPrdP = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdVrdV = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdQrdQ = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tdKrdK = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});


    // for lse read
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}); // same buffer as accS
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    static_assert(decltype(size<0>(taccScS))::value == 8);
    Tensor taccScS_row = logical_divide(taccScS, Shape<_1>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<V>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    static_assert(size<0>(tSrS) * size<1>(tSrS) == size<0>(lse) && "row of acc and lse not match");
    // misc

    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    const int tail_m = param.seq_len_q % kBlockM;

    constexpr int k_tile = ceil_div(kHeadDim, kBlockK);

    cutlass::NumericConverter<T, float> converter;
    // clear accumulator
    clear(tdVrdV);
    clear(tdKrdK);
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        if (m_block == max_m_block - 1 and tail_m != 0) {
            mQ = make_tensor(make_gmem_ptr(mQ.data()),
                             make_layout(
                                 make_shape(tail_m, Int<kHeadDim>{}, _1{}),
                                 make_stride(param.q_r_stride, _1{}, _1{})));
            mdO = make_tensor(make_gmem_ptr(mdO.data()),
                              make_layout(
                                  make_shape(tail_m, Int<kHeadDim>{}, _1{}),
                                  make_stride(param.o_r_stride, _1{}, _1{})));
            mdOt = make_tensor(make_gmem_ptr(mdOt.data()),
                               make_layout(
                                   make_shape(Int<kHeadDim>{}, tail_m, _1{}),
                                   make_stride(_1{}, param.o_r_stride, _1{})));
            mdQaccum = make_tensor(make_gmem_ptr(mdQaccum.data()),
                                   make_layout(
                                       make_shape(tail_m, Int<kHeadDim>{}, _1{}),
                                       make_stride(param.dq_r_stride, _1{}, _1{})));
            mQt = make_tensor(make_gmem_ptr(mQt.data()),
                              make_layout(
                                  make_shape(Int<kHeadDim>{}, tail_m, _1{}),
                                  make_stride(_1{}, param.q_r_stride, _1{})));
            // Tensor mK = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
            //                         make_layout(
            //                             shapeK,
            //                             make_stride(param.k_r_stride, _1{}, _1{})));

            tileloadQ = typename Trait::TiledLoadQ{mQ};
            // auto tileloadK = typename Trait::TiledLoadK{mK};
            tileloaddO = typename Trait::TiledLoaddO{mdO};
            tileloaddOt = typename Trait::TiledLoaddOt{mdOt};
            tileloaddQ = typename Trait::TiledLoaddQ{mdQaccum};
            tileloadQt = typename Trait::TiledLoadQt{mQt};
            tilesavedQ = typename Trait::TiledSavedQ{mdQaccum};
        }
        clear(tSrS);
        clear(tdPrdP);
        clear(tdQrdQ);
        // S=QKt
        gemm_ker<k_tile>(tSrS, tSrQ, tSrKt, tQgQ, tQrQ, tKtgKt, tKtrKt, tiled_mma_sdp, tileloadQ, tileloadKt, thr_copy_q, thr_copy_kt);
        load_1colvec(lse, mLSE, taccScS_row);
        Tensor dP_sum = make_fragment_like(lse);
        load_1colvec(dP_sum, mdPsum, taccScS_row);
        Tensor scores = make_tensor(tSrS.data(), convert_layout_acc_layout(tSrS.layout()));

        // P=softmax(S,lse)
        scale_apply_exp2(scores, lse, param.scale_softmax_log2);
        auto tSrSl = make_tensor_like<T>(tSrS);
        convert_type(converter, tSrS, tSrSl);
        copy(tilesaveP, tSrSl, tPgP); // save P to internal buffers
        // dP=dO*Vt
        gemm_ker<k_tile>(tdPrdP, tdPrdO, tdPrV, tdOgdO, tdOrdO, tVgV, tVrV, tiled_mma_sdp, tileloaddO, tileloadV, thr_copy_do, thr_copy_v);
        Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
        // dS=P(dP-sum_row(P))*scale
        softmax_backward(scores, dP_sum, dS, param.scale_softmax);
        auto tdPrdPl = make_tensor_like<T>(tdPrdP);
        convert_type(converter, tdPrdP, tdPrdPl);

        // dV=Pt*dO
        gemm_ker(tdVrdV, tdVrPt, tdVrdOt, tPtgPt, tPtrPt, tdOtgdOt, tdOtrdOt, tiled_mma_dkv, tileloadPt, tileloaddOt, thr_copy_pt, thr_copy_dot);
        sycl::group_barrier(group);

        copy(tilesavedP, tdPrdPl, tdPgdP); // save dP to buffer after P used by dV
        sycl::group_barrier(group);

        if (n_block > 0)
            copy(tileloaddQ, tdQgdQ, tdQrdQ);
        // dQ=dP*K
        gemm_ker<k_tile>(tdQrdQ, tdQrdP, tdQrK, tdPgdPa, tdPrdPa, tKgK, tKrK, tiled_mma_dq, tileloaddP, tileloadK, thr_copy_dp, thr_copy_k);
        copy(tilesavedQ, tdQrdQ, tdQgdQ);
        // dK=dPt*Q
        gemm_ker<k_tile>(tdKrdK, tdKrdPt, tdKrQt, tdPtgdPt, tdPtrdPt, tQtgQt, tQtrQt, tiled_mma_dkv, tileloaddPt, tileloadQt, thr_copy_dpt, thr_copy_qt);
        // update ptr/atom copy
        mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
        mdO.data() = mdO.data() + int(kBlockM * param.o_r_stride);
        mdOt.data() = mdOt.data() + int(kBlockM * param.o_r_stride);
        mdQaccum.data() = mdQaccum.data() + int(kBlockM * param.dq_r_stride);
        mQt.data() = mQt.data() + int(kBlockM * param.q_r_stride);
        mLSE.data() = mLSE.data() + int(kBlockM);
        mdPsum.data() = mdPsum.data() + int(kBlockM);

        tileloadQ = typename Trait::TiledLoadQ{mQ};
        tileloaddO = typename Trait::TiledLoaddO{mdO};
        tileloaddOt = typename Trait::TiledLoaddOt{mdOt};
        tileloaddQ = typename Trait::TiledLoaddQ{mdQaccum};
        tileloadQt = typename Trait::TiledLoadQt{mQt};
        tilesavedQ = typename Trait::TiledSavedQ{mdQaccum};

    }
    auto tdVrdVl = make_tensor_like<T>(tdVrdV);
    convert_type(converter, tdVrdV, tdVrdVl);
    copy(tilesavedV, tdVrdVl, tdVgdV);
    auto tdKrdKl = make_tensor_like<T>(tdKrdK);
    convert_type(converter,tdKrdK, tdKrdKl);
    copy(tilesavedK, tdKrdKl, tdKgdK);
}

template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout),
                                     get<1>(layout)),
                         get<2>(layout));
    return l;
}

template<bool Is_even_M, class T>
void
compute_o_dot_do(T &trait, Param<typename T::DType> &param,
                 const int m_block, const int bidb, const int bidh) {
    // The thread index.
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    using DType = typename T::DType;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t o_offset = bofst.o_offset(bidb, bidh, m_block * kBlockM);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, m_block * kBlockM);

    using ShapeO = Shape<
        std::conditional_t <Is_even_M, Int<kBlockM>, int>,
        Int<kHeadDim>, Int<1>>;
    using ShapeP = Shape<
        std::conditional_t <Is_even_M, Int<kBlockM>, int>>;
    ShapeO O_shape;
    ShapeP dP_shape;
    if constexpr(Is_even_M) {
        O_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, _1{});
        dP_shape = make_shape(Int<kBlockM>{});
    } else {
        O_shape = make_shape(param.tail_m, Int<kHeadDim>{}, _1{});
        dP_shape = make_shape(param.tail_m);
    }
    Shape dQ_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, _1{});

    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                             make_layout(
                                 O_shape,
                                 make_stride(param.o_r_stride, _1{}, _1{})));
    Tensor mO = make_tensor(make_gmem_ptr(param.o_ptr + o_offset),
                            make_layout(
                                O_shape,
                                make_stride(param.o_r_stride, _1{}, _1{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      dQ_shape,
                                      make_stride(param.dq_r_stride, _1{})));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + dpsum_offset),
                                make_layout(
                                    dP_shape,
                                    Stride<_1>{}));

    Shape tile_dO = typename T::TileShapedQ{};
    auto tileloaddO = typename T::TiledLoaddP{mdO};
    auto tileloadO = typename T::TiledLoaddP{mO};
    auto tilesavedQ = typename T::TiledSavedQ{mdQaccum};

    typename T::TiledMmadQ tiled_mma_dq;
    auto thr_mma_do = tiled_mma_dq.get_slice(compat::local_id::x());

    Tensor mO_coord = cute::get_xe_tensor(O_shape);
    Tensor dQ_coord = cute::get_xe_tensor(dQ_shape);

    Tensor gdO = local_tile(mO_coord, select<0, 1>(tile_dO), make_coord(0, 0, 0));
    Tensor gdQ = local_tile(dQ_coord, select<0, 1>(tile_dO), make_coord(0, 0, 0));

    Tensor tdOgdO = thr_mma_do.partition_C(gdO);
    Tensor tdQgdQ = thr_mma_do.partition_C(gdQ);

    Tensor tdQrdQ = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tdOrdO = make_fragment_like<DType>(tdQrdQ);
    Tensor tdOrO = make_fragment_like<DType>(tdQrdQ);
    clear(tdOrdO);
    clear(tdOrO);
    clear(tdQrdQ);
    copy(tilesavedQ, tdQrdQ, tdQgdQ);
    copy(tileloaddO, tdOgdO, tdOrdO);
    copy(tileloadO, tdOgdO, tdOrO);

    Tensor dO_reshaped = make_tensor(tdOrdO.data(), convert_layout_2d_layout(tdOrdO.layout()));
    Tensor O_reshaped = make_tensor(tdOrO.data(), dO_reshaped.layout());
    constexpr int kGmemThreadsPerRow = kBlockM / size<0>(dO_reshaped);
    Tensor tdOrdP = make_fragment_like<float>(size<0>(dO_reshaped));
    constexpr int NUM_ROW_PER_THD = size<0>(dO_reshaped); // 32
    constexpr int NUM_SG_PER_ROW = kNSGs / (kBlockM / size<0>(dO_reshaped)); // 4
    constexpr int NUM_SG_PER_BLK_M = kBlockM / NUM_ROW_PER_THD; // 2

    const int sg_local_id = sg.get_local_id();
    const int sg_group_id = sg.get_group_id();
    const int sg_group_id_M = sg_group_id % NUM_SG_PER_BLK_M;
    const int sg_group_id_N = sg_group_id / NUM_SG_PER_BLK_M;
    auto smem = compat::local_mem<float[kNSGs * NUM_ROW_PER_THD]>();
    Tensor stensor = make_tensor(make_smem_ptr(smem), make_shape(Int<NUM_ROW_PER_THD>{}, Int<NUM_SG_PER_ROW>{}, Int<NUM_SG_PER_BLK_M>{}));

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(dO_reshaped); ++mi) {
        float dP_sum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(dO_reshaped); ++ni) {
            dP_sum = dP_sum + (float)dO_reshaped(mi, ni) * (float)O_reshaped(mi, ni);
        }
        tdOrdP(mi) = dP_sum;
    }
    /*
     * reduce within subgroup
     */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(tdOrdP); ++i) {
        tdOrdP(i) = reduce_over_group(sg, tdOrdP(i), sycl::plus<>());
    }
    /*
     * store to smem
     */
    if (sg_local_id == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(tdOrdP); ++i) {
            stensor(i, sg_group_id_N, sg_group_id_M) = tdOrdP(i);
        }
    }

    sycl::group_barrier(group);
    /*
     * reduce all sgs in the same row
     */
    if (sg_local_id == 0 and sg_group_id_N == 0) {
        for (int i = 0; i < size<0>(tdOrdP); ++i) {
            tdOrdP(i) = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < NUM_SG_PER_ROW; ++j) {
                tdOrdP(i) += stensor(i, j, sg_group_id_M);
            }
        }
    }
    // /*
    //  * broadcast to all threads in the sg 0
    //  */
    // CUTLASS_PRAGMA_UNROLL
    // for (int i = 0; i < size<0>(tdOrdP); ++i) {
    //     tdOrdP(i) = sycl::group_broadcast(sg, tdOrdP(i), 0);
    // }
    /*
     * write back to global memory
     */
    if constexpr(Is_even_M) {
        if (sg_local_id == 0 and sg_group_id_N == 0) {
            const int offset = sg_group_id_M * NUM_ROW_PER_THD;
            for (int i = 0; i < size<0>(tdOrdP); ++i) {
                mdPsum(i + offset) = tdOrdP(i);
            }
        }
    } else {
        if (sg_local_id == 0 and sg_group_id_N == 0) {
            const int offset = sg_group_id_M * NUM_ROW_PER_THD;
            int j = offset;
            for (int i = 0; i < size<0>(tdOrdP) and j < param.tail_m; ++i,++j) {
                mdPsum(j) = tdOrdP(i);
            }
        }
    }
}

template<class T>
void
mha_backward(T trait,
             Param<typename T::DType> param) {
    const int bidb = BlockIdxZ();
    const int bidh = BlockIdxY();
    constexpr bool parallel_seq_kv = false;
    // const int max_n_block = ceil_div(param.seq_len_kv, trait.kBlockN);
    for (int n_block = 0; n_block < param.n_block; ++n_block)
        dq_dk_dv_1colblock<parallel_seq_kv, true>(trait, param, bidb, bidh, n_block);
    if (param.tail_n > 0)
        dq_dk_dv_1colblock<parallel_seq_kv, false>(trait, param, bidb, bidh, param.n_block, param.tail_n);
}

template<class T>
void
mha_dot_do_o(T trait,
             Param<typename T::DType> param) {
    // The block index for the M dimension.
    const int m_block = BlockIdxX();
    // The block index for the batch.
    const int bidb = BlockIdxZ();
    // The block index for the head.
    const int bidh = BlockIdxY();;
    if (m_block == param.m_block - 1 and param.tail_m > 0) {
        compute_o_dot_do<false>(trait, param, m_block, bidb, bidh);
    } else {
        compute_o_dot_do<true>(trait, param, m_block, bidb, bidh);
    }
}

template <class T>
void
convert_dq(T &trait, Param<typename T::DType> &param, int m_block, int bidb, int bidh) {
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    using DType = typename T::DType;
    using VType = typename T::VType;

    auto bofst = Boffset(param);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);
    VType * dQaccum = param.dqaccum_ptr + dq_offset;
    DType * dQ = param.dq_ptr + q_offset;

    int tail_m = param.seq_len_q - m_block * kBlockM;
    int m = ThreadIdxX();
    if (m < tail_m) {
        for (int h = 0; h < kHeadDim; ++h) {
            dQ[m * param.q_r_stride + h] = static_cast<DType>(dQaccum[m * param.dq_r_stride + h]);
        }
    }
}

template <bool Is_even_M, class T>
void
convert_dq(T &trait, Param<typename T::DType> &param, int m_block, int bidb, int bidh) {
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    using DType = typename T::DType;
    using VType = typename T::VType;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;

    auto bofst = Boffset(param);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);
    using ShapeQ = Shape<
        std::conditional_t<Is_even_M, Int<kBlockM>, int>,
        Int<kHeadDim>, _1>;
    ShapeQ shapeQ;
    if constexpr (Is_even_M) {
        shapeQ = make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, _1{});
    } else {
        shapeQ = make_shape(param.tail_m, Int<kHeadDim>{}, _1{});
    }

    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      shapeQ,
                                      make_stride(param.dq_r_stride, _1{}, _1{})));
    Tensor mdQ = make_tensor(make_gmem_ptr(param.dq_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{}, _1{})));

    Shape tile_dq = typename T::TileShapedQ{};

    auto tileloaddQ = typename T::TiledLoaddQ{mdQaccum};
    auto tilesavedQ = typename T::TiledSavedV{mdQ};


    typename T::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_slice(first_thread_in_sg_idx);

    Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
    Tensor gdQ = local_tile(mQ_coord, select<0, 1>(tile_dq), make_coord(0, 0, 0)); // dump dQ

    Tensor tdQgdQ = thr_mma_dq.partition_C(gdQ); // save to dq
    Tensor tdQrdQaccum = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    Tensor tdQrdQ = make_fragment_like<DType>(tdQrdQaccum);
    copy(tileloaddQ, tdQgdQ, tdQrdQaccum);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tdQrdQ); ++i) {
        tdQrdQ(i) = static_cast<DType>(tdQrdQaccum(i));
    }
    copy(tilesavedQ, tdQrdQ, tdQgdQ);
}

template<class T>
void
mhd_convert_dq(T trait,
               Param<typename T::DType>  param) {
    // The block index for the M dimension.
    const int m_block = BlockIdxX();
    // The block index for the batch.
    const int bidb = BlockIdxZ();
    // The block index for the head.
    const int bidh = BlockIdxY();
    if (param.tail_m > 0 and m_block == param.m_block - 1) {
        convert_dq<false>(trait, param, m_block, bidb, bidh);
    } else {
        convert_dq<true>(trait, param, m_block, bidb, bidh);
    }
}

template<typename T, class ProblemShape, int kBlockM, int kBlockN, int kHeadDim, bool is_bhsd>
void launch_mha_backward_headdim(ProblemShape problem_shape,
                                 const T *do_d,
                                 const T *o_d,
                                 const T *q_d,
                                 const T *k_d,
                                 const T *v_d,
                                 const float *lse_d,
                                 float *odo_d,
                                 float *dqaccum_d,
                                 T *dq_d,
                                 T *dk_d,
                                 T *dv_d,
                                 T *s_d,
                                 T *dp_d,
                                 const int seq_len_q_pad,
                                 const int seq_len_kv_pad) {
    constexpr int numSGs = 8;
    constexpr int kBlockK = 32;
    auto trait = FAKernel<T, kHeadDim, kBlockM, kBlockN, kBlockK, numSGs>{};

    const int BATCH = get<0>(problem_shape);
    const int NUM_HEAD_Q = get<1>(problem_shape);
    const int NUM_HEAD_KV = get<2>(problem_shape);
    const int SEQ_LEN_Q = get<3>(problem_shape);
    const int SEQ_LEN_KV = get<4>(problem_shape);
    const int N_BLOCK = SEQ_LEN_KV / kBlockN;
    const int tail_n = SEQ_LEN_KV % kBlockN;
    const int M_BLOCK = ceil_div(SEQ_LEN_Q, kBlockM);
    const int tail_m = SEQ_LEN_Q % kBlockM;
    T * pbuff = compat::malloc<T>(BATCH * NUM_HEAD_Q * seq_len_kv_pad * kBlockM);
    auto param = Param<T>(do_d, o_d, q_d, k_d, v_d, lse_d, odo_d,
                          dqaccum_d, dq_d, dk_d, dv_d, s_d, dp_d, pbuff,
                          1 / sqrt(static_cast<float>(kHeadDim)));
    param.batch = BATCH;
    param.num_head_q = NUM_HEAD_Q;
    param.num_head_kv = NUM_HEAD_KV;
    param.seq_len_q = SEQ_LEN_Q;
    param.seq_len_kv = SEQ_LEN_KV;
    param.head_dim = kHeadDim;
    param.n_block = N_BLOCK;
    param.tail_n = tail_n;
    param.m_block = M_BLOCK;
    param.tail_m = tail_m;
    param.seq_len_kv_pad = seq_len_kv_pad;
    param.seq_len_q_pad = seq_len_q_pad;
    if constexpr(is_bhsd) {
        setup_bhsd_stride(param);
    } else {
        setup_bshd_stride(param);
    }

    auto dimGrid0 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
    auto dimBlock0 = compat::dim3(size(numSGs * trait.SubgroupSize), size(1), size(1));
    compat::experimental::launch_properties launch_props0{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    compat::experimental::kernel_properties kernel_props0{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy0{dimGrid0, dimBlock0, launch_props0, kernel_props0};
    auto event0 = compat::experimental::launch<
        mha_dot_do_o<decltype(trait)>>(policy0,
                                       trait,
                                       param);
    EventManager::getInstance().addEvent(event0);

    auto dimGrid1 = compat::dim3(size(1), size(param.num_head_q), size(param.batch));
    assert((trait.num_head_q % trait.num_head_kv == 0) && "num_head_q must be dividable by num_head_kv");
    assert((trait.num_head_q >= trait.num_head_kv) && "num_head_q must be bigger than or equal to num_head_kv");
    auto dimBlock1 = compat::dim3(size(numSGs * trait.SubgroupSize), size(1), size(1));
    // auto dimBlock = compat::dim3(size(trait.tiled_mma_sdp));

    compat::experimental::launch_properties launch_props1{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    compat::experimental::kernel_properties kernel_props1{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy1{dimGrid1, dimBlock1, launch_props1, kernel_props1};
    auto event1 = compat::experimental::launch<
        mha_backward<decltype(trait)>>(policy1,
                                       trait,
                                       param);
    EventManager::getInstance().addEvent(event1);

    auto dimGrid2 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
    auto dimBlock2 = compat::dim3(size(numSGs * trait.SubgroupSize), size(1), size(1));
    compat::experimental::launch_properties launch_props2{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    compat::experimental::kernel_properties kernel_props2{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    compat::experimental::launch_policy policy2{dimGrid2, dimBlock2, launch_props2, kernel_props2};
    auto event2 = compat::experimental::launch<
        mhd_convert_dq<decltype(trait)>>(policy2,
                                         trait,
                                         param);
    EventManager::getInstance().addEvent(event2);
}

template<typename T, class ProblemShape, int kBlockM, int kBlockN, bool is_bhsd>
void launch_mha_backward(ProblemShape problem_shape,
                         const T *do_d,
                         const T *o_d,
                         const T *q_d,
                         const T *k_d,
                         const T *v_d,
                         const float *lse_d,
                         float *odo_d,
                         float *dqaccum_d,
                         T *dq_d,
                         T *dk_d,
                         T *dv_d,
                         T *s_d,
                         T *dp_d,
                         const int seq_len_q_pad,
                         const int seq_len_kv_pad) {
    const int headdim = get<5>(problem_shape);
    if (headdim == 64) {
        constexpr int kHeadDim = 64;
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN, kHeadDim, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 96) {
        constexpr int kHeadDim = 96;
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN, kHeadDim, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 128) {
        constexpr int kHeadDim = 128;
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN, kHeadDim, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 192) {
        constexpr int kHeadDim = 192;
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN, kHeadDim, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else if (headdim == 256) {
        constexpr int kHeadDim = 256;
        launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN, kHeadDim, is_bhsd>(
            problem_shape,
            do_d, o_d, q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d,
            seq_len_q_pad, seq_len_kv_pad);
    } else {
        assert(false && "only support headdim 64,96,128,192,256");
    }
}

int main(int argc, char**argv) {
    // using T = cute::bfloat16_t;
    using T = cute::half_t;
    using V = float;
    std::string data_file = "mha.npz";
    // read qkv
    cnpy::NpyArray q_npy = cnpy::npz_load(data_file, "q");
    cnpy::NpyArray k_npy = cnpy::npz_load(data_file, "k");
    cnpy::NpyArray v_npy = cnpy::npz_load(data_file, "v");

    // read grad output
    cnpy::NpyArray do_npy = cnpy::npz_load(data_file, "grad");
    cnpy::NpyArray o_npy = cnpy::npz_load(data_file, "out");

    // read lse
    cnpy::NpyArray lse_npy = cnpy::npz_load(data_file, "lse");
    // read odo
    cnpy::NpyArray odo_npy = cnpy::npz_load(data_file, "odo");

    // read grad reference
    cnpy::NpyArray dq_npy = cnpy::npz_load(data_file, "q_grad");
    cnpy::NpyArray dk_npy = cnpy::npz_load(data_file, "k_grad");
    cnpy::NpyArray dv_npy = cnpy::npz_load(data_file, "v_grad");
    cnpy::NpyArray dp_npy = cnpy::npz_load(data_file, "p_grad");
    cnpy::NpyArray ds_npy = cnpy::npz_load(data_file, "s_grad");

    // read shape
    cnpy::NpyArray shape = cnpy::npz_load(data_file, "shape");

    int64_t BATCH = shape.data<int>()[0];
    int64_t NUM_HEAD_Q = shape.data<int>()[1];
    int64_t NUM_HEAD_KV = shape.data<int>()[2];
    int64_t SEQ_LEN_QO = shape.data<int>()[3];
    int64_t SEQ_LEN_KV = shape.data<int>()[4];
    int64_t HEAD_SIZE_QK = shape.data<int>()[5];
    int64_t HEAD_SIZE_VO = shape.data<int>()[6];
    bool is_causal = shape.data<int>()[7];
    bool is_bhsd = shape.data<int>()[8];
    assert(HEAD_SIZE_QK == HEAD_SIZE_VO && "only support head_size_qk==head_size_vo");
    constexpr int kBlockN = 32;
    constexpr int kBlockM = 64;
    int64_t SEQ_LEN_QO_PAD = ceil_div(SEQ_LEN_QO, kBlockM) * kBlockM;
    int64_t SEQ_LEN_KV_PAD = ceil_div(SEQ_LEN_KV, kBlockN) * kBlockN;
    printf("batch %d nh_q %d nh_k %d sq_q %d(%d) sq_k %d(%d) hd_q %d hd_v %d causal %d bhsd %d\n", BATCH, NUM_HEAD_Q, NUM_HEAD_KV, SEQ_LEN_QO, SEQ_LEN_QO_PAD, SEQ_LEN_KV, SEQ_LEN_KV_PAD, HEAD_SIZE_QK, HEAD_SIZE_VO, is_causal, is_bhsd);
    // read_args(argc, argv, 1, &BATCH);
    // read_args(argc, argv, 2, &NUM_HEAD_Q);
    // read_args(argc, argv, 3, &NUM_HEAD_KV);
    // read_args(argc, argv, 4, &SEQ_LEN_QO);
    // read_args(argc, argv, 5, &SEQ_LEN_KV);
    // read_args(argc, argv, 6, &HEAD_SIZE_QK);
    // read_args(argc, argv, 7, &HEAD_SIZE_VO);

    // alloc qkv
    T *q_d = compat::malloc<T>(q_npy.num_vals);
    T *k_d = compat::malloc<T>(k_npy.num_vals);
    T *v_d = compat::malloc<T>(v_npy.num_vals);

    // alloc ps
    T *p_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    T *s_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);

    // alloc lse, odo
    V *lse_d = compat::malloc<V>(lse_npy.num_vals);
    V *odo_d = compat::malloc<V>(odo_npy.num_vals);

    // alloc grad output
    T *do_d = compat::malloc<T>(do_npy.num_vals);
    T *o_d = compat::malloc<T>(o_npy.num_vals);

    // alloc grad test on device
    T *dq_d = compat::malloc<T>(dq_npy.num_vals);
    V *dqaccum_d = compat::malloc<V>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);
    T *dk_d = compat::malloc<T>(dk_npy.num_vals);
    T *dv_d = compat::malloc<T>(dv_npy.num_vals);
    T *dp_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    // copy qkv
    compat::memcpy<T>(q_d, q_npy.data<T>(), q_npy.num_vals);
    compat::memcpy<T>(k_d, k_npy.data<T>(), k_npy.num_vals);
    compat::memcpy<T>(v_d, v_npy.data<T>(), v_npy.num_vals);

    // copy grad output
    compat::memcpy<T>(do_d, do_npy.data<T>(), do_npy.num_vals);
    compat::memcpy<T>(o_d, o_npy.data<T>(), o_npy.num_vals);

    // copy lse
    compat::memcpy<V>(lse_d, lse_npy.data<V>(), lse_npy.num_vals);

    // copy odo
    // compat::memcpy<V>(odo_d, odo_npy.data<V>(), odo_npy.num_vals);

    auto problem_shape = ProblemShapeRegular(BATCH, NUM_HEAD_Q, NUM_HEAD_KV, SEQ_LEN_QO, SEQ_LEN_KV, HEAD_SIZE_QK, HEAD_SIZE_VO);
    if (is_bhsd) {
        launch_mha_backward<T, decltype(problem_shape), kBlockM, kBlockN, true>(
            problem_shape,
            do_d, o_d,
            q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);
    } else {
        launch_mha_backward<T, decltype(problem_shape), kBlockM, kBlockN, false>(
            problem_shape,
            do_d, o_d,
            q_d, k_d, v_d,
            lse_d, odo_d,
            dqaccum_d, dq_d, dk_d, dv_d,
            s_d, dp_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD);
    }
    float atol = 1e-3f;
    float rtol = 1e-3f;

    std::vector<V> odo_test(odo_npy.num_vals);
    compat::memcpy<V>(odo_test.data(), odo_d, odo_test.size());
    compat::wait_and_throw();
    printf("odo val: ");
    verify(odo_npy.data<V>(), odo_test.data(), BATCH, NUM_HEAD_Q, SEQ_LEN_QO, atol, rtol);

    compat::wait_and_throw();
    std::vector<T> dv_test(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_VO);
    compat::memcpy<T>(dv_test.data(), dv_d, dv_test.size());
    compat::wait_and_throw();
    printf("dV val: ");
    verify(dv_npy.data<T>(), dv_test.data(), BATCH * NUM_HEAD_KV, SEQ_LEN_KV, HEAD_SIZE_VO, atol, rtol);

    std::vector<T> dk_test(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_QK);
    compat::memcpy<T>(dk_test.data(), dk_d, dk_test.size());
    compat::wait_and_throw();
    printf("dK val: ");
    verify(dk_npy.data<T>(), dk_test.data(), BATCH * NUM_HEAD_KV, SEQ_LEN_KV, HEAD_SIZE_QK, atol, rtol);

    std::vector<T> dq_test(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    compat::memcpy<T>(dq_test.data(), dq_d, dq_test.size());
    compat::wait_and_throw();
    printf("dQ val: ");
    verify(dq_npy.data<T>(), dq_test.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, HEAD_SIZE_QK, atol, rtol);
}
