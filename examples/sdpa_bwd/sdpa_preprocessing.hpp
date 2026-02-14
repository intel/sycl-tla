#pragma once

template<bool Is_even_M, class T>
void
compute_o_dot_do(T &trait, Param<typename T::DType> &param,
                 const int m_block, const int bidb, const int bidh) {
    // The thread index.
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    constexpr int SubgroupSize = T::SubgroupSize;
    using DType = typename T::DType;
    using VType = typename T::VType;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = Boffset(param);

    const index_t o_offset = bofst.o_offset(bidb, bidh, m_block * kBlockM);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, m_block * kBlockM);

    using ShapeO = Shape<
        std::conditional_t <Is_even_M, Int<kBlockM>, int>,
        Int<kHeadDim>>;
    using ShapeP = Shape<
        std::conditional_t <Is_even_M, Int<kBlockM>, int>>;
    ShapeO O_shape;
    ShapeP dP_shape;
    if constexpr(Is_even_M) {
        O_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
        dP_shape = make_shape(Int<kBlockM>{});
    } else {
        O_shape = make_shape(param.tail_m, Int<kHeadDim>{});
        dP_shape = make_shape(param.tail_m);
    }
    auto dQ_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});

    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                             make_layout(
                                 O_shape,
                                 make_stride(param.o_r_stride, _1{})));
    Tensor mO = make_tensor(make_gmem_ptr(param.o_ptr + o_offset),
                            make_layout(
                                O_shape,
                                make_stride(param.o_r_stride, _1{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                                      make_stride(param.dq_r_stride, _1{})));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + dpsum_offset),
                                make_layout(
                                    dP_shape,
                                    Stride<_1>{}));
    using ThreadLayout = Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>,
                                Stride<Int<SubgroupSize>, _1>>;
    using ValueLayout = std::conditional_t<
        kHeadDim == 96,
        Layout<Shape<_1, _2>>,
        std::conditional_t<
            kHeadDim == 192,
            Layout<Shape<_1, _4>>,
            Layout<Shape<_1, Int<kHeadDim / SubgroupSize>>>>>;
    using OdOType = cutlass::AlignedArray<DType, size(ValueLayout{})>;
    using OdOAtom = Copy_Atom<UniversalCopy<OdOType>, DType>;
    using dQType = cutlass::AlignedArray<VType, size(ValueLayout{})>;
    using dQAtom = Copy_Atom<UniversalCopy<dQType>, VType>;

    auto tileload_odo = make_tiled_copy(OdOAtom{},
                                        ThreadLayout{},
                                        ValueLayout{});
    auto tileload_dq = make_tiled_copy(dQAtom{},
                                       ThreadLayout{},
                                       ValueLayout{});

    auto thr_load_odo = tileload_odo.get_thread_slice(ThreadIdxX());
    auto thr_load_dq = tileload_dq.get_thread_slice(ThreadIdxX());

    Tensor thr_tile_do_S = thr_load_odo.partition_S(mdO);
    Tensor thr_tile_o_S = thr_load_odo.partition_S(mO);
    Tensor thr_tile_dq_D = thr_load_dq.partition_D(mdQaccum);
    Tensor rdQ = make_fragment_like(thr_tile_dq_D);
    Tensor rdO = make_fragment_like<DType>(rdQ);
    Tensor rO = make_fragment_like<DType>(rdQ);
    Tensor cO = make_identity_tensor(dQ_shape);
    Tensor tcO = thr_load_odo.partition_S(cO);
    Tensor tcO_row = logical_divide(tcO, Shape<_1>{})(make_coord(0, 0), _, 0);
    Layout rdO_layout = rdO.layout();
    Tensor rdO_2d = make_tensor(rdO.data(),
                                make_layout(get<1>(rdO_layout),
                                            make_layout(get<0>(rdO_layout),
                                                        get<2>(rdO_layout))));
    Tensor rO_2d = make_tensor(rO.data(),
                               rdO_2d.layout());

    constexpr int NumValperCol = size<0>(rdO_2d);
    auto smem = compat::local_mem<VType[kNSGs * SubgroupSize * NumValperCol]>();
    auto stensor = make_tensor(make_smem_ptr(smem),
                               make_layout(
                                   Shape<
                                   Int<NumValperCol>,
                                   Int<kNSGs>,
                                   Int<SubgroupSize>>{}));
    clear(rdO_2d);
    clear(rO_2d);
    if constexpr(Is_even_M) {
        copy(tileload_odo, thr_tile_do_S, rdO);
        copy(tileload_odo, thr_tile_o_S, rO);
    } else {
        for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
            if (get<0>(tcO_row(mi)) < param.tail_m) {
                copy(tileload_odo, thr_tile_do_S(_, mi, _), rdO(_, mi, _));
                copy(tileload_odo, thr_tile_o_S(_, mi, _), rO(_, mi, _));
            }
        }
    }
    int sg_group_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
        float accum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(rdO_2d); ++ni) {
            accum = accum + (float)rdO_2d(mi, ni) * (float)rO_2d(mi, ni);
        }
        stensor(mi, sg_group_id, sg_local_id) = accum;
    }
    // sycl::group_barrier(group);
    if (sg_local_id == 0) {
        float accum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NumValperCol; ++mi) {
            float accum = 0.0f;
            // reduce within subgroup
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < SubgroupSize; ++ni) {
                accum += stensor(mi, sg_group_id, ni);
            }
            if constexpr(Is_even_M) {
                mdPsum(get<0>(tcO_row(mi))) = accum;
            } else {
                if (get<0>(tcO_row(mi)) < param.tail_m) {
                    mdPsum(get<0>(tcO_row(mi))) = accum;
                }
            }
        }
    }
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
    const int bidh = BlockIdxY();
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

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template<class Engine0, class Layout0,
         class Engine1, class Layout1>
CUTLASS_DEVICE
void
convert_type(Tensor<Engine0, Layout0> const & src,
             Tensor<Engine1, Layout1> & dst) {
    using From_type = typename Engine0::value_type;
    using To_type = typename Engine1::value_type;
    constexpr int numel = decltype(size(src))::value;
    cutlass::NumericConverter<To_type, From_type> convert_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < numel; ++i) {
        dst(i) = convert_op(src(i));
    }
}

template <bool Is_even_M, class T>
void
convert_dq(T &trait, Param<typename T::DType> &param, int m_block, int bidb, int bidh) {
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    constexpr int SubgroupSize = T::SubgroupSize;
    using DType = typename T::DType;
    using VType = typename T::VType;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;

    auto bofst = Boffset(param);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);
    using ShapeQ = Shape<
        std::conditional_t<Is_even_M, Int<kBlockM>, int>,
        Int<kHeadDim>>;
    ShapeQ shapeQ;
    if constexpr (Is_even_M) {
        shapeQ = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
    } else {
        shapeQ = make_shape(param.tail_m, Int<kHeadDim>{});
    }

    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                      make_stride(param.dq_r_stride, _1{})));
    Tensor mdQ = make_tensor(make_gmem_ptr(param.dq_ptr + q_offset),
                            make_layout(
                                shapeQ,
                                make_stride(param.q_r_stride, _1{})));

    using ThreadLayout = Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>,
                                Stride<Int<SubgroupSize>, _1>>;
    using ValueLayout = std::conditional_t<
        kHeadDim == 96,
        Layout<Shape<_1, _2>>,
        std::conditional_t<
            kHeadDim == 192,
            Layout<Shape<_1, _4>>,
            Layout<Shape<_1, Int<kHeadDim / SubgroupSize>>>>>;

    using dQaccumType = cutlass::AlignedArray<VType, size(ValueLayout{})>;
    using dQaccumAtom = Copy_Atom<UniversalCopy<dQaccumType>, VType>;
    using dQType = cutlass::AlignedArray<DType, size(ValueLayout{})>;
    using dQAtom = Copy_Atom<UniversalCopy<dQType>, DType>;

    auto tileload_dQaccum = make_tiled_copy(dQaccumAtom{},
                                            ThreadLayout{},
                                            ValueLayout{});
    auto tilesave_dQ = make_tiled_copy(dQAtom{},
                                       ThreadLayout{},
                                       ValueLayout{});

    auto thr_load_dQaccum = tileload_dQaccum.get_thread_slice(ThreadIdxX());
    auto thr_save_dQ = tilesave_dQ.get_thread_slice(ThreadIdxX());

    Tensor thr_tile_dQaccum_S = thr_load_dQaccum.partition_S(mdQaccum);
    Tensor thr_tile_dQ_D = thr_save_dQ.partition_D(mdQ);

    Tensor rdQaccum = make_fragment_like(thr_tile_dQaccum_S);

    copy(tileload_dQaccum, thr_tile_dQaccum_S, rdQaccum);
    if constexpr(Is_even_M) {
        Tensor rdQ = convert_type<DType>(rdQaccum);
        copy(tilesave_dQ, rdQ, thr_tile_dQ_D);
    } else {
        Tensor rdQ = make_tensor_like<DType>(rdQaccum);
        convert_type(rdQaccum, rdQ);
        Tensor accQ = make_identity_tensor(shapeQ);
        Tensor taccQ = thr_save_dQ.partition_D(accQ);
        Tensor taccQ_rc = logical_divide(taccQ, Shape<_1> {})(0, _, 0);
        for (int m = 0; m < size<1>(thr_tile_dQ_D); ++m) {
            int row_m = get<0>(taccQ_rc(m));
            if (row_m < param.tail_m) {
                auto thr_tile_dq = thr_tile_dQ_D(_, m, _);
                auto r_dq = rdQ(_, m, _);
                copy(r_dq, thr_tile_dq);
            }
        }
    }
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
