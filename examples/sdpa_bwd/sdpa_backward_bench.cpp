#include "sdpa_kernel.hpp"
#include "sdpa_util.hpp"
#include "reference_gpu.hpp"

template<class...> class mhaodoDeviceName;
template<class...> class mhabwdDeviceName;
template<class...> class mhacvtDeviceName;

using DurTuple = std::tuple<std::chrono::duration<long, std::nano>,
                            std::chrono::duration<long, std::nano>,
                            std::chrono::duration<long, std::nano>,
                            std::chrono::duration<long, std::nano>>;

template<typename T>
void
uniform_init(int seed, T *dst, size_t N, float a = -1.0f, float b = 1.0f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(a, b);
    for (int i = 0; i < N; ++i) {
        dst[i] = static_cast<T>(dis(gen));
    }
}

template<typename T>
void
norm_init(int seed, T *dst, size_t N, float c = 0.0f, float d = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(c, d);
    for (int i = 0; i < N; ++i) {
        dst[i] = static_cast<T>(dis(gen));
    }
}

template<typename T, class ProblemShape, int kBlockM, int kBlockN,
         int kHeadDim, int kNSGs, int AtomLayoutMSdP, int AtomLayoutNdKV,
         int AtomLayoutMdQ, bool is_causal, bool is_bhsd>
DurTuple
launch_mha_backward_headdim(ProblemShape problem_shape,
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
                            const int seq_len_kv_pad,
                            int count) {
    auto trait = FAKernel<T, kHeadDim, kBlockM, kBlockN, kNSGs,
                          AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                          is_causal>{};

    const int BATCH = get<0>(problem_shape);
    const int NUM_HEAD_Q = get<1>(problem_shape);
    const int NUM_HEAD_KV = get<2>(problem_shape);
    const int SEQ_LEN_Q = get<3>(problem_shape);
    const int SEQ_LEN_KV = get<4>(problem_shape);
    const int N_BLOCK = ceil_div(SEQ_LEN_KV, kBlockN);
    const int tail_n = SEQ_LEN_KV % kBlockN;
    const int M_BLOCK = ceil_div(SEQ_LEN_Q, kBlockM);
    const int tail_m = SEQ_LEN_Q % kBlockM;
    T * pbuff = compat::malloc<T>(BATCH * NUM_HEAD_Q * seq_len_kv_pad * 2 * kBlockM);
    auto param = Param<T>(do_d, o_d, q_d, k_d, v_d, lse_d, odo_d,
                          dqaccum_d, dq_d, dk_d, dv_d, pbuff,
                          1 / sqrt(static_cast<float>(kHeadDim)));
    param.batch = BATCH;
    param.num_head_q = NUM_HEAD_Q;
    param.num_head_kv = NUM_HEAD_KV;
    param.num_qh_per_kvh = NUM_HEAD_Q / NUM_HEAD_KV;
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
    printf("Launching mha backward kernel with %d times\n", count);
    auto start = std::chrono::high_resolution_clock::now();
    auto dur0 = start - start;
    auto dur1 = start - start;
    auto dur2 = start - start;
    for (int iii=0; iii < count; ++iii) {
        auto start0 = std::chrono::high_resolution_clock::now();
        auto dimGrid0 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
        auto dimBlock0 = compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props0{
            // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
        };
        compat::experimental::kernel_properties kernel_props0{
            sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
        compat::experimental::launch_policy policy0{dimGrid0, dimBlock0, launch_props0, kernel_props0};
        auto event0 = compat::experimental::launch<
            mha_dot_do_o<decltype(trait)>,
            mhaodoDeviceName<decltype(trait)>>(policy0,
                                               trait,
                                               param);
        EventManager::getInstance().addEvent(event0);
        compat::wait_and_throw();
        auto end0 = std::chrono::high_resolution_clock::now();
        dur0 += end0 - start0;

        auto start1 = std::chrono::high_resolution_clock::now();
        auto dimGrid1 = compat::dim3(size(param.n_block),
                                     size(param.num_head_q), size(param.batch));
        assert((param.num_head_q % param.num_head_kv == 0) && "num_head_q must be dividable by num_head_kv");
        assert((param.num_head_q >= param.num_head_kv) && "num_head_q must be bigger than or equal to num_head_kv");
        auto dimBlock1 = compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));

        compat::experimental::launch_properties launch_props1{
            sycl::ext::oneapi::experimental::work_group_scratch_size(trait.smem_size),
        };
        compat::experimental::kernel_properties kernel_props1{
            sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
        compat::experimental::launch_policy policy1{dimGrid1, dimBlock1, launch_props1, kernel_props1};
        auto event1 = compat::experimental::launch<
            mha_backward_parallel<decltype(trait)>,
            mhabwdDeviceName<decltype(trait)>>(policy1,
                                               trait,
                                               param);
        EventManager::getInstance().addEvent(event1);
        compat::wait_and_throw();
        auto end1 = std::chrono::high_resolution_clock::now();
        dur1 += end1 - start1;

        auto start2 = std::chrono::high_resolution_clock::now();
        auto dimGrid2 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
        auto dimBlock2 = compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props2{
            // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
        };
        compat::experimental::kernel_properties kernel_props2{
            sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
        compat::experimental::launch_policy policy2{dimGrid2, dimBlock2, launch_props2, kernel_props2};
        auto event2 = compat::experimental::launch<
            mhd_convert_dq<decltype(trait)>,
            mhacvtDeviceName<decltype(trait)>>(policy2,
                                               trait,
                                               param);
        EventManager::getInstance().addEvent(event2);
        compat::wait_and_throw();
        auto end2 = std::chrono::high_resolution_clock::now();
        dur2 += end2 - start2;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - start;
    DurTuple res = make_tuple(dur0, dur1, dur2, dur);
    return res;
}

template<typename T, class ProblemShape, int kMPad, int kNPad, bool is_causal, bool is_bhsd>
DurTuple
launch_mha_backward(ProblemShape problem_shape,
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
                    const int seq_len_kv_pad,
                    int count) {
    const int headdim = get<5>(problem_shape);
    if (headdim == 64) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 64;
        constexpr int kHeadDim = 64;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 4;
        constexpr int AtomLayoutNdKV = 2;
        constexpr int AtomLayoutMdQ = 2;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        return launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
                                        problem_shape,
                                        do_d, o_d, q_d, k_d, v_d,
                                        lse_d, odo_d,
                                        dqaccum_d, dq_d, dk_d, dv_d,
                                        s_d, dp_d,
                                        seq_len_q_pad, seq_len_kv_pad, count);
    } else if (headdim == 96) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 64;
        constexpr int kHeadDim = 96;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 2;
        constexpr int AtomLayoutNdKV = 4;
        constexpr int AtomLayoutMdQ = 4;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        return launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
                                        problem_shape,
                                        do_d, o_d, q_d, k_d, v_d,
                                        lse_d, odo_d,
                                        dqaccum_d, dq_d, dk_d, dv_d,
                                        s_d, dp_d,
                                        seq_len_q_pad, seq_len_kv_pad, count);
    } else if (headdim == 128) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 64;
        constexpr int kHeadDim = 128;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 2;
        constexpr int AtomLayoutNdKV = 4;
        constexpr int AtomLayoutMdQ = 4;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        return launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
                                        problem_shape,
                                        do_d, o_d, q_d, k_d, v_d,
                                        lse_d, odo_d,
                                        dqaccum_d, dq_d, dk_d, dv_d,
                                        s_d, dp_d,
                                        seq_len_q_pad, seq_len_kv_pad, count);
    } else if (headdim == 192) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 32;
        constexpr int kHeadDim = 192;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 4;
        constexpr int AtomLayoutNdKV = 2;
        constexpr int AtomLayoutMdQ = 2;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        return launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
                                        problem_shape,
                                        do_d, o_d, q_d, k_d, v_d,
                                        lse_d, odo_d,
                                        dqaccum_d, dq_d, dk_d, dv_d,
                                        s_d, dp_d,
                                        seq_len_q_pad, seq_len_kv_pad, count);
    } else if (headdim == 256) {
        constexpr int kBlockM = 64;
        constexpr int kBlockN = 32;
        constexpr int kHeadDim = 256;
        constexpr int kNSGs = 8;
        constexpr int AtomLayoutMSdP = 4;
        constexpr int AtomLayoutNdKV = 2;
        constexpr int AtomLayoutMdQ = 2;
        static_assert(kBlockM <=  kMPad, "kBlockM must be less than or equal to kMPad");
        static_assert(kBlockN <=  kNPad, "kBlockN must be less than or equal to kNPad");
        return launch_mha_backward_headdim<T, ProblemShape, kBlockM, kBlockN,
                                    kHeadDim, kNSGs,
                                    AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                    is_causal, is_bhsd>(
                                        problem_shape,
                                        do_d, o_d, q_d, k_d, v_d,
                                        lse_d, odo_d,
                                        dqaccum_d, dq_d, dk_d, dv_d,
                                        s_d, dp_d,
                                        seq_len_q_pad, seq_len_kv_pad, count);
    } else {
        assert(false && "only support headdim 64,96,128,192,256");
    }
    return DurTuple{};
}

template<typename T, typename V, class ProblemShape>
DurTuple
launch_mha_wrapper(ProblemShape problem_shape, bool is_causal, bool is_bhsd, int count, bool checksum) {
    const int BATCH = get<0>(problem_shape);
    const int NUM_HEAD_Q = get<1>(problem_shape);
    const int NUM_HEAD_KV = get<2>(problem_shape);
    const int SEQ_LEN_QO = get<3>(problem_shape);
    const int SEQ_LEN_KV = get<4>(problem_shape);
    const int HEAD_SIZE_QK = get<5>(problem_shape);
    const int HEAD_SIZE_VO = get<6>(problem_shape);

    constexpr int kBlockN = 64;
    constexpr int kBlockM = 64;
    int64_t SEQ_LEN_QO_PAD = ceil_div(SEQ_LEN_QO, kBlockM) * kBlockM;
    int64_t SEQ_LEN_KV_PAD = ceil_div(SEQ_LEN_KV, kBlockN) * kBlockN;

    // alloc host memory for qkv
    std::vector<T> q_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    std::vector<T> k_h(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_QK);
    std::vector<T> v_h(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_VO);

    // alloc host memory for ps
    std::vector<T> p_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    std::vector<T> ds_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);

    // alloc host memory for lse, odo
    std::vector<V> lse_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO);
    std::vector<V> odo_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO);

    // alloc host memory for grad output
    std::vector<T> do_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_VO);
    std::vector<T> o_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_VO);

    // alloc host memory for grad test
    std::vector<T> dq_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    std::vector<T> dk_h(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_QK);
    std::vector<T> dv_h(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_VO);

    std::vector<V> dqaccum_h(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);

    int seed = 123;
    // init qkv
    norm_init(seed + 1, q_h.data(), q_h.size());
    norm_init(seed + 2, k_h.data(), k_h.size());
    norm_init(seed + 3, v_h.data(), v_h.size());

    // init grad output
    norm_init(seed + 5, do_h.data(), do_h.size());
    
    // compute o and lse from qkv using forward pass
    sdpa_forward_reference_gpu<T, V>(q_h.data(), k_h.data(), v_h.data(),
                                     is_causal, is_bhsd,
                                     BATCH, NUM_HEAD_Q, NUM_HEAD_KV,
                                     SEQ_LEN_QO, SEQ_LEN_KV,
                                     HEAD_SIZE_QK, HEAD_SIZE_VO,
                                     o_h.data(), lse_h.data());

    // alloc qkv
    T *q_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    T *k_d = compat::malloc<T>(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_QK);
    T *v_d = compat::malloc<T>(BATCH * NUM_HEAD_KV * SEQ_LEN_KV * HEAD_SIZE_VO);

    // alloc ps
    T *p_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);
    T *ds_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * SEQ_LEN_KV_PAD);

    // alloc lse, odo
    V *lse_d = compat::malloc<V>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO);
    V *odo_d = compat::malloc<V>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO);

    // alloc grad output
    T *do_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_VO);
    T *o_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_VO);

    // alloc grad test on device
    T *dq_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    T *dk_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_QK);
    T *dv_d = compat::malloc<T>(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_VO);

    V *dqaccum_d = compat::malloc<V>(BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);
    // init dqaccum
    compat::fill(dqaccum_d, 0.0f, BATCH * NUM_HEAD_Q * SEQ_LEN_QO_PAD * HEAD_SIZE_QK);

    // copy qkv
    compat::memcpy<T>(q_d, q_h.data(), q_h.size());
    compat::memcpy<T>(k_d, k_h.data(), k_h.size());
    compat::memcpy<T>(v_d, v_h.data(), v_h.size());

    // copy grad output
    compat::memcpy<T>(do_d, do_h.data(), do_h.size());
    compat::memcpy<T>(o_d, o_h.data(), o_h.size());

    // copy lse
    compat::memcpy<V>(lse_d, lse_h.data(), lse_h.size());
    DurTuple res;
    if (is_bhsd) {
        if (is_causal)
            res = launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, true, true>(
                                    problem_shape,
                                    do_d, o_d,
                                    q_d, k_d, v_d,
                                    lse_d, odo_d,
                                    dqaccum_d, dq_d, dk_d, dv_d,
                                    p_d, ds_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD, count);
        else
            res = launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, false, true>(
                                    problem_shape,
                                    do_d, o_d,
                                    q_d, k_d, v_d,
                                    lse_d, odo_d,
                                    dqaccum_d, dq_d, dk_d, dv_d,
                                    p_d, ds_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD, count);
    } else {
        if (is_causal) {
            res = launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, true, false>(
                                    problem_shape,
                                    do_d, o_d,
                                    q_d, k_d, v_d,
                                    lse_d, odo_d,
                                    dqaccum_d, dq_d, dk_d, dv_d,
                                    p_d, ds_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD, count);
        } else {
            res = launch_mha_backward<T, decltype(problem_shape),
                                kBlockM, kBlockN, false, false>(
                                    problem_shape,
                                    do_d, o_d,
                                    q_d, k_d, v_d,
                                    lse_d, odo_d,
                                    dqaccum_d, dq_d, dk_d, dv_d,
                                    p_d, ds_d, SEQ_LEN_QO_PAD, SEQ_LEN_KV_PAD, count);
        }
    }

    std::vector<V> odo_ref(BATCH * NUM_HEAD_Q * SEQ_LEN_QO);
    std::vector<V> dqaccum_ref(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    std::vector<T> dq_ref(BATCH * NUM_HEAD_Q * SEQ_LEN_QO * HEAD_SIZE_QK);
    std::vector<T> dk_ref(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_QK);
    std::vector<T> dv_ref(BATCH * NUM_HEAD_Q * SEQ_LEN_KV * HEAD_SIZE_VO);
    if (checksum) {
        sdpa_backward_reference_gpu<T, V>(q_h.data(), k_h.data(), v_h.data(),
                                      o_h.data(), do_h.data(), lse_h.data(),
                                      is_causal, is_bhsd,
                                      BATCH, NUM_HEAD_Q, NUM_HEAD_KV,
                                      SEQ_LEN_QO, SEQ_LEN_KV,
                                      HEAD_SIZE_QK, HEAD_SIZE_VO,
                                      odo_ref.data(), dqaccum_ref.data(),
                                      dq_ref.data(), dk_ref.data(), dv_ref.data());
        float atol = 1e-2f;
        float rtol = 1e-2f;
        compat::memcpy<V>(odo_h.data(), odo_d, odo_h.size());
        compat::memcpy<V>(dqaccum_h.data(), dqaccum_d, dqaccum_h.size());
        compat::memcpy<T>(dq_h.data(), dq_d, dq_h.size());
        compat::memcpy<T>(dk_h.data(), dk_d, dk_h.size());
        compat::memcpy<T>(dv_h.data(), dv_d, dv_h.size());
        compat::wait_and_throw();
        printf("odo val: ");
        verify(odo_ref.data(), odo_h.data(), BATCH, NUM_HEAD_Q, SEQ_LEN_QO, atol, rtol);
        printf("dq val: ");
        verify(dq_ref.data(), dq_h.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_QO, HEAD_SIZE_QK, atol, rtol);
        printf("dk val: ");
        verify(dk_ref.data(), dk_h.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_KV, HEAD_SIZE_QK, atol, rtol);
        printf("dv val: ");
        verify(dv_ref.data(), dv_h.data(), BATCH * NUM_HEAD_Q, SEQ_LEN_KV, HEAD_SIZE_VO, atol, rtol);
    }

    compat::free(q_d);
    compat::free(k_d);
    compat::free(v_d);

    compat::free(p_d);
    compat::free(ds_d);

    compat::free(lse_d);
    compat::free(odo_d);

    compat::free(do_d);
    compat::free(o_d);

    compat::free(dq_d);
    compat::free(dk_d);
    compat::free(dv_d);
    compat::free(dqaccum_d);
    return res;
}
int main(int argc, const char**argv) {
    // using T = cute::bfloat16_t;
    using V = float;

    Options options;
    options.parse(argc, argv);
    if (options.help) {
        options.print_usage(std::cout);
        return 0;
    }
    int64_t BATCH = options.batch;
    int64_t NUM_HEAD_Q = options.num_heads_q;
    int64_t NUM_HEAD_KV = options.num_heads_kv;
    int64_t SEQ_LEN_QO = options.seq_len_qo;
    int64_t SEQ_LEN_KV = options.seq_len_kv;
    int64_t HEAD_SIZE_QK = options.head_size_qk;
    int64_t HEAD_SIZE_VO = options.head_size_vo;
    bool is_causal = options.is_causal;
    bool is_bhsd = options.is_bhsd;
    constexpr int kBlockN = 64;
    constexpr int kBlockM = 64;
    int64_t SEQ_LEN_QO_PAD = ceil_div(SEQ_LEN_QO, kBlockM) * kBlockM;
    int64_t SEQ_LEN_KV_PAD = ceil_div(SEQ_LEN_KV, kBlockN) * kBlockN;
    assert(HEAD_SIZE_QK == HEAD_SIZE_VO && "only support head_size_qk==head_size_vo");
    printf("batch %d nh_q %d nh_k %d sq_q %d(%d) sq_k %d(%d) hd_q %d hd_v %d causal %d bhsd %d\n", BATCH, NUM_HEAD_Q, NUM_HEAD_KV, SEQ_LEN_QO, SEQ_LEN_QO_PAD, SEQ_LEN_KV, SEQ_LEN_KV_PAD, HEAD_SIZE_QK, HEAD_SIZE_VO, is_causal, is_bhsd);



    auto problem_shape = ProblemShapeRegular(BATCH, NUM_HEAD_Q, NUM_HEAD_KV,
                                             SEQ_LEN_QO, SEQ_LEN_KV, HEAD_SIZE_QK, HEAD_SIZE_VO);

    DurTuple res;
    if (options.is_bf16) {
        using T = cute::bfloat16_t;
        launch_mha_wrapper<T, V>(problem_shape, is_causal, is_bhsd, 1, true);
    } else {
        using T = cute::half_t;
        launch_mha_wrapper<T, V>(problem_shape, is_causal, is_bhsd, 1, true);
    }
    float atol = 3e-3f;
    float rtol = 3e-3f;
    // verify
    // benchmark
    int count = options.iterations;
    if (options.is_bf16) {
        using T = cute::bfloat16_t;
        res = launch_mha_wrapper<T, V>(problem_shape, is_causal, is_bhsd, count, false);
    } else {
        using T = cute::half_t;
        res = launch_mha_wrapper<T, V>(problem_shape, is_causal, is_bhsd, count, false);
    }

    auto [dur0, dur1, dur2, dur] = res;
    printf("mha dot do kernel time, odo :%f mha bwd: %f dq: %f total: %f us\n",
           std::chrono::duration_cast<std::chrono::microseconds>(dur0).count() / float(count),
           std::chrono::duration_cast<std::chrono::microseconds>(dur1).count() / float(count),
           std::chrono::duration_cast<std::chrono::microseconds>(dur2).count() / float(count),
           std::chrono::duration_cast<std::chrono::microseconds>(dur).count() / float(count));
    return 0;
}
