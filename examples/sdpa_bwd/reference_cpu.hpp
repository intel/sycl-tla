#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include <cmath>
#include <cstring>
#include <vector>

inline void compute_odo(float *odo_output, const std::vector<float> &o,
                        const std::vector<float> &do_data, int seq_len,
                        int head_size) {
    for (int i = 0; i < seq_len; ++i) {
        float sum = 0.0f;
        for (int d = 0; d < head_size; ++d) {
            sum += o[i * head_size + d] * do_data[i * head_size + d];
        }
        odo_output[i] = sum;
    }
}

inline void apply_causal_mask(std::vector<float> &S, int seq_len_q,
                              int seq_len_k, bool is_causal) {
    if (!is_causal)
        return;

    int offset = seq_len_k - seq_len_q;

    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < seq_len_k; ++j) {
            if (j > i + offset) {
                S[i * seq_len_k + j] = -INFINITY;
            }
        }
    }
}

inline void compute_softmax_with_lse(std::vector<float> &P,
                                     const std::vector<float> &S,
                                     const float *lse, int seq_len_q,
                                     int seq_len_k) {
    P.resize(seq_len_q * seq_len_k);

    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < seq_len_k; ++j) {
            int idx = i * seq_len_k + j;
            P[idx] = expf(S[idx] - lse[i]);
        }
    }
}

inline void compute_softmax_backward(std::vector<float> &dS,
                                     const std::vector<float> &P,
                                     const std::vector<float> &dP,
                                     const float *oDo, int seq_len_q,
                                     int seq_len_k) {
    dS.resize(seq_len_q * seq_len_k);

    for (int i = 0; i < seq_len_q; ++i) {
        float oDo_val = oDo[i];
        for (int j = 0; j < seq_len_k; ++j) {
            int idx = i * seq_len_k + j;

            dS[idx] = P[idx] * (dP[idx] - oDo_val);
        }
    }
}

inline int compute_strided_index(int b, int h, int s, int d, int num_head,
                                 int seq_len, int head_size, bool is_bhsd) {
    if (is_bhsd) {

        return ((b * num_head + h) * seq_len + s) * head_size + d;
    } else {

        return ((b * seq_len + s) * num_head + h) * head_size + d;
    }
}

inline int get_lse_index(int b, int h, int s, int num_head, int seq_len) {
    return (b * num_head + h) * seq_len + s;
}

inline int get_attention_matrix_index(int b, int h, int s_q, int s_k,
                                      int num_head_qo, int seq_len_qo,
                                      int seq_len_kv, bool is_bhsd) {
    if (is_bhsd) {
        return ((b * num_head_qo + h) * seq_len_qo + s_q) * seq_len_kv + s_k;
    } else {
        return ((b * seq_len_qo + s_q) * num_head_qo + h) * seq_len_kv + s_k;
    }
}

template <typename T>
inline void copy_strided_to_float(std::vector<float> &dst, const T *src, int b,
                                  int h, int seq_len, int head_size,
                                  int num_head, bool is_bhsd) {
    dst.resize(seq_len * head_size);
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < head_size; ++d) {
            int src_idx = compute_strided_index(b, h, s, d, num_head, seq_len,
                                                head_size, is_bhsd);
            dst[s * head_size + d] = static_cast<float>(src[src_idx]);
        }
    }
}

template <typename T>
inline void write_float_to_strided(T *dst, const std::vector<float> &src, int b,
                                   int h, int seq_len, int head_size,
                                   int num_head, bool is_bhsd,
                                   bool accumulate = false) {
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < head_size; ++d) {
            int dst_idx = compute_strided_index(b, h, s, d, num_head, seq_len,
                                                head_size, is_bhsd);
            if (accumulate) {
                dst[dst_idx] = static_cast<T>(static_cast<float>(dst[dst_idx]) +
                                              src[s * head_size + d]);
            } else {
                dst[dst_idx] = static_cast<T>(src[s * head_size + d]);
            }
        }
    }
}

template <typename T>
inline void write_attention_matrix(T *dst, const std::vector<float> &src, int b,
                                   int h, int seq_len_qo, int seq_len_kv,
                                   int num_head_qo, bool is_bhsd) {
    if (dst == nullptr)
        return;
    for (int i = 0; i < seq_len_qo; ++i) {
        for (int j = 0; j < seq_len_kv; ++j) {
            int dst_idx = get_attention_matrix_index(b, h, i, j, num_head_qo,
                                                     seq_len_qo, seq_len_kv, is_bhsd);
            dst[dst_idx] = static_cast<T>(src[i * seq_len_kv + j]);
        }
    }
}

inline void compute_lse_and_softmax_cpu(std::vector<float> &P, float *lse,
                                        const std::vector<float> &S,
                                        int seq_len_q, int seq_len_k) {
    P.resize(seq_len_q * seq_len_k);
    for (int i = 0; i < seq_len_q; ++i) {
        float max_val = -INFINITY;
        for (int j = 0; j < seq_len_k; ++j) {
            float v = S[i * seq_len_k + j];
            if (v > max_val)
                max_val = v;
        }
        if (max_val == -INFINITY) {
            lse[i] = -INFINITY;
            for (int j = 0; j < seq_len_k; ++j)
                P[i * seq_len_k + j] = 0.0f;
        } else {
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len_k; ++j)
                sum_exp += expf(S[i * seq_len_k + j] - max_val);
            lse[i] = max_val + logf(sum_exp);
            for (int j = 0; j < seq_len_k; ++j)
                P[i * seq_len_k + j] = expf(S[i * seq_len_k + j] - lse[i]);
        }
    }
}

template <typename T, typename V>
void sdpa_forward_reference_cpu(

    T *q_ptr, T *k_ptr, T *v_ptr,

    bool is_causal, bool is_bhsd,

    int batch, int num_head_qo, int num_head_kv, int seq_len_qo, int seq_len_kv,
    int head_size_qk, int head_size_vo,

    T *o_ptr, V *lse_ptr) {
    using LayoutRowMajor = cutlass::layout::RowMajor;
    using LayoutColMajor = cutlass::layout::ColumnMajor;
    using ComputeType = float;

    const float scale = 1.0f / sqrtf(static_cast<float>(head_size_qk));
    const int num_group = num_head_qo / num_head_kv;

    for (int b = 0; b < batch; ++b) {
        for (int h_qo = 0; h_qo < num_head_qo; ++h_qo) {
            int h_kv = h_qo / num_group;

            std::vector<float> Q_float, K_float, V_float;
            copy_strided_to_float(Q_float, q_ptr, b, h_qo, seq_len_qo, head_size_qk,
                                  num_head_qo, is_bhsd);
            copy_strided_to_float(K_float, k_ptr, b, h_kv, seq_len_kv, head_size_qk,
                                  num_head_kv, is_bhsd);
            copy_strided_to_float(V_float, v_ptr, b, h_kv, seq_len_kv, head_size_vo,
                                  num_head_kv, is_bhsd);

            std::vector<float> S(seq_len_qo * seq_len_kv, 0.0f);
            cutlass::TensorRef<float, LayoutRowMajor> ref_Q(Q_float.data(),
                                                            head_size_qk);
            cutlass::TensorRef<float, LayoutColMajor> ref_KT(K_float.data(),
                                                             head_size_qk);
            cutlass::TensorRef<float, LayoutRowMajor> ref_S(S.data(), seq_len_kv);
            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor, float, LayoutColMajor, float, LayoutRowMajor,
                ComputeType, ComputeType>(
                    {seq_len_qo, seq_len_kv, head_size_qk}, scale, ref_Q,
                    cutlass::ComplexTransform::kNone, ref_KT,
                    cutlass::ComplexTransform::kNone, 0.0f, ref_S, ref_S);

            apply_causal_mask(S, seq_len_qo, seq_len_kv, is_causal);

            std::vector<float> P;
            int lse_offset = get_lse_index(b, h_qo, 0, num_head_qo, seq_len_qo);
            compute_lse_and_softmax_cpu(P, lse_ptr + lse_offset, S, seq_len_qo,
                                        seq_len_kv);

            std::vector<float> O_float(seq_len_qo * head_size_vo, 0.0f);
            cutlass::TensorRef<float, LayoutRowMajor> ref_P(P.data(), seq_len_kv);
            cutlass::TensorRef<float, LayoutRowMajor> ref_V(V_float.data(),
                                                            head_size_vo);
            cutlass::TensorRef<float, LayoutRowMajor> ref_O(O_float.data(),
                                                            head_size_vo);
            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor, float, LayoutRowMajor, float, LayoutRowMajor,
                ComputeType, ComputeType>(
                    {seq_len_qo, head_size_vo, seq_len_kv}, 1.0f, ref_P,
                    cutlass::ComplexTransform::kNone, ref_V,
                    cutlass::ComplexTransform::kNone, 0.0f, ref_O, ref_O);

            write_float_to_strided<T>(o_ptr, O_float, b, h_qo, seq_len_qo,
                                      head_size_vo, num_head_qo, is_bhsd, false);
        }
    }
}

template <typename T, typename V>
void sdpa_backward_reference_cpu(

    T *q_ptr, T *k_ptr, T *v_ptr, T *o_ptr, T *do_ptr, V *lse_ptr,

    bool is_causal, bool is_bhsd,

    int batch, int num_head_qo, int num_head_kv, int seq_len_qo, int seq_len_kv,
    int head_size_qk, int head_size_vo,

    V *odo_ptr, V *dqaccum_ptr, T *dq_ptr, T *dk_ptr, T *dv_ptr) {
    using LayoutRowMajor = cutlass::layout::RowMajor;
    using LayoutColMajor = cutlass::layout::ColumnMajor;
    using ComputeType = float;

    const float scale = 1.0f / sqrtf(static_cast<float>(head_size_qk));
    const int num_group = num_head_qo / num_head_kv;

    for (int b = 0; b < batch; ++b) {
        for (int h_qo = 0; h_qo < num_head_qo; ++h_qo) {
            int h_kv = h_qo / num_group;
            bool is_first_in_group = (h_qo % num_group == 0);

            std::vector<float> Q_float, K_float, V_float, O_float, dO_float;

            copy_strided_to_float(Q_float, q_ptr, b, h_qo, seq_len_qo, head_size_qk,
                                  num_head_qo, is_bhsd);
            copy_strided_to_float(K_float, k_ptr, b, h_kv, seq_len_kv, head_size_qk,
                                  num_head_kv, is_bhsd);
            copy_strided_to_float(V_float, v_ptr, b, h_kv, seq_len_kv, head_size_vo,
                                  num_head_kv, is_bhsd);
            copy_strided_to_float(O_float, o_ptr, b, h_qo, seq_len_qo, head_size_vo,
                                  num_head_qo, is_bhsd);
            copy_strided_to_float(dO_float, do_ptr, b, h_qo, seq_len_qo, head_size_vo,
                                  num_head_qo, is_bhsd);

            std::vector<float> S(seq_len_qo * seq_len_kv, 0.0f);

            cutlass::TensorRef<float, LayoutRowMajor> ref_Q(Q_float.data(),
                                                            head_size_qk);
            cutlass::TensorRef<float, LayoutColMajor> ref_KT(K_float.data(),
                                                             head_size_qk);
            cutlass::TensorRef<float, LayoutRowMajor> ref_S(S.data(), seq_len_kv);

            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor, float, LayoutColMajor, float, LayoutRowMajor,
                ComputeType, ComputeType>(
                    {seq_len_qo, seq_len_kv, head_size_qk}, scale, ref_Q,
                    cutlass::ComplexTransform::kNone, ref_KT,
                    cutlass::ComplexTransform::kNone, 0.0f, ref_S, ref_S);

            apply_causal_mask(S, seq_len_qo, seq_len_kv, is_causal);

            std::vector<float> P;
            int lse_offset = get_lse_index(b, h_qo, 0, num_head_qo, seq_len_qo);
            compute_softmax_with_lse(P, S, lse_ptr + lse_offset, seq_len_qo,
                                     seq_len_kv);

            int odo_offset = get_lse_index(b, h_qo, 0, num_head_qo, seq_len_qo);
            compute_odo(odo_ptr + odo_offset, O_float, dO_float, seq_len_qo,
                        head_size_vo);

            std::vector<float> dV_float(seq_len_kv * head_size_vo, 0.0f);

            cutlass::TensorRef<float, LayoutColMajor> ref_PT(P.data(), seq_len_kv);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dO(dO_float.data(),
                                                             head_size_vo);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dV(dV_float.data(),
                                                             head_size_vo);

            cutlass::reference::host::GemmComplex<
                float, LayoutColMajor, float, LayoutRowMajor, float, LayoutRowMajor,
                ComputeType, ComputeType>(
                    {seq_len_kv, head_size_vo, seq_len_qo}, 1.0f, ref_PT,
                    cutlass::ComplexTransform::kNone, ref_dO,
                    cutlass::ComplexTransform::kNone, 0.0f, ref_dV, ref_dV);

            std::vector<float> dP(seq_len_qo * seq_len_kv, 0.0f);

            cutlass::TensorRef<float, LayoutColMajor> ref_VT(V_float.data(),
                                                             head_size_vo);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dP(dP.data(), seq_len_kv);

            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor, float, LayoutColMajor, float, LayoutRowMajor,
                ComputeType, ComputeType>(
                    {seq_len_qo, seq_len_kv, head_size_vo}, 1.0f, ref_dO,
                    cutlass::ComplexTransform::kNone, ref_VT,
                    cutlass::ComplexTransform::kNone, 0.0f, ref_dP, ref_dP);

            std::vector<float> dS;
            compute_softmax_backward(dS, P, dP, odo_ptr + odo_offset, seq_len_qo,
                                     seq_len_kv);

            std::vector<float> dQ_float(seq_len_qo * head_size_qk, 0.0f);

            cutlass::TensorRef<float, LayoutRowMajor> ref_dS(dS.data(), seq_len_kv);
            cutlass::TensorRef<float, LayoutRowMajor> ref_K(K_float.data(),
                                                            head_size_qk);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dQ(dQ_float.data(),
                                                             head_size_qk);

            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor, float, LayoutRowMajor, float, LayoutRowMajor,
                ComputeType, ComputeType>(
                    {seq_len_qo, head_size_qk, seq_len_kv}, scale, ref_dS,
                    cutlass::ComplexTransform::kNone, ref_K,
                    cutlass::ComplexTransform::kNone, 0.0f, ref_dQ, ref_dQ);

            std::vector<float> dK_float(seq_len_kv * head_size_qk, 0.0f);

            cutlass::TensorRef<float, LayoutColMajor> ref_dST(dS.data(), seq_len_kv);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dK(dK_float.data(),
                                                             head_size_qk);

            cutlass::reference::host::GemmComplex<
                float, LayoutColMajor, float, LayoutRowMajor, float, LayoutRowMajor,
                ComputeType, ComputeType>(
                    {seq_len_kv, head_size_qk, seq_len_qo}, scale, ref_dST,
                    cutlass::ComplexTransform::kNone, ref_Q,
                    cutlass::ComplexTransform::kNone, 0.0f, ref_dK, ref_dK);

            write_float_to_strided<T>(dq_ptr, dQ_float, b, h_qo, seq_len_qo,
                                      head_size_qk, num_head_qo, is_bhsd, false);

            write_float_to_strided<V>(dqaccum_ptr, dQ_float, b, h_qo, seq_len_qo,
                                      head_size_qk, num_head_qo, is_bhsd, false);

            write_float_to_strided<T>(dk_ptr, dK_float, b, h_kv, seq_len_kv,
                                      head_size_qk, num_head_kv, is_bhsd,
                                      !is_first_in_group);
            write_float_to_strided<T>(dv_ptr, dV_float, b, h_kv, seq_len_kv,
                                      head_size_vo, num_head_kv, is_bhsd,
                                      !is_first_in_group);
        }
    }
}
