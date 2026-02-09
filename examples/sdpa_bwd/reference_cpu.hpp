#pragma once

#include <cmath>
#include <vector>
#include <cstring>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/reference/host/gemm_complex.h"

// ========================================
// Helper: Compute o * do (row-wise dot product)
// ========================================

inline void compute_odo(
    float* odo_output,           // Output: [seq_len]
    const std::vector<float>& o, // [seq_len, head_size]
    const std::vector<float>& do_data, // [seq_len, head_size]
    int seq_len,
    int head_size
) {
    for (int i = 0; i < seq_len; ++i) {
        float sum = 0.0f;
        for (int d = 0; d < head_size; ++d) {
            sum += o[i * head_size + d] * do_data[i * head_size + d];
        }
        odo_output[i] = sum;
    }
}

// ========================================
// Helper: Apply causal mask to attention scores (bottom-right aligned)
// ========================================

inline void apply_causal_mask(
    std::vector<float>& S,  // [seq_len_q, seq_len_k], modified in-place
    int seq_len_q,
    int seq_len_k,
    bool is_causal
) {
    if (!is_causal) return;
    
    // Bottom-right aligned causal mask
    // The last query token can attend to all key tokens
    int offset = seq_len_k - seq_len_q;
    
    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < seq_len_k; ++j) {
            if (j > i + offset) {
                S[i * seq_len_k + j] = -INFINITY;
            }
        }
    }
}

// ========================================
// Helper: Compute P = softmax(S) using pre-computed LSE
// ========================================

inline void compute_softmax_with_lse(
    std::vector<float>& P,        // Output: [seq_len_q, seq_len_k]
    const std::vector<float>& S,  // Input: [seq_len_q, seq_len_k]
    const float* lse,             // Input: [seq_len_q]
    int seq_len_q,
    int seq_len_k
) {
    P.resize(seq_len_q * seq_len_k);
    
    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < seq_len_k; ++j) {
            int idx = i * seq_len_k + j;
            P[idx] = expf(S[idx] - lse[i]);
        }
    }
}

// ========================================
// Helper: Compute dS = P * (dP - oDo) [Softmax Backward]
// ========================================

inline void compute_softmax_backward(
    std::vector<float>& dS,         // Output: [seq_len_q, seq_len_k]
    const std::vector<float>& P,    // Input: [seq_len_q, seq_len_k] (softmax output)
    const std::vector<float>& dP,   // Input: [seq_len_q, seq_len_k] (gradient w.r.t. P)
    const float* oDo,               // Input: [seq_len_q] (row-wise dot product)
    int seq_len_q,
    int seq_len_k
) {
    dS.resize(seq_len_q * seq_len_k);
    
    for (int i = 0; i < seq_len_q; ++i) {
        float oDo_val = oDo[i];
        for (int j = 0; j < seq_len_k; ++j) {
            int idx = i * seq_len_k + j;
            // Softmax backward: dS = P * (dP - sum(P * dP))
            // Here sum(P * dP) is pre-computed as oDo for flash attention
            dS[idx] = P[idx] * (dP[idx] - oDo_val);
        }
    }
}

// ========================================
// Helper: Generic index calculation
// ========================================

inline int compute_strided_index(
    int b, int h, int s, int d,
    int num_head, int seq_len, int head_size,
    bool is_bhsd
) {
    if (is_bhsd) {
        // [batch, num_head, seq_len, head_size]
        return ((b * num_head + h) * seq_len + s) * head_size + d;
    } else {
        // [batch, seq_len, num_head, head_size]
        return ((b * seq_len + s) * num_head + h) * head_size + d;
    }
}

inline int get_lse_index(
    int b, int h, int s,
    int num_head, int seq_len
) {
    return (b * num_head + h) * seq_len + s;
}

inline int get_attention_matrix_index(
    int b, int h, int s_q, int s_k,
    int num_head_qo, int seq_len_qo, int seq_len_kv,
    bool is_bhsd
) {
    if (is_bhsd) {
        return ((b * num_head_qo + h) * seq_len_qo + s_q) * seq_len_kv + s_k;
    } else {
        return ((b * seq_len_qo + s_q) * num_head_qo + h) * seq_len_kv + s_k;
    }
}

// ========================================
// Helper: Generic copy strided tensor to contiguous float buffer
// ========================================

template<typename T>
inline void copy_strided_to_float(
    std::vector<float>& dst,
    const T* src,
    int b, int h, int seq_len, int head_size,
    int num_head, bool is_bhsd
) {
    dst.resize(seq_len * head_size);
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < head_size; ++d) {
            int src_idx = compute_strided_index(b, h, s, d, num_head, seq_len, head_size, is_bhsd);
            dst[s * head_size + d] = static_cast<float>(src[src_idx]);
        }
    }
}

// ========================================
// Helper: Generic write float buffer back to strided tensor
// ========================================

template<typename T>
inline void write_float_to_strided(
    T* dst,
    const std::vector<float>& src,
    int b, int h, int seq_len, int head_size,
    int num_head, bool is_bhsd,
    bool accumulate = false
) {
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < head_size; ++d) {
            int dst_idx = compute_strided_index(b, h, s, d, num_head, seq_len, head_size, is_bhsd);
            if (accumulate) {
                dst[dst_idx] = static_cast<T>(
                    static_cast<float>(dst[dst_idx]) + src[s * head_size + d]
                );
            } else {
                dst[dst_idx] = static_cast<T>(src[s * head_size + d]);
            }
        }
    }
}

// ========================================
// Helper: Write attention matrix (P or dS) for verification
// ========================================

template<typename T>
inline void write_attention_matrix(
    T* dst,
    const std::vector<float>& src,
    int b, int h, int seq_len_qo, int seq_len_kv,
    int num_head_qo, bool is_bhsd
) {
    if (dst == nullptr) return;
    for (int i = 0; i < seq_len_qo; ++i) {
        for (int j = 0; j < seq_len_kv; ++j) {
            int dst_idx = get_attention_matrix_index(b, h, i, j, num_head_qo, seq_len_qo, seq_len_kv, is_bhsd);
            dst[dst_idx] = static_cast<T>(src[i * seq_len_kv + j]);
        }
    }
}

// ========================================
// Main SDPA Backward Function
// ========================================

template<typename T, typename V>
void sdpa_backward_reference_cpu(
    // Input pointers
    T* q_ptr,              // [batch, num_head_qo, seq_len_qo, head_size_qk] or [batch, seq_len_qo, num_head_qo, head_size_qk]
    T* k_ptr,              // [batch, num_head_kv, seq_len_kv, head_size_qk] or [batch, seq_len_kv, num_head_kv, head_size_qk]
    T* v_ptr,              // [batch, num_head_kv, seq_len_kv, head_size_vo] or [batch, seq_len_kv, num_head_kv, head_size_vo]
    T* o_ptr,              // [batch, num_head_qo, seq_len_qo, head_size_vo] or [batch, seq_len_qo, num_head_qo, head_size_vo]
    T* do_ptr,             // [batch, num_head_qo, seq_len_qo, head_size_vo] or [batch, seq_len_qo, num_head_qo, head_size_vo]
    V* lse_ptr,        // [batch, num_head_qo, seq_len_qo]
    
    // Config
    bool is_causal,
    bool is_bhsd,
    
    // Dimensions
    int batch,
    int num_head_qo,
    int num_head_kv,
    int seq_len_qo,
    int seq_len_kv,
    int head_size_qk,
    int head_size_vo,
    
    // Output pointers
    V* odo_ptr,        // [batch, num_head_qo, seq_len_qo]
    V* dqaccum_ptr,        // [batch, num_head_qo, seq_len_qo, head_size_qk] or [batch, seq_len_qo, num_head_qo, head_size_qk]
    T* dq_ptr,             // [batch, num_head_qo, seq_len_qo, head_size_qk] or [batch, seq_len_qo, num_head_qo, head_size_qk]
    T* dk_ptr,             // [batch, num_head_kv, seq_len_kv, head_size_qk] or [batch, seq_len_kv, num_head_kv, head_size_qk]
    T* dv_ptr             // [batch, num_head_kv, seq_len_kv, head_size_vo] or [batch, seq_len_kv, num_head_kv, head_size_vo]
) {
    using LayoutRowMajor = cutlass::layout::RowMajor;
    using LayoutColMajor = cutlass::layout::ColumnMajor;
    using ComputeType = float;
    
    const float scale = 1.0f / sqrtf(static_cast<float>(head_size_qk));
    const int num_group = num_head_qo / num_head_kv;
    
    // ========================================
    // Process each batch and query head
    // ========================================
    
    for (int b = 0; b < batch; ++b) {
        for (int h_qo = 0; h_qo < num_head_qo; ++h_qo) {
            int h_kv = h_qo / num_group;
            bool is_first_in_group = (h_qo % num_group == 0);
            
            // ========================================
            // Step 0: Load data into float buffers
            // ========================================
            
            std::vector<float> Q_float, K_float, V_float, O_float, dO_float;
            
            copy_strided_to_float(Q_float, q_ptr, b, h_qo, seq_len_qo, head_size_qk, num_head_qo, is_bhsd);
            copy_strided_to_float(K_float, k_ptr, b, h_kv, seq_len_kv, head_size_qk, num_head_kv, is_bhsd);
            copy_strided_to_float(V_float, v_ptr, b, h_kv, seq_len_kv, head_size_vo, num_head_kv, is_bhsd);
            copy_strided_to_float(O_float, o_ptr, b, h_qo, seq_len_qo, head_size_vo, num_head_qo, is_bhsd);
            copy_strided_to_float(dO_float, do_ptr, b, h_qo, seq_len_qo, head_size_vo, num_head_qo, is_bhsd);
            
            // ========================================
            // Step 1: Compute S = Q @ K^T (scaled)
            // ========================================
            
            std::vector<float> S(seq_len_qo * seq_len_kv, 0.0f);
            
            cutlass::TensorRef<float, LayoutRowMajor> ref_Q(Q_float.data(), head_size_qk);
            cutlass::TensorRef<float, LayoutColMajor> ref_KT(K_float.data(), head_size_qk);
            cutlass::TensorRef<float, LayoutRowMajor> ref_S(S.data(), seq_len_kv);
            
            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor,
                float, LayoutColMajor,
                float, LayoutRowMajor,
                ComputeType, ComputeType
            >(
                {seq_len_qo, seq_len_kv, head_size_qk},
                scale, ref_Q, cutlass::ComplexTransform::kNone,
                ref_KT, cutlass::ComplexTransform::kNone,
                0.0f, ref_S, ref_S
            );
            
            // ========================================
            // Step 2: Apply causal mask to S
            // ========================================
            
            apply_causal_mask(S, seq_len_qo, seq_len_kv, is_causal);
            
            // ========================================
            // Step 3: Compute P = softmax(S) using LSE
            // ========================================
            
            std::vector<float> P;
            int lse_offset = get_lse_index(b, h_qo, 0, num_head_qo, seq_len_qo);
            compute_softmax_with_lse(P, S, lse_ptr + lse_offset, seq_len_qo, seq_len_kv);
            
            // ========================================
            // Step 4: Compute oDo = sum(O * dO, dim=-1)
            // ========================================
            
            int odo_offset = get_lse_index(b, h_qo, 0, num_head_qo, seq_len_qo);
            compute_odo(odo_ptr + odo_offset, O_float, dO_float, seq_len_qo, head_size_vo);
            
            // ========================================
            // Step 5: Compute dV = P^T @ dO
            // ========================================
            
            std::vector<float> dV_float(seq_len_kv * head_size_vo, 0.0f);
            
            cutlass::TensorRef<float, LayoutColMajor> ref_PT(P.data(), seq_len_kv);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dO(dO_float.data(), head_size_vo);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dV(dV_float.data(), head_size_vo);
            
            cutlass::reference::host::GemmComplex<
                float, LayoutColMajor,
                float, LayoutRowMajor,
                float, LayoutRowMajor,
                ComputeType, ComputeType
            >(
                {seq_len_kv, head_size_vo, seq_len_qo},
                1.0f, ref_PT, cutlass::ComplexTransform::kNone,
                ref_dO, cutlass::ComplexTransform::kNone,
                0.0f, ref_dV, ref_dV
            );
            
            // ========================================
            // Step 6: Compute dP = dO @ V^T
            // ========================================
            
            std::vector<float> dP(seq_len_qo * seq_len_kv, 0.0f);
            
            cutlass::TensorRef<float, LayoutColMajor> ref_VT(V_float.data(), head_size_vo);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dP(dP.data(), seq_len_kv);
            
            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor,
                float, LayoutColMajor,
                float, LayoutRowMajor,
                ComputeType, ComputeType
            >(
                {seq_len_qo, seq_len_kv, head_size_vo},
                1.0f, ref_dO, cutlass::ComplexTransform::kNone,
                ref_VT, cutlass::ComplexTransform::kNone,
                0.0f, ref_dP, ref_dP
            );
            
            // ========================================
            // Step 7: Compute dS = P * (dP - oDo) [Softmax Backward]
            // ========================================
            
            std::vector<float> dS;
            compute_softmax_backward(dS, P, dP, odo_ptr + odo_offset, seq_len_qo, seq_len_kv);
            
            // ========================================
            // Step 8: Compute dQ = (scale * dS) @ K
            // ========================================
            
            std::vector<float> dQ_float(seq_len_qo * head_size_qk, 0.0f);
            
            cutlass::TensorRef<float, LayoutRowMajor> ref_dS(dS.data(), seq_len_kv);
            cutlass::TensorRef<float, LayoutRowMajor> ref_K(K_float.data(), head_size_qk);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dQ(dQ_float.data(), head_size_qk);
            
            cutlass::reference::host::GemmComplex<
                float, LayoutRowMajor,
                float, LayoutRowMajor,
                float, LayoutRowMajor,
                ComputeType, ComputeType
            >(
                {seq_len_qo, head_size_qk, seq_len_kv},
                scale, ref_dS, cutlass::ComplexTransform::kNone,
                ref_K, cutlass::ComplexTransform::kNone,
                0.0f, ref_dQ, ref_dQ
            );
            
            // ========================================
            // Step 9: Compute dK = (scale * dS)^T @ Q
            // ========================================
            
            std::vector<float> dK_float(seq_len_kv * head_size_qk, 0.0f);
            
            cutlass::TensorRef<float, LayoutColMajor> ref_dST(dS.data(), seq_len_kv);
            cutlass::TensorRef<float, LayoutRowMajor> ref_dK(dK_float.data(), head_size_qk);
            
            cutlass::reference::host::GemmComplex<
                float, LayoutColMajor,
                float, LayoutRowMajor,
                float, LayoutRowMajor,
                ComputeType, ComputeType
            >(
                {seq_len_kv, head_size_qk, seq_len_qo},
                scale, ref_dST, cutlass::ComplexTransform::kNone,
                ref_Q, cutlass::ComplexTransform::kNone,
                0.0f, ref_dK, ref_dK
            );
            
            // ========================================
            // Step 10: Write results back
            // ========================================
            
            // Write dQ with T type (fp16/bf16)
            write_float_to_strided<T>(dq_ptr, dQ_float, b, h_qo, seq_len_qo, head_size_qk, num_head_qo, is_bhsd, false);
            
            // Write dQ_accum with V type (float for accumulator)
            write_float_to_strided<V>(dqaccum_ptr, dQ_float, b, h_qo, seq_len_qo, head_size_qk, num_head_qo, is_bhsd, false);
            
            // Write dK and dV with T type (e.g., fp16/bf16), with accumulation for GQA
            write_float_to_strided<T>(dk_ptr, dK_float, b, h_kv, seq_len_kv, head_size_qk, num_head_kv, is_bhsd, !is_first_in_group);
            write_float_to_strided<T>(dv_ptr, dV_float, b, h_kv, seq_len_kv, head_size_vo, num_head_kv, is_bhsd, !is_first_in_group);
        }
    }
}
