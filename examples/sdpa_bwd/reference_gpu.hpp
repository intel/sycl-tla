#pragma once

#include <cmath>
#include <vector>
#include <cstring>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

// SYCL memory management
#include "cute/util/compat.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"

// ========================================
// Helper: Generic index calculation (non-template version for GPU)
// ========================================

inline int compute_strided_index_gpu(
    int b, int h, int s, int d,
    int num_head, int seq_len, int head_size,
    bool is_bhsd
) {
    if (is_bhsd) {
        return ((b * num_head + h) * seq_len + s) * head_size + d;
    } else {
        return ((b * seq_len + s) * num_head + h) * head_size + d;
    }
}

// ========================================
// GPU Kernel: Compute o * do (row-wise dot product)
// ========================================

template<typename T, typename V>
void compute_odo_kernel(
    V* odo_output,
    const T* o,
    const T* do_data,
    int seq_len,
    int head_size,
    sycl::nd_item<1> item
) {
    int i = item.get_global_id(0);
    if (i < seq_len) {
        V sum = static_cast<V>(0.0f);
        for (int d = 0; d < head_size; ++d) {
            sum += static_cast<V>(o[i * head_size + d]) *
                   static_cast<V>(do_data[i * head_size + d]);
        }
        odo_output[i] = sum;
    }
}

// ========================================
// GPU Kernel: Apply causal mask
// ========================================

template<typename T, typename V>
void apply_causal_mask_kernel(
    V* S,
    int seq_len_q,
    int seq_len_k,
    sycl::nd_item<2> item
) {
    int i = item.get_global_id(0);
    int j = item.get_global_id(1);

    if (i < seq_len_q && j < seq_len_k) {
        int offset = seq_len_k - seq_len_q;
        if (j > i + offset) {
            S[i * seq_len_k + j] = static_cast<V>(-INFINITY);
        }
    }
}

// ========================================
// GPU Kernel: Compute LSE and P = softmax(S)
// ========================================

template<typename T, typename V>
void compute_lse_and_softmax_kernel(
    V* P,
    V* lse,
    const V* S,
    int seq_len_q,
    int seq_len_k,
    sycl::nd_item<1> item
) {
    int i = item.get_global_id(0);
    if (i < seq_len_q) {
        // Find max value in row i for numerical stability
        V max_val = -INFINITY;
        for (int j = 0; j < seq_len_k; ++j) {
            V s_val = S[i * seq_len_k + j];
            if (s_val > max_val) {
                max_val = s_val;
            }
        }

        // Compute sum of exp(S[i,j] - max_val)
        V sum_exp = 0.0f;
        for (int j = 0; j < seq_len_k; ++j) {
            V s_val = S[i * seq_len_k + j];
            sum_exp += sycl::exp(s_val - max_val);
        }

        // Compute lse[i] = max_val + log(sum_exp)
        lse[i] = max_val + sycl::log(sum_exp);

        // Compute P[i,j] = exp(S[i,j] - lse[i])
        for (int j = 0; j < seq_len_k; ++j) {
            V s_val = S[i * seq_len_k + j];
            P[i * seq_len_k + j] = sycl::exp(s_val - lse[i]);
        }
    }
}

// ========================================
// GPU Kernel: Compute P = softmax(S) using LSE
// ========================================

template<typename T, typename V>
void compute_softmax_with_lse_kernel(
    V* P,
    const V* S,
    const V* lse,
    int seq_len_q,
    int seq_len_k,
    sycl::nd_item<2> item
) {
    int i = item.get_global_id(0);
    int j = item.get_global_id(1);

    if (i < seq_len_q && j < seq_len_k) {
        int idx = i * seq_len_k + j;
        V s_val = S[idx];
        V p_val = sycl::exp(s_val - lse[i]);
        P[idx] = p_val;
    }
}

// ========================================
// GPU Kernel: Compute dS = P * (dP - oDo) [Softmax Backward]
// ========================================

template<typename T, typename V>
void compute_softmax_backward_kernel(
    V* dS,
    const V* P,
    const V* dP,
    const V* oDo,
    int seq_len_q,
    int seq_len_k,
    sycl::nd_item<2> item
) {
    int i = item.get_global_id(0);
    int j = item.get_global_id(1);

    if (i < seq_len_q && j < seq_len_k) {
        int idx = i * seq_len_k + j;
        V p_val = P[idx];
        V dp_val = dP[idx];
        V odo_val = oDo[i];
        V ds_val = p_val * (dp_val - odo_val);
        dS[idx] = ds_val;
    }
}

// ========================================
// GPU Kernel: Convert V to T
// ========================================

template<typename T, typename V>
void convert_V_to_T_kernel(
    T* dst,
    const V* src,
    int size,
    sycl::nd_item<1> item
) {
    int i = item.get_global_id(0);
    if (i < size) {
        dst[i] = static_cast<T>(src[i]);
    }
}

// ========================================
// Helper: Convert V to T
// ========================================

template<typename T, typename V>
void convert_V_to_T(
    sycl::queue& q,
    T* dst,
    const V* src,
    int size
) {
    sycl::range<1> grid((size + 255) / 256);
    sycl::range<1> block(256);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(grid * block, block),
            [=](sycl::nd_item<1> item) {
                convert_V_to_T_kernel<T, V>(dst, src, size, item);
            }
        );
    }).wait();
}

// ========================================
// Helper: Convert T to V
// ========================================

template<typename T, typename V>
void convert_T_to_V(
    sycl::queue& q,
    V* dst,
    const T* src,
    int size
) {
    sycl::range<1> grid((size + 255) / 256);
    sycl::range<1> block(256);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(grid * block, block),
            [=](sycl::nd_item<1> item) {
                int i = item.get_global_id(0);
                if (i < size) {
                    dst[i] = static_cast<V>(src[i]);
                }
            }
        );
    }).wait();
}

// ========================================
// Helper: Device GEMM wrapper
// ========================================

template<typename T, typename V>
void device_gemm(
    int M, int N, int K,
    V alpha,
    const T* A, int lda, bool trans_A,
    const T* B, int ldb, bool trans_B,
    V beta,
    V* C, int ldc
) {
    using LayoutRowMajor = cutlass::layout::RowMajor;
    using LayoutColMajor = cutlass::layout::ColumnMajor;

    if (!trans_A && !trans_B) {
        // C = A @ B
        cutlass::TensorRef<T, LayoutRowMajor> ref_A(const_cast<T*>(A), lda);
        cutlass::TensorRef<T, LayoutRowMajor> ref_B(const_cast<T*>(B), ldb);
        cutlass::TensorRef<V, LayoutRowMajor> ref_C(C, ldc);

        cutlass::reference::device::GemmComplex<
            T, LayoutRowMajor,
            T, LayoutRowMajor,
            V, LayoutRowMajor,
            float, float
        >(
            {M, N, K},
            alpha, ref_A, cutlass::ComplexTransform::kNone,
            ref_B, cutlass::ComplexTransform::kNone,
            beta, ref_C, ref_C,
            0.0f, 1, 0, 0, 0, 0
        );
    } else if (!trans_A && trans_B) {
        // C = A @ B^T
        cutlass::TensorRef<T, LayoutRowMajor> ref_A(const_cast<T*>(A), lda);
        cutlass::TensorRef<T, LayoutColMajor> ref_BT(const_cast<T*>(B), ldb);
        cutlass::TensorRef<V, LayoutRowMajor> ref_C(C, ldc);

        cutlass::reference::device::GemmComplex<
            T, LayoutRowMajor,
            T, LayoutColMajor,
            V, LayoutRowMajor,
            float, float
        >(
            {M, N, K},
            alpha, ref_A, cutlass::ComplexTransform::kNone,
            ref_BT, cutlass::ComplexTransform::kNone,
            beta, ref_C, ref_C,
            0.0f, 1, 0, 0, 0, 0
        );
    } else if (trans_A && !trans_B) {
        // C = A^T @ B
        cutlass::TensorRef<T, LayoutColMajor> ref_AT(const_cast<T*>(A), lda);
        cutlass::TensorRef<T, LayoutRowMajor> ref_B(const_cast<T*>(B), ldb);
        cutlass::TensorRef<V, LayoutRowMajor> ref_C(C, ldc);

        cutlass::reference::device::GemmComplex<
            T, LayoutColMajor,
            T, LayoutRowMajor,
            V, LayoutRowMajor,
            float, float
        >(
            {M, N, K},
            alpha, ref_AT, cutlass::ComplexTransform::kNone,
            ref_B, cutlass::ComplexTransform::kNone,
            beta, ref_C, ref_C,
            0.0f, 1, 0, 0, 0, 0
        );
    }

    compat::wait_and_throw();
}

// ========================================
// Helper: Copy strided tensor to contiguous device buffer
// ========================================

template<typename T>
void copy_strided_to_device(
    T* d_dst,
    const T* h_src,
    int b, int h, int seq_len, int head_size,
    int num_head, bool is_bhsd
) {
    std::vector<T> temp(seq_len * head_size);
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < head_size; ++d) {
            int src_idx = compute_strided_index_gpu(b, h, s, d, num_head, seq_len, head_size, is_bhsd);
            temp[s * head_size + d] = h_src[src_idx];
        }
    }
    // Use compat::memcpy<T>(dest, src, count)
    compat::memcpy<T>(d_dst, temp.data(), seq_len * head_size);
}

template<typename T>
void write_device_to_strided(
    T* h_dst,
    const T* d_src,
    int b, int h, int seq_len, int head_size,
    int num_head, bool is_bhsd,
    bool accumulate
) {
    std::vector<T> temp(seq_len * head_size);
    // Use compat::memcpy<T>(dest, src, count)
    compat::memcpy<T>(temp.data(), d_src, seq_len * head_size);
    compat::wait_and_throw();

    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < head_size; ++d) {
            int dst_idx = compute_strided_index_gpu(b, h, s, d, num_head, seq_len, head_size, is_bhsd);
            if (accumulate) {
                h_dst[dst_idx] = static_cast<T>(
                    static_cast<float>(h_dst[dst_idx]) + static_cast<float>(temp[s * head_size + d])
                );
            } else {
                h_dst[dst_idx] = temp[s * head_size + d];
            }
        }
    }
}

// ========================================
// Main SDPA Backward GPU Function
// ========================================

template<typename T, typename V>
void sdpa_backward_reference_gpu(
    // Input pointers (host)
    T* q_ptr,
    T* k_ptr,
    T* v_ptr,
    T* o_ptr,
    T* do_ptr,
    V* lse_ptr,

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

    // Output pointers (host)
    V* odo_ptr,
    V* dqaccum_ptr,
    T* dq_ptr,
    T* dk_ptr,
    T* dv_ptr
) {
    // Get default queue
    auto q = compat::get_default_queue();

    const float scale = 1.0f / sqrtf(static_cast<float>(head_size_qk));
    const int num_group = num_head_qo / num_head_kv;

    // ========================================
    // Allocate device memory once (reused for all batches and heads)
    // ========================================

    T* d_Q = compat::malloc<T>(seq_len_qo * head_size_qk);
    T* d_K = compat::malloc<T>(seq_len_kv * head_size_qk);
    T* d_V = compat::malloc<T>(seq_len_kv * head_size_vo);
    T* d_O = compat::malloc<T>(seq_len_qo * head_size_vo);
    T* d_dO = compat::malloc<T>(seq_len_qo * head_size_vo);

    V* d_S = compat::malloc<V>(seq_len_qo * seq_len_kv);
    V* d_P = compat::malloc<V>(seq_len_qo * seq_len_kv);
    T* d_P_T = compat::malloc<T>(seq_len_qo * seq_len_kv);  // T version for GEMM
    V* d_dS = compat::malloc<V>(seq_len_qo * seq_len_kv);
    T* d_dS_T = compat::malloc<T>(seq_len_qo * seq_len_kv);  // T version for GEMM

    T* d_dV = compat::malloc<T>(seq_len_kv * head_size_vo);
    T* d_dQ = compat::malloc<T>(seq_len_qo * head_size_qk);
    T* d_dK = compat::malloc<T>(seq_len_kv * head_size_qk);

    V* d_oDo = compat::malloc<V>(seq_len_qo);
    V* d_lse = compat::malloc<V>(seq_len_qo);

    // Allocate V buffers for GEMM outputs
    V* d_dV_V = compat::malloc<V>(seq_len_kv * head_size_vo);
    V* d_dP_V = compat::malloc<V>(seq_len_qo * seq_len_kv);
    V* d_dQ_V = compat::malloc<V>(seq_len_qo * head_size_qk);
    V* d_dK_V = compat::malloc<V>(seq_len_kv * head_size_qk);

    for (int b = 0; b < batch; ++b) {
        for (int h_qo = 0; h_qo < num_head_qo; ++h_qo) {
            int h_kv = h_qo / num_group;

            // ========================================
            // Copy input data from host to device
            // ========================================

            copy_strided_to_device(d_Q, q_ptr, b, h_qo, seq_len_qo, head_size_qk, num_head_qo, is_bhsd);
            copy_strided_to_device(d_K, k_ptr, b, h_kv, seq_len_kv, head_size_qk, num_head_kv, is_bhsd);
            copy_strided_to_device(d_V, v_ptr, b, h_kv, seq_len_kv, head_size_vo, num_head_kv, is_bhsd);
            copy_strided_to_device(d_O, o_ptr, b, h_qo, seq_len_qo, head_size_vo, num_head_qo, is_bhsd);
            copy_strided_to_device(d_dO, do_ptr, b, h_qo, seq_len_qo, head_size_vo, num_head_qo, is_bhsd);

            // Copy LSE
            int lse_offset = (b * num_head_qo + h_qo) * seq_len_qo;
            compat::memcpy<V>(d_lse, lse_ptr + lse_offset, seq_len_qo);

            compat::wait_and_throw();

            // ========================================
            // Step 1: Compute S = Q @ K^T (scaled)
            // ========================================

            device_gemm<T, V>(seq_len_qo, seq_len_kv, head_size_qk,
                              scale, d_Q, head_size_qk, false,
                              d_K, head_size_qk, true,
                              0.0f, d_S, seq_len_kv);

            // ========================================
            // Step 2: Apply causal mask
            // ========================================

            if (is_causal) {
                sycl::range<2> grid((seq_len_qo + 15) / 16, (seq_len_kv + 15) / 16);
                sycl::range<2> block(16, 16);

                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<2>(grid * block, block),
                        [=](sycl::nd_item<2> item) {
                            apply_causal_mask_kernel<T, V>(d_S, seq_len_qo, seq_len_kv, item);
                        }
                    );
                }).wait();
            }

            // ========================================
            // Step 3: Compute P = softmax(S) using LSE
            // ========================================

            {
                sycl::range<2> grid((seq_len_qo + 15) / 16, (seq_len_kv + 15) / 16);
                sycl::range<2> block(16, 16);

                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<2>(grid * block, block),
                        [=](sycl::nd_item<2> item) {
                            compute_softmax_with_lse_kernel<T, V>(d_P, d_S, d_lse, seq_len_qo, seq_len_kv, item);
                        }
                    );
                }).wait();
            }

            // Convert P from V to T for GEMM
            convert_V_to_T<T, V>(q, d_P_T, d_P, seq_len_qo * seq_len_kv);

            // ========================================
            // Step 4: Compute oDo = sum(O * dO, dim=-1)
            // ========================================

            {
                sycl::range<1> grid((seq_len_qo + 255) / 256);
                sycl::range<1> block(256);

                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<1>(grid * block, block),
                        [=](sycl::nd_item<1> item) {
                            compute_odo_kernel<T, V>(d_oDo, d_O, d_dO, seq_len_qo, head_size_vo, item);
                        }
                    );
                }).wait();
            }

            // ========================================
            // Step 5: Compute dV = P^T @ dO
            // ========================================

            device_gemm<T, V>(seq_len_kv, head_size_vo, seq_len_qo,
                              1.0f, d_P_T, seq_len_kv, true,
                              d_dO, head_size_vo, false,
                              0.0f, d_dV_V, head_size_vo);

            // Convert V to T
            convert_V_to_T<T, V>(q, d_dV, d_dV_V, seq_len_kv * head_size_vo);

            // ========================================
            // Step 6: Compute dP = dO @ V^T
            // ========================================

            device_gemm<T, V>(seq_len_qo, seq_len_kv, head_size_vo,
                              1.0f, d_dO, head_size_vo, false,
                              d_V, head_size_vo, true,
                              0.0f, d_dP_V, seq_len_kv);

            // ========================================
            // Step 7: Compute dS = P * (dP - oDo)
            // ========================================

            {
                sycl::range<2> grid((seq_len_qo + 15) / 16, (seq_len_kv + 15) / 16);
                sycl::range<2> block(16, 16);

                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<2>(grid * block, block),
                        [=](sycl::nd_item<2> item) {
                            compute_softmax_backward_kernel<T, V>(d_dS, d_P, d_dP_V, d_oDo, seq_len_qo, seq_len_kv, item);
                        }
                    );
                }).wait();
            }

            // Convert dS from V to T for GEMM
            convert_V_to_T<T, V>(q, d_dS_T, d_dS, seq_len_qo * seq_len_kv);

            // ========================================
            // Step 8: Compute dQ = (scale * dS) @ K
            // ========================================

            device_gemm<T, V>(seq_len_qo, head_size_qk, seq_len_kv,
                              scale, d_dS_T, seq_len_kv, false,
                              d_K, head_size_qk, false,
                              0.0f, d_dQ_V, head_size_qk);

            // Convert V to T
            convert_V_to_T<T, V>(q, d_dQ, d_dQ_V, seq_len_qo * head_size_qk);

            // ========================================
            // Step 9: Compute dK = (scale * dS)^T @ Q
            // ========================================

            device_gemm<T, V>(seq_len_kv, head_size_qk, seq_len_qo,
                              scale, d_dS_T, seq_len_kv, true,
                              d_Q, head_size_qk, false,
                              0.0f, d_dK_V, head_size_qk);

            // Convert V to T
            convert_V_to_T<T, V>(q, d_dK, d_dK_V, seq_len_kv * head_size_qk);

            // ========================================
            // Step 10: Copy results back to host
            // ========================================

            write_device_to_strided(dq_ptr, d_dQ, b, h_qo, seq_len_qo, head_size_qk, num_head_qo, is_bhsd, false);
            write_device_to_strided(dk_ptr, d_dK, b, h_qo, seq_len_kv, head_size_qk, num_head_qo, is_bhsd, false);
            write_device_to_strided(dv_ptr, d_dV, b, h_qo, seq_len_kv, head_size_vo, num_head_qo, is_bhsd, false);

            // Copy oDo back
            int odo_offset = (b * num_head_qo + h_qo) * seq_len_qo;
            compat::memcpy<V>(odo_ptr + odo_offset, d_oDo, seq_len_qo);

            // For dqaccum, copy from d_dQ to float accumulator
            if (dqaccum_ptr) {
                std::vector<T> temp_dq(seq_len_qo * head_size_qk);
                compat::memcpy<T>(temp_dq.data(), d_dQ, seq_len_qo * head_size_qk);
                compat::wait_and_throw();

                for (int s = 0; s < seq_len_qo; ++s) {
                    for (int d = 0; d < head_size_qk; ++d) {
                        int dst_idx = compute_strided_index_gpu(b, h_qo, s, d, num_head_qo, seq_len_qo, head_size_qk, is_bhsd);
                        dqaccum_ptr[dst_idx] = static_cast<V>(temp_dq[s * head_size_qk + d]);
                    }
                }
            }
        }
    }

    // ========================================
    // Free device memory
    // ========================================

    compat::free(d_Q);
    compat::free(d_K);
    compat::free(d_V);
    compat::free(d_O);
    compat::free(d_dO);
    compat::free(d_S);
    compat::free(d_P);
    compat::free(d_P_T);
    compat::free(d_dS);
    compat::free(d_dS_T);
    compat::free(d_dV);
    compat::free(d_dQ);
    compat::free(d_dK);
    compat::free(d_oDo);
    compat::free(d_lse);
    compat::free(d_dV_V);
    compat::free(d_dP_V);
    compat::free(d_dQ_V);
    compat::free(d_dK_V);
}

// ========================================
// SDPA Forward GPU Function (compute O from Q, K, V)
// ========================================

template<typename T, typename V>
void sdpa_forward_reference_gpu(
    // Input pointers (host)
    T* q_ptr,
    T* k_ptr,
    T* v_ptr,

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

    // Output pointers (host)
    T* o_ptr,
    V* lse_ptr
) {
    // Get default queue
    auto q = compat::get_default_queue();

    const float scale = 1.0f / sqrtf(static_cast<float>(head_size_qk));
    const int num_group = num_head_qo / num_head_kv;

    // ========================================
    // Allocate device memory once (reused for all batches and heads)
    // ========================================

    T* d_Q = compat::malloc<T>(seq_len_qo * head_size_qk);
    T* d_K = compat::malloc<T>(seq_len_kv * head_size_qk);
    T* d_V = compat::malloc<T>(seq_len_kv * head_size_vo);
    T* d_O = compat::malloc<T>(seq_len_qo * head_size_vo);

    V* d_S = compat::malloc<V>(seq_len_qo * seq_len_kv);
    V* d_P = compat::malloc<V>(seq_len_qo * seq_len_kv);
    T* d_P_T = compat::malloc<T>(seq_len_qo * seq_len_kv);  // T version for GEMM
    V* d_lse = compat::malloc<V>(seq_len_qo);
    V* d_O_V = compat::malloc<V>(seq_len_qo * head_size_vo);  // V version for GEMM output

    for (int b = 0; b < batch; ++b) {
        for (int h_qo = 0; h_qo < num_head_qo; ++h_qo) {
            int h_kv = h_qo / num_group;

            // ========================================
            // Copy input data from host to device
            // ========================================

            copy_strided_to_device(d_Q, q_ptr, b, h_qo, seq_len_qo, head_size_qk, num_head_qo, is_bhsd);
            copy_strided_to_device(d_K, k_ptr, b, h_kv, seq_len_kv, head_size_qk, num_head_kv, is_bhsd);
            copy_strided_to_device(d_V, v_ptr, b, h_kv, seq_len_kv, head_size_vo, num_head_kv, is_bhsd);

            compat::wait_and_throw();

            // ========================================
            // Step 1: Compute S = Q @ K^T (scaled)
            // ========================================

            device_gemm<T, V>(seq_len_qo, seq_len_kv, head_size_qk,
                              scale, d_Q, head_size_qk, false,
                              d_K, head_size_qk, true,
                              0.0f, d_S, seq_len_kv);

            // ========================================
            // Step 2: Apply causal mask
            // ========================================

            if (is_causal) {
                sycl::range<2> grid((seq_len_qo + 15) / 16, (seq_len_kv + 15) / 16);
                sycl::range<2> block(16, 16);

                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<2>(grid * block, block),
                        [=](sycl::nd_item<2> item) {
                            apply_causal_mask_kernel<T, V>(d_S, seq_len_qo, seq_len_kv, item);
                        }
                    );
                }).wait();
            }

            // ========================================
            // Step 3: Compute LSE and P = softmax(S)
            // ========================================

            {
                sycl::range<1> grid((seq_len_qo + 255) / 256);
                sycl::range<1> block(256);

                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(
                        sycl::nd_range<1>(grid * block, block),
                        [=](sycl::nd_item<1> item) {
                            compute_lse_and_softmax_kernel<T, V>(d_P, d_lse, d_S, seq_len_qo, seq_len_kv, item);
                        }
                    );
                }).wait();
            }

            // Convert P from V to T for GEMM (P*V)
            convert_V_to_T<T, V>(q, d_P_T, d_P, seq_len_qo * seq_len_kv);

            // ========================================
            // Step 4: Compute O = P @ V (output is V type)
            // ========================================

            device_gemm<T, V>(seq_len_qo, head_size_vo, seq_len_kv,
                              1.0f, d_P_T, seq_len_kv, false,
                              d_V, head_size_vo, false,
                              0.0f, d_O_V, head_size_vo);

            // Convert O from V to T
            convert_V_to_T<T, V>(q, d_O, d_O_V, seq_len_qo * head_size_vo);

            // ========================================
            // Copy results back to host
            // ========================================

            write_device_to_strided(o_ptr, d_O, b, h_qo, seq_len_qo, head_size_vo, num_head_qo, is_bhsd, false);

            // Copy LSE back
            int lse_offset = (b * num_head_qo + h_qo) * seq_len_qo;
            compat::memcpy<V>(lse_ptr + lse_offset, d_lse, seq_len_qo);
        }
    }

    // ========================================
    // Free device memory
    // ========================================

    compat::free(d_Q);
    compat::free(d_K);
    compat::free(d_V);
    compat::free(d_O);
    compat::free(d_S);
    compat::free(d_P);
    compat::free(d_P_T);
    compat::free(d_lse);
    compat::free(d_O_V);
}
