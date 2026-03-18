/* Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/tensor_sg.hpp>
#include <cute/util/sycl_vec.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <cmath>
#include <chrono>

#include "cutlass_unit_test.h"
#include "utils.hpp"

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

// ============================================================================
// Constants (preserve hyperparameters as requested)
// ============================================================================
constexpr int kSubgroupSize = 16;
constexpr int kNSGs = 8;
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kNumThreads = kSubgroupSize * kNSGs;

// ============================================================================
// Helper: convert_layout_2d_layout (from sdpa_backward.hpp)
// ============================================================================
template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout),
                                     get<1>(layout)),
                         get<2>(layout));
    return l;
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
void scale_apply_exp2_cpu_reference(
    float* tensor,           // [M, N] in row-major
    const float* max_vec,    // [N]
    int M, int N,
    float scale,
    bool is_even_M = true,
    int tail_m = 0)
{
    for (int n = 0; n < N; ++n) {
        float max_scaled = (max_vec[n] == -INFINITY) ? 0.f : max_vec[n] * M_LOG2E;
        
        int M_limit = is_even_M ? M : tail_m;
        for (int m = 0; m < M_limit; ++m) {
            int idx = m * N + n;
            tensor[idx] = exp2f(tensor[idx] * scale - max_scaled);
        }
    }
}

// ============================================================================
// GPU Kernel using CuTe API
// ============================================================================
template <bool Is_even_M, int M, int N>
class ScaleApplyExp2Kernel;

template <bool Is_even_M, int M, int N>
void scale_apply_exp2_kernel(
    float* d_tensor,        // device memory [M, N]
    const float* d_max,     // device memory [N]
    float scale,
    int tail_m)
{
    // Get thread info
    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_id();
    
    // Each subgroup processes a portion of the N dimension
    // With kNSGs=8 subgroups, each processes N/8 columns (for N >= kNSGs)
    // For N < kNSGs, distribute N columns among first N subgroups
    constexpr int N_per_sg = (N >= kNSGs) ? (N / kNSGs) : 1;
    
    // Early exit if this subgroup has no work (when N < kNSGs)
    if constexpr(N < kNSGs) {
        if (sg_id >= N) return;
    }
    
    int n_start = (N >= kNSGs) ? (sg_id * N_per_sg) : sg_id;
    
    // Process M dimension: each thread in subgroup processes different rows
    // With subgroup size 16, stride through M dimension
    constexpr int M_stride = kSubgroupSize;
    
    for (int ni = 0; ni < N_per_sg; ++ni) {
        int n = n_start + ni;
        
        // Load max value for this column
        float max_val = d_max[n];
        float max_scaled = (max_val == -INFINITY) ? 0.f : max_val * M_LOG2E;
        
        // Each thread processes rows based on lane_id with stride
        for (int m = lane_id; m < M; m += M_stride) {
            if constexpr(Is_even_M) {
                int idx = m * N + n;
                float val = d_tensor[idx];
                val = exp2f(val * scale - max_scaled);
                d_tensor[idx] = val;
            } else {
                if (m < tail_m) {
                    int idx = m * N + n;
                    float val = d_tensor[idx];
                    val = exp2f(val * scale - max_scaled);
                    d_tensor[idx] = val;
                }
            }
        }
    }
}

// ============================================================================
// Test Function
// ============================================================================
template <bool Is_even_M, int M, int N>
void test_scale_apply_exp2(int iterations = 100) {
    static_assert(M == kBlockM, "M must equal kBlockM");
    static_assert(N == kBlockN, "N must equal kBlockN");
    static_assert(M % kSubgroupSize == 0, "M must be divisible by subgroup size");
    static_assert(N % kNSGs == 0, "N must be divisible by number of subgroups");
    
    const float scale = 1.44269504f;  // log2(e), typical softmax scale
    const int tail_m = Is_even_M ? M : (M - 8);  // simulate tail case
    
    // Allocate and initialize host memory
    cutlass::host_vector<float> h_tensor(M * N);
    cutlass::host_vector<float> h_max(N);
    cutlass::host_vector<float> h_tensor_ref(M * N);
    cutlass::host_vector<float> h_tensor_out(M * N);
    
    // Fill with random values
    std::random_device rd;
    std::mt19937 gen(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> max_dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * N; ++i) {
        h_tensor[i] = dist(gen);
        h_tensor_ref[i] = h_tensor[i];
    }
    
    for (int i = 0; i < N; ++i) {
        h_max[i] = max_dist(gen);
    }
    
    // Compute CPU reference
    scale_apply_exp2_cpu_reference(h_tensor_ref.data(), h_max.data(), M, N, scale, Is_even_M, tail_m);
    
    // Allocate device memory
    cutlass::device_vector<float> d_tensor = h_tensor;
    cutlass::device_vector<float> d_max = h_max;
    
    // Launch kernel
    launch<scale_apply_exp2_kernel<Is_even_M, M, N>, ScaleApplyExp2Kernel<Is_even_M, M, N>>(
        launch_policy{
            compat::dim3(1), 
            compat::dim3(kNumThreads),
            kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
        },
        d_tensor.data(),
        d_max.data(),
        scale,
        tail_m
    );
    
    // Copy result back
    h_tensor_out = d_tensor;
    
    // ========================================================================
    // 1. Accuracy Test: Compare all values
    // ========================================================================
    int error_count = 0;
    float max_rel_error = 0.0f;
    float max_abs_error = 0.0f;
    
    int check_limit = Is_even_M ? M : tail_m;
    
    for (int m = 0; m < check_limit; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * N + n;
            float ref_val = h_tensor_ref[idx];
            float gpu_val = h_tensor_out[idx];
            
            float abs_error = std::abs(ref_val - gpu_val);
            float rel_error = std::abs(ref_val) > 1e-6f ? abs_error / std::abs(ref_val) : abs_error;
            
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            
            // Check tolerance (adjust based on your requirements)
            if (rel_error > 1e-4f && abs_error > 1e-5f) {
                if (error_count < 10) {  // Print first 10 errors
                    printf("Mismatch at [%d, %d]: ref=%.6f, gpu=%.6f, abs_err=%.6e, rel_err=%.6e\n",
                           m, n, ref_val, gpu_val, abs_error, rel_error);
                }
                error_count++;
            }
        }
    }
    
    printf("Accuracy Test (M=%d, N=%d, is_even_M=%d):\n", M, N, Is_even_M);
    printf("  Errors: %d / %d\n", error_count, check_limit * N);
    printf("  Max absolute error: %.6e\n", max_abs_error);
    printf("  Max relative error: %.6e\n", max_rel_error);
    
    EXPECT_EQ(error_count, 0);
    EXPECT_LT(max_rel_error, 1e-4f);
    
    // ========================================================================
    // 2. Performance Test
    // ========================================================================
    if (iterations > 0) {
        // Warmup
        for (int i = 0; i < 10; ++i) {
            launch<scale_apply_exp2_kernel<Is_even_M, M, N>, ScaleApplyExp2Kernel<Is_even_M, M, N>>(
                launch_policy{
                    compat::dim3(1), 
                    compat::dim3(kNumThreads),
                    kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
                },
                d_tensor.data(),
                d_max.data(),
                scale,
                tail_m
            );
        }
        
        compat::get_default_queue().wait();
        
        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            launch<scale_apply_exp2_kernel<Is_even_M, M, N>, ScaleApplyExp2Kernel<Is_even_M, M, N>>(
                launch_policy{
                    compat::dim3(1), 
                    compat::dim3(kNumThreads),
                    kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
                },
                d_tensor.data(),
                d_max.data(),
                scale,
                tail_m
            );
        }
        
        compat::get_default_queue().wait();
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_time_us = (elapsed_ms * 1000.0) / iterations;
        
        // Calculate throughput
        size_t ops = static_cast<size_t>(check_limit) * N * 3;  // 3 ops: mul, sub, exp2
        double gflops = (ops * iterations) / (elapsed_ms * 1e6);
        
        size_t bytes = static_cast<size_t>(check_limit) * N * sizeof(float) * 2;  // read + write
        double bandwidth_gb_s = (bytes * iterations) / (elapsed_ms * 1e6);
        
        printf("Performance Test (%d iterations):\n", iterations);
        printf("  Average time: %.3f us\n", avg_time_us);
        printf("  Throughput: %.2f GFLOPS\n", gflops);
        printf("  Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    }
}

// ============================================================================
// Test Cases
// ============================================================================
TEST(ScaleApplyExp2Test, EvenM_64x64) {
    test_scale_apply_exp2<true, kBlockM, kBlockN>(100);
}

TEST(ScaleApplyExp2Test, UnevenM_64x64) {
    test_scale_apply_exp2<false, kBlockM, kBlockN>(100);
}

TEST(ScaleApplyExp2Test, Accuracy_EvenM) {
    test_scale_apply_exp2<true, kBlockM, kBlockN>(0);  // accuracy only
}

TEST(ScaleApplyExp2Test, Accuracy_UnevenM) {
    test_scale_apply_exp2<false, kBlockM, kBlockN>(0);  // accuracy only
}

// ============================================================================
// Optimized Version - GPU Kernel (to be optimized with inline asm, vectorization, etc.)
// ============================================================================
template <bool Is_even_M, int M, int N>
class ScaleApplyExp2OptimizedKernel;

template <bool Is_even_M, int M, int N>
void scale_apply_exp2_optimized_kernel(
    float* d_tensor,        // device memory [M, N]
    const float* d_max,     // device memory [N]
    float scale,
    int tail_m)
{
#ifdef __SYCL_DEVICE_ONLY__
    // Get thread info
    auto local_id = int(compat::get_nd_item<1>().get_local_id(0));
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_id = sg.get_group_linear_id();
    int lane_id = sg.get_local_id();
    
    // Each subgroup processes a portion of the N dimension
    // With kNSGs=8 subgroups, each processes N/8 columns (for N >= kNSGs)
    // For N < kNSGs (e.g., N=1), only some subgroups are used
    constexpr int N_per_sg = (N >= kNSGs) ? (N / kNSGs) : 1;
    constexpr int active_sgs = (N >= kNSGs) ? kNSGs : N;
    
    // Early exit for subgroups that don't process any data (when N < kNSGs)
    if constexpr(N < kNSGs) {
        if (sg_id >= N) return;
    }
    
    int n_start = (N >= kNSGs) ? (sg_id * N_per_sg) : sg_id;
    
    // Each lane processes M/16 = 4 rows
    constexpr int M_per_lane = M / kSubgroupSize;
    static_assert(M_per_lane == 4, "Optimized for M=64, subgroup_size=16");
    
    // Process each column in this subgroup's range
    for (int ni = 0; ni < N_per_sg; ++ni) {
        int n = n_start + ni;
        
        // Load max value for this column and compute neg_max_scaled
        float max_val = d_max[n];
        float neg_max_scaled = (max_val == -INFINITY) ? 0.f : -(max_val * M_LOG2E);
        
        if constexpr(Is_even_M) {
            int m_base = lane_id;
            
            // Use intel::float4 to represent 4 values per lane
            // In GRF: 16 lanes × 4 floats = 64 floats = 256 bytes (4 GRFs)
            // Layout: interleaved per element across lanes (round-robin)
            intel::float4 vals;
            
            if constexpr(N == 1) {
                // N=1 optimization: Use CuTe tensor API for consecutive memory
                // Global memory layout: [row0, row1, ..., row63] (consecutive)
                
                // Step 1: Create local storage and tensor view
                float local_data[4];
                auto local_vals = make_tensor(&local_data[0], make_layout(Int<4>{}));
                
                // Step 2: Copy data from global memory to local tensor
                // Each lane needs values at: [lane_id, lane_id+16, lane_id+32, lane_id+48]
                auto gmem_vec = make_tensor(make_gmem_ptr(d_tensor),
                                            make_layout(Int<64>{}));
                local_vals(0) = gmem_vec(m_base + 0 * kSubgroupSize);
                local_vals(1) = gmem_vec(m_base + 1 * kSubgroupSize);
                local_vals(2) = gmem_vec(m_base + 2 * kSubgroupSize);
                local_vals(3) = gmem_vec(m_base + 3 * kSubgroupSize);
                
                // Step 3: Recast local tensor data to intel::float4
                vals = *recast_ptr<intel::float4>(local_vals.data());
            } else {
                // General case: Strided memory access with column offset
                vals[0] = d_tensor[(m_base + 0 * kSubgroupSize) * N + n];
                vals[1] = d_tensor[(m_base + 1 * kSubgroupSize) * N + n];
                vals[2] = d_tensor[(m_base + 2 * kSubgroupSize) * N + n];
                vals[3] = d_tensor[(m_base + 3 * kSubgroupSize) * N + n];
            }
            
            // Intel Xe inline assembly using vector register declarations
            // GRF layout: vals[0][lane0...15], vals[1][lane0...15], vals[2][lane0...15], vals[3][lane0...15]
            // Each element occupies 16 lanes worth of data (64 bytes) in round-robin layout
            __asm__ volatile (
                "{\n"
                ".decl VALS v_type=G type=F num_elts=64 alias=<%0,0>\n"
                ".decl SCALE v_type=G type=F num_elts=16 alias=<%1,0>\n"
                ".decl NEG_MAX v_type=G type=F num_elts=16 alias=<%2,0>\n"
                "mad (M1, 16) VALS(0,0)<1>  VALS(0,0)<1;1,0>  SCALE(0,0)<0;1,0> NEG_MAX(0,0)<0;1,0>\n"
                "mad (M1, 16) VALS(0,16)<1> VALS(0,16)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<0;1,0>\n"
                "mad (M1, 16) VALS(0,32)<1> VALS(0,32)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<0;1,0>\n"
                "mad (M1, 16) VALS(0,48)<1> VALS(0,48)<1;1,0> SCALE(0,0)<0;1,0> NEG_MAX(0,0)<0;1,0>\n"
                "exp (M1, 16) VALS(0,0)<1>  VALS(0,0)<1;1,0>\n"
                "exp (M1, 16) VALS(0,16)<1> VALS(0,16)<1;1,0>\n"
                "exp (M1, 16) VALS(0,32)<1> VALS(0,32)<1;1,0>\n"
                "exp (M1, 16) VALS(0,48)<1> VALS(0,48)<1;1,0>\n"
                "}\n"
                : "+rw"(vals)
                : "rw"(scale), "rw"(neg_max_scaled)
            );
            
            if constexpr(N == 1) {
                // N=1 optimization: Use CuTe tensor API for consecutive memory store
                // vals was modified by inline asm, now write back to global memory
                
                // Step 1: Recast vals to local tensor
                auto local_vals = make_tensor(recast_ptr<float>(&vals),
                                              make_layout(Int<4>{}));
                
                // Step 2: Create gmem tensor view
                auto gmem_vec = make_tensor(make_gmem_ptr(d_tensor),
                                            make_layout(Int<64>{}));
                
                // Step 3: Copy from local tensor to global memory with stride
                gmem_vec(m_base + 0 * kSubgroupSize) = local_vals(0);
                gmem_vec(m_base + 1 * kSubgroupSize) = local_vals(1);
                gmem_vec(m_base + 2 * kSubgroupSize) = local_vals(2);
                gmem_vec(m_base + 3 * kSubgroupSize) = local_vals(3);
            } else {
                // General case: Explicit scatter to strided memory
                d_tensor[(m_base + 0 * kSubgroupSize) * N + n] = vals[0];
                d_tensor[(m_base + 1 * kSubgroupSize) * N + n] = vals[1];
                d_tensor[(m_base + 2 * kSubgroupSize) * N + n] = vals[2];
                d_tensor[(m_base + 3 * kSubgroupSize) * N + n] = vals[3];
            }
        } else {
            // Handle tail case with bounds checking
            int m_base = lane_id;
            
            // Process up to 4 values, checking bounds
            for (int i = 0; i < M_per_lane; ++i) {
                int m = m_base + i * kSubgroupSize;
                if (m < tail_m) {
                    int idx = m * N + n;
                    float val = d_tensor[idx];
                    
                    // Single value inline assembly
                    __asm__ volatile (
                        "mad (M1, 16) %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<0;1,0> %3(0,0)<1;1,0>\n\t"
                        "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>"
                        : "=rw"(val)
                        : "0"(val), "rw"(scale), "rw"(neg_max_scaled)
                    );
                    
                    d_tensor[idx] = val;
                }
            }
        }
    }
#else
    // Fallback for host compilation (shouldn't be called)
    (void)d_tensor; (void)d_max; (void)scale; (void)tail_m;
#endif
}

// ============================================================================
// Test Function for Optimized Version
// ============================================================================
template <bool Is_even_M, int M, int N>
void test_scale_apply_exp2_optimized(int iterations = 100) {
    static_assert(M == kBlockM, "M must equal kBlockM");
    static_assert(M % kSubgroupSize == 0, "M must be divisible by subgroup size");
    // For N < kNSGs (e.g., N=1), only one subgroup will process the data
    static_assert(N == kBlockN || N == 1, "N must equal kBlockN or 1 for specialized path");
    
    const float scale = 1.44269504f;  // log2(e), typical softmax scale
    const int tail_m = Is_even_M ? M : (M - 8);  // simulate tail case
    
    // Allocate and initialize host memory
    cutlass::host_vector<float> h_tensor(M * N);
    cutlass::host_vector<float> h_max(N);
    cutlass::host_vector<float> h_tensor_ref(M * N);
    cutlass::host_vector<float> h_tensor_baseline(M * N);
    cutlass::host_vector<float> h_tensor_out(M * N);
    
    // Fill with random values
    std::random_device rd;
    std::mt19937 gen(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> max_dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * N; ++i) {
        h_tensor[i] = dist(gen);
        h_tensor_ref[i] = h_tensor[i];
        h_tensor_baseline[i] = h_tensor[i];
    }
    
    for (int i = 0; i < N; ++i) {
        h_max[i] = max_dist(gen);
    }
    
    // Compute CPU reference
    scale_apply_exp2_cpu_reference(h_tensor_ref.data(), h_max.data(), M, N, scale, Is_even_M, tail_m);
    
    // Allocate device memory
    cutlass::device_vector<float> d_tensor = h_tensor;
    cutlass::device_vector<float> d_tensor_baseline = h_tensor_baseline;
    cutlass::device_vector<float> d_max = h_max;
    
    // Run baseline version for comparison
    launch<scale_apply_exp2_kernel<Is_even_M, M, N>, ScaleApplyExp2Kernel<Is_even_M, M, N>>(
        launch_policy{
            compat::dim3(1), 
            compat::dim3(kNumThreads),
            kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
        },
        d_tensor_baseline.data(),
        d_max.data(),
        scale,
        tail_m
    );
    h_tensor_baseline = d_tensor_baseline;
    
    // Launch optimized kernel
    launch<scale_apply_exp2_optimized_kernel<Is_even_M, M, N>, ScaleApplyExp2OptimizedKernel<Is_even_M, M, N>>(
        launch_policy{
            compat::dim3(1), 
            compat::dim3(kNumThreads),
            kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
        },
        d_tensor.data(),
        d_max.data(),
        scale,
        tail_m
    );
    
    // Copy result back
    h_tensor_out = d_tensor;
    
    // ========================================================================
    // 1. Accuracy Test: Compare with CPU reference
    // ========================================================================
    int error_count_cpu = 0;
    int error_count_baseline = 0;
    float max_rel_error_cpu = 0.0f;
    float max_abs_error_cpu = 0.0f;
    float max_rel_error_baseline = 0.0f;
    
    int check_limit = Is_even_M ? M : tail_m;
    
    for (int m = 0; m < check_limit; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * N + n;
            float ref_val = h_tensor_ref[idx];
            float baseline_val = h_tensor_baseline[idx];
            float gpu_val = h_tensor_out[idx];
            
            // Compare with CPU reference
            float abs_error_cpu = std::abs(ref_val - gpu_val);
            float rel_error_cpu = std::abs(ref_val) > 1e-6f ? abs_error_cpu / std::abs(ref_val) : abs_error_cpu;
            
            max_abs_error_cpu = std::max(max_abs_error_cpu, abs_error_cpu);
            max_rel_error_cpu = std::max(max_rel_error_cpu, rel_error_cpu);
            
            if (rel_error_cpu > 1e-4f && abs_error_cpu > 1e-5f) {
                if (error_count_cpu < 5) {
                    printf("  CPU mismatch at [%d, %d]: ref=%.6f, gpu=%.6f, abs_err=%.6e, rel_err=%.6e\n",
                           m, n, ref_val, gpu_val, abs_error_cpu, rel_error_cpu);
                }
                error_count_cpu++;
            }
            
            // Compare with baseline version
            float abs_error_baseline = std::abs(baseline_val - gpu_val);
            float rel_error_baseline = std::abs(baseline_val) > 1e-6f ? abs_error_baseline / std::abs(baseline_val) : abs_error_baseline;
            
            max_rel_error_baseline = std::max(max_rel_error_baseline, rel_error_baseline);
            
            if (rel_error_baseline > 1e-6f && abs_error_baseline > 1e-7f) {
                if (error_count_baseline < 5) {
                    printf("  Baseline mismatch at [%d, %d]: baseline=%.6f, optimized=%.6f, abs_err=%.6e, rel_err=%.6e\n",
                           m, n, baseline_val, gpu_val, abs_error_baseline, rel_error_baseline);
                }
                error_count_baseline++;
            }
        }
    }
    
    printf("Optimized Accuracy Test (M=%d, N=%d, is_even_M=%d):\n", M, N, Is_even_M);
    printf("  vs CPU Reference:\n");
    printf("    Errors: %d / %d\n", error_count_cpu, check_limit * N);
    printf("    Max absolute error: %.6e\n", max_abs_error_cpu);
    printf("    Max relative error: %.6e\n", max_rel_error_cpu);
    printf("  vs Baseline GPU:\n");
    printf("    Errors: %d / %d\n", error_count_baseline, check_limit * N);
    printf("    Max relative error: %.6e\n", max_rel_error_baseline);
    
    EXPECT_EQ(error_count_cpu, 0);
    EXPECT_LT(max_rel_error_cpu, 1e-4f);
    EXPECT_EQ(error_count_baseline, 0);
    EXPECT_LT(max_rel_error_baseline, 1e-6f);
    
    // ========================================================================
    // 2. Performance Test
    // ========================================================================
    if (iterations > 0) {
        // Warmup baseline
        for (int i = 0; i < 10; ++i) {
            launch<scale_apply_exp2_kernel<Is_even_M, M, N>, ScaleApplyExp2Kernel<Is_even_M, M, N>>(
                launch_policy{
                    compat::dim3(1), 
                    compat::dim3(kNumThreads),
                    kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
                },
                d_tensor_baseline.data(),
                d_max.data(),
                scale,
                tail_m
            );
        }
        compat::get_default_queue().wait();
        
        // Time baseline
        auto start_baseline = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            launch<scale_apply_exp2_kernel<Is_even_M, M, N>, ScaleApplyExp2Kernel<Is_even_M, M, N>>(
                launch_policy{
                    compat::dim3(1), 
                    compat::dim3(kNumThreads),
                    kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
                },
                d_tensor_baseline.data(),
                d_max.data(),
                scale,
                tail_m
            );
        }
        compat::get_default_queue().wait();
        auto end_baseline = std::chrono::high_resolution_clock::now();
        
        // Warmup optimized
        for (int i = 0; i < 10; ++i) {
            launch<scale_apply_exp2_optimized_kernel<Is_even_M, M, N>, ScaleApplyExp2OptimizedKernel<Is_even_M, M, N>>(
                launch_policy{
                    compat::dim3(1), 
                    compat::dim3(kNumThreads),
                    kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
                },
                d_tensor.data(),
                d_max.data(),
                scale,
                tail_m
            );
        }
        compat::get_default_queue().wait();
        
        // Time optimized
        auto start_optimized = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            launch<scale_apply_exp2_optimized_kernel<Is_even_M, M, N>, ScaleApplyExp2OptimizedKernel<Is_even_M, M, N>>(
                launch_policy{
                    compat::dim3(1), 
                    compat::dim3(kNumThreads),
                    kernel_properties{sycl_exp::sub_group_size<kSubgroupSize>}
                },
                d_tensor.data(),
                d_max.data(),
                scale,
                tail_m
            );
        }
        compat::get_default_queue().wait();
        auto end_optimized = std::chrono::high_resolution_clock::now();
        
        double elapsed_baseline_ms = std::chrono::duration<double, std::milli>(end_baseline - start_baseline).count();
        double elapsed_optimized_ms = std::chrono::duration<double, std::milli>(end_optimized - start_optimized).count();
        
        double avg_time_baseline_us = (elapsed_baseline_ms * 1000.0) / iterations;
        double avg_time_optimized_us = (elapsed_optimized_ms * 1000.0) / iterations;
        
        // Calculate throughput
        size_t ops = static_cast<size_t>(check_limit) * N * 3;  // 3 ops: mul, sub, exp2
        double gflops_baseline = (ops * iterations) / (elapsed_baseline_ms * 1e6);
        double gflops_optimized = (ops * iterations) / (elapsed_optimized_ms * 1e6);
        
        size_t bytes = static_cast<size_t>(check_limit) * N * sizeof(float) * 2;  // read + write
        double bandwidth_baseline_gb_s = (bytes * iterations) / (elapsed_baseline_ms * 1e6);
        double bandwidth_optimized_gb_s = (bytes * iterations) / (elapsed_optimized_ms * 1e6);
        
        double speedup = avg_time_baseline_us / avg_time_optimized_us;
        
        printf("Performance Test (%d iterations):\n", iterations);
        printf("  Baseline:\n");
        printf("    Average time: %.3f us\n", avg_time_baseline_us);
        printf("    Throughput: %.2f GFLOPS\n", gflops_baseline);
        printf("    Bandwidth: %.2f GB/s\n", bandwidth_baseline_gb_s);
        printf("  Optimized:\n");
        printf("    Average time: %.3f us\n", avg_time_optimized_us);
        printf("    Throughput: %.2f GFLOPS\n", gflops_optimized);
        printf("    Bandwidth: %.2f GB/s\n", bandwidth_optimized_gb_s);
        printf("  Speedup: %.2fx\n", speedup);
    }
}

// ============================================================================
// Test Cases for Optimized Version
// ============================================================================
TEST(ScaleApplyExp2OptimizedTest, EvenM_64x64) {
    test_scale_apply_exp2_optimized<true, kBlockM, kBlockN>(100);
}

TEST(ScaleApplyExp2OptimizedTest, UnevenM_64x64) {
    test_scale_apply_exp2_optimized<false, kBlockM, kBlockN>(100);
}

TEST(ScaleApplyExp2OptimizedTest, Accuracy_EvenM) {
    test_scale_apply_exp2_optimized<true, kBlockM, kBlockN>(0);  // accuracy only
}

TEST(ScaleApplyExp2OptimizedTest, Accuracy_UnevenM) {
    test_scale_apply_exp2_optimized<false, kBlockM, kBlockN>(0);  // accuracy only
}

// ============================================================================
// Test Cases for N=1 Optimization (Direct Recast Path)
// ============================================================================
TEST(ScaleApplyExp2OptimizedTest, N1_EvenM_64x1) {
    test_scale_apply_exp2_optimized<true, kBlockM, 1>(100);
}

TEST(ScaleApplyExp2OptimizedTest, N1_Accuracy_EvenM) {
    test_scale_apply_exp2_optimized<true, kBlockM, 1>(0);  // accuracy only
}
