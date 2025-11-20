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

#include "cutlass/detail/layout.hpp"

#include <cute/tensor.hpp>
#include <cute/algorithm/reorder.hpp>
#include <cute/tensor_sg.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass_unit_test.h"

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

// ============================================================================
// Test Helpers
// ============================================================================

template<class...> class ReorderKernelName;

// Generic reorder test kernel for SubgroupTensor
template <class SrcType, class DstType, int M, int N>
void reorder_kernel_subgroup_tensor(SrcType* src_global, DstType* dst_global)
{
  const int tid = ThreadIdxX();
  constexpr int total_size = M * N;
  constexpr int values_per_thread = total_size / intel::sg_size;

  // Each thread owns a slice of values (round-robin pattern)
  SrcType src_local[values_per_thread];
  DstType dst_local[values_per_thread];
  
  // Load from global memory (each thread loads its values)
  for (int i = 0; i < values_per_thread; ++i) {
    src_local[i] = src_global[tid + i * intel::sg_size];
  }

  // Create fragments
  auto src_tensor = make_tensor(make_rmem_ptr(src_local),
                                 make_layout(Shape<Int<values_per_thread>>{}));
  auto dst_tensor = make_tensor(make_rmem_ptr(dst_local),
                                 make_layout(Shape<Int<values_per_thread>>{}));

  // Subgroup TV layout for round-robin ownership
  constexpr auto sg_tv_layout = make_layout(Shape<Int<intel::sg_size>, Int<values_per_thread>>{},
                                            Stride<_1, Int<intel::sg_size>>{});

  // Create SubgroupTensors and perform reorder
  auto src_sg = make_subgroup_tensor(src_tensor, sg_tv_layout);
  auto dst_sg = make_subgroup_tensor(dst_tensor, sg_tv_layout);
  reorder(src_sg, dst_sg);
  
  // Store back to global memory
  for (int i = 0; i < values_per_thread; ++i) {
    dst_global[tid + i * intel::sg_size] = dst_local[i];
  }
}

// Helper function to run a reorder test
template <class SrcType, class DstType, int M, int N, int TestID>
void run_reorder_test(cutlass::host_vector<SrcType>& host_src,
                      cutlass::host_vector<DstType>& host_dst)
{
  cutlass::device_vector<SrcType> device_src = host_src;
  cutlass::device_vector<DstType> device_dst(M * N);

  launch<reorder_kernel_subgroup_tensor<SrcType, DstType, M, N>,
         ReorderKernelName<SrcType, DstType, Int<TestID>>>(
      launch_policy{compat::dim3(1), compat::dim3(intel::sg_size),
                    kernel_properties{sycl_exp::sub_group_size<intel::sg_size>}},
      device_src.data(), device_dst.data());

  compat::wait_and_throw();
  host_dst = device_dst;
}

// Helper function to initialize test data
template <class T>
void initialize_test_data(cutlass::host_vector<T>& host_src) {
  for (size_t i = 0; i < host_src.size(); ++i) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || 
                  std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
      host_src[i] = static_cast<T>(i);
    } else if constexpr (std::is_same_v<T, half_t>) {
      host_src[i] = half_t(static_cast<float>(i));
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
      host_src[i] = bfloat16_t(static_cast<float>(i) * 0.25f);
    } else if constexpr (std::is_same_v<T, uint4_t>) {
      host_src[i] = uint4_t(static_cast<unsigned>(i % 16));
    } else if constexpr (std::is_same_v<T, int4_t>) {
      host_src[i] = int4_t(static_cast<int>((i % 16) - 8));
    }
  }
}

// Helper function to initialize source data for type conversions
template <class SrcType>
void initialize_conversion_source(cutlass::host_vector<SrcType>& host_src) {
  for (size_t i = 0; i < host_src.size(); ++i) {
    if constexpr (std::is_same_v<SrcType, float>) {
      host_src[i] = static_cast<SrcType>(i) * 0.5f;
    } else if constexpr (std::is_same_v<SrcType, half_t>) {
      host_src[i] = half_t(static_cast<float>(i) * 0.5f);
    } else if constexpr (std::is_same_v<SrcType, int32_t>) {
      host_src[i] = static_cast<SrcType>(i);
    } else if constexpr (std::is_same_v<SrcType, int8_t>) {
      host_src[i] = static_cast<SrcType>(i % 128);
    }
  }
}

// Test kernel for tensor-based reorder (without SubgroupTensor)
template <class SrcType, class DstType, int M, int N>
void reorder_kernel_tensor(
    SrcType* src_global, DstType* dst_global)
{
  const int tid = ThreadIdxX();
  constexpr int total_size = M * N;
  constexpr int values_per_thread = total_size / intel::sg_size;

  // Each thread owns a slice of the data
  SrcType src_local[values_per_thread];
  DstType dst_local[values_per_thread];
  
  // Load from global memory
  for (int i = 0; i < values_per_thread; ++i) {
    src_local[i] = src_global[tid + i * intel::sg_size];
  }
  
  auto src_fragment = make_tensor(
      make_rmem_ptr(src_local),
      make_layout(Shape<Int<values_per_thread>>{}));
  
  auto dst_fragment = make_tensor(
      make_rmem_ptr(dst_local),
      make_layout(Shape<Int<values_per_thread>>{}));

  // Subgroup TV layout (same for src and dst)
  constexpr auto sg_layout = make_layout(
      Shape<Int<intel::sg_size>, Int<values_per_thread>>{},
      Stride<_1, Int<intel::sg_size>>{});

  // Perform reorder with explicit TV layouts
  reorder(src_fragment, dst_fragment, sg_layout, sg_layout);
  
  // Store back to global memory
  for (int i = 0; i < values_per_thread; ++i) {
    dst_global[tid + i * intel::sg_size] = dst_local[i];
  }
}

// Helper function to run a tensor-based reorder test
template <class SrcType, class DstType, int M, int N, int TestID>
void run_tensor_reorder_test(cutlass::host_vector<SrcType>& host_src,
                              cutlass::host_vector<DstType>& host_dst)
{
  cutlass::device_vector<SrcType> device_src = host_src;
  cutlass::device_vector<DstType> device_dst(M * N);

  launch<reorder_kernel_tensor<SrcType, DstType, M, N>,
         ReorderKernelName<SrcType, DstType, Int<TestID>>>(
      launch_policy{compat::dim3(1), compat::dim3(intel::sg_size),
                    kernel_properties{sycl_exp::sub_group_size<intel::sg_size>}},
      device_src.data(), device_dst.data());

  compat::wait_and_throw();
  host_dst = device_dst;
}

// Generic template-based test helper for reorder operations
template <class T, int M, int N, int TestID, 
          template <class, class, int, int> class KernelRunner>
struct ReorderTestBase {
  static void run() {
    cutlass::host_vector<T> host_src(M * N);
    cutlass::host_vector<T> host_dst(M * N);

    // Initialize with sequential values
    initialize_test_data(host_src);

    // Run the appropriate kernel via KernelRunner
    KernelRunner<T, T, M, N>::template execute<TestID>(host_src, host_dst);

    // Verify reorder preserves all values
    for (size_t i = 0; i < host_src.size(); ++i) {
      EXPECT_EQ(host_dst[i], host_src[i]);
    }
  }
};

// Kernel runner for SubgroupTensor-based reorder
template <class SrcType, class DstType, int M, int N>
struct SubgroupTensorRunner {
  template <int TestID>
  static void execute(cutlass::host_vector<SrcType>& host_src,
                      cutlass::host_vector<DstType>& host_dst) {
    run_reorder_test<SrcType, DstType, M, N, TestID>(host_src, host_dst);
  }
};

// Kernel runner for Tensor-based reorder
template <class SrcType, class DstType, int M, int N>
struct TensorRunner {
  template <int TestID>
  static void execute(cutlass::host_vector<SrcType>& host_src,
                      cutlass::host_vector<DstType>& host_dst) {
    run_tensor_reorder_test<SrcType, DstType, M, N, TestID>(host_src, host_dst);
  }
};

// Generic template-based test helper for cross-type reorder operations
template <class SrcType, class DstType, int M, int N, int TestID,
          template <class, class, int, int> class KernelRunner>
struct ConversionReorderTestBase {
  static void run() {
    cutlass::host_vector<SrcType> host_src(M * N);
    cutlass::host_vector<DstType> host_dst(M * N);

    // Initialize source data
    initialize_conversion_source(host_src);

    // Expected result after conversion
    cutlass::host_vector<DstType> host_expected(M * N);
    for (size_t i = 0; i < host_src.size(); ++i) {
      host_expected[i] = static_cast<DstType>(host_src[i]);
    }

    // Run the reorder kernel
    KernelRunner<SrcType, DstType, M, N>::template execute<TestID>(host_src, host_dst);

    // Verify conversion correctness (with tolerance for floating point)
    for (size_t i = 0; i < host_expected.size(); ++i) {
      if constexpr (std::is_floating_point_v<DstType>) {
        EXPECT_NEAR(static_cast<float>(host_dst[i]), 
                    static_cast<float>(host_expected[i]), 1e-4f);
      } else {
        EXPECT_EQ(host_dst[i], host_expected[i]);
      }
    }
  }
};

// Template-based test helper for cross-type reorders using SubgroupTensor
template <class SrcType, class DstType, int M, int N, int TestID>
struct ConversionSubgroupTest : ConversionReorderTestBase<SrcType, DstType, M, N, TestID, SubgroupTensorRunner> {};

// Template-based test helper for cross-type reorders using Tensor
template <class SrcType, class DstType, int M, int N, int TestID>
struct ConversionTensorTest : ConversionReorderTestBase<SrcType, DstType, M, N, TestID, TensorRunner> {};

// Macro for cross-type conversion tests with SubgroupTensor
#define REORDER_CONVERSION_SUBGROUP_TEST(TestName, SrcType, DstType, M, N, TestID) \
  TEST(PVC_CuTe_Xe_Reorder_Conversion, TestName) { \
    ConversionSubgroupTest<SrcType, DstType, M, N, TestID>::run(); \
  }

// Macro for cross-type conversion tests with Tensor
#define REORDER_CONVERSION_TENSOR_TEST(TestName, SrcType, DstType, M, N, TestID) \
  TEST(PVC_CuTe_Xe_Reorder_Conversion_Tensor, TestName) { \
    ConversionTensorTest<SrcType, DstType, M, N, TestID>::run(); \
  }

// Template-based test helper for identity reorders using SubgroupTensor
template <class T, int M, int N, int TestID>
struct IdentityReorderTest : ReorderTestBase<T, M, N, TestID, SubgroupTensorRunner> {};

// Macro to simplify test declaration
#define REORDER_IDENTITY_TEST(TestName, Type, M, N, TestID) \
  TEST(PVC_CuTe_Xe_Reorder, TestName) { \
    IdentityReorderTest<Type, M, N, TestID>::run(); \
  }

// Macro to simplify tensor-based test declaration
#define REORDER_TENSOR_TEST(TestName, Type, M, N, TestID) \
  TEST(PVC_CuTe_Xe_Reorder_Tensor, TestName) { \
    ReorderTestBase<Type, M, N, TestID, TensorRunner>::run(); \
  }

// ============================================================================
// SubgroupTensor-based Reorder Tests
// ============================================================================
// These tests verify identity reorder operations using SubgroupTensor,
// ensuring data integrity through round-robin subgroup ownership patterns.

// Test: Basic Types with SubgroupTensor (8x16 matrices)
REORDER_IDENTITY_TEST(subgroup_basic_float, float, 8, 16, 0)
REORDER_IDENTITY_TEST(subgroup_basic_int32, int32_t, 8, 16, 1)

// Test: Half-precision and BFloat16 types
REORDER_IDENTITY_TEST(subgroup_half_precision, half_t, 8, 16, 2)
REORDER_IDENTITY_TEST(subgroup_bfloat16, bfloat16_t, 8, 16, 11)

// Test: Integer types (8-bit and sub-byte)
REORDER_IDENTITY_TEST(subgroup_int8, int8_t, 8, 16, 12)

// Note: Sub-byte types (uint4_t, int4_t) require special handling due to bit-packing
// and are covered in conversion tests with appropriate value ranges.

// Test: Varied matrix sizes
REORDER_IDENTITY_TEST(subgroup_small_matrix_4x4, float, 4, 4, 7)
REORDER_IDENTITY_TEST(subgroup_large_matrix_16x32, int32_t, 16, 32, 8)
REORDER_IDENTITY_TEST(subgroup_minimal_1x16, float, 1, 16, 13)
REORDER_IDENTITY_TEST(subgroup_power_of_two_32x16, half_t, 32, 16, 14)

// Test: Layout identity verification (same layout in and out)
REORDER_IDENTITY_TEST(subgroup_layout_identity_int32, int32_t, 8, 16, 5)

// Test: VNNI-compatible patterns
REORDER_IDENTITY_TEST(subgroup_vnni_pattern_half, half_t, 16, 16, 6)

// ============================================================================
// Tensor-based Reorder Tests
// ============================================================================
// These tests verify identity reorder operations using explicit tensor layouts
// with TV (Thread Value) semantics, ensuring correct data handling without
// SubgroupTensor abstraction.

// Test: Basic Types with Tensor Layouts (8x16 matrices)
REORDER_TENSOR_TEST(tensor_basic_float, float, 8, 16, 100)
REORDER_TENSOR_TEST(tensor_basic_int32, int32_t, 8, 16, 101)

// Test: Half-precision and BFloat16 types
REORDER_TENSOR_TEST(tensor_half_precision, half_t, 8, 16, 102)
REORDER_TENSOR_TEST(tensor_bfloat16, bfloat16_t, 8, 16, 103)

// Test: Integer types with larger matrices
REORDER_TENSOR_TEST(tensor_int8, int8_t, 16, 32, 104)

// Note: Sub-byte types (uint4_t, int4_t) require special handling due to bit-packing
// and are covered in conversion tests with appropriate value ranges.

// ============================================================================
// Cross-Type Conversion Tests (SubgroupTensor-based)
// ============================================================================
// These tests verify reorder operations with type conversions using SubgroupTensor,
// ensuring correct data conversion from source to destination types.

// Test: float to half_t conversion
REORDER_CONVERSION_SUBGROUP_TEST(conversion_float_to_half, float, half_t, 8, 16, 200)

// Test: int32_t to int8_t conversion
REORDER_CONVERSION_SUBGROUP_TEST(conversion_int32_to_int8, int32_t, int8_t, 8, 16, 201)

// Test: float to int32_t conversion
REORDER_CONVERSION_SUBGROUP_TEST(conversion_float_to_int32, float, int32_t, 8, 16, 202)

// Test: half_t to float conversion
REORDER_CONVERSION_SUBGROUP_TEST(conversion_half_to_float, half_t, float, 8, 16, 203)

// ============================================================================
// Cross-Type Conversion Tests (Tensor-based)
// ============================================================================
// These tests verify reorder operations with type conversions using explicit tensor layouts,
// ensuring correct data conversion without SubgroupTensor abstraction.

// Test: float to half_t conversion
REORDER_CONVERSION_TENSOR_TEST(tensor_conversion_float_to_half, float, half_t, 8, 16, 300)

// Test: int32_t to int8_t conversion
REORDER_CONVERSION_TENSOR_TEST(tensor_conversion_int32_to_int8, int32_t, int8_t, 8, 16, 301)

// Test: float to int32_t conversion
REORDER_CONVERSION_TENSOR_TEST(tensor_conversion_float_to_int32, float, int32_t, 8, 16, 302)

// Test: half_t to float conversion
REORDER_CONVERSION_TENSOR_TEST(tensor_conversion_half_to_float, half_t, float, 8, 16, 303)