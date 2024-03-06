/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#  include "cutlass/util/cublas_wrappers.hpp"
#endif

namespace utils {

    template <typename T>
    struct Vector {
    public:
        T vector_;
        int size_;

        Vector(int size) : vector_(size), size_(size) {
        }

        template<class T_>
        Vector(const T_ &v) : vector_(v), size_(static_cast<int>(vector_.size())) {}

        CUTE_HOST
        auto data() {
          auto data = vector_.data();
          if constexpr (!std::is_pointer_v<decltype(data)>) {
            return data.get();
          } else {
            return data;
          }
        }

        CUTE_HOST
        const auto data() const {
          return vector_.data();
        }

        CUTE_HOST
        auto size() {
          return size_;
        }

        CUTE_HOST
        auto& operator[](int const i) {
          return vector_[i];
        }
    };

#if defined(CUTLASS_ENABLE_SYCL)
    template<class T>
    class SyclVector {
    public:
        T* vector_;
        int size_;

        SyclVector(int size) : vector_(sycl::malloc_device<T>(size, syclcompat::get_default_queue())), size_(size) {
        }

        SyclVector(const thrust::host_vector<T> &host_vector)
                : SyclVector(host_vector.data(), host_vector.size()) {
        }

        CUTE_HOST
        auto data() {
          return vector_;
        }

        CUTE_HOST
        const auto data() const {
          return vector_;
        }

        CUTE_HOST
        auto size() {
          return size_;
        }

        CUTE_HOST
        auto& operator[](int const i) {
          return vector_[i];
        }
    private:
        SyclVector(const T* host_data, int size)
                : vector_(sycl::malloc_device<T>(size, syclcompat::get_default_queue())), size_(size) {
          auto q = syclcompat::get_default_queue();
          q.memcpy(vector_, host_data, size * sizeof(T)).wait();
        }
    };

    template <typename T>
    class DeviceVector : public Vector<SyclVector<T>> {
    public:
        DeviceVector(const Vector<thrust::host_vector<T>> &host_vector)
                : Vector<SyclVector<T>>(host_vector.vector_) {
        }
    };

#else
    template <typename T>
    class DeviceVector : public Vector<thrust::device_vector<T>> {
    public:
        DeviceVector(const Vector<thrust::host_vector<T>> &host_vector)
          : Vector<thrust::device_vector<T>>(host_vector.vector_) {
        }
    };
#endif

    template <typename T>
    class HostVector : public Vector<thrust::host_vector<T>> {
    public:
#if defined(CUTLASS_ENABLE_SYCL)
        HostVector(const Vector<SyclVector<T>> &device_vector)
                : Vector<thrust::host_vector<T>>(device_vector.size_) {
          auto q = syclcompat::get_default_queue();
          q.memcpy(Vector<thrust::host_vector<T>>::vector_.data(), device_vector.data(), device_vector.size_ * sizeof(T));
        }
#else
        HostVector(const Vector<thrust::device_vector<T>> &device_vector)
                : Vector<thrust::host_vector<T>>(device_vector.vector_) {
        }
#endif
    };

/**
 * @brief Tests GEMM operation.
 *
 * This function tests the GEMM operation. It generates random matrices
 * for input A and B, performs the GEMM operation, and compares the results
 * with the output obtained from cuBLAS if available.
 *
 * @tparam gemm The GEMM function to test.
 * @tparam TA The data type of matrix A.
 * @tparam TB The data type of matrix B.
 * @tparam TC The data type of matrix C.
 * @tparam TI The data type of the alpha and beta parameters.
 * @tparam TAT The data type to cast matrix A to before the operation (default is TA).
 * @tparam TBT The data type to cast matrix B to before the operation (default is TB).
 *
 * @param m The number of rows in matrix A and C.
 * @param n The number of columns in matrix B and C.
 * @param k The number of columns in matrix A and rows in matrix B.
 */
    template<auto gemm, typename TA, typename TB, typename TC, typename TI,
            typename TAT = TA, typename TBT = TB>
    void test_gemm(int m, int n, int k)
    {
      std::cout << "M = " << m << std::endl;
      std::cout << "N = " << n << std::endl;
      std::cout << "K = " << k << std::endl;

      utils::HostVector<TA> h_A(m*k);
      utils::HostVector<TB> h_B(n*k);
      utils::HostVector<TC> h_C(m*n);

      for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TAT>( 4*(rand() / double(RAND_MAX)) - 4 );
      for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TBT>( 4*(rand() / double(RAND_MAX)) - 4 );
      for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

      utils::DeviceVector<TA> d_A = h_A;
      utils::DeviceVector<TA> d_B = h_B;
      utils::DeviceVector<TA> d_C = h_C;

      TI alpha = 1.0;
      TI beta  = 0.0;

      double tflops = (2.0*m*n*k) * 1e-12;

      const int timing_iterations = 100;
      GPU_Clock timer;

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
      //
      // cuBLas
      //

      cublasHandle_t handle;
      cublasCreate(&handle);

      // Run once
      blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                         m, n, k,
                         &alpha,
                         d_A.data(), m,
                         d_B.data(), n,
                         &beta,
                         d_C.data(), m);
      CUTE_CHECK_LAST();

      utils::HostVector<TC> cublas_result = d_C;

      // Timing iterations
      timer.start();
      for (int i = 0; i < timing_iterations; ++i) {
        blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           m, n, k,
                           &alpha,
                           d_A.data(), m,
                           d_B.data(), n,
                           &beta,
                           d_C.data(), m);
      }
      double cublas_time = timer.seconds() / timing_iterations;
      CUTE_CHECK_LAST();
      printf("CUBLAS_GEMM:   [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cublas_time, cublas_time*1000);

#else

      std::cout << "Verification by comparison with cuBLAS is disabled, "
    "either because the CMake option CUTLASS_ENABLE_CUBLAS "
    "was explicitly set to OFF, or because CMake could not find cuBLAS.  "
    "If you would like to enable verification with cuBLAS, "
    "please set the CMake option CUTLASS_ENABLE_CUBLAS to ON, "
    "rerun CMake, and recompile this example.\n";

#endif // CUTLASS_ENABLE_CUBLAS

      //
      // CuTe
      //

      // Run once (and check)
      d_C = h_C;
      gemm(m, n, k,
           alpha,
           d_A.data(), m,
           d_B.data(), n,
           beta,
           d_C.data(), m);
      CUTE_CHECK_LAST();

      // Timing iterations

      timer.start();
      for (int i = 0; i < timing_iterations; ++i) {
        gemm(m, n, k,
              alpha,
              d_A.data(), m,
              d_B.data(), n,
              beta,
              d_C.data(), m);
      }

      double cute_time = timer.seconds() / timing_iterations;
      CUTE_CHECK_LAST();
      printf("CUTE_GEMM:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);

      utils::HostVector<TC> cute_result = d_C;

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
      printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

      auto host_matrix_to_const_column_major_cute_tensor =
              [](const auto& X, int num_rows, int num_cols, int LDX) {
                  const auto shape = cute::Shape<int, int>{num_rows, num_cols};
                  const auto strides = cute::Stride<int, int>{1, LDX};
                  return cute::make_tensor(X.data(), cute::make_layout(shape, strides));
              };

      const auto A_view = host_matrix_to_const_column_major_cute_tensor(h_A, m, k, m);
      // B^T is k x n, so B is n x k.
      const auto B_view = host_matrix_to_const_column_major_cute_tensor(h_B, n, k, n);
      const auto C_computed_view = host_matrix_to_const_column_major_cute_tensor(cute_result, m, n, m);
      const auto C_expected_view = host_matrix_to_const_column_major_cute_tensor(cublas_result, m, n, m);
      print_matrix_multiply_mollified_relative_error("float", A_view, B_view, C_computed_view, C_expected_view);

#endif // CUTLASS_ENABLE_CUBLAS
    }

/**
 * @brief Launches a kernel function.
 *
 * This function launches a kernel function either using SYCL or CUDA.
 *
 * @tparam F The kernel function to launch.
 * @tparam Grid The type representing grid dimensions for the kernel launch.
 * @tparam Block The type representing block dimensions for the kernel launch.
 * @tparam Args The types of arguments to pass to the kernel function.
 *
 * @param grid The grid dimensions for the kernel launch.
 * @param threads The block dimensions for the kernel launch.
 * @param args The arguments to pass to the kernel function.
 */
    template <auto F, typename Grid, typename Block, typename... Args>
    void launch_kernel(const Grid &grid, const Block &block, Args... args) {
#if defined(CUTLASS_ENABLE_SYCL)
      syclcompat::launch<F>(grid, block, args...);
#else
      F<<< grid, block, 0, 0 >>>(args...);
#endif
    }

} // namespace utils

