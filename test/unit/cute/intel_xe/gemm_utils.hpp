/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass_unit_test.h"

#include "cutlass/device_kernel.h"
#include "cutlass_unit_test.h"
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

using namespace cute;
using namespace cutlass;

#define SUBGROUP_SIZE (16)

#define CUTLASS_ENABLE_DEBUG_PRINTS (0)
#define LOG_GROUP (0)
#define LOG_THREAD (0)

template <class atype, class btype, class ctype>
void verify(uint32_t m, uint32_t n, uint32_t k, atype *A, btype *B, ctype *C,
            ctype *D, bool row_a, bool row_b) {
  std::vector<ctype> h_D(m * n);

  syclcompat::memcpy<ctype>(h_D.data(), D, m * n);
  syclcompat::wait();

  int cnt = 0;
  bool is_normal = true;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int z = 0; z < k; z++) {
        auto b = row_b ? B[z * n + j] : B[z + j * k];
        C[i * n + j] += A[i * k + z] * b;
      }

      ctype val = h_D.data()[i * n + j];
      ctype expect = C[i * n + j];

      if (isnormal(val) && isnormal(expect)) {
        auto error = abs((expect - val) / val);
        if (error > 0.01f) {
          cnt++;
        }
      } else {
        is_normal = false;
      }
    }
  }

  EXPECT_EQ(cnt, 0);
  EXPECT_EQ(is_normal, true);
}

template <typename T> static void fill_matrix(std::vector<T> &M) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<float> dist((T)0.0, (T)1.0);
  std::generate(std::begin(M), std::end(M),
                [&] { return static_cast<T>(dist(rng)); });
}

template <class kernel> void run(uint32_t m, uint32_t n, uint32_t k) {

  using TA = kernel::TA;
  using TB = kernel::TB;
  using TC = kernel::TC;

  std::vector<TA> h_A(m * k);
  std::vector<TB> h_B(n * k);
  std::vector<TC> h_C(m * n);
  h_C.clear();

  fill_matrix(h_A);
  fill_matrix(h_B);

  auto d_A = syclcompat::malloc<TA>(m * k);
  auto d_B = syclcompat::malloc<TB>(k * n);
  auto d_C = syclcompat::malloc<TC>(m * n);

  syclcompat::memcpy<TA>(d_A, h_A.data(), m * k);
  syclcompat::memcpy<TB>(d_B, h_B.data(), k * n);
  syclcompat::memcpy<TC>(d_C, h_C.data(), m * n);
  syclcompat::wait();

  auto dimBlock = syclcompat::dim3(
      ceil_div(kernel::wg_tile_m, kernel::sg_tile_m),
      SUBGROUP_SIZE * ceil_div(kernel::wg_tile_n, kernel::sg_tile_n));
  auto dimGrid = syclcompat::dim3(size(ceil_div(m, kernel::wg_tile_m)),
                                  size(ceil_div(n, kernel::wg_tile_n)));

  syclcompat::experimental::launch<kernel::func, SUBGROUP_SIZE>(
      dimGrid, dimBlock, d_A, d_B, d_C, m, n, k);

  syclcompat::wait();

  verify(m, n, k, h_A.data(), h_B.data(), h_C.data(), d_C,
         kernel::is_a_row_major, kernel::is_b_row_major);

  syclcompat::free(d_A);
  syclcompat::free(d_B);
  syclcompat::free(d_C);
}
