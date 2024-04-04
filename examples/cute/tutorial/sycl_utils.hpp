/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

using namespace std::chrono;

void print_device_info(sycl::queue& queue, std::ostream& output_stream) {
  output_stream << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << " ";
  output_stream << "Num CUs: " << queue.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
}

template <class Kernel, typename... Args>
std::pair<double, double> benchmark(syclcompat::dim3&& grid_dim, syclcompat::dim3&& block_dim, sycl::queue& queue,
                                    int timing_iterations, int warmup_iterations, std::size_t flops, Args... args) {
  for (int i = 0; i < warmup_iterations; i++) {
    syclcompat::launch<Kernel>(grid_dim, block_dim, queue, args...).wait_and_throw();
  }

  double total_time = 0;
  for (int i = 0; i < timing_iterations; i++) {
    auto t1 = high_resolution_clock::now();
    syclcompat::launch<Kernel>(grid_dim, block_dim, queue, args...).wait();
    auto t2 = high_resolution_clock::now();
    total_time += duration_cast<microseconds>(t2 - t1).count();
  }
  // Returns GigaFlops
  double average_time_in_seconds = (total_time / timing_iterations) * 1e-6;
  return {(static_cast<double>(flops) / average_time_in_seconds) * 1e-9, total_time * 1e-3};
}

template <typename Foo, typename... Args>
std::pair<double, double> benchmark(Foo&& foo, int timing_iterations, int warmup_iterations, std::size_t flops,
                                    Args... args) {
  for (int i = 0; i < warmup_iterations; i++) {
    foo(args...).wait_and_throw();
  }

  double total_time = 0;
  for (int i = 0; i < timing_iterations; i++) {
    auto t1 = high_resolution_clock::now();
    foo(args...).wait();
    auto t2 = high_resolution_clock::now();
    total_time += duration_cast<microseconds>(t2 - t1).count();
  }
  // Returns GigaFlops
  double average_time_in_seconds = (total_time / timing_iterations) * 1e-6;
  return {(static_cast<double>(flops) / average_time_in_seconds) * 1e-9, total_time * 1e-3};
}