/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

// This example demonstrates cute::print() function for printing various CuTe objects
// cute::print() works on both host and device for Layouts, Tensors, Shapes, Strides, etc.

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// Device kernel that prints CuTe objects on device (only thread 0)
void print_kernel(sycl::nd_item<1> item) {
  // Only print from thread 0 to avoid duplicate output
  if (thread0()) {
    // Print various CuTe types
    printf("\n=== CuTe Print Demo (Device) ===\n\n");
    
    // 1. Print a simple shape
    printf("1. Shape (4,8):\n");
    auto shape = make_shape(4, 8);
    print(shape);
    printf("\n\n");
    
    // 2. Print a stride
    printf("2. Stride (1,4):\n");
    auto stride = make_stride(1, 4);
    print(stride);
    printf("\n\n");
    
    // 3. Print a layout
    printf("3. Layout ((4,8),(1,4)):\n");
    auto layout = make_layout(make_shape(4, 8), make_stride(1, 4));
    print(layout);
    printf("\n\n");
    
    // 4. Print a hierarchical shape
    printf("4. Hierarchical Shape ((2,2),(4,2)):\n");
    auto hier_shape = make_shape(make_shape(2, 2), make_shape(4, 2));
    print(hier_shape);
    printf("\n\n");
    
    // 5. Print a hierarchical layout
    printf("5. Hierarchical Layout (((2,2),(4,2)),((1,8),(2,16))):\n");
    auto hier_layout = make_layout(
        make_shape(make_shape(2, 2), make_shape(4, 2)),
        make_stride(make_stride(1, 8), make_stride(2, 16))
    );
    print(hier_layout);
    printf("\n\n");
  }
}

void host_print_examples() {
  printf("\n=== CuTe Print Demo (Host) ===\n\n");
  
  // 1. Print a simple integer
  printf("1. Integer: ");
  print(42);
  printf("\n\n");
  
  // 2. Print a shape
  printf("2. Shape (4,8,16):\n");
  auto shape = make_shape(4, 8, 16);
  print(shape);
  printf("\n\n");
  
  // 3. Print a stride
  printf("3. Stride (1,4,32):\n");
  auto stride = make_stride(1, 4, 32);
  print(stride);
  printf("\n\n");
  
  // 4. Print a layout
  printf("4. Column-major Layout for 4x8 matrix:\n");
  auto col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
  print(col_major);
  printf("\n\n");
  
  printf("5. Row-major Layout for 4x8 matrix:\n");
  auto row_major = make_layout(make_shape(4, 8), make_stride(8, 1));
  print(row_major);
  printf("\n\n");
  
  // 6. Print a tensor
  printf("6. Simple Tensor:\n");
  float* data = new float[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto tensor = make_tensor(data, make_layout(make_shape(3, 4), make_stride(1, 3)));
  print(tensor);
  delete[] data;
  printf("\n\n");
  
  // 7. Print coordinates
  printf("7. Coordinates (2,3,1):\n");
  auto coord = make_coord(2, 3, 1);
  print(coord);
  printf("\n\n");
}

int main(int argc, char** argv) {
  
  // First, demonstrate printing on the host
  host_print_examples();
  
  // Then demonstrate printing on the device
  printf("\n--- Launching device kernel for device printing ---\n");
  
  sycl::queue q(sycl::default_selector_v);
  
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
                   [=](sycl::nd_item<1> item) {
      print_kernel(item);
    });
  }).wait();
  
  printf("\n=== Print Demo Complete ===\n\n");
  
  return 0;
}
