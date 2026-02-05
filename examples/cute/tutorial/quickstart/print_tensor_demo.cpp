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

// This example demonstrates cute::print_tensor() function
// print_tensor() displays rank-1, rank-2, rank-3, or rank-4 tensors as formatted tables
// showing the actual values stored in the tensor

#include <cute/tensor.hpp>

using namespace cute;

int main(int argc, char** argv) {
  
  printf("\n=== CuTe print_tensor() Demo ===\n\n");
  printf("print_tensor() displays tensor values in multidimensional table format\n\n");
  
  // Example 1: Rank-1 tensor (vector)
  printf("1. Rank-1 tensor (vector of 8 elements):\n\n");
  float* vec_data = new float[8]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  auto vector = make_tensor(vec_data, make_layout(make_shape(8)));
  print_tensor(vector);
  delete[] vec_data;
  printf("\n\n");
  
  // Example 2: Rank-2 tensor (matrix) - Column major
  printf("2. Rank-2 tensor (4x6 matrix, column-major):\n\n");
  float* mat_data = new float[24];
  for (int i = 0; i < 24; i++) mat_data[i] = static_cast<float>(i);
  auto matrix = make_tensor(mat_data, make_layout(make_shape(4, 6), make_stride(1, 4)));
  print_tensor(matrix);
  printf("\n\n");
  
  // Example 3: Rank-2 tensor (matrix) - Row major
  printf("3. Rank-2 tensor (4x6 matrix, row-major):\n");
  printf("   Note how values change compared to column-major\n\n");
  auto matrix_row = make_tensor(mat_data, make_layout(make_shape(4, 6), make_stride(6, 1)));
  print_tensor(matrix_row);
  delete[] mat_data;
  printf("\n\n");
  
  // Example 4: Rank-3 tensor (3D array)
  printf("4. Rank-3 tensor (2x3x4 array):\n");
  printf("   Useful for batch processing or 3D data\n\n");
  float* tensor3d_data = new float[24];
  for (int i = 0; i < 24; i++) tensor3d_data[i] = static_cast<float>(i * 10);
  auto tensor3d = make_tensor(tensor3d_data, make_shape(2, 3, 4));
  print_tensor(tensor3d);
  delete[] tensor3d_data;
  printf("\n\n");
  
  // Example 5: Simple 4x8 matrix
  printf("5. Simple 4x8 matrix:\n");
  printf("   Standard row-major layout\n\n");
  float* simple_data = new float[32];
  for (int i = 0; i < 32; i++) simple_data[i] = static_cast<float>(i);
  auto simple_tensor = make_tensor(simple_data, make_layout(make_shape(4, 8), make_stride(8, 1)));
  print_tensor(simple_tensor);
  delete[] simple_data;
  printf("\n\n");
  
  // Example 6: Small example for verification
  printf("6. Small 3x3 identity-like pattern:\n\n");
  float* small_data = new float[9]{1.0f, 0.0f, 0.0f,
                                   0.0f, 2.0f, 0.0f,
                                   0.0f, 0.0f, 3.0f};
  auto small_tensor = make_tensor(small_data, make_layout(make_shape(3, 3), make_stride(1, 3)));
  print_tensor(small_tensor);
  delete[] small_data;
  printf("\n\n");
  
  // Example 7: Demonstrate after a copy operation
  printf("7. Tensor after initialization (simulating copy result):\n");
  printf("   This is useful for verifying data after copy operations\n\n");
  float* result_data = new float[12];
  for (int i = 0; i < 12; i++) result_data[i] = static_cast<float>(i * 2 + 1);
  auto result_tensor = make_tensor(result_data, make_layout(make_shape(3, 4), make_stride(1, 3)));
  print_tensor(result_tensor);
  delete[] result_data;
  printf("\n\n");
  
  printf("=== print_tensor() Demo Complete ===\n\n");
  printf("Key observations:\n");
  printf("- Displays actual values in the tensor\n");
  printf("- Works for rank-1 through rank-4 tensors\n");
  printf("- Excellent for debugging after copy/transform operations\n");
  printf("- Shows how layout affects the logical view of data\n");
  printf("- Use to verify tiled copies and memory operations\n\n");
  
  return 0;
}
