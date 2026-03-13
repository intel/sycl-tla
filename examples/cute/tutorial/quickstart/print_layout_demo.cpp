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

// This example demonstrates cute::print_layout() function
// print_layout() displays any rank-2 layout as a plain text table
// showing the mapping from coordinates to indices

#include <cute/tensor.hpp>

using namespace cute;

int main(int argc, char** argv) {
  
  printf("\n=== CuTe print_layout() Demo ===\n\n");
  printf("print_layout() displays only rank-2 layouts as tables showing coordinate->index mapping.\nUse this for initial understanding. Refer layout examples for more complex cases.\n\n");
  
  // Example 1: Column-major layout (standard matrix layout)
  printf("1. Column-major 4x8 layout (stride (1,4)):\n");
  printf("   Each column is contiguous in memory\n\n");
  auto col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
  print_layout(col_major);
  printf("\n\n");
  
  // Example 2: Row-major layout
  printf("2. Row-major 4x8 layout (stride (8,1)):\n");
  printf("   Each row is contiguous in memory\n\n");
  auto row_major = make_layout(make_shape(4, 8), make_stride(8, 1));
  print_layout(row_major);
  printf("\n\n");
  
  // Example 3: Transposed layout
  printf("3. Transposed 6x4 layout (was 4x6 column-major):\n");
  printf("   Swapping dimensions\n\n");
  auto transposed = make_layout(make_shape(6, 4), make_stride(4, 1));
  print_layout(transposed);
  printf("\n\n");
  
  // Example 4: Diagonal-stride layout
  printf("4. Diagonal access pattern 6x6:\n");
  printf("   Non-standard stride pattern\n\n");
  auto diagonal = make_layout(make_shape(6, 6), make_stride(1, 7));
  print_layout(diagonal);
  printf("\n\n");
  
  // Example 5: Strided layout (every other element)
  printf("5. Strided layout 4x4 (stride (2,8)):\n");
  printf("   Skips every other element\n\n");
  auto strided = make_layout(make_shape(4, 4), make_stride(2, 8));
  print_layout(strided);
  printf("\n\n");
  
  // Example 6: Smaller layout for clarity
  printf("6. Small 3x3 column-major layout:\n");
  printf("   Easy to verify mapping by hand\n\n");
  auto small = make_layout(make_shape(3, 3), make_stride(1, 3));
  print_layout(small);
  printf("\n\n");
  
  // Example 7: Blocked layout
  printf("7. Blocked 8x8 layout with stride 16:\n");
  printf("   Useful for cache-blocked algorithms\n\n");
  auto blocked = make_layout(make_shape(8, 8), make_stride(1, 16));
  print_layout(blocked);
  printf("\n\n");
  
  printf("=== print_layout() Demo Complete ===\n\n");
  printf("Key observations:\n");
  printf("- Numbers in the table show the linear index in memory\n");
  printf("- Rows represent the first dimension, columns the second\n");
  printf("- You can visualize data access patterns and memory layout\n");
  printf("- Useful for debugging tiling and partitioning strategies\n\n");
  
  return 0;
}
