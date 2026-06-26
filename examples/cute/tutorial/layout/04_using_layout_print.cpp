/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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

// Tutorial 4: Using Layout and Print 1D, 2D
// Demonstrates how to use layouts to map coordinates to indices and visualize them

#include <cute/tensor.hpp>

using namespace cute;

template <class Layout>
void print_1d_mapping(Layout const& layout) {
  printf("  Index mapping: [");
  for (int i = 0; i < size(layout); ++i) {
    printf("%d", int(layout(i)));
    if (i < size(layout) - 1) printf(", ");
  }
  printf("]\n");
}

void easy_examples() {
  printf("\n=== EASY EXAMPLES: 1D and Simple 2D Layouts ===\n\n");
  
  printf("Example 1: Simple 1D layout\n");
  auto layout_1d = make_layout(8);
  printf("Layout: "); print(layout_1d); printf("\n");
  print_1d_mapping(layout_1d);
  printf("\n");
  
  printf("Example 2: 3x4 column-major layout\n");
  auto col_major = make_layout(make_shape(3, 4));
  printf("Layout: "); print(col_major); printf("\n");
  printf("  Visualization as 2D table:\n");
  print_layout(col_major);
  printf("\n");
  
  printf("Example 3: 3x4 row-major layout\n");
  auto row_major = make_layout(make_shape(3, 4), LayoutRight{});
  printf("Layout: "); print(row_major); printf("\n");
  print_layout(row_major);
  printf("\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Custom Strides and Padding ===\n\n");
  
  printf("Example 4: Strided 1D layout (every 2nd element)\n");
  auto strided = make_layout(make_shape(6), make_stride(2));
  printf("Layout: "); print(strided); printf("\n");
  print_1d_mapping(strided);
  printf("\n");
  
  printf("Example 5: Padded 2D layout\n");
  auto padded = make_layout(make_shape(4, 4), make_stride(1, 6));
  printf("Layout: "); print(padded); printf("\n");
  printf("  Size: %d, Cosize: %d (padding: %d)\n",
         int(size(padded)), int(cosize(padded)),
         int(cosize(padded) - size(padded)));
  print_layout(padded);
  printf("\n");
  
  printf("Example 6: Transposed view\n");
  auto transposed = make_layout(make_shape(5, 3), make_stride(5, 1));
  printf("Layout: "); print(transposed); printf("\n");
  print_layout(transposed);
  printf("\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Complex Access Patterns ===\n\n");
  
  printf("Example 7: Diagonal access pattern\n");
  auto diagonal = make_layout(make_shape(6, 6), make_stride(1, 7));
  printf("Layout: "); print(diagonal); printf("\n");
  print_layout(diagonal);
  printf("\n");
  
  printf("Example 8: Blocked layout with 8 elements of discrete blocks\n");
  auto blocked = make_layout(make_shape(8, 8), make_stride(1, 16));
  printf("Layout: "); print(blocked); printf("\n");
  printf("  First 8x8 block:\n");
  print_layout(blocked);
  printf("\n");
  
  printf("Example 9: Checkerboard pattern\n");
  auto checker = make_layout(make_shape(4, 4), make_stride(2, 8));
  printf("Layout: "); print(checker); printf("\n");
  print_layout(checker);
  printf("\n");
}

int main() {
  printf("\n");
  printf("============================================================\n");
  printf(" CuTe Tutorial 4: Using Layout and Print 1D, 2D\n");
  printf("============================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("============================================================\n");
  printf(" Key Points:\n");
  printf("   - layout(coord) maps coordinates to linear index\n");
  printf("   - print_layout() visualizes 2D layouts as tables\n");
  printf("   - Strides control memory access patterns\n");
  printf("============================================================\n\n");
  
  return 0;
}
