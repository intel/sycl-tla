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

// Tutorial 1: Rank, Depth, Shape, Stride, Size, and Cosize
// 
// This example demonstrates the fundamental properties of CuTe layouts:
// - rank: number of elements in a layout (Tuple)
// - depth: level of hierarchical nesting. For example: Number of tuple layers. For ingle integer layouts, depth is 0.
// - shape: the coordinate space dimensions
// - stride: how coordinates map to indices / Space betweeen indices
// - size: product of all shape dimensions (domain size)
// - cosize: size of the codomain (maximum index + 1)

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic 1D and 2D Layouts ===\n\n");
  
  // Example 1: Simple 1D layout (rank-1)
  printf("Example 1: Simple 1D layout of 8 elements\n");
  auto layout_1d = make_layout(Int<8>{});
  printf("Layout: ");
  print(layout_1d);
  printf("\n");
  printf("  rank:   %d\n", int(rank(layout_1d)));
  printf("  depth:  %d\n", int(depth(layout_1d)));
  printf("  shape:  ");
  print(shape(layout_1d));
  printf("\n");
  printf("  stride: ");
  print(stride(layout_1d));
  printf("\n");
  printf("  size:   %d (domain size)\n", int(size(layout_1d)));
  printf("  cosize: %d (codomain size)\n\n", int(cosize(layout_1d)));
  
  // Example 2: Simple 2D layout (rank-2, column-major)
  printf("Example 2: 4x6 column-major layout\n");
  auto layout_2d = make_layout(make_shape(4, 6));
  printf("Layout: ");
  print(layout_2d);
  printf("\n");
  printf("  rank:   %d (2D has 2 modes)\n", int(rank(layout_2d)));
  printf("  depth:  %d (flat, no nesting)\n", int(depth(layout_2d)));
  printf("  shape:  ");
  print(shape(layout_2d));
  printf("\n");
  printf("  stride: ");
  print(stride(layout_2d));
  printf(" (column-major: stride-1 in first mode)\n");
  printf("  size:   %d (4 * 6 = 24)\n", int(size(layout_2d)));
  printf("  cosize: %d (last element is at index 23)\n\n", int(cosize(layout_2d)));
  
  // Example 3: Row-major 2D layout
  printf("Example 3: 4x6 row-major layout\n");
  auto layout_row = make_layout(make_shape(4, 6), LayoutRight{});
  printf("Layout: ");
  print(layout_row);
  printf("\n");
  printf("  rank:   %d\n", int(rank(layout_row)));
  printf("  depth:  %d\n", int(depth(layout_row)));
  printf("  stride: ");
  print(stride(layout_row));
  printf(" (row-major: stride-1 in second mode)\n");
  printf("  size:   %d (same domain size)\n", int(size(layout_row)));
  printf("  cosize: %d (same codomain size)\n\n", int(cosize(layout_row)));
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Hierarchical Layouts ===\n\n");
  
  // Example 4: Hierarchical 2D layout (depth > 1)
  printf("Example 4: Hierarchical layout (2, (2,3))\n");
  auto hier_shape = make_shape(2, make_shape(2, 3));
  auto hier_layout = make_layout(hier_shape);
  printf("Layout: ");
  print(hier_layout);
  printf("\n");
  printf("  rank:   %d (outer tuple has 2 elements)\n", int(rank(hier_layout)));
  printf("  depth:  %d (has nested tuple)\n", int(depth(hier_layout)));
  printf("  shape:  ");
  print(shape(hier_layout));
  printf("\n");
  printf("  stride: ");
  print(stride(hier_layout));
  printf("\n");
  printf("  size:   %d (2 * 2 * 3 = 12, product of all)\n", int(size(hier_layout)));
  printf("  cosize: %d\n\n", int(cosize(hier_layout)));
  print_layout(hier_layout);
  print_layout(make_layout(make_shape(2, 6), make_stride(1, 2)));
  
  // Example 5: Custom stride layout with gaps
  printf("Example 5: Layout with gaps (stride creates holes)\n");
  auto gap_layout = make_layout(make_shape(4, 3), make_stride(3, 12));
  printf("Layout: ");
  print(gap_layout);
  printf("\n");
  printf("  rank:   %d\n", int(rank(gap_layout)));
  printf("  size:   %d (domain: 4 * 3 = 12 elements)\n", int(size(gap_layout)));
  printf("  cosize: %d (codomain: last index + 1 = 3*3 + 12*2 + 1)\n", int(cosize(gap_layout)));
  printf("  Note: size < cosize means the layout has 'gaps'\n\n");
  
  // Example 6: Multi-level hierarchy
  printf("Example 6: Three-level hierarchy ((2,2),(3,2))\n");
  auto deep_shape = make_shape(make_shape(2, 2), make_shape(3, 2));
  auto deep_layout = make_layout(deep_shape);
  printf("Layout: ");
  print(deep_layout);
  printf("\n");
  printf("  rank:   %d (top level has 2 modes)\n", int(rank(deep_layout)));
  printf("  depth:  %d (two levels of nesting)\n", int(depth(deep_layout)));
  printf("  size:   %d (2*2*3*2 = 24)\n", int(size(deep_layout)));
  printf("  cosize: %d\n\n", int(cosize(deep_layout)));
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Complex Hierarchical and Strided Layouts ===\n\n");
  
  // Example 7: Complex tiled layout
  printf("Example 7: Tiled layout for 8x8 matrix with 2x4 tiles\n");
  auto tile_shape = make_shape(make_shape(2, 4), make_shape(4, 2));
  auto tile_stride = make_stride(make_stride(1, 2), make_stride(8, 32));
  auto tiled_layout = make_layout(tile_shape, tile_stride);
  printf("Layout: ");
  print(tiled_layout);
  printf("\n");
  printf("  Interpretation: 4x2 grid of 2x4 tiles\n");
  printf("  rank:   %d\n", int(rank(tiled_layout)));
  printf("  depth:  %d\n", int(depth(tiled_layout)));
  printf("  size:   %d ((2*4) * (4*2) = 64)\n", int(size(tiled_layout)));
  printf("  cosize: %d\n\n", int(cosize(tiled_layout)));
  
  // Example 8: Strided sub-sampling layout
  printf("Example 8: Strided sub-sampling (every other element)\n");
  auto subsample = make_layout(make_shape(3, 4), make_stride(2, 10));
  printf("Layout: ");
  print(subsample);
  printf("\n");
  printf("  size:   %d (samples 12 elements)\n", int(size(subsample)));
  printf("  cosize: %d (spans much larger range)\n", int(cosize(subsample)));
  printf("  Sparsity: %.2f%% (size/cosize ratio)\n\n", 
         100.0 * size(subsample) / cosize(subsample));
  
  // Example 9: Deep hierarchy with mixed static/dynamic
  printf("Example 9: Mixed static/dynamic deep hierarchy\n");
  auto mixed_shape = make_shape(
    Int<2>{},
    make_shape(4, make_shape(Int<3>{}, 2))
  );
  auto mixed_layout = make_layout(mixed_shape);
  printf("Layout: ");
  print(mixed_layout);
  printf("\n");
  printf("  rank:   %d\n", int(rank(mixed_layout)));
  printf("  depth:  %d (three levels deep)\n", int(depth(mixed_layout)));
  printf("  size:   %d (2 * 4 * 3 * 2 = 48)\n", int(size(mixed_layout)));
  printf("  cosize: %d\n", int(cosize(mixed_layout)));
  printf("  Note: Mix of compile-time (Int<N>) and run-time (int) dimensions\n\n");
  
  // Example 10: Transposed blocked layout
  printf("Example 10: Transposed blocked layout\n");
  auto block_shape = make_shape(make_shape(4, 2), make_shape(2, 4));
  auto block_stride = make_stride(make_stride(16, 1), make_stride(2, 64));
  auto blocked = make_layout(block_shape, block_stride);
  printf("Layout: ");
  print(blocked);
  printf("\n");
  printf("  rank:   %d\n", int(rank(blocked)));
  printf("  depth:  %d\n", int(depth(blocked)));
  printf("  size:   %d\n", int(size(blocked)));
  printf("  cosize: %d\n", int(cosize(blocked)));
  printf("  Cache-friendly blocked memory access pattern\n\n");
}

int main() {
  printf("\n");
  printf("======================================================================\n");
  printf(" CuTe Layout Tutorial 1: Rank, Depth, Shape, Stride, Size, Cosize\n");
  printf("======================================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("======================================================================\n");
  printf(" Summary:\n");
  printf("   rank(L):   Number of modes (dimensions) in the layout\n");
  printf("   depth(L):  Level of hierarchical nesting (0 = flat)\n");
  printf("   shape(L):  Coordinate space dimensions\n");
  printf("   stride(L): How coordinates map to linear indices\n");
  printf("   size(L):   Product of shape (total elements in domain)\n");
  printf("   cosize(L): Maximum index + 1 (codomain size)\n");
  printf("   \n");
  printf("   Key insight: size <= cosize\n");
  printf("   When size < cosize, the layout has 'gaps' in memory\n");
  printf("======================================================================\n\n");
  
  return 0;
}
