/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Tutorial 5: Layout Compatibility
// Demonstrates congruent(), compatible(), and when layouts can be used together

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Congruence and Basic Compatibility. Note the congreunt check is applicable for compile time as well as long as the shapes and strides are static===\n\n");
  
  printf("Example 1: Congruent shapes and strides\n");
  auto shape1 = make_shape(4, 6);
  auto stride1 = make_stride(1, 4);
  printf("  Shape:  "); print(shape1); printf("\n");
  printf("  Stride: "); print(stride1); printf("\n");
  printf("  Congruent: %s\n\n", congruent(shape1, stride1) ? "YES" : "NO");
  
  printf("Example 2: Non-congruent (mismatched ranks)\n");
  auto shape2 = make_shape(4, 6, 3);
  auto stride2 = make_stride(1, 4);
  printf("  Shape:  "); print(shape2); printf(" (rank 3)\n");
  printf("  Stride: "); print(stride2); printf(" (rank 2)\n");
  printf("  Congruent: %s\n\n", congruent(shape2, stride2) ? "YES" : "NO");
  
  printf("Example 3: Same size, different shapes\n");
  auto layout_a = make_layout(make_shape(4, 6));
  auto layout_b = make_layout(make_shape(3, 8));
  printf("  Layout A: "); print(layout_a); printf(" size=%d\n", int(size(layout_a)));
  printf("  Layout B: "); print(layout_b); printf(" size=%d\n", int(size(layout_b)));
  printf("  Same size but different shapes\n\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Hierarchical Congruence ===\n\n");
  
  printf("Example 4: Hierarchical congruence\n");
  auto hier_shape = make_shape(2, make_shape(3, 4));
  auto hier_stride = make_stride(1, make_stride(2, 6));
  printf("  Shape:  "); print(hier_shape); printf("\n");
  printf("  Stride: "); print(hier_stride); printf("\n");
  printf("  Congruent: %s (matching hierarchy)\n\n",
         congruent(hier_shape, hier_stride) ? "YES" : "NO");
  
  printf("Example 5: Non-congruent hierarchy\n");
  auto flat_stride = make_stride(1, 2);
  printf("  Shape:  "); print(hier_shape); printf(" (nested)\n");
  printf("  Stride: "); print(flat_stride); printf(" (flat)\n");
  printf("  Congruent: %s (hierarchy mismatch)\n\n",
         congruent(hier_shape, flat_stride) ? "YES" : "NO");
  
  printf("Example 6: Compatible layouts for composition\n");
  auto layout_4x6 = make_layout(make_shape(4, 6));
  auto layout_6x2 = make_layout(make_shape(6, 2));
  printf("  Layout A: "); print(layout_4x6); printf("\n");
  printf("  Layout B: "); print(layout_6x2); printf("\n");
  printf("  Can compose if size(A) matches shape elements of B\n\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Advanced Compatibility Checks ===\n\n");
  
  printf("Example 7: Mixed static/dynamic congruence\n");
  auto static_shape = make_shape(Int<4>{}, Int<6>{});
  auto dynamic_stride = make_stride(1, 4);
  printf("  Shape:  "); print(static_shape); printf(" (static)\n");
  printf("  Stride: "); print(dynamic_stride); printf(" (dynamic)\n");
  printf("  Congruent: %s (types don't need to match)\n\n",
         congruent(static_shape, dynamic_stride) ? "YES" : "NO");
  
  printf("Example 8: Deep hierarchy congruence\n");
  auto deep_shape = make_shape(2, make_shape(3, make_shape(4, 5)));
  auto deep_stride = make_stride(1, make_stride(2, make_stride(6, 24)));
  printf("  Shape:  "); print(deep_shape); printf("\n");
  printf("  Stride: "); print(deep_stride); printf("\n");
  printf("  Congruent: %s\n", congruent(deep_shape, deep_stride) ? "YES" : "NO");
  printf("  Depth: %d\n\n", int(depth(deep_shape)));
  
  printf("Example 9: Validating layout construction\n");
  auto valid_layout = make_layout(make_shape(4, 6, 8), make_stride(1, 4, 24));
  printf("  Layout: "); print(valid_layout); printf("\n");
  printf("  Shape and stride are congruent by construction\n");
  printf("  static_assert(congruent(shape, stride)) would pass\n\n");
  
  printf("Example 10: Compatibility for tensor operations\n");
  auto src_layout = make_layout(make_shape(8, 8));
  auto dst_layout = make_layout(make_shape(8, 8), LayoutRight{});
  printf("  Source:      "); print(src_layout); printf("\n");
  printf("  Destination: "); print(dst_layout); printf("\n");
  printf("  Same shape, different strides\n");
  printf("  Compatible for copy if shapes match\n\n");
}

int main() {
  printf("\n");
  printf("============================================================\n");
  printf(" CuTe Tutorial 5: Layout Compatibility\n");
  printf("============================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("============================================================\n");
  printf(" Summary:\n");
  printf("   congruent(shape, stride): Checks structural match\n");
  printf("   - Same rank and hierarchy required\n");
  printf("   - Type (static/dynamic) doesn't matter\n");
  printf("   \n");
  printf("   Layouts are compatible when:\n");
  printf("   - Shapes match for element-wise operations\n");
  printf("   - Size matches for flat access\n");
  printf("   - Congruence ensures valid construction\n");
  printf("============================================================\n\n");
  
  return 0;
}
