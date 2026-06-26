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

// Tutorial 5: Layout Compatibility
// Demonstrates congruent(), compatible(), and when layouts can be used together

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Congruence and Basic Compatibility. Note the congreunt check is applicable for compile time as well as long as the shapes and strides are static===\n\n");
  
  printf("Example 1: Congruent shapes and strides\n");
  printf("  congruent(shape, stride): checks same rank AND same hierarchy (tuple vs scalar)\n");
  printf("  at every level. Static vs dynamic integer type does NOT matter.\n");
  auto shape1 = make_shape(4, 6);
  auto stride1 = make_stride(1, 4);
  printf("  Shape:  "); print(shape1); printf("  rank=%d  depth=%d\n", int(rank(shape1)), int(depth(shape1)));
  printf("  Stride: "); print(stride1); printf("  rank=%d  depth=%d\n", int(rank(stride1)), int(depth(stride1)));
  printf("  Congruent: %s  (same rank=2, same depth=1)\n\n", congruent(shape1, stride1) ? "YES" : "NO");
  
  printf("Example 2: Non-congruent (mismatched ranks)\n");
  printf("  Each mode of shape needs exactly one stride -- ranks must match.\n");
  auto shape2 = make_shape(4, 6, 3);
  auto stride2 = make_stride(1, 4);
  printf("  Shape:  "); print(shape2); printf("  rank=%d\n", int(rank(shape2)));
  printf("  Stride: "); print(stride2); printf("  rank=%d\n", int(rank(stride2)));
  printf("  Congruent: %s  (rank %d != %d)\n\n", congruent(shape2, stride2) ? "YES" : "NO",
         int(rank(shape2)), int(rank(stride2)));
  
  printf("Example 3: Same size, different shapes\n");
  printf("  size(Layout) = product of all shape elements (domain cardinality).\n");
  printf("  Two layouts with the same size share the same domain, but per-mode\n");
  printf("  extents differ -- they are NOT congruent as layouts.\n");
  auto layout_a = make_layout(make_shape(4, 6));
  auto layout_b = make_layout(make_shape(3, 8));
  printf("  Layout A: "); print(layout_a); printf("  size=%d\n", int(size(layout_a)));
  printf("  Layout B: "); print(layout_b); printf("  size=%d\n", int(size(layout_b)));
  printf("  Same size=%d but shape(A)!= shape(B) => not structurally identical\n\n",
         int(size(layout_a)));
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Hierarchical Congruence ===\n\n");
  
  printf("Example 4: Hierarchical congruence\n");
  printf("  depth() measures nesting: 0=scalar, 1=flat tuple, 2=tuple-of-tuples.\n");
  printf("  congruent requires matching structure at EVERY level of the hierarchy.\n");
  auto hier_shape = make_shape(2, make_shape(3, 4));
  auto hier_stride = make_stride(1, make_stride(2, 6));
  printf("  Shape:  "); print(hier_shape); printf("  rank=%d depth=%d\n",
         int(rank(hier_shape)), int(depth(hier_shape)));
  printf("  Stride: "); print(hier_stride); printf("  rank=%d depth=%d\n",
         int(rank(hier_stride)), int(depth(hier_stride)));
  printf("  Congruent: %s (rank and depth both match)\n\n",
         congruent(hier_shape, hier_stride) ? "YES" : "NO");
  
  printf("Example 5: Non-congruent hierarchy\n");
  printf("  Shape mode-1 is a nested tuple (3,4); stride mode-1 must also be a tuple.\n");
  printf("  A scalar stride cannot distinguish the two sub-strides inside (3,4).\n");
  auto flat_stride = make_stride(1, 2);
  printf("  Shape:  "); print(hier_shape); printf("  depth=%d (nested)\n", int(depth(hier_shape)));
  printf("  Stride: "); print(flat_stride); printf("  depth=%d (flat)\n", int(depth(flat_stride)));
  printf("  Congruent: %s (depth mismatch at mode-1)\n\n",
         congruent(hier_shape, flat_stride) ? "YES" : "NO");
  
  printf("Example 6: Compatible layouts for composition\n");
  printf("  For R = A o B: B's output values must land inside A's domain.\n");
  printf("  Condition: cosize(B) <= size(A).  cosize(L) = L(size(L)-1)+1.\n");
  auto layout_4x6 = make_layout(make_shape(4, 6));
  auto layout_6x2 = make_layout(make_shape(6, 2));
  printf("  Layout A: "); print(layout_4x6); printf("  size=%d\n", int(size(layout_4x6)));
  printf("  Layout B: "); print(layout_6x2);
  printf("  size=%d  cosize=%d\n", int(size(layout_6x2)), int(cosize(layout_6x2)));
  printf("  cosize(B)=%d <= size(A)=%d => compatible for composition\n\n",
         int(cosize(layout_6x2)), int(size(layout_4x6)));
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Advanced Compatibility Checks ===\n\n");
  
  printf("Example 7: Mixed static/dynamic congruence\n");
  printf("  Static integers (Int<N>) and dynamic integers (int) are interchangeable.\n");
  printf("  congruent only inspects structure, not the integer kind.\n");
  printf("  When both shape and stride are static, the check happens at compile time.\n");
  auto static_shape = make_shape(Int<4>{}, Int<6>{});
  auto dynamic_stride = make_stride(1, 4);
  printf("  Shape   (static) : "); print(static_shape);   printf("\n");
  printf("  Stride  (dynamic): "); print(dynamic_stride); printf("\n");
  printf("  Congruent: %s (structure matches regardless of static/dynamic)\n\n",
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
  printf("  For element-wise copy: shape(src) must equal shape(dst).\n");
  printf("  Strides can differ -- that just means a different memory traversal order.\n");
  auto src_layout = make_layout(make_shape(8, 8));                 // col-major
  auto dst_layout = make_layout(make_shape(8, 8), LayoutRight{});  // row-major
  printf("  Source (col-major): "); print(src_layout); printf("\n");
  printf("  Dest   (row-major): "); print(dst_layout); printf("\n");
  printf("  shape match: %s  stride match: %s\n",
         (shape(src_layout) == shape(dst_layout)) ? "YES" : "NO",
         (stride(src_layout) == stride(dst_layout)) ? "YES" : "NO");
  printf("  src(2,3)=%d  dst(2,3)=%d  (same logical coord, different address)\n",
         int(src_layout(2,3)), int(dst_layout(2,3)));
  printf("  Compatible for copy: same domain (size=%d), different memory layout\n\n",
         int(size(src_layout)));
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
