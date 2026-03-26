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

// Tutorial 7: Index Mapping  
// Demonstrates mapping from coordinates to linear indices using layout(coord) or crd2idx

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic Coordinate to Index ===\n\n");
  
  printf("Example 1: 1D coordinate to index\n");
  auto layout_1d = make_layout(8);
  printf("Layout: "); print(layout_1d); printf("\n");
  for (int i = 0; i < 8; ++i) {
    int idx = crd2idx(i, shape(layout_1d), stride(layout_1d));
    printf("  coord %d -> index %d\n", i, idx);
  }
  printf("\n");
  
  printf("Example 2: 2D coordinate to index (column-major)\n");
  auto col_major = make_layout(make_shape(3, 4));
  printf("Layout: "); print(col_major); printf("\n");
  printf("  Visual representation (row coord = i, col coord = j):\n");
  printf("       0     1     2     3    <== col j\n");
  printf("    +-----+-----+-----+-----+\n");
  for (int i = 0; i < 3; ++i) {
    printf(" %d  ", i);
    for (int j = 0; j < 4; ++j) {
      int idx = crd2idx(make_coord(i, j), shape(col_major), stride(col_major));
      printf("|  %2d ", idx);
    }
    printf("|\n");
    printf("    +-----+-----+-----+-----+\n");
  }
  printf("  ^\n  row i\n\n");
  
  printf("Example 3: Using layout(coord) shorthand\n");
  auto layout = make_layout(make_shape(4, 5));
  printf("Layout: "); print(layout); printf("\n");
  printf("  layout(coord) is shorthand for crd2idx:\n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 5; ++j) {
      int idx1 = layout(i, j);
      int idx2 = crd2idx(make_coord(i, j), shape(layout), stride(layout));
      printf("  layout(%d,%d) = %d, crd2idx((%d,%d)) = %d\n", i, j, idx1, i, j, idx2);
    }
  }
  printf("\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Different Stride Patterns ===\n\n");
  
  printf("Example 4: Row-major coordinate to index\n");
  auto row_major = make_layout(make_shape(3, 4), LayoutRight{});
  printf("Layout: "); print(row_major); printf("\n");
  printf("  Visual representation (row coord = i, col coord = j):\n");
  printf("       0     1     2     3    <== col j\n");
  printf("    +-----+-----+-----+-----+\n");
  for (int i = 0; i < 3; ++i) {
    printf(" %d  ", i);
    for (int j = 0; j < 4; ++j) {
      int idx = crd2idx(make_coord(i, j), shape(row_major), stride(row_major));
      printf("|  %2d ", idx);
    }
    printf("|\n");
    printf("    +-----+-----+-----+-----+\n");
  }
  printf("  ^\n  row i\n\n");
  
  printf("Example 5: Custom strided layout\n");
  auto strided = make_layout(make_shape(3, 4), make_stride(1, 5));
  printf("Layout: "); print(strided); printf("\n");
  printf("  Note: stride=(1,5) creates gaps (cosize > size)\n");
  printf("  Visual representation (indices have gaps):\n");
  printf("       0     1     2     3    <== col j\n");
  printf("    +-----+-----+-----+-----+\n");
  for (int i = 0; i < 3; ++i) {
    printf(" %d  ", i);
    for (int j = 0; j < 4; ++j) {
      int idx = crd2idx(make_coord(i, j), shape(strided), stride(strided));
      printf("|  %2d ", idx);
    }
    printf("|\n");
    printf("    +-----+-----+-----+-----+\n");
  }
  printf("  ^\n  row i\n");
  printf("  Size: %d, Cosize: %d (indices 0-20 but only 12 used)\n\n", 
         int(size(strided)), int(cosize(strided)));
  
  printf("Example 6: 3D coordinate to index mapping\n");
  auto layout_3d = make_layout(make_shape(2, 3, 4));
  printf("Layout: "); print(layout_3d); printf("\n");
  printf("  Sample coordinate to index mappings:\n");
  for (int i : {0, 1}) {
    for (int j : {0, 1, 2}) {
      for (int k : {0, 2}) {
        int idx = crd2idx(make_coord(i, j, k), shape(layout_3d), stride(layout_3d));
        printf("  coord(%d,%d,%d) -> index %d\n", i, j, k, idx);
      }
    }
  }
  printf("\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Hierarchical Index Mapping ===\n\n");
  
  printf("Example 7: Hierarchical coordinate to index\n");
  auto hier = make_layout(make_shape(make_shape(2, 3), make_shape(4, 2)));
  printf("Layout: "); print(hier); printf("\n");
  printf("  Hierarchical coordinates map to linear indices:\n");
  for (int i : {0, 1}) {
    for (int j : {0, 1, 2}) {
      for (int k : {0, 2}) {
        for (int l : {0, 1}) {
          auto coord = make_coord(make_coord(i, j), make_coord(k, l));
          int idx = crd2idx(coord, shape(hier), stride(hier));
          printf("  coord((%d,%d),(%d,%d)) -> index %d\n", i, j, k, l, idx);
        }
      }
    }
  }
  printf("\n");
  
  printf("Example 8: Equivalent coordinate representations\n");
  auto shape_h = make_shape(_3{}, make_shape(_2{}, _3{}));
  auto stride_h = make_stride(_3{}, make_stride(_12{}, _1{}));
  printf("Shape: "); print(shape_h); printf(", Stride: "); print(stride_h); printf("\n");
  printf("  Visual representation for hierarchical shape (3,(2,3)):\n");
  printf("       0     1     2     3     4     5     <== 1-D col coord\n");
  printf("     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)\n");
  printf("    +-----+-----+-----+-----+-----+-----+\n");
  for (int i = 0; i < 3; ++i) {
    printf(" %d  ", i);
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        auto coord = make_coord(i, make_coord(j, k));
        int idx = crd2idx(coord, shape_h, stride_h);
        printf("|  %2d ", idx);
      }
    }
    printf("|\n");
    printf("    +-----+-----+-----+-----+-----+-----+\n");
  }
  printf("  ^\n  row i\n\n");
  printf("  All these coords map to same index:\n");
  printf("  crd2idx(16, shape, stride)              -> %d\n", 
         int(crd2idx(16, shape_h, stride_h)));
  printf("  crd2idx(_16{}, shape, stride)           -> ");
  print(crd2idx(_16{}, shape_h, stride_h)); printf("\n");
  printf("  crd2idx((1,5), shape, stride)           -> %d\n",
         int(crd2idx(make_coord(1, 5), shape_h, stride_h)));
  printf("  crd2idx((_1{},5), shape, stride)        -> ");
  print(crd2idx(make_coord(_1{}, 5), shape_h, stride_h)); printf("\n");
  printf("  crd2idx((1,(1,2)), shape, stride)       -> %d\n",
         int(crd2idx(make_coord(1, make_coord(1, 2)), shape_h, stride_h)));
  printf("  crd2idx((_1{},(_1{},_2{})), shape, str) -> ");
  print(crd2idx(make_coord(_1{}, make_coord(_1{}, _2{})), shape_h, stride_h)); printf("\n");
  printf("\n");
  
  printf("Example 9: Tiled layout index mapping\n");
  auto tiled = make_layout(
    make_shape(make_shape(2, 4), make_shape(3, 2)),
    make_stride(make_stride(1, 2), make_stride(8, 32))
  );
  printf("Layout: "); print(tiled); printf("\n");
  printf("  Shape: ((2,4),(3,2)) = (within-tile, tile)\n");
  printf("  Stride: ((1,2),(8,32))\n");
  printf("  \n");
  printf("  Sample mappings (showing subset):\n");
  printf("  Tile (0,0):\n");
  for (int wi = 0; wi < 2; ++wi) {
    for (int ti = 0; ti < 4; ++ti) {
      auto coord = make_coord(make_coord(wi, ti), make_coord(0, 0));
      int idx = crd2idx(coord, shape(tiled), stride(tiled));
      printf("    within-tile(%d,%d), tile(0,0) -> index %d\n", wi, ti, idx);
    }
  }
  printf("  Tile (1,0):\n");
  for (int wi = 0; wi < 2; ++wi) {
    for (int ti = 0; ti < 2; ++ti) {  // Show first 2 only
      auto coord = make_coord(make_coord(wi, ti), make_coord(1, 0));
      int idx = crd2idx(coord, shape(tiled), stride(tiled));
      printf("    within-tile(%d,%d), tile(1,0) -> index %d\n", wi, ti, idx);
    }
  }
  printf("  Tile (0,1):\n");
  for (int wi = 0; wi < 2; ++wi) {
    for (int ti = 0; ti < 2; ++ti) {  // Show first 2 only
      auto coord = make_coord(make_coord(wi, ti), make_coord(0, 1));
      int idx = crd2idx(coord, shape(tiled), stride(tiled));
      printf("    within-tile(%d,%d), tile(0,1) -> index %d\n", wi, ti, idx);
    }
  }
  printf("\n");
  
  printf("Example 10: Formula verification\n");
  auto layout = make_layout(make_shape(3, 4), make_stride(2, 10));
  printf("Layout: "); print(layout); printf("\n");
  printf("  Verifying: index = coord[0]*stride[0] + coord[1]*stride[1]\n");
  printf("           = i*2 + j*10\n");
  printf("  Visual representation:\n");
  printf("       0     1     2     3    <== col j\n");
  printf("    +-----+-----+-----+-----+\n");
  for (int i = 0; i < 3; ++i) {
    printf(" %d  ", i);
    for (int j = 0; j < 4; ++j) {
      int idx = crd2idx(make_coord(i, j), shape(layout), stride(layout));
      printf("|  %2d ", idx);
    }
    printf("|\n");
    printf("    +-----+-----+-----+-----+\n");
  }
  printf("  ^\n  row i\n\n");
  printf("  Manual verification:\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      int idx = crd2idx(make_coord(i, j), shape(layout), stride(layout));
      int manual = i * 2 + j * 10;
      printf("  coord(%d,%d): crd2idx=%d, manual=%d %s\n", 
             i, j, idx, manual, (idx == manual ? "✓" : "✗"));
    }
  }
  printf("\n");
}

int main() {
  printf("\n");
  printf("============================================================\n");
  printf(" CuTe Tutorial 7: Index Mapping\n");
  printf("============================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("============================================================\n");
  printf(" Summary:\n");
  printf("   INDEX MAPPING: Coordinate to Index\n");
  printf("   \n");
  printf("   crd2idx(coord, shape, stride): Coordinate -> Index\n");
  printf("   layout(coord): Shorthand for crd2idx\n");
  printf("   \n");
  printf("   Formula: index = sum(coord[i] * stride[i])\n");
  printf("   - Takes natural coordinate (or any equivalent coord)\n");
  printf("   - Computes inner product with strides\n");
  printf("   - Returns linear index into memory\n");
  printf("   \n");
  printf("   Key insight: Multiple equivalent coordinates\n");
  printf("   (1D, 2D, hierarchical) all map to the same index\n");
  printf("   \n");
  printf("   See Tutorial 6 for COORDINATE MAPPING (idx2crd)\n");
  printf("============================================================\n\n");
  
  return 0;
}
