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

// Tutorial 6: Coordinate Mapping
// Demonstrates how layouts map multi-dimensional coordinates to linear indices

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic Coordinate Mapping ===\n\n");
  
  printf("Example 1: 1D to natural coordinate\n");
  auto shape_1d = make_shape(8);
  printf("Shape: "); print(shape_1d); printf("\n");
  printf("  1D coord -> natural coord:\n");
  for (int i = 0; i < 8; ++i) {
    auto nat_coord = idx2crd(i, shape_1d);
    printf("  %d -> ", i); print(nat_coord); printf("\n");
  }
  printf("\n");
  
  printf("Example 2: 1D index to 2D natural coordinate\n");
  auto shape_2d = make_shape(3, 4);
  printf("Shape: "); print(shape_2d); printf("\n");
  printf("  Selected 1D indices map to 2D natural coords:\n");
  printf("  idx2crd(0, shape)  -> "); print(idx2crd(0, shape_2d)); printf("\n");
  printf("  idx2crd(1, shape)  -> "); print(idx2crd(1, shape_2d)); printf("\n");
  printf("  idx2crd(5, shape)  -> "); print(idx2crd(5, shape_2d)); printf("\n");
  printf("  idx2crd(11, shape) -> "); print(idx2crd(11, shape_2d)); printf("\n");
  printf("\n");
  
  printf("Example 3: Equivalent coordinate representations\n");
  auto shape = make_shape(3, 4);
  printf("Shape: "); print(shape); printf("\n");
  printf("  All these map to the same natural coordinate:\n");
  printf("  idx2crd(5, shape)           -> "); print(idx2crd(5, shape)); printf("\n");
  printf("  idx2crd(_5{}, shape)        -> "); print(idx2crd(_5{}, shape)); printf("\n");
  printf("  idx2crd((2,1), shape)       -> "); print(idx2crd(make_coord(2,1), shape)); printf("\n");
  printf("  idx2crd((_2{},1), shape)    -> "); print(idx2crd(make_coord(_2{},1), shape)); printf("\n");
  printf("\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Hierarchical Shapes ===\n\n");
  
  printf("Example 4: 2D coords to nested 2D natural coords\n");
  auto shape_hier = make_shape(3, make_shape(2, 3));
  printf("Shape: "); print(shape_hier); printf("\n");
  printf("  2D coords (i,j) map to natural hierarchical (i,(j0,j1)):\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 6; ++j) {
      auto nat_coord = idx2crd(make_coord(i, j), shape_hier);
      printf("  idx2crd((%d,%d), shape) -> ", i, j); print(nat_coord); printf("\n");
    }
  }
  printf("\n");
  
  printf("Example 5: 3D coordinate transformations\n");
  auto shape_3d = make_shape(2, 3, 4);
  printf("Shape: "); print(shape_3d); printf("\n");
  printf("  Selected 1D coords -> natural 3D coords:\n");
  printf("  0  -> "); print(idx2crd(0, shape_3d)); printf("\n");
  printf("  1  -> "); print(idx2crd(1, shape_3d)); printf("\n");
  printf("  2  -> "); print(idx2crd(2, shape_3d)); printf("\n");
  printf("  6  -> "); print(idx2crd(6, shape_3d)); printf("\n");
  printf("  23 -> "); print(idx2crd(23, shape_3d)); printf("\n");
  printf("\n");
  
  printf("Example 6: Equivalent representations (hierarchical shape)\n");
  auto shape_h = make_shape(_3{}, make_shape(_2{}, _3{}));
  printf("Shape: "); print(shape_h); printf("\n");
  printf("  All map to natural coord (1,(1,2)):\n");
  printf("  idx2crd(16, shape)                -> "); print(idx2crd(16, shape_h)); printf("\n");
  printf("  idx2crd(_16{}, shape)             -> "); print(idx2crd(_16{}, shape_h)); printf("\n");
  printf("  idx2crd((1,5), shape)             -> "); print(idx2crd(make_coord(1,5), shape_h)); printf("\n");
  printf("  idx2crd((_1{},5), shape)          -> "); print(idx2crd(make_coord(_1{},5), shape_h)); printf("\n");
  printf("  idx2crd((1,(1,2)), shape)         -> "); print(idx2crd(make_coord(1,make_coord(1,2)), shape_h)); printf("\n");
  printf("  idx2crd((_1{},(1,_2{})), shape)   -> "); print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape_h)); printf("\n");
  printf("\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Complex Coordinate Transformations ===\n\n");
  
  printf("Example 7: Nested hierarchical coordinate mapping\n");
  auto shape_complex = make_shape(make_shape(2, 4), make_shape(3, 2));
  printf("Shape: "); print(shape_complex); printf("\n");
  printf("  1D coords to natural nested coords (first 10):\n");
  for (int idx = 0; idx < 10; ++idx) {
    auto nat_coord = idx2crd(idx, shape_complex);
    printf("  %2d -> ", idx); print(nat_coord); printf("\n");
  }
  printf("\n");
  
  printf("Example 8: 2D coords to deep hierarchical shape\n");
  auto shape_deep = make_shape(2, make_shape(3, make_shape(2, 4)));
  printf("Shape: "); print(shape_deep); printf("\n");
  printf("  2D coords (i,j) map to natural deeply-nested (i,(j0,(j1,j2))):\n");
  for (int i = 0; i < 2; ++i) {
    for (int j : {0, 5, 12, 23}) {
      auto nat_coord = idx2crd(make_coord(i, j), shape_deep);
      printf("  idx2crd((%d,%2d), shape) -> ", i, j); print(nat_coord); printf("\n");
    }
  }
  printf("\n");
  
  printf("Example 9: Verifying equivalent representations\n");
  auto shape_multi = make_shape(make_shape(_2{}, _2{}), make_shape(_3{}, _2{}));
  printf("Shape: "); print(shape_multi); printf("\n");
  printf("  All these map to the same natural coord:\n");
  printf("  idx2crd(7, shape)                 -> "); print(idx2crd(7, shape_multi)); printf("\n");
  printf("  idx2crd(_7{}, shape)              -> "); print(idx2crd(_7{}, shape_multi)); printf("\n");
  printf("  idx2crd((3,1), shape)             -> "); print(idx2crd(make_coord(3,1), shape_multi)); printf("\n");
  printf("  idx2crd((_3{},_1{}), shape)       -> "); print(idx2crd(make_coord(_3{},_1{}), shape_multi)); printf("\n");
  printf("  idx2crd(((1,1),(1,0)), shape)     -> "); print(idx2crd(make_coord(make_coord(1,1),make_coord(1,0)), shape_multi)); printf("\n");
  printf("  idx2crd(((_1{},_1{}),(_1{},_0{})),s) -> "); print(idx2crd(make_coord(make_coord(_1{},_1{}),make_coord(_1{},_0{})), shape_multi)); printf("\n");
  printf("\n");
}

int main() {
  printf("\n");
  printf("============================================================\n");
  printf(" CuTe Tutorial 6: Coordinate Mapping\n");
  printf("============================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("============================================================\n");
  printf(" Summary:\n");
  printf("   idx2crd(coord, shape): Maps any coordinate to natural coord\n");
  printf("   \n");
  printf("   KEY INSIGHT: Multiple coordinate representations can refer\n");
  printf("   to the same element! A shape accepts:\n");
  printf("     - 1D coordinates (single integer 0 to size-1)\n");
  printf("     - N-D coordinates (tuple matching shape rank)\n");
  printf("     - Hierarchical coordinates (nested for nested shapes)\n");
  printf("   \n");
  printf("   Example: For shape (3,(2,3)), coord 16, (1,5), and \n");
  printf("   (1,(1,2)) all map to the same natural coord: (1,(1,2))\n");
  printf("   \n");
  printf("   Colexicographical order: rightmost mode varies fastest\n");
  printf("   \n");
  printf("   Note: layout(coord) does index mapping (coord->index)\n");
  printf("============================================================\n\n");
  
  return 0;
}
