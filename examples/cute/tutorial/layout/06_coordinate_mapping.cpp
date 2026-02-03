/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Tutorial 6: Coordinate Mapping
// Demonstrates how layouts map multi-dimensional coordinates to linear indices

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic Coordinate Mapping ===\n\n");
  
  printf("Example 1: 1D coordinate mapping\n");
  auto layout_1d = make_layout(8);
  printf("Layout: "); print(layout_1d); printf("\n");
  for (int i = 0; i < 8; ++i) {
    printf("  coord(%d) -> index %d\n", i, int(layout_1d(i)));
  }
  printf("\n");
  
  printf("Example 2: 2D column-major mapping\n");
  auto col_major = make_layout(make_shape(3, 4));
  printf("Layout: "); print(col_major); printf("\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("  coord(%d,%d) -> index %d\n", i, j, int(col_major(i, j)));
    }
  }
  printf("\n");
  
  printf("Example 3: 2D row-major mapping\n");
  auto row_major = make_layout(make_shape(3, 4), LayoutRight{});
  printf("Layout: "); print(row_major); printf("\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("  coord(%d,%d) -> index %d\n", i, j, int(row_major(i, j)));
    }
  }
  printf("\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Custom Strides and 3D ===\n\n");
  
  printf("Example 4: Custom strided mapping\n");
  auto strided = make_layout(make_shape(3, 3), make_stride(2, 10));
  printf("Layout: "); print(strided); printf("\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int idx = layout(strided)(i, j);
      printf("  coord(%d,%d) -> index %d\n", i, j, int(idx));
    }
  }
  printf("  Formula: index = i*stride[0] + j*stride[1] = i*2 + j*10\n\n");
  
  printf("Example 5: 3D coordinate mapping\n");
  auto layout_3d = make_layout(make_shape(2, 3, 4));
  printf("Layout: "); print(layout_3d); printf("\n");
  printf("  Selected coordinates:\n");
  printf("  coord(0,0,0) -> %d\n", int(layout_3d(0,0,0)));
  printf("  coord(1,0,0) -> %d\n", int(layout_3d(1,0,0)));
  printf("  coord(0,1,0) -> %d\n", int(layout_3d(0,1,0)));
  printf("  coord(0,0,1) -> %d\n", int(layout_3d(0,0,1)));
  printf("  coord(1,2,3) -> %d\n\n", int(layout_3d(1,2,3)));
  
  printf("Example 6: Hierarchical coordinate mapping\n");
  auto hier = make_layout(make_shape(2, make_shape(3, 4)));
  printf("Layout: "); print(hier); printf("\n");
  printf("  Mode 0 coord 1, Mode 1 coord (2,3):\n");
  auto coord = make_coord(1, make_coord(2, 3));
  printf("  coord(1, (2,3)) -> index %d\n", int(hier(coord)));
  // Invalid: hier(1, 2, 3) would require 3 modes but layout has rank-2
  printf("\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Complex Mappings ===\n\n");
  
  printf("Example 7: Tiled coordinate mapping\n");
  auto tiled = make_layout(
    make_shape(make_shape(2, 4), make_shape(3, 2)),
    make_stride(make_stride(1, 2), make_stride(8, 32))
  );
  printf("Layout: "); print(tiled); printf("\n");
  printf("  Tile (0,0), Element (0,0): %d\n", int(tiled(make_coord(make_coord(0,0), make_coord(0,0)))));
  printf("  Tile (0,0), Element (1,0): %d\n", int(tiled(make_coord(make_coord(1,0), make_coord(0,0)))));
  printf("  Tile (0,0), Element (0,1): %d\n", int(tiled(make_coord(make_coord(0,1), make_coord(0,0)))));
  printf("  Tile (1,0), Element (0,0): %d\n", int(tiled(make_coord(make_coord(0,0), make_coord(1,0)))));
  printf("  Tile (0,1), Element (0,0): %d\n\n", int(tiled(make_coord(make_coord(0,0), make_coord(0,1)))));
  
  printf("Example 8: Diagonal coordinate mapping\n");
  auto diagonal = make_layout(make_shape(5, 5), make_stride(1, 6));
  printf("Layout: "); print(diagonal); printf("\n");
  printf("  Diagonal elements:\n");
  for (int i = 0; i < 5; ++i) {
    printf("  coord(%d,%d) -> index %d\n", i, i, int(diagonal(i, i)));
  }
  printf("\n");
  
  printf("Example 9: Blocked strided mapping\n");
  auto blocked = make_layout(make_shape(4, 4), make_stride(1, 8));
  printf("Layout: "); print(blocked); printf("\n");
  printf("  All coordinates:\n");
  for (int i = 0; i < 4; ++i) {
    printf("  Row %d: ", i);
    for (int j = 0; j < 4; ++j) {
      printf("%2d ", int(blocked(i, j)));
    }
    printf("\n");
  }
  printf("\n");
  
  printf("Example 10: Complex hierarchical mapping\n");
  auto complex = make_layout(
    make_shape(make_shape(2, 2), make_shape(3, 2)),
    make_stride(make_stride(1, 2), make_stride(4, 16))
  );
  printf("Layout: "); print(complex); printf("\n");
  printf("  Mapping all coordinates of first tile:\n");
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      printf("  Within-tile coord(%d,%d), tile(0,0) -> %d\n",
             i, j, int(complex(make_coord(make_coord(i, j), make_coord(0, 0)))));
    }
  }
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
  printf("   layout(i): 1D coordinate to index\n");
  printf("   layout(i,j): 2D coordinate to index\n");
  printf("   layout(i,j,k,...): N-D coordinate to index\n");
  printf("   \n");
  printf("   Formula: index = sum(coord[i] * stride[i])\n");
  printf("   \n");
  printf("   Hierarchical layouts accept nested coordinates\n");
  printf("   or can be flattened with multiple arguments\n");
  printf("============================================================\n\n");
  
  return 0;
}
