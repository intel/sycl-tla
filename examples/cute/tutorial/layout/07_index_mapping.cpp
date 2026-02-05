/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Tutorial 7: Index Mapping  
// Demonstrates inverse mapping from linear index back to coordinates

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic Index to Coordinate ===\n\n");
  
  printf("Example 1: 1D index to coordinate\n");
  auto layout_1d = make_layout(8);
  printf("Layout: "); print(layout_1d); printf("\n");
  for (int idx = 0; idx < 8; ++idx) {
    auto coord = idx2crd(idx, shape(layout_1d));
    printf("  index %d -> coord ", idx); print(coord); printf("\n");
  }
  printf("\n");
  
  printf("Example 2: 2D column-major index to coordinate\n");
  auto col_major = make_layout(make_shape(3, 4));
  printf("Layout: "); print(col_major); printf("\n");
  for (int idx = 0; idx < 12; ++idx) {
    auto coord = idx2crd(idx, shape(col_major), stride(col_major));
    printf("  index %d -> coord ", idx); print(coord); printf("\n");
  }
  printf("\n");
  
  printf("Example 3: Verifying round-trip mapping\n");
  auto layout = make_layout(make_shape(4, 5));
  printf("Layout: "); print(layout); printf("\n");
  printf("  Forward and back:\n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 5; ++j) {
      int idx = layout(i, j);
      auto coord = idx2crd(idx, shape(layout), stride(layout));
      printf("  (%d,%d) -> %d -> ", i, j, idx);
      print(coord);
      printf("\n");
    }
  }
  printf("\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Non-Trivial Strides ===\n\n");
  
  printf("Example 4: Row-major index to coordinate\n");
  auto row_major = make_layout(make_shape(3, 4), LayoutRight{});
  printf("Layout: "); print(row_major); printf("\n");
  for (int idx = 0; idx < 12; ++idx) {
    auto coord = idx2crd(idx, shape(row_major), stride(row_major));
    printf("  index %d -> coord ", idx); print(coord); printf("\n");
  }
  printf("\n");
  
  printf("Example 5: Custom strided layout\n");
  auto strided = make_layout(make_shape(3, 4), make_stride(1, 5));
  printf("Layout: "); print(strided); printf("\n");
  printf("  Note: Has gaps (cosize > size)\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      int idx = strided(i, j);
      printf("  coord(%d,%d) -> index %d\n", i, j, idx);
    }
  }
  printf("\n");
  
  printf("Example 6: 3D layout index mapping\n");
  auto layout_3d = make_layout(make_shape(2, 3, 4));
  printf("Layout: "); print(layout_3d); printf("\n");
  printf("  Sample index to coord mappings:\n");
  for (int idx : {0, 1, 2, 6, 12, 23}) {
    auto coord = idx2crd(idx, shape(layout_3d), stride(layout_3d));
    printf("  index %d -> coord ", idx); print(coord); printf("\n");
  }
  printf("\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Hierarchical and Complex ===\n\n");
  
  printf("Example 7: Hierarchical index to coordinate\n");
  auto hier = make_layout(make_shape(make_shape(2, 3), make_shape(4, 2)));
  printf("Layout: "); print(hier); printf("\n");
  printf("  Linear index to hierarchical coordinate:\n");
  for (int idx : {0, 1, 2, 6, 12, 24, 47}) {
    auto coord = idx2crd(idx, shape(hier), stride(hier));
    printf("  index %d -> coord ", idx); print(coord); printf("\n");
  }
  printf("\n");
  
  printf("Example 8: Tiled layout index mapping\n");
  auto tiled = make_layout(
    make_shape(make_shape(2, 4), make_shape(3, 2)),
    make_stride(make_stride(1, 2), make_stride(8, 32))
  );
  printf("Layout: "); print(tiled); printf("\n");
  printf("  Mapping indices back to tile + within-tile coords:\n");
  for (int idx : {0, 1, 2, 8, 32, 40}) {
    auto coord = idx2crd(idx, shape(tiled), stride(tiled));
    printf("  index %d -> coord ", idx); print(coord); printf("\n");
  }
  printf("\n");
  
  printf("Example 9: Verifying bijection for injective layout\n");
  auto bijective = make_layout(make_shape(4, 6));
  printf("Layout: "); print(bijective); printf("\n");
  printf("  Verifying all indices map to unique coordinates:\n");
  bool all_unique = true;
  for (int idx1 = 0; idx1 < size(bijective); ++idx1) {
    auto coord1 = idx2crd(idx1, shape(bijective), stride(bijective));
    int idx2 = bijective(coord1);
    if (idx1 != idx2) {
      printf("  ERROR: index %d != %d after round-trip\n", idx1, idx2);
      all_unique = false;
    }
  }
  printf("  Result: %s\n\n", all_unique ? "All bijective!" : "Failed");
  
  printf("Example 10: Non-injective layout (with gaps)\n");
  auto gaps = make_layout(make_shape(3, 4), make_stride(2, 10));
  printf("Layout: "); print(gaps); printf("\n");
  printf("  Size: %d, Cosize: %d (has gaps)\n",
         int(size(gaps)), int(cosize(gaps)));
  printf("  Coordinates map to non-contiguous indices:\n");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      int idx = gaps(i, j);
      printf("  (%d,%d) -> %d\n", i, j, idx);
    }
  }
  printf("  Indices %d to %d are not covered by any coordinate\n\n",
         int(size(gaps)), int(cosize(gaps)-1));
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
  printf("   idx2crd(index, shape, stride): Linear index to coordinate\n");
  printf("   \n");
  printf("   For injective layouts (no gaps):\n");
  printf("   - Every coordinate maps to unique index\n");
  printf("   - Every index maps to unique coordinate\n");
  printf("   - Round-trip is identity: coord == idx2crd(layout(coord))\n");
  printf("   \n");
  printf("   For non-injective layouts (with gaps):\n");
  printf("   - Some indices don't correspond to any coordinate\n");
  printf("   - size < cosize indicates gaps\n");
  printf("============================================================\n\n");
  
  return 0;
}
