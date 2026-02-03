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

// Tutorial 3: Layout Construction
//
// This example demonstrates various ways to construct CuTe layouts:
// - make_layout(shape): Default column-major strides
// - make_layout(shape, LayoutLeft{}): Explicit column-major
// - make_layout(shape, LayoutRight{}): Row-major strides
// - make_layout(shape, stride): Custom strides
// - Static vs dynamic integers in construction
// - Hierarchical layout construction

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic Layout Construction ===\n\n");
  
  // Example 1: Default construction (column-major)
  printf("Example 1: Default construction - column-major\n");
  auto col_default = make_layout(make_shape(4, 6));
  printf("  make_layout(make_shape(4, 6))\n");
  printf("  Result: ");
  print(col_default);
  printf("\n");
  printf("  Stride: ");
  print(stride(col_default));
  printf(" (column-major: leftmost stride is 1)\n\n");
  
  // Example 2: Explicit column-major with LayoutLeft
  printf("Example 2: Explicit column-major with LayoutLeft\n");
  auto col_explicit = make_layout(make_shape(4, 6), LayoutLeft{});
  printf("  make_layout(make_shape(4, 6), LayoutLeft{})\n");
  printf("  Result: ");
  print(col_explicit);
  printf("\n");
  printf("  Same as default!\n\n");
  
  // Example 3: Row-major with LayoutRight
  printf("Example 3: Row-major with LayoutRight\n");
  auto row = make_layout(make_shape(4, 6), LayoutRight{});
  printf("  make_layout(make_shape(4, 6), LayoutRight{})\n");
  printf("  Result: ");
  print(row);
  printf("\n");
  printf("  Stride: ");
  print(stride(row));
  printf(" (row-major: rightmost stride is 1)\n\n");
  
  // Example 4: Custom strides
  printf("Example 4: Custom stride specification\n");
  auto custom = make_layout(make_shape(3, 4), make_stride(1, 5));
  printf("  make_layout(make_shape(3, 4), make_stride(1, 5))\n");
  printf("  Result: ");
  print(custom);
  printf("\n");
  printf("  Creates gaps in memory (stride 5 > shape 3)\n\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Static vs Dynamic and Hierarchical ===\n\n");
  
  // Example 5: Static integers (compile-time known)
  printf("Example 5: Static integer construction\n");
  auto static_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));
  printf("  make_layout(make_shape(Int<4>{}, Int<8>{}))\n");
  printf("  Result: ");
  print(static_layout);
  printf("\n");
  printf("  _N notation means compile-time constant\n\n");
  
  // Example 6: Mixed static and dynamic
  printf("Example 6: Mixed static/dynamic construction\n");
  int dynamic_dim = 6;
  auto mixed = make_layout(make_shape(Int<4>{}, dynamic_dim));
  printf("  make_layout(make_shape(Int<4>{}, 6))\n");
  printf("  Result: ");
  print(mixed);
  printf("\n");
  printf("  Can mix compile-time and run-time dimensions\n\n");
  
  // Example 7: Hierarchical shape construction
  printf("Example 7: Hierarchical layout (tiled)\n");
  auto tiled = make_layout(
    make_shape(make_shape(2, 4), make_shape(3, 2))
  );
  printf("  make_layout(make_shape(make_shape(2,4), make_shape(3,2)))\n");
  printf("  Result: ");
  print(tiled);
  printf("\n");
  printf("  Interprets as 3x2 grid of 2x4 tiles\n");
  printf("  Total size: %d elements\n\n", int(size(tiled)));
  
  // Example 8: Custom hierarchical strides
  printf("Example 8: Hierarchical with custom strides\n");
  auto tiled_custom = make_layout(
    make_shape(make_shape(2, 4), make_shape(3, 2)),
    make_stride(make_stride(1, 2), make_stride(8, 32))
  );
  printf("  Layout: ");
  print(tiled_custom);
  printf("\n");
  printf("  Inner tile is column-major (stride 1,2)\n");
  printf("  Tiles are strided by 8 and 32\n\n");
  
  // Example 9: 3D layout
  printf("Example 9: 3D layout construction\n");
  auto layout_3d = make_layout(make_shape(4, 5, 6));
  printf("  make_layout(make_shape(4, 5, 6))\n");
  printf("  Result: ");
  print(layout_3d);
  printf("\n");
  printf("  Default strides: 1, 4, 20 (column-major)\n\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Complex and Specialized Layouts ===\n\n");
  
  // Example 10: Padded layout (for alignment)
  printf("Example 10: Padded layout for alignment\n");
  auto padded = make_layout(make_shape(15, 8), make_stride(1, 16));
  printf("  make_layout(make_shape(15, 8), make_stride(1, 16))\n");
  printf("  Result: ");
  print(padded);
  printf("\n");
  printf("  Stride 16 > shape 15: adds padding for alignment\n");
  printf("  Size: %d, Cosize: %d (wasted space: %d)\n\n",
         int(size(padded)), int(cosize(padded)),
         int(cosize(padded) - size(padded)));
  
  // Example 11: Swizzled/blocked layout for cache
  printf("Example 11: Blocked layout for cache efficiency\n");
  auto blocked = make_layout(
    make_shape(make_shape(Int<8>{}, Int<4>{}), make_shape(4, 8)),
    make_stride(make_stride(Int<1>{}, Int<8>{}), make_stride(32, 256))
  );
  printf("  Layout: ");
  print(blocked);
  printf("\n");
  printf("  8x4 blocks arranged in 4x8 grid\n");
  printf("  Optimized for cache line access\n\n");
  
  // Example 12: Transposed layout view
  printf("Example 12: Constructing transposed layout\n");
  auto original = make_layout(make_shape(6, 4), make_stride(1, 6));
  auto transposed = make_layout(make_shape(4, 6), make_stride(6, 1));
  printf("  Original:   ");
  print(original);
  printf("\n");
  printf("  Transposed: ");
  print(transposed);
  printf("\n");
  printf("  Same memory, different logical view\n\n");
  
  // Example 13: Multi-level deep hierarchy
  printf("Example 13: Deep hierarchical construction\n");
  auto deep = make_layout(
    make_shape(
      Int<2>{},
      make_shape(
        3,
        make_shape(Int<4>{}, 2)
      )
    )
  );
  printf("  Layout: ");
  print(deep);
  printf("\n");
  printf("  Three levels of hierarchy\n");
  printf("  Depth: %d, Total size: %d\n\n", int(depth(deep)), int(size(deep)));
  
  // Example 14: Strided subsampling layout
  printf("Example 14: Strided subsampling\n");
  auto subsample = make_layout(make_shape(8, 8), make_stride(2, 32));
  printf("  make_layout(make_shape(8, 8), make_stride(2, 32))\n");
  printf("  Result: ");
  print(subsample);
  printf("\n");
  printf("  Samples every 2nd element in first dim\n");
  printf("  Large stride in second dim\n");
  printf("  Sparsity: %.1f%%\n\n", 100.0 * size(subsample) / cosize(subsample));
  
  // Example 15: Complex tiled + padded + hierarchical
  printf("Example 15: Complex combination\n");
  auto complex = make_layout(
    make_shape(
      make_shape(Int<16>{}, Int<8>{}),  // Tile size
      make_shape(7, 5)                    // Number of tiles
    ),
    make_stride(
      make_stride(Int<1>{}, Int<16>{}), // Tile is column-major
      make_stride(128, 1024)              // Tiles are padded/strided
    )
  );
  printf("  Layout: ");
  print(complex);
  printf("\n");
  printf("  16x8 tiles in 7x5 grid with padding\n");
  printf("  Size: %d, Cosize: %d\n", int(size(complex)), int(cosize(complex)));
  printf("  Memory efficiency: %.1f%%\n\n",
         100.0 * size(complex) / cosize(complex));
}

int main() {
  printf("\n");
  printf("======================================================================\n");
  printf(" CuTe Layout Tutorial 3: Layout Construction\n");
  printf("======================================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("======================================================================\n");
  printf(" Construction Summary:\n");
  printf("   make_layout(shape)               - Default (column-major)\n");
  printf("   make_layout(shape, LayoutLeft)   - Column-major\n");
  printf("   make_layout(shape, LayoutRight)  - Row-major\n");
  printf("   make_layout(shape, stride)       - Custom strides\n");
  printf("   \n");
  printf("   Integers:\n");
  printf("   - Int<N>{}     - Static (compile-time)\n");
  printf("   - int{N}       - Dynamic (run-time)\n");
  printf("   - Can mix both in same layout\n");
  printf("   \n");
  printf("   Hierarchical:\n");
  printf("   - Use make_shape(make_shape(...), ...) for nesting\n");
  printf("   - Enables tiling, blocking, and complex access patterns\n");
  printf("======================================================================\n\n");
  
  return 0;
}
