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

// Tutorial 2: Hierarchical Access Functions
//
// This example demonstrates hierarchical access to nested layouts:
// - get<I>(layout): Access the I-th mode
// - get<I,J>(layout): Access the J-th sub-mode of the I-th mode
// - rank<I>(layout): Rank of the I-th mode
// - depth<I>(layout): Depth of the I-th mode  
// - shape<I>(layout): Shape of the I-th mode
// - size<I>(layout): Size of the I-th mode
// - stride<I>(layout): Stride of the I-th mode

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Accessing Modes in Simple Layouts ===\n\n");
  
  // Example 1: Accessing modes of a 2D layout
  printf("Example 1: Accessing modes of a 3x4 layout\n");
  auto layout = make_layout(make_shape(3, 4));
  printf("Layout: ");
  print(layout);
  printf("\n");
  printf("  rank:     %d\n", int(rank(layout)));
  printf("  shape<0>: ");
  print(shape<0>(layout));
  printf(" (first mode size)\n");
  printf("  shape<1>: ");
  print(shape<1>(layout));
  printf(" (second mode size)\n");
  printf("  size<0>:  %d\n", int(size<0>(layout)));
  printf("  size<1>:  %d\n", int(size<1>(layout)));
  printf("  stride<0>: ");
  print(stride<0>(layout));
  printf("\n");
  printf("  stride<1>: ");
  print(stride<1>(layout));
  printf("\n\n");
  
  // Example 2: Using get<> to extract sub-layouts
  printf("Example 2: Extracting sub-layouts with get<I>()\n");
  auto layout_4x6 = make_layout(make_shape(4, 6), make_stride(1, 4));
  printf("Full layout: ");
  print(layout_4x6);
  printf("\n");
  auto mode0 = get<0>(layout_4x6);
  auto mode1 = get<1>(layout_4x6);
  printf("  get<0>(): ");
  print(mode0);
  printf(" (first mode as 1D layout)\n");
  printf("  get<1>(): ");
  print(mode1);
  printf(" (second mode as 1D layout)\n\n");
  
  // Example 3: rank<I>() for each mode
  printf("Example 3: Rank of each mode\n");
  auto layout_3d = make_layout(make_shape(2, 3, 4));
  printf("Layout: ");
  print(layout_3d);
  printf("\n");
  printf("  rank:      %d (total modes)\n", int(rank(layout_3d)));
  printf("  rank<0>:   %d (mode 0 is rank-1)\n", int(rank<0>(layout_3d)));
  printf("  rank<1>:   %d (mode 1 is rank-1)\n", int(rank<1>(layout_3d)));
  printf("  rank<2>:   %d (mode 2 is rank-1)\n\n", int(rank<2>(layout_3d)));
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: Hierarchical Access in Nested Layouts ===\n\n");
  
  // Example 4: Two-level hierarchy
  printf("Example 4: Accessing nested modes with get<I,J>()\n");
  auto hier_shape = make_shape(4, make_shape(2, 3));
  auto hier_layout = make_layout(hier_shape);
  printf("Layout: ");
  print(hier_layout);
  printf("\n");
  printf("  shape:     ");
  print(shape(hier_layout));
  printf("\n");
  printf("  shape<0>:  ");
  print(shape<0>(hier_layout));
  printf(" (first mode)\n");
  printf("  shape<1>:  ");
  print(shape<1>(hier_layout));
  printf(" (second mode, nested)\n");
  printf("  shape<1,0>: ");
  print(shape<1,0>(hier_layout));
  printf(" (first sub-mode of mode 1)\n");
  printf("  shape<1,1>: ");
  print(shape<1,1>(hier_layout));
  printf(" (second sub-mode of mode 1)\n");
  printf("  size<1>:    %d (2 * 3 = 6)\n", int(size<1>(hier_layout)));
  printf("  size<1,0>:  %d\n", int(size<1,0>(hier_layout)));
  printf("  size<1,1>:  %d\n\n", int(size<1,1>(hier_layout)));
  
  // Example 5: Depth of nested modes
  printf("Example 5: Depth of nested modes\n");
  auto deep = make_layout(make_shape(make_shape(2, 2), 5));
  printf("Layout: ");
  print(deep);
  printf("\n");
  printf("  depth:     %d (overall depth)\n", int(depth(deep)));
  printf("  depth<0>:  %d (mode 0 has nested tuple)\n", int(depth<0>(deep)));
  printf("  depth<1>:  %d (mode 1 is flat)\n", int(depth<1>(deep)));
  printf("  depth<0,0>: %d (innermost is flat)\n", int(depth<0,0>(deep)));
  printf("  depth<0,1>: %d (innermost is flat)\n\n", int(depth<0,1>(deep)));
  
  // Example 6: Strides in nested layouts
  printf("Example 6: Accessing strides hierarchically\n");
  auto tiled = make_layout(
    make_shape(make_shape(2, 4), make_shape(3, 2)),
    make_stride(make_stride(1, 2), make_stride(8, 48))
  );
  printf("Layout: ");
  print(tiled);
  printf("\n");
  printf("  stride<0>:   ");
  print(stride<0>(tiled));
  printf(" (outer mode strides)\n");
  printf("  stride<1>:   ");
  print(stride<1>(tiled));
  printf(" (tile strides)\n");
  printf("  stride<0,0>: ");
  print(stride<0,0>(tiled));
  printf("\n");
  printf("  stride<0,1>: ");
  print(stride<0,1>(tiled));
  printf("\n");
  printf("  stride<1,0>: ");
  print(stride<1,0>(tiled));
  printf("\n");
  printf("  stride<1,1>: ");
  print(stride<1,1>(tiled));
  printf("\n\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Deep Hierarchies and Complex Access ===\n\n");
  
  // Example 7: Three-level hierarchy
  printf("Example 7: Three-level hierarchical access\n");
  auto deep3 = make_layout(
    make_shape(
      2,
      make_shape(3, make_shape(4, 2))
    )
  );
  printf("Layout: ");
  print(deep3);
  printf("\n");
  printf("  depth:       %d (three levels)\n", int(depth(deep3)));
  printf("  rank:        %d (top level)\n", int(rank(deep3)));
  printf("  rank<1>:     %d (second level)\n", int(rank<1>(deep3)));
  printf("  rank<1,1>:   %d (third level)\n", int(rank<1,1>(deep3)));
  printf("  shape<1,1,0>: ");
  print(shape<1,1,0>(deep3));
  printf("\n");
  printf("  shape<1,1,1>: ");
  print(shape<1,1,1>(deep3));
  printf("\n");
  printf("  size<1,1>:    %d (4 * 2 = 8)\n", int(size<1,1>(deep3)));
  printf("  size<1>:      %d (3 * 4 * 2 = 24)\n\n", int(size<1>(deep3)));
  
  // Example 8: Mixed static and dynamic hierarchy
  printf("Example 8: Navigating mixed static/dynamic hierarchy\n");
  auto mixed = make_layout(
    make_shape(
      make_shape(Int<2>{}, 4),
      make_shape(Int<3>{}, make_shape(5, Int<6>{}))
    )
  );
  printf("Layout: ");
  print(mixed);
  printf("\n");
  printf("  Accessing different levels:\n");
  printf("  shape<0,0>:   ");
  print(shape<0,0>(mixed));
  printf(" (static)\n");
  printf("  shape<0,1>:   ");
  print(shape<0,1>(mixed));
  printf(" (dynamic)\n");
  printf("  shape<1,0>:   ");
  print(shape<1,0>(mixed));
  printf(" (static)\n");
  printf("  shape<1,1,0>: ");
  print(shape<1,1,0>(mixed));
  printf(" (dynamic)\n");
  printf("  shape<1,1,1>: ");
  print(shape<1,1,1>(mixed));
  printf(" (static)\n\n");
  
  // Example 9: Iterating through modes programmatically
  printf("Example 9: Programmatic iteration through modes\n");
  auto multi = make_layout(make_shape(2, 3, 4, 5));
  printf("Layout: ");
  print(multi);
  printf("\n");
  printf("  Mode sizes: ");
  printf("%d, ", int(size<0>(multi)));
  printf("%d, ", int(size<1>(multi)));
  printf("%d, ", int(size<2>(multi)));
  printf("%d\n", int(size<3>(multi)));
  printf("  Mode strides: ");
  printf("%d, ", int(stride<0>(multi)));
  printf("%d, ", int(stride<1>(multi)));
  printf("%d, ", int(stride<2>(multi)));
  printf("%d\n\n", int(stride<3>(multi)));
  
  // Example 10: Complex tiled layout access
  printf("Example 10: Accessing complex tiled layout\n");
  auto complex_tile = make_layout(
    make_shape(
      make_shape(Int<4>{}, Int<8>{}),  // Tile shape
      make_shape(3, 2)                   // Grid of tiles
    ),
    make_stride(
      make_stride(Int<1>{}, Int<4>{}),  // Within-tile strides
      make_stride(32, 256)                // Between-tile strides
    )
  );
  printf("Layout: ");
  print(complex_tile);
  printf("\n");
  printf("  Tile dimensions:\n");
  printf("    Inner tile shape<0>: ");
  print(shape<0>(complex_tile));
  printf(" (4x8)\n");
  printf("    Tile grid shape<1>:  ");
  print(shape<1>(complex_tile));
  printf(" (3x2)\n");
  printf("  Total coverage: %d elements\n", int(size(complex_tile)));
  printf("  Cosize: %d (memory footprint)\n\n", int(cosize(complex_tile)));
}

int main() {
  printf("\n");
  printf("======================================================================\n");
  printf(" CuTe Layout Tutorial 2: Hierarchical Access Functions\n");
  printf("======================================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("======================================================================\n");
  printf(" Summary:\n");
  printf("   get<I>(L):      Extract the I-th mode as a sub-layout\n");
  printf("   get<I,J>(L):    Extract J-th sub-mode of I-th mode\n");
  printf("   shape<I>(L):    Shape of the I-th mode\n");
  printf("   stride<I>(L):   Stride of the I-th mode\n");
  printf("   size<I>(L):     Size of the I-th mode\n");
  printf("   rank<I>(L):     Rank of the I-th mode\n");
  printf("   depth<I>(L):    Depth of the I-th mode\n");
  printf("   \n");
  printf("   Use these to navigate and query hierarchical layouts!\n");
  printf("======================================================================\n\n");
  
  return 0;
}
