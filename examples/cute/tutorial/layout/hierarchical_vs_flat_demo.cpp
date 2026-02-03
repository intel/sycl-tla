/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Demonstrates why hierarchical layouts are needed even when they map to same memory

#include <cute/tensor.hpp>

using namespace cute;

int main() {
  
  printf("\n=== Why Hierarchical Layouts Matter ===\n\n");
  
  // Both layouts access the same 12 memory locations
  auto hier_layout = make_layout(make_shape(Int<2>{}, make_shape(Int<2>{}, Int<3>{})),
                                  make_stride(Int<1>{}, make_stride(Int<2>{}, Int<4>{})));
  auto flat_layout = make_layout(make_shape(Int<2>{}, Int<6>{}), 
                                  make_stride(Int<1>{}, Int<2>{}));
  
  printf("Hierarchical: "); print(hier_layout); printf("\n");
  printf("Flat:         "); print(flat_layout); printf("\n\n");
  
  printf("=== 1. MEMORY ACCESS: Both access same locations ===\n");
  printf("Hierarchical coordinates → memory:\n");
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 3; k++) {
        printf("  (%d,(%d,%d)) → %d\n", i, j, k, int(hier_layout(i, make_coord(j,k))));
      }
    }
  }
  
  printf("\nFlat coordinates → memory:\n");
  for (int i = 0; i < 2; i++) {
    for (int c = 0; c < 6; c++) {
      printf("  (%d,%d) → %d\n", i, c, int(flat_layout(i, c)));
    }
  }
  
  printf("\n=== 2. RANK & DEPTH: Different logical structure ===\n");
  printf("Hierarchical: rank=%d, depth=%d\n", int(rank(hier_layout)), int(depth(hier_layout)));
  printf("Flat:         rank=%d, depth=%d\n", int(rank(flat_layout)), int(depth(flat_layout)));
  printf("→ Hierarchical can be accessed at different nesting levels!\n\n");
  
  printf("=== 3. HIERARCHICAL ACCESS: Only possible with hierarchy ===\n");
  // With hierarchical layout, you can access the nested dimension
  auto second_dim = hier_layout.shape<1>();  // Gets the (2,3) structure
  printf("Hierarchical 2nd dimension shape: "); print(second_dim); printf("\n");
  printf("  Inner shape<0>: %d\n", int(shape<0>(second_dim)));
  printf("  Inner shape<1>: %d\n", int(shape<1>(second_dim)));
  
  auto flat_second = flat_layout.shape<1>();  // Just gets 6
  printf("\nFlat 2nd dimension shape: "); print(flat_second); printf("\n");
  printf("  (No internal structure - just an integer)\n\n");
  
  printf("=== 4. TILING/PARTITIONING: Hierarchy enables grouping ===\n");
  printf("Hierarchical layout naturally represents:\n");
  printf("  - 2 thread groups\n");
  printf("  - Each group handles 2 tiles\n");
  printf("  - Each tile has 3 elements\n");
  printf("This structure is PRESERVED and can be used by algorithms!\n\n");
  
  printf("Flat layout represents:\n");
  printf("  - 2 rows × 6 columns\n");
  printf("  - No information about how to partition/tile\n");
  printf("  - Algorithm must manually compute grouping\n\n");
  
  printf("=== 5. EXAMPLE: Partitioning among threads ===\n");
  // Suppose we want to assign work to 4 threads
  // Hierarchical makes this natural:
  printf("With hierarchical (2,(2,3)):\n");
  printf("  Thread 0 gets: (0,(0,:)) → indices 0,4,8\n");
  printf("  Thread 1 gets: (0,(1,:)) → indices 2,6,10\n");
  printf("  Thread 2 gets: (1,(0,:)) → indices 1,5,9\n");
  printf("  Thread 3 gets: (1,(1,:)) → indices 3,7,11\n");
  printf("  → Structure makes assignment obvious!\n\n");
  
  printf("With flat (2,6):\n");
  printf("  Need to manually compute partitioning\n");
  printf("  Structure is lost - harder to reason about\n\n");
  
  printf("=== 6. REAL USE CASE: GPU Thread-to-Data Mapping ===\n");
  printf("In GPU GEMM, you often have:\n");
  printf("  - Threadblock tile: (BLK_M, BLK_N)\n");
  printf("  - Each thread handles: (THR_M, THR_N)\n");
  printf("  - Each thread loads: (VEC_M, VEC_N) per iteration\n\n");
  
  printf("Hierarchical layout: ((BLK_M,(THR_M,VEC_M)), (BLK_N,(THR_N,VEC_N)))\n");
  printf("  → Encodes 3 levels of the algorithm structure!\n");
  printf("  → CuTe can partition/tile using this structure\n\n");
  
  printf("Flat layout: (BLK_M*THR_M*VEC_M, BLK_N*THR_N*VEC_N)\n");
  printf("  → Structure is lost\n");
  printf("  → Cannot automatically partition\n\n");
  
  printf("=== CONCLUSION ===\n");
  printf("Hierarchical and flat layouts access the SAME MEMORY,\n");
  printf("but hierarchical layouts encode ALGORITHMIC STRUCTURE\n");
  printf("that CuTe operations can leverage for:\n");
  printf("  • Thread partitioning\n");
  printf("  • Tiling strategies\n");
  printf("  • Vectorized access\n");
  printf("  • Code clarity and maintainability\n\n");
  
  printf("Think of it as: both describe the same data in memory,\n");
  printf("but hierarchical layouts add *semantic annotations*\n");
  printf("that make algorithms easier to write and understand.\n\n");
  
  return 0;
}
