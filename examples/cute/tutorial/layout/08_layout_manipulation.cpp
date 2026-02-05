/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Tutorial 8: Layout Manipulation
// Demonstrates composition, complement, logical_divide, and other layout operations

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic Manipulation ===\n\n");
  
  printf("Example 1: Composition of layouts\n");
  // Inner layout: 4×3 = 12 elements with column-major strides (4,3):(_1,4)
  auto inner = make_layout(make_shape(4, 3));
  // Outer layout: 6×5 = 30 elements with column-major strides (6,5):(_1,6)
  auto outer = make_layout(make_shape(6, 5));
  printf("  Inner: "); print(inner); printf(" [12 elements]\n");
  printf("  Outer: "); print(outer); printf(" [30 elements]\n");
  
  // Composition: composition(inner, outer)(coord) = outer[inner(coord)]
  // Step 1: Inner maps (i,j) → offset ∈ [0,12)
  // Step 2: That offset is reinterpreted as coordinate in outer's space
  // Step 3: Outer maps that coordinate → final memory address
  //
  // Visual comparison:
  //   Outer's full (6,5) space:     Inner's 12 offsets in outer:
  //     j=0  j=1  j=2  j=3  j=4       j=0  j=1  j=2  j=3  j=4
  //   i=0  0    6   12   18   24   i=0  0    6    ·    ·    ·
  //   i=1  1    7   13   19   25   i=1  1    7    ·    ·    ·
  //   i=2  2    8   14   20   26   i=2  2    8    ·    ·    ·
  //   i=3  3    9   15   21   27   i=3  3    9    ·    ·    ·
  //   i=4  4   10   16   22   28   i=4  4   10    ·    ·    ·
  //   i=5  5   11   17   23   29   i=5  5   11    ·    ·    ·
  //
  // Result: Composed layout ((4,1),(1,5)) describes a 4×5 = 20 element coordinate space
  //   - Inner's 12 coordinates index into this 20-element composed layout
  //   - Composed layout size: 20 coordinate positions
  //   - If further nested (composed as outer for another 12-element inner):
  //     Total scalars = 20 × 12 = 240 elements
  //
  // CUTLASS Use Cases - When to use composition:
  // =============================================
  // 1. Thread → Warp → Block Hierarchy:
  //    inner = thread_layout(8)         // 8 threads per warp
  //    outer = warp_layout(4)           // 4 warps per block
  //    composed = composition(inner, outer)  // Maps thread_id → block-level position
  //    Use: Determine which thread in the block handles which data element
  //
  // 2. Register → Shared Memory → Global Memory:
  //    inner = register_tile(8,8)       // 8×8 elements in registers per thread
  //    outer = smem_layout(16,16)       // 16×16 shared memory tile
  //    composed = composition(inner, outer)  // Maps register coords → smem addresses
  //    Use: Each thread's register tile maps to a sub-region of shared memory
  //
  // 3. Logical Tile → Physical Swizzle Pattern:
  //    inner = logical_tile(16,16)      // Logical 16×16 data tile
  //    outer = swizzle_pattern(...)     // Bank-conflict-free memory pattern
  //    composed = composition(inner, outer)  // Maps logical → physical with swizzle
  //    Use: Access data logically but store it in a bank-conflict-avoiding pattern
  //
  // 4. MMA Fragment → Thread Distribution:
  //    inner = mma_fragment(16,8)       // MMA instruction's output shape
  //    outer = thread_distribution(32)  // How 32 threads collectively own MMA results
  //    composed = composition(inner, outer)  // Maps fragment coord → owning thread
  //    Use: Determine which thread owns each element of the matrix multiply result
  //
  // 5. Multi-level Tiling (GEMM K-loop):
  //    inner = k_tile(32)               // 32 elements per K-iteration
  //    outer = total_k_layout(128)      // 128 total K elements
  //    composed = composition(inner, outer)  // Maps tile_iteration → K-range
  //    Use: Track which K-tile iteration accesses which slice of global K dimension
  //
  // Benefits:
  // - Separates logical indexing from physical memory layout
  // - Enables modular changes (swap swizzle pattern without changing tile logic)
  // - Compile-time composition ensures zero runtime overhead
  // - Reuse same inner layout with different outer layouts for different memory spaces
  
  auto composed = composition(inner, outer);
  auto composed_size = size(composed);  // Shape product: (4×1)×(1×5) = 20
  printf("  composition(inner, outer): "); print(composed); 
  printf(" [%d coordinate positions]\n", (int)composed_size);
  printf("  Composed layout size: %d (not inner's 12 or outer's 30)\n", (int)composed_size);
  printf("  If nested further: %d × 12 = %d total scalar elements\n\n",
         (int)composed_size, (int)composed_size * 12);
  
  printf("Example 2: Complement (find remaining space). \n when a sub-group processes a tile [0, N), complement generates the layout for the next sub-group to process [N, 2N), \n ensuring no overlap and complete coverage of data.\n");

  auto partial = make_layout(make_shape(12), make_stride(_1{}));  // Rank-1 with static stride
  printf("  Layout: "); print(partial); printf("\n");
  auto comp = complement(partial, 24);
  printf("  complement(layout, 24): "); print(comp); printf("\n");
  printf("  Gives layout for elements [12, 24)\n\n");
  
  printf("Example 3: Logical divide (tiling)\n");
  auto layout = make_layout(make_shape(12));
  auto tile_shape = Int<4>{};
  printf("  Layout: "); print(layout); printf("\n");
  printf("  Tile shape: "); print(tile_shape); printf("\n");
  auto tiled = logical_divide(layout, tile_shape);
  printf("  logical_divide(layout, 4): "); print(tiled); printf("\n");
  printf("  Creates 3 tiles of 4 elements each\n\n");
  
  printf("Example 4: CUTLASS Use Case - Thread/Warp Hierarchical Layouts\n");
  printf("  CUTLASS commonly uses composition for multi-level parallelism:\n\n");
  
  // Scenario: 128-thread block organized as 4 warps × 32 threads/warp
  // accessing a 128-element tile
  
  printf("  4a) Thread within warp layout:\n");
  auto thread_in_warp = make_layout(make_shape(32));  // 32 threads per warp
  printf("      Thread-in-warp: "); print(thread_in_warp); printf("\n");
  
  printf("  4b) Warp within block layout:\n");
  auto warp_in_block = make_layout(make_shape(4));    // 4 warps per block
  printf("      Warp-in-block:  "); print(warp_in_block); printf("\n");
  
  printf("  4c) Composed thread layout (thread → warp → block):\n");
  auto thread_in_block = composition(thread_in_warp, warp_in_block);
  printf("      Thread-in-block: "); print(thread_in_block); 
  printf(" [%d threads total]\n", (int)size(thread_in_block));
  printf("      Use: Global thread ID from (warp_id, thread_id) coordinates\n\n");
  
  printf("  4d) Data tile layout (128 elements):\n");
  auto data_tile = make_layout(make_shape(128));
  printf("      Data tile:      "); print(data_tile); printf("\n");
  
  printf("  4e) Partition data by thread hierarchy:\n");
  auto data_per_thread = logical_divide(data_tile, shape(thread_in_block));
  printf("      Data/thread:    "); print(data_per_thread); printf("\n");
  printf("      Each of 128 threads gets 1 element\n\n");
  
  printf("  WHY USE COMPOSITION IN CUTLASS (Intel Xe2/Xe4 Architecture):\n");
  printf("  • Work-item hierarchy: (work-item → sub-group → work-group → grid) addressing\n");
  printf("  • Tiled memory access: (element → vector → tile → matrix)\n");
  printf("  • Register partitioning: (register → work-item → sub-group fragment)\n");
  printf("  • Multi-level tiling: (inner tile → outer tile → global matrix)\n");
  printf("  • DPAS atom mapping: (work-item value → sub-group accumulator → work-group tile)\n");
  
  printf("  PRACTICAL CUTLASS PATTERNS FOR INTEL XE:\n");
  printf("  1. Work-group tile composition:\n");
  printf("     auto wg_tile = composition(sg_tile, wg_layout);\n");
  printf("     Maps sub-group-local coords → work-group-wide tile offsets\n");
  printf("     (Xe2: 16 work-items/sub-group, Xe4: SIMT-style execution)\n\n");
  
  printf("  2. Shared Local Memory (SLM) swizzle:\n");
  printf("     auto slm_layout = composition(unswizzled, swizzle_pattern);\n");
  printf("     Applies bank conflict avoidance to SLM addresses\n");
  printf("     (Intel Xe: 32-way banked SLM, 128B cache lines)\n\n");
  
  printf("  3. Global → SLM → Register hierarchy:\n");
  printf("     auto gmem_layout = composition(composition(reg, slm), gmem);\n");
  printf("     Three-level addressing for pipelined data movement\n");
  printf("     (Xe prefetch intrinsics: 2D block load, LSC cache hints)\n\n");
}

void medium_examples() {
  printf("\n=== MEDIUM EXAMPLES: 2D Tiling and Reshaping ===\n\n");
  
  printf("Example 5: 2D logical divide\n");
  auto layout_2d = make_layout(make_shape(8, 12));
  auto tile = make_shape(Int<2>{}, Int<4>{});
  printf("  Layout: "); print(layout_2d); printf("\n");
  printf("  Tile: "); print(tile); printf("\n");
  auto tiled_2d = logical_divide(layout_2d, tile);
  printf("  logical_divide(layout, tile): "); print(tiled_2d); printf("\n");
  printf("  Creates (4,3) grid of (2,4) tiles\n\n");
  
  printf("Example 6: Flatten hierarchical layout\n");
  auto hier = make_layout(make_shape(make_shape(2, 3), make_shape(4, 5)));
  printf("  Hierarchical: "); print(hier); printf("\n");
  auto flat = coalesce(hier);
  printf("  coalesce(hier): "); print(flat); printf("\n");
  printf("  Flattens to simpler representation\n\n");
  
  printf("Example 7: Blocking a layout\n");
  auto linear = make_layout(make_shape(24));
  auto block_size = Int<4>{};
  printf("  Linear layout: "); print(linear); printf("\n");
  auto blocked = logical_divide(linear, block_size);
  printf("  Block into size 4: "); print(blocked); printf("\n\n");
  
  printf("Example 8: Composition for thread partitioning\n");
  auto data_layout = make_layout(make_shape(64));
  auto thread_layout = make_layout(make_shape(8));
  printf("  Data layout (64 elements): "); print(data_layout); printf("\n");
  printf("  Thread layout (8 threads): "); print(thread_layout); printf("\n");
  auto per_thread = logical_divide(data_layout, shape(thread_layout));
  printf("  Data per thread: "); print(per_thread); printf("\n");
  printf("  Each thread gets 8 elements\n\n");
}

void hard_examples() {
  printf("\n=== HARD EXAMPLES: Complex Manipulations ===\n\n");
  
  printf("Example 9: Multi-level tiling\n");
  auto matrix = make_layout(make_shape(32, 32));
  auto tile_16x16 = make_shape(Int<16>{}, Int<16>{});
  auto tile_4x4 = make_shape(Int<4>{}, Int<4>{});
  printf("  Matrix: "); print(matrix); printf("\n");
  auto level1 = logical_divide(matrix, tile_16x16);
  printf("  First tiling (16x16): "); print(level1); printf("\n");
  // Further divide the inner tiles
  auto inner_tiles = get<0>(level1);
  auto level2_inner = logical_divide(inner_tiles, tile_4x4);
  printf("  Inner tile divided (4x4): "); print(level2_inner); printf("\n\n");
  
  printf("Example 10: Composition for address calculation\n");
  printf("  Scenario: Map a 3×4 logical tile through a strided memory layout\n");
  printf("  Inner layout: logical tile coordinates (i,j) ∈ [0,3)×[0,4)\n");
  printf("  Outer layout: physical memory with non-unit strides\n");
  printf("  Composition: tile_coords → memory_offsets\n\n");
  
  // Outer layout: 6×8 buffer with strides (2, 12) - non-contiguous access pattern
  auto memory_layout = make_layout(make_shape(6, 8), make_stride(2, 12));
  printf("  Physical memory layout (6×8) with stride (2,12): "); print(memory_layout); printf("\n");
  print_layout(memory_layout);
  
  // Inner layout: 3×4 logical tile with simple column-major strides
  auto tile_layout = make_layout(make_shape(3, 4), make_stride(1, 3));
  printf("\n  Logical tile layout (3×4) with stride (1,3): "); print(tile_layout); printf("\n");
  print_layout(tile_layout);
  
  // Composition: maps tile coordinates through memory layout
  auto composed = composition(tile_layout, memory_layout);
  printf("\n  Composed layout (tile coords → memory offsets): "); print(composed); printf("\n");
  print_layout(composed);
  
  printf("\n  === HOW COMPOSITION CHANGES THE LAYOUT ===\n");
  printf("  Step 1: tile_layout(i,j) maps tile coordinates to linear offsets [0,11]\n");
  printf("  Step 2: That linear offset k is used as a coordinate in memory_layout\n");
  printf("  Step 3: memory_layout(k) maps to final memory address\n\n");
  
  printf("  Example trace for tile[1,2]:\n");
  printf("    tile_layout(1,2) = 1×1 + 2×3 = 7    (tile's internal offset)\n");
  printf("    Treat 7 as coordinate in memory: memory_layout divides 7 → (row=7%%6=1, col=7//6=1)\n");
  printf("    memory_layout(1,1) = 1×2 + 1×12 = 14 (actual memory offset)\n");
  printf("    Therefore: composed(1,2) = %d\n\n", composed(1,2));
  
  printf("  VISUAL OVERLAY: Tile positions in physical memory\n");
  printf("  Showing which memory addresses the 3×4 tile accesses:\n\n");
  printf("       Col 0  Col 1  Col 2  Col 3  Col 4  Col 5  Col 6  Col 7\n");
  printf("      +------+------+------+------+------+------+------+------+\n");
  for (int i = 0; i < 6; i++) {
    printf("  R%d  |", i);
    for (int j = 0; j < 8; j++) {
      int mem_offset = memory_layout(i, j);
      bool found = false;
      // Check if this memory location is accessed by the tile
      for (int ti = 0; ti < 3 && !found; ti++) {
        for (int tj = 0; tj < 4 && !found; tj++) {
          if (composed(ti, tj) == mem_offset) {
            printf(" T%d,%d |", ti, tj);
            found = true;
          }
        }
      }
      if (!found) {
        printf(" %3d  |", mem_offset);
      }
    }
    printf("\n      +------+------+------+------+------+------+------+------+\n");
  }
  
  printf("  • Before composition: tile_layout has simple strides (1,3)\n");
  printf("  • After composition: inherits memory_layout's complex strides (2,12)\n");
  printf("  • Tile[0,0]→mem[%d], Tile[0,1]→mem[%d]: stride changed from 3→%d!\n",
         composed(0,0), composed(0,1), composed(0,1) - composed(0,0));
  printf("  • Composition = automatic address translation without manual offset math\n");
  printf("  • Intel Xe use: Map sub-group tile coords → SLM/global memory addresses\n\n");


  printf("Example 11: Filter layout (select sub-modes)\n");
  auto layout_4d = make_layout(make_shape(2, 3, 4, 5));
  printf("  4D layout: "); print(layout_4d); printf("\n");
  printf("  Can use get<I> to select specific modes\n");
  auto mode01 = make_layout(
    make_shape(shape<0>(layout_4d), shape<1>(layout_4d)),
    make_stride(stride<0>(layout_4d), stride<1>(layout_4d))
  );
  printf("  Modes 0,1 only: "); print(mode01); printf("\n\n");
  
  printf("Example 12: Swizzled composition for bank conflict avoidance\n");
  printf("  === WHAT IS SWIZZLE? ===\n");
  printf("  Swizzle = Permuting memory addresses to avoid bank conflicts in banked memory\n");
  printf("  Bank conflict occurs when multiple work-items access different addresses\n");
  printf("  in the same memory bank simultaneously, serializing the accesses.\n\n");
  
  printf("  Intel Xe SLM Architecture:\n");
  printf("  • 32-way banked (32 independent banks)\n");
  printf("  • 4-byte bank width (each bank handles 4 consecutive bytes)\n");
  printf("  • Bank_ID = (address >> 2) %% 32\n");
  printf("  • Conflict: Multiple work-items → same bank → serialized access\n\n");
  
  printf("  === EXAMPLE: 8×8 Matrix in SLM (32 bytes/row = 8 banks/row) ===\n\n");
  
  // Without swizzle: simple column-major layout
  auto unswizzled = make_layout(make_shape(8, 8), make_stride(1, 8));
  printf("  WITHOUT SWIZZLE - Column-major (8,8):(_1,8):\n");
  print_layout(unswizzled);
  
  printf("\n  Bank assignment (showing bank IDs for each element):\n");
  printf("       Col 0  Col 1  Col 2  Col 3  Col 4  Col 5  Col 6  Col 7\n");
  printf("      +------+------+------+------+------+------+------+------+\n");
  for (int i = 0; i < 8; i++) {
    printf("  R%d  |", i);
    for (int j = 0; j < 8; j++) {
      int addr = unswizzled(i, j) * 4;  // Address in bytes (4 bytes per float)
      int bank = (addr >> 2) % 32;
      printf("  B%02d |", bank);
    }
    printf("\n      +------+------+------+------+------+------+------+------+\n");
  }
  
  printf("\n  PROBLEM: Column access pattern (reading column 0: rows 0-7)\n");
  printf("  Without swizzle: All 8 elements → Banks 0,1,2,3,4,5,6,7\n");
  printf("  If 16 work-items access 2 columns simultaneously:\n");
  printf("    Work-items 0-7 access column 0 → Banks 0-7\n");
  printf("    Work-items 8-15 access column 1 → Banks 8-15\n");
  printf("  ✓ No conflict (different banks)\n\n");
  
  printf("  But if work-items access same column at different times:\n");
  printf("    All access bank pattern 0,1,2,3,4,5,6,7 → No conflicts per column\n");
  printf("  Actually, this simple case has no conflicts! Let's see a REAL conflict:\n\n");
  
  printf("  === REAL CONFLICT SCENARIO: Transposed Access ===\n");
  printf("  If sub-group reads row 0, then row 1, then row 2...\n");
  printf("  Each row access: work-items 0-7 read consecutive columns of same row\n");
  printf("  Row 0: addresses 0,8,16,24,32,40,48,56 → banks 0,8,16,24,0,8,16,24\n");
  printf("  CONFLICT! Work-items 0&4 both→bank0, 1&5→bank8, 2&6→bank16, 3&7→bank24\n");
  printf("  Result: 2-way bank conflicts → 50%% efficiency!\n\n");
  
  printf("  === WITH SWIZZLE: XOR-based permutation ===\n");
  printf("  Swizzle pattern: XOR row and column indices\n");
  printf("  swizzled_col = col XOR (row %% 4)\n");
  printf("  This spreads consecutive row accesses across different banks\n\n");
  
  // Create a swizzled layout using composition
  // Inner: 8×8 tile layout
  auto tile_8x8 = make_layout(make_shape(8, 8), make_stride(1, 8));
  
  // Outer: swizzle pattern (simplified - real swizzle uses bit manipulation)
  // For demonstration, create a pattern that shifts column based on row
  auto swizzle_pattern = make_layout(make_shape(8, 8), make_stride(1, 9)); // Stride (1,9) creates offset
  
  printf("  Swizzle via stride manipulation: (8,8):(_1,9)\n");
  printf("  Instead of stride 8, use stride 9 to create diagonal offset pattern\n");
  print_layout(swizzle_pattern);
  
  printf("\n  Bank assignment after swizzle:\n");
  printf("       Col 0  Col 1  Col 2  Col 3  Col 4  Col 5  Col 6  Col 7\n");
  printf("      +------+------+------+------+------+------+------+------+\n");
  for (int i = 0; i < 8; i++) {
    printf("  R%d  |", i);
    for (int j = 0; j < 8; j++) {
      int addr = swizzle_pattern(i, j) * 4;  // Address in bytes
      int bank = (addr >> 2) % 32;
      printf("  B%02d |", bank);
    }
    printf("\n      +------+------+------+------+------+------+------+------+\n");
  }
  
  printf("\n  === INTEL XE SWIZZLE USAGE ===\n");
  printf("  WHEN TO USE SWIZZLE:\n");
  printf("  1. Matrix Transpose in SLM:\n");
  printf("     • Load column-major from global → store row-major to SLM\n");
  printf("     • Swizzle prevents bank conflicts during stores\n\n");
  
  printf("  2. GEMM Shared Memory Tiles:\n");
  printf("     • A matrix tile: K×M layout (K along columns)\n");
  printf("     • B matrix tile: K×N layout (K along rows)\n");
  printf("     • Swizzle B tile to avoid conflicts when broadcasting K values\n\n");
  
  printf("  3. Sub-group Reductions:\n");
  printf("     • 16 work-items write partial sums to SLM\n");
  printf("     • Without swizzle: all write to same bank pattern\n");
  printf("     • With swizzle: spread across banks\n\n");
  
  printf("  4. Block Load/Store:\n");
  printf("     • Intel Xe 2D block load (16×16 or 8×32 blocks)\n");
  printf("     • Swizzle ensures uniform bank utilization\n\n");
  
  printf("  HOW TO IMPLEMENT IN CUTLASS:\n");
  printf("  auto unswizzled_slm = make_layout(shape, stride);\n");
  printf("  auto swizzle = Swizzle<3,0,3>{}; // CuTe swizzle: B=3 bits, M=0, S=3\n");
  printf("  auto swizzled_slm = composition(unswizzled_slm, swizzle);\n\n");
  
  printf("  PERFORMANCE IMPACT:\n");
  printf("  • Without swizzle: Up to 32× slowdown (worst case)\n");
  printf("  • With swizzle: Near-theoretical bandwidth\n");
  printf("  • Xe2 (BMG): 128 KB SLM @ ~1 TB/s (conflict-free)\n");
  printf("  • Xe4: Similar architecture, SIMT benefits from swizzle\n\n");
  
  printf("Example 13: Zipped layouts for interleaving\n");
  auto layout_a = make_layout(make_shape(8), make_stride(2));  // Even indices
  auto layout_b = make_layout(make_shape(8), make_stride(2));  // Odd indices  
  printf("  Layout A (stride 2): "); print(layout_a); printf("\n");
  printf("  Layout B (stride 2): "); print(layout_b); printf("\n");
  printf("  Can create interleaved access patterns\n\n");
  
  printf("Example 14: Transpose via layout manipulation\n");
  auto orig = make_layout(make_shape(4, 6));
  auto trans = make_layout(make_shape(shape<1>(orig), shape<0>(orig)),
                          make_stride(stride<1>(orig), stride<0>(orig)));
  printf("  Original: "); print(orig); printf("\n");
  printf("  Transposed: "); print(trans); printf("\n");
  printf("  Swapped shape and stride modes\n\n");
}

int main() {
  printf("\n");
  printf("============================================================\n");
  printf(" CuTe Tutorial 8: Layout Manipulation\n");
  printf("============================================================\n");
  
  easy_examples();
  medium_examples();
  hard_examples();
  
  printf("============================================================\n");
  printf(" Summary:\n");
  printf("   composition(A, B): Compose two layouts\n");
  printf("   complement(L, N): Find complement to size N\n");
  printf("   logical_divide(L, tile): Tile/partition layout\n");
  printf("   coalesce(L): Flatten hierarchical layout\n");
  printf("   \n");
  printf("   These enable:\n");
  printf("   - Tiling for cache/thread mapping\n");
  printf("   - Address calculation optimization\n");
  printf("   - Layout transformations (transpose, swizzle)\n");
  printf("   - Hierarchical memory access patterns\n");
  printf("============================================================\n\n");
  
  return 0;
}
