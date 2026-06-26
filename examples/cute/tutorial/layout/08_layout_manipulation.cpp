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

// Tutorial 8: Layout Manipulation
// Demonstrates composition, complement, logical_divide, and other layout operations

#include <cute/tensor.hpp>

using namespace cute;

void easy_examples() {
  printf("\n=== EASY EXAMPLES: Basic Manipulation ===\n\n");
  
  // ---------------------------------------------------------------------------
  // Example 1: Step-by-step composition walkthrough
  //   Functional composition R := A o B,  R(c) := A(B(c))
  //   A = (6,2):(8,2)   B = (4,3):(3,1)
  //   (Follows the Layout Algebra doc, Example 1 exactly)
  // ---------------------------------------------------------------------------
  printf("Example 1: Step-by-step composition  A=(6,2):(8,2)  o  B=(4,3):(3,1)\n");
  printf("  R(c) := A(B(c))\n\n");

  auto A1 = make_layout(make_shape(6, 2), make_stride(8, 2));
  auto B1 = make_layout(make_shape(4, 3), make_stride(3, 1));
  printf("  A = "); print(A1); printf("\n");
  printf("  B = "); print(B1); printf("\n\n");

  // Step 1 — left-distributive split of B into its sublayouts
  printf("  Step 1) Left-distributive split over B's modes:\n");
  printf("    B = (4:3, 3:1)\n");
  printf("    A o B = (A o 4:3,  A o 3:1)\n\n");

  // Branch 1 — A o 4:3   (s=4, d=3)
  printf("  Step 2) Compute A o 4:3\n");
  printf("    a) Strided layout (A / d=3):\n");
  printf("       (6,2):(8,2)  /  3  =  (6/3, 2):(8*3, 2)  =  (2,2):(24,2)\n");
  printf("          shape[0]=6 is divisible by d=3  =>  6/3=2  and  stride[0]=8*3=24\n");
  printf("          shape[1] and stride[1] are unchanged\n");
  printf("    b) Shape-compatible keep (A / 3) %% s=4:\n");
  printf("       (2,2):(24,2)  %%  4  =  (2,2):(24,2)\n");
  printf("          need 4 elements total: 2*2=4 exactly spans both modes, no truncation\n");
  auto B1_m0 = make_layout(make_shape(4), make_stride(3));        // sublayout B mode-0: 4:3
  auto r1_branch0 = composition(A1, B1_m0);
  printf("    CuTe: composition(A, 4:3) = "); print(r1_branch0); printf("\n\n");

  // Branch 2 — A o 3:1   (s=3, d=1)
  printf("  Step 3) Compute A o 3:1\n");
  printf("    a) Strided layout (A / d=1):  trivial\n");
  printf("       (6,2):(8,2)  /  1  =  (6,2):(8,2)\n");
  printf("    b) Shape-compatible keep (A / 1) %% s=3:\n");
  printf("       (6,2):(8,2)  %%  3  =  (3,1):(8,2)\n");
  printf("          3 < shape[0]=6, so keep only first 3 from mode-0; mode-1 collapses to 1\n");
  auto B1_m1 = make_layout(make_shape(3), make_stride(1));        // sublayout B mode-1: 3:1
  auto r1_branch1 = composition(A1, B1_m1);
  printf("    CuTe: composition(A, 3:1) = "); print(r1_branch1); printf("\n\n");

  // Reassemble and coalesce
  printf("  Step 4) Reassemble branches and coalesce each mode:\n");
  printf("    R = ((2,2), 3) : ((24,2), 8)\n");
  printf("    Coalesced mode-0: (2,2):(24,2) stays as-is (strides not contiguous)\n");
  printf("    Coalesced mode-1:  3:8  (scalar — already coalesced)\n\n");
  auto R1 = composition(A1, B1);
  printf("  CuTe direct: composition(A, B) = "); print(R1);
  printf("\n  Expected from docs:  ((2,2),3):((24,2),8)\n\n");

  // Verify a few values
  printf("  Spot-check  R(c) = A(B(c)):\n");
  for (int c = 0; c < 4; ++c) {
    printf("    R(%2d) = A(B(%2d)) = A(%2d) = %2d  |  layout: %2d\n",
           c, c, (int)B1(c), (int)A1(B1(c)), (int)R1(c));
  }
  printf("\n");

  // ---------------------------------------------------------------------------
  // Example 1b: Composition with clear shape/stride divisibility
  //   A = (10,2):(16,4)   B = (5,4):(1,5)
  //   (Follows the Layout Algebra doc, Example 3 exactly)
  // ---------------------------------------------------------------------------
  printf("Example 1b: Step-by-step composition  A=(10,2):(16,4)  o  B=(5,4):(1,5)\n");
  printf("  Demonstrates clean divisibility: 5 divides 10, stride 5 divides 10\n\n");

  auto A2 = make_layout(make_shape(10, 2), make_stride(16, 4));
  auto B2 = make_layout(make_shape( 5, 4), make_stride( 1, 5));
  printf("  A = "); print(A2); printf("\n");
  printf("  B = "); print(B2); printf("\n\n");

  printf("  Step 1) Left-distributive split over B's modes:\n");
  printf("    B = (5:1, 4:5)\n");
  printf("    A o B = (A o 5:1,  A o 4:5)\n\n");

  // Branch 1 — A o 5:1   (s=5, d=1)
  printf("  Step 2) Compute A o 5:1\n");
  printf("    a) Strided layout (A / d=1):  trivial\n");
  printf("       (10,2):(16,4)  /  1  =  (10,2):(16,4)\n");
  printf("    b) Shape-compatible keep %% s=5:\n");
  printf("       (10,2):(16,4)  %%  5  =  (5,1):(16,4)\n");
  printf("          5 < shape[0]=10, keep first 5 from mode-0; mode-1 collapses to 1\n");
  auto B2_m0 = make_layout(make_shape(5), make_stride(1));
  auto r2_branch0 = composition(A2, B2_m0);
  printf("    CuTe: composition(A, 5:1) = "); print(r2_branch0);
  printf("  =>  coalesced: 5:16\n\n");

  // Branch 2 — A o 4:5   (s=4, d=5)
  printf("  Step 3) Compute A o 4:5\n");
  printf("    a) Strided layout (A / d=5):\n");
  printf("       shape[0]=10 is divisible by d=5  =>  10/5=2, stride[0]=16*5=80\n");
  printf("       (10,2):(16,4)  /  5  =  (2,2):(80,4)\n");
  printf("    b) Shape-compatible keep %% s=4:\n");
  printf("       (2,2):(80,4)  %%  4  =  (2,2):(80,4)\n");
  printf("          2*2=4 exactly spans both modes, no truncation\n");
  auto B2_m1 = make_layout(make_shape(4), make_stride(5));
  auto r2_branch1 = composition(A2, B2_m1);
  printf("    CuTe: composition(A, 4:5) = "); print(r2_branch1); printf("\n\n");

  // Reassemble
  printf("  Step 4) Reassemble and by-mode coalesce:\n");
  printf("    R = ((5,1):(16,4),  (2,2):(80,4))\n");
  printf("    Coalesce mode-0:  (5,1):(16,4)  =>  5:16  (mode-1 is size-1, drops)\n");
  printf("    Mode-1 stays:     (2,2):(80,4)  (non-contiguous, cannot coalesce further)\n");
  printf("    Final: (5,(2,2)):(16,(80,4))\n\n");
  auto R2 = composition(A2, B2);
  printf("  CuTe direct: composition(A, B) = "); print(R2);
  printf("\n  Expected from docs:  (5,(2,2)):(16,(80,4))\n\n");

  // Spot-check
  printf("  Spot-check  R(c) = A(B(c)):\n");
  for (int c = 0; c < 5; ++c) {
    printf("    R(%2d) = A(B(%2d)) = A(%2d) = %3d  |  layout: %3d\n",
           c, c, (int)B2(c), (int)A2(B2(c)), (int)R2(c));
  }
  printf("\n");
  
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
  
  printf("  WHY USE COMPOSITION IN CUTLASS (Intel Xe2 Architecture):\n");
  printf("  • Work-item hierarchy: (work-item → sub-group → work-group → grid) addressing\n");
  printf("  • Tiled memory access: (element → vector → tile → matrix)\n");
  printf("  • Register partitioning: (register → work-item → sub-group fragment)\n");
  printf("  • Multi-level tiling: (inner tile → outer tile → global matrix)\n");
  printf("  • DPAS atom mapping: (work-item value → sub-group accumulator → work-group tile)\n");
  
  printf("  PRACTICAL CUTLASS PATTERNS FOR INTEL XE:\n");
  printf("  1. Work-group tile composition:\n");
  printf("     auto wg_tile = composition(sg_tile, wg_layout);\n");
  printf("     Maps sub-group-local coords → work-group-wide tile offsets\n");
  printf("     (Xe2: 16 work-items/sub-group)\n\n");
  
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
