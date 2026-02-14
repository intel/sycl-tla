/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// Demonstrates the difference between static Int<N> and dynamic int

#include <cute/tensor.hpp>
#include <type_traits>

using namespace cute;

int main() {
  
  printf("\n=== Static Int<N>{} vs Dynamic int ===\n\n");
  
  // Static layout with Int<4>{}, Int<8>{}
  auto static_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));
  
  // Dynamic layout with regular integers
  auto dynamic_layout = make_layout(make_shape(4, 8));
  
  printf("1. PRINT REPRESENTATION:\n");
  printf("   Static:  ");
  print(static_layout);
  printf("\n");
  printf("   Dynamic: ");
  print(dynamic_layout);
  printf("\n");
  printf("   → Static uses _N notation (underscore means compile-time constant)\n\n");
  
  printf("2. COMPILE-TIME vs RUNTIME:\n");
  printf("   Static Int<4>{}:\n");
  printf("     - Value known at COMPILE TIME\n");
  printf("     - Encoded in the TYPE system\n");
  printf("     - Zero runtime storage cost\n\n");
  printf("   Dynamic int{4}:\n");
  printf("     - Value known at RUNTIME\n");
  printf("     - Stored in memory/registers\n");
  printf("     - Has storage cost\n\n");
  
  printf("3. TYPE INFORMATION:\n");
  auto static_shape = static_layout.shape();
  auto dynamic_shape = dynamic_layout.shape();
  
  printf("   Static shape type: tuple<Int<4>, Int<8>>\n");
  printf("   Dynamic shape type: tuple<int, int>\n");
  printf("   → Static: types encode values! Dynamic: just stores values\n\n");
  
  printf("4. COMPILE-TIME QUERIES:\n");
  // With static, you can query at compile time
  constexpr auto static_val = Int<4>{};
  constexpr int compile_time_value = static_val;  // Can use in constexpr!
  printf("   Static Int<4>{} can be used in constexpr contexts\n");
  printf("   Compiler KNOWS the value = %d at compile time\n\n", compile_time_value);
  
  // Dynamic requires runtime
  int dynamic_val = 4;
  printf("   Dynamic int{4} is only known at runtime\n");
  printf("   Value = %d (must be evaluated at runtime)\n\n", dynamic_val);
  
  printf("5. OPTIMIZATION IMPACT:\n\n");
  printf("   With STATIC Int<4>{}:\n");
  printf("   ✓ Compiler can unroll loops\n");
  printf("   ✓ Can eliminate bounds checks\n");
  printf("   ✓ Better register allocation\n");
  printf("   ✓ Constant propagation\n");
  printf("   ✓ Dead code elimination\n");
  printf("   Example: for(int i=0; i<4; i++) → fully unrolled!\n\n");
  
  printf("   With DYNAMIC int{4}:\n");
  printf("   ✗ Must generate loop code\n");
  printf("   ✗ Runtime bounds checks needed\n");
  printf("   ✗ Less aggressive optimization\n");
  printf("   ✗ Variable is in a register/memory\n\n");
  
  printf("6. PRACTICAL EXAMPLE - Array Access:\n\n");
  printf("   // Static version\n");
  printf("   for_each(make_shape(Int<4>{}), [&](auto i) {\n");
  printf("     // Compiler KNOWS i ∈ {0,1,2,3}\n");
  printf("     // Can fully unroll this loop!\n");
  printf("   });\n\n");
  
  printf("   // Dynamic version\n");
  printf("   for(int i=0; i<4; i++) {\n");
  printf("     // Compiler must generate loop\n");
  printf("     // Cannot assume bounds without analysis\n");
  printf("   }\n\n");
  
  printf("7. MEMORY REPRESENTATION:\n");
  printf("   Static layout size:  %zu bytes (types, minimal storage)\n", sizeof(static_layout));
  printf("   Dynamic layout size: %zu bytes (stores values)\n", sizeof(dynamic_layout));
  printf("   → Static layouts are often smaller!\n\n");
  
  printf("8. WHEN TO USE EACH:\n\n");
  printf("   Use STATIC Int<N>{} when:\n");
  printf("   ✓ Size is known at compile time (tile sizes, block sizes)\n");
  printf("   ✓ Want maximum performance\n");
  printf("   ✓ Want compile-time guarantees\n");
  printf("   ✓ GPU kernels with fixed sizes\n");
  printf("   Examples: threadblock tiles (16x16), warp sizes (32), vector widths\n\n");
  
  printf("   Use DYNAMIC int when:\n");
  printf("   ✓ Size is only known at runtime (user input, data-dependent)\n");
  printf("   ✓ Need flexibility\n");
  printf("   ✓ Size varies between runs\n");
  printf("   Examples: batch size, problem size from command line\n\n");
  
  printf("9. MIXING STATIC AND DYNAMIC:\n");
  int batch_size = 5;  // Runtime value
  auto mixed_layout = make_layout(
    make_shape(Int<16>{}, Int<16>{}, batch_size)  // Mix!
  );
  printf("   Layout: ");
  print(mixed_layout);
  printf("\n");
  printf("   First two dimensions: static (tile size 16x16)\n");
  printf("   Third dimension: dynamic (batch size from runtime)\n");
  printf("   → Get optimization for fixed dims, flexibility for variable dim!\n\n");
  
  printf("10. REAL GPU EXAMPLE:\n\n");
  printf("   Typical GEMM kernel:\n");
  printf("   auto layout = make_layout(\n");
  printf("     make_shape(\n");
  printf("       Int<128>{},    // M tile (compile-time)\n");
  printf("       Int<128>{},    // N tile (compile-time)\n");
  printf("       batch         // Batch size (runtime)\n");
  printf("     )\n");
  printf("   );\n\n");
  printf("   Why?\n");
  printf("   • Tile sizes (128x128) are fixed by algorithm\n");
  printf("   • Compiler can optimize tile processing\n");
  printf("   • Batch size varies per problem\n");
  printf("   • Best of both worlds!\n\n");
  
  printf("11. THE FUNDAMENTAL DIFFERENCE:\n\n");
  printf("   Int<4>{}:  A TYPE that represents the value 4\n");
  printf("   int{4}:    A VARIABLE that stores the value 4\n\n");
  printf("   Think of it as:\n");
  printf("   Int<4>{} = std::integral_constant<int, 4>  (type-level)\n");
  printf("   int{4}   = int x = 4;                      (value-level)\n\n");
  
  printf("=== CONCLUSION ===\n\n");
  printf("Both create the SAME LAYOUT at runtime (same memory access pattern),\n");
  printf("but Int<N>{} gives the compiler MORE INFORMATION to optimize with!\n\n");
  printf("Rule of thumb:\n");
  printf("  Use Int<N>{} whenever possible for performance-critical GPU code!\n\n");
  
  return 0;
}
