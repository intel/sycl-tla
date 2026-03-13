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

// This example demonstrates cute::print_latex() function
// print_latex() generates LaTeX code for pretty-printing Layout, TiledCopy, and TiledMMA objects
// The output can be compiled with pdflatex to create nicely formatted colored tables

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

int main(int argc, char** argv) {
  
  printf("\n=== CuTe print_latex() Demo ===\n\n");
  printf("print_latex() generates LaTeX code for visualizing CuTe layouts\n");
  printf("Copy the output to a .tex file and compile with: pdflatex file.tex\n\n");
  
  printf("===========================================\n");
  printf("LaTeX Document Begin\n");
  printf("===========================================\n\n");
  
  // LaTeX document preamble
  printf("\\documentclass{article}\n");
  printf("\\usepackage[margin=0.5in]{geometry}\n");
  printf("\\usepackage{xcolor}\n");
  printf("\\usepackage{colortbl}\n");
  printf("\\usepackage{amsmath}\n");
  printf("\\begin{document}\n\n");
  printf("\\section*{CuTe Layout Visualizations}\n\n");
  
  // Example 1: Simple column-major layout
  printf("\\subsection*{1. Column-major 8x8 Layout}\n");
  printf("Standard column-major storage for a matrix.\n\n");
  auto col_major = make_layout(make_shape(8, 8), make_stride(1, 8));
  print_latex(col_major);
  printf("\n\n");
  
  // Example 2: Row-major layout
  printf("\\subsection*{2. Row-major 8x8 Layout}\n");
  printf("Row-major storage for comparison.\n\n");
  auto row_major = make_layout(make_shape(8, 8), make_stride(8, 1));
  print_latex(row_major);
  printf("\n\n");
  
  // Example 3: Larger column-major layout
  printf("\\subsection*{3. Larger 16x16 Column-major Layout}\n");
  printf("Scaled up version of standard layout.\n\n");
  auto large_col = make_layout(make_shape(16, 16), make_stride(1, 16));
  print_latex(large_col);
  printf("\n\n");
  
  // Example 4: Strided layout
  printf("\\subsection*{4. Strided 12x12 Layout}\n");
  printf("Non-contiguous memory access pattern.\n\n");
  auto strided_large = make_layout(make_shape(12, 12), make_stride(2, 24));
  print_latex(strided_large);
  printf("\n\n");
  
  // Example 5: Transposed layout
  printf("\\subsection*{5. Transposed 8x8 Layout}\n");
  printf("Column-major transposed to row-major view.\n\n");
  auto transposed_8x8 = make_layout(make_shape(8, 8), make_stride(8, 1));
  print_latex(transposed_8x8);
  printf("\n\n");
  
  // Example 6: Smaller layout for clarity
  printf("\\subsection*{6. Small 4x4 Layout}\n");
  printf("Smaller example for educational purposes.\n\n");
  auto small = make_layout(make_shape(4, 4), make_stride(1, 4));
  print_latex(small);
  printf("\n\n");
  
  // Example 7: TiledCopy example (if available)
  printf("\\subsection*{7. Copy Atom Visualization}\n");
  printf("Shows how a Copy\\_Atom maps threads to data.\n\n");
  using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
  auto tiled_copy = make_tiled_copy(CopyAtom{},
                                   Layout<Shape<_4,_8>>{},
                                   Layout<Shape<_1,_1>>{});
  print_latex(tiled_copy);
  printf("\n\n");
  
  // LaTeX document end
  printf("\\end{document}\n\n");
  
  printf("===========================================\n");
  printf("LaTeX Document End\n");
  printf("===========================================\n\n");
  
  printf("Instructions:\n");
  printf("1. Copy all LaTeX code above (from \\documentclass to \\end{document})\n");
  printf("2. Save to a file, e.g., cute_layouts.tex\n");
  printf("3. Compile with: pdflatex cute_layouts.tex\n");
  printf("4. View the generated PDF for colored layout visualizations\n\n");
  
  printf("The colored tables help visualize:\n");
  printf("- Memory access patterns\n");
  printf("- Thread-to-data mappings\n");
  printf("- Tiling strategies\n");
  printf("- Bank conflict patterns\n\n");
  
  printf("=== print_latex() Demo Complete ===\n\n");
  
  return 0;
}
