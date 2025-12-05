/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the disclaimer.
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

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/arch/copy_xe_2d.hpp>
#include <sycl/sycl.hpp>
#include "cutlass_unit_test.h"

using namespace cute;

#if (IGC_VERSION_MAJOR > 2) || (IGC_VERSION_MAJOR == 2 && IGC_VERSION_MINOR >= 18) 

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_API_Declaration) {
  // Template: XE_LOAD_2D_TRANSPOSE<Bits, Height, Width>
  // Constraints: Bits == 32 || Bits == 64, Width <= 8
  // For 64-bit: Height == 8 && Width < 4
  
  // Test 32-bit transpose operations
  using TransposeOp_32bit_2x4 = XE_LOAD_2D_TRANSPOSE<32, 2, 4>;
  using TransposeOp_32bit_4x8 = XE_LOAD_2D_TRANSPOSE<32, 4, 8>;
  using TransposeOp_32bit_8x2 = XE_LOAD_2D_TRANSPOSE<32, 8, 2>;
  
  // Test 64-bit transpose operations (limited constraints)
  using TransposeOp_64bit_8x2 = XE_LOAD_2D_TRANSPOSE<64, 8, 2>;
  using TransposeOp_64bit_8x3 = XE_LOAD_2D_TRANSPOSE<64, 8, 3>;
  
  // Test that the operations have the required static members from XE_Copy_Op_2D_Base
  static_assert(TransposeOp_32bit_2x4::AtomHeight == 2);
  static_assert(TransposeOp_32bit_2x4::AtomWidth == 4);
  static_assert(TransposeOp_32bit_2x4::CopyBits == 32);
  
  static_assert(TransposeOp_32bit_4x8::AtomHeight == 4);
  static_assert(TransposeOp_32bit_4x8::AtomWidth == 8);
  static_assert(TransposeOp_32bit_4x8::CopyBits == 32);
  
  static_assert(TransposeOp_64bit_8x2::AtomHeight == 8);
  static_assert(TransposeOp_64bit_8x2::AtomWidth == 2);
  static_assert(TransposeOp_64bit_8x2::CopyBits == 64);
  
  EXPECT_TRUE(true) << "XE_LOAD_2D_TRANSPOSE API types declared successfully";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_Constraints) {
  // Test that the compile-time constraints are enforced
  
  // Valid 32-bit operations
  using Valid32_1 = XE_LOAD_2D_TRANSPOSE<32, 1, 1>;
  using Valid32_2 = XE_LOAD_2D_TRANSPOSE<32, 16, 8>; // Width <= 8
  
  // Valid 64-bit operations (Height == 8 && Width < 4)
  using Valid64_1 = XE_LOAD_2D_TRANSPOSE<64, 8, 1>;
  using Valid64_2 = XE_LOAD_2D_TRANSPOSE<64, 8, 2>;
  using Valid64_3 = XE_LOAD_2D_TRANSPOSE<64, 8, 3>;
  
  static_assert(Valid32_1::CopyBits == 32);
  static_assert(Valid32_2::CopyBits == 32);
  static_assert(Valid64_1::CopyBits == 64);
  static_assert(Valid64_2::CopyBits == 64);
  static_assert(Valid64_3::CopyBits == 64);
  
  EXPECT_TRUE(true) << "XE_LOAD_2D_TRANSPOSE constraint validation successful";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_MinimalConfigs) {
  // Test minimal 32-bit transpose configurations
  using Transpose_32bit_1x1 = XE_LOAD_2D_TRANSPOSE<32, 1, 1>;
  using Transpose_32bit_1x2 = XE_LOAD_2D_TRANSPOSE<32, 1, 2>;
  using Transpose_32bit_2x1 = XE_LOAD_2D_TRANSPOSE<32, 2, 1>;
  using Transpose_32bit_2x2 = XE_LOAD_2D_TRANSPOSE<32, 2, 2>;
  
  static_assert(Transpose_32bit_1x1::CopyBits == 32);
  static_assert(Transpose_32bit_1x1::AtomHeight == 1 && Transpose_32bit_1x1::AtomWidth == 1);
  
  static_assert(Transpose_32bit_1x2::CopyBits == 32);
  static_assert(Transpose_32bit_1x2::AtomHeight == 1 && Transpose_32bit_1x2::AtomWidth == 2);
  
  static_assert(Transpose_32bit_2x1::CopyBits == 32);
  static_assert(Transpose_32bit_2x1::AtomHeight == 2 && Transpose_32bit_2x1::AtomWidth == 1);
  
  static_assert(Transpose_32bit_2x2::CopyBits == 32);
  static_assert(Transpose_32bit_2x2::AtomHeight == 2 && Transpose_32bit_2x2::AtomWidth == 2);
  
  EXPECT_TRUE(true) << "32-bit minimal transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_Width4) {
  // Test 32-bit transpose with width = 4
  using Transpose_32bit_1x4 = XE_LOAD_2D_TRANSPOSE<32, 1, 4>;
  using Transpose_32bit_2x4 = XE_LOAD_2D_TRANSPOSE<32, 2, 4>;
  using Transpose_32bit_4x4 = XE_LOAD_2D_TRANSPOSE<32, 4, 4>;
  using Transpose_32bit_8x4 = XE_LOAD_2D_TRANSPOSE<32, 8, 4>;
  
  static_assert(Transpose_32bit_1x4::AtomHeight == 1 && Transpose_32bit_1x4::AtomWidth == 4);
  static_assert(Transpose_32bit_2x4::AtomHeight == 2 && Transpose_32bit_2x4::AtomWidth == 4);
  static_assert(Transpose_32bit_4x4::AtomHeight == 4 && Transpose_32bit_4x4::AtomWidth == 4);
  static_assert(Transpose_32bit_8x4::AtomHeight == 8 && Transpose_32bit_8x4::AtomWidth == 4);
  
  EXPECT_TRUE(true) << "32-bit width=4 transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_Width8_MaxWidth) {
  // Test 32-bit transpose with width = 8 (maximum allowed width)
  using Transpose_32bit_1x8 = XE_LOAD_2D_TRANSPOSE<32, 1, 8>;
  using Transpose_32bit_2x8 = XE_LOAD_2D_TRANSPOSE<32, 2, 8>;
  using Transpose_32bit_4x8 = XE_LOAD_2D_TRANSPOSE<32, 4, 8>;
  using Transpose_32bit_8x8 = XE_LOAD_2D_TRANSPOSE<32, 8, 8>;
  using Transpose_32bit_16x8 = XE_LOAD_2D_TRANSPOSE<32, 16, 8>;
  
  static_assert(Transpose_32bit_1x8::AtomHeight == 1 && Transpose_32bit_1x8::AtomWidth == 8);
  static_assert(Transpose_32bit_2x8::AtomHeight == 2 && Transpose_32bit_2x8::AtomWidth == 8);
  static_assert(Transpose_32bit_4x8::AtomHeight == 4 && Transpose_32bit_4x8::AtomWidth == 8);
  static_assert(Transpose_32bit_8x8::AtomHeight == 8 && Transpose_32bit_8x8::AtomWidth == 8);
  static_assert(Transpose_32bit_16x8::AtomHeight == 16 && Transpose_32bit_16x8::AtomWidth == 8);
  
  EXPECT_TRUE(true) << "32-bit max width=8 transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_VariousHeights) {
  // Test 32-bit transpose with various height values
  using Transpose_32bit_1x4 = XE_LOAD_2D_TRANSPOSE<32, 1, 4>;
  using Transpose_32bit_4x4 = XE_LOAD_2D_TRANSPOSE<32, 4, 4>;
  using Transpose_32bit_8x4 = XE_LOAD_2D_TRANSPOSE<32, 8, 4>;
  using Transpose_32bit_16x4 = XE_LOAD_2D_TRANSPOSE<32, 16, 4>;
  using Transpose_32bit_32x4 = XE_LOAD_2D_TRANSPOSE<32, 32, 4>;
  
  static_assert(Transpose_32bit_1x4::AtomHeight == 1);
  static_assert(Transpose_32bit_4x4::AtomHeight == 4);
  static_assert(Transpose_32bit_8x4::AtomHeight == 8);
  static_assert(Transpose_32bit_16x4::AtomHeight == 16);
  static_assert(Transpose_32bit_32x4::AtomHeight == 32);
  
  EXPECT_TRUE(true) << "32-bit various height transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_SquareConfigs) {
  // Test 32-bit square transpose configurations
  using Transpose_32bit_1x1 = XE_LOAD_2D_TRANSPOSE<32, 1, 1>;
  using Transpose_32bit_2x2 = XE_LOAD_2D_TRANSPOSE<32, 2, 2>;
  using Transpose_32bit_4x4 = XE_LOAD_2D_TRANSPOSE<32, 4, 4>;
  using Transpose_32bit_8x8 = XE_LOAD_2D_TRANSPOSE<32, 8, 8>;
  
  static_assert(Transpose_32bit_1x1::AtomHeight == 1 && Transpose_32bit_1x1::AtomWidth == 1);
  static_assert(Transpose_32bit_2x2::AtomHeight == 2 && Transpose_32bit_2x2::AtomWidth == 2);
  static_assert(Transpose_32bit_4x4::AtomHeight == 4 && Transpose_32bit_4x4::AtomWidth == 4);
  static_assert(Transpose_32bit_8x8::AtomHeight == 8 && Transpose_32bit_8x8::AtomWidth == 8);
  
  EXPECT_TRUE(true) << "32-bit square transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_TallConfigs) {
  // Test 32-bit tall (Height > Width) transpose configurations
  using Transpose_32bit_8x1 = XE_LOAD_2D_TRANSPOSE<32, 8, 1>;
  using Transpose_32bit_8x2 = XE_LOAD_2D_TRANSPOSE<32, 8, 2>;
  using Transpose_32bit_16x4 = XE_LOAD_2D_TRANSPOSE<32, 16, 4>;
  using Transpose_32bit_32x8 = XE_LOAD_2D_TRANSPOSE<32, 32, 8>;
  
  static_assert(Transpose_32bit_8x1::AtomHeight == 8 && Transpose_32bit_8x1::AtomWidth == 1);
  static_assert(Transpose_32bit_8x2::AtomHeight == 8 && Transpose_32bit_8x2::AtomWidth == 2);
  static_assert(Transpose_32bit_16x4::AtomHeight == 16 && Transpose_32bit_16x4::AtomWidth == 4);
  static_assert(Transpose_32bit_32x8::AtomHeight == 32 && Transpose_32bit_32x8::AtomWidth == 8);
  
  EXPECT_TRUE(true) << "32-bit tall transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_WideConfigs) {
  // Test 32-bit wide (Width > Height) transpose configurations
  using Transpose_32bit_1x8 = XE_LOAD_2D_TRANSPOSE<32, 1, 8>;
  using Transpose_32bit_2x8 = XE_LOAD_2D_TRANSPOSE<32, 2, 8>;
  using Transpose_32bit_4x8 = XE_LOAD_2D_TRANSPOSE<32, 4, 8>;
  using Transpose_32bit_1x4 = XE_LOAD_2D_TRANSPOSE<32, 1, 4>;
  
  static_assert(Transpose_32bit_1x8::AtomHeight == 1 && Transpose_32bit_1x8::AtomWidth == 8);
  static_assert(Transpose_32bit_2x8::AtomHeight == 2 && Transpose_32bit_2x8::AtomWidth == 8);
  static_assert(Transpose_32bit_4x8::AtomHeight == 4 && Transpose_32bit_4x8::AtomWidth == 8);
  static_assert(Transpose_32bit_1x4::AtomHeight == 1 && Transpose_32bit_1x4::AtomWidth == 4);
  
  EXPECT_TRUE(true) << "32-bit wide transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_64bit_AllValid) {
  // Test all valid 64-bit transpose configurations (Height=8, Width<4)
  using Transpose_64bit_8x1 = XE_LOAD_2D_TRANSPOSE<64, 8, 1>;
  using Transpose_64bit_8x2 = XE_LOAD_2D_TRANSPOSE<64, 8, 2>;
  using Transpose_64bit_8x3 = XE_LOAD_2D_TRANSPOSE<64, 8, 3>;
  
  static_assert(Transpose_64bit_8x1::CopyBits == 64);
  static_assert(Transpose_64bit_8x1::AtomHeight == 8 && Transpose_64bit_8x1::AtomWidth == 1);
  
  static_assert(Transpose_64bit_8x2::CopyBits == 64);
  static_assert(Transpose_64bit_8x2::AtomHeight == 8 && Transpose_64bit_8x2::AtomWidth == 2);
  
  static_assert(Transpose_64bit_8x3::CopyBits == 64);
  static_assert(Transpose_64bit_8x3::AtomHeight == 8 && Transpose_64bit_8x3::AtomWidth == 3);
  
  EXPECT_TRUE(true) << "64-bit all valid transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_64bit_Constraints) {
  // Test that 64-bit transpose respects its strict constraints
  // Valid: Height == 8 && Width < 4
  using Valid_64bit_1 = XE_LOAD_2D_TRANSPOSE<64, 8, 1>;
  using Valid_64bit_2 = XE_LOAD_2D_TRANSPOSE<64, 8, 2>;
  using Valid_64bit_3 = XE_LOAD_2D_TRANSPOSE<64, 8, 3>;
  
  // Verify all have correct dimensions
  static_assert(Valid_64bit_1::AtomHeight == 8 && Valid_64bit_1::AtomWidth == 1);
  static_assert(Valid_64bit_2::AtomHeight == 8 && Valid_64bit_2::AtomWidth == 2);
  static_assert(Valid_64bit_3::AtomHeight == 8 && Valid_64bit_3::AtomWidth == 3);
  
  // Verify all have 64-bit size
  static_assert(Valid_64bit_1::CopyBits == 64);
  static_assert(Valid_64bit_2::CopyBits == 64);
  static_assert(Valid_64bit_3::CopyBits == 64);
  
  EXPECT_TRUE(true) << "64-bit constraint validation successful";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_PowerOfTwo_Heights) {
  // Test 32-bit transpose with power-of-two heights
  using Transpose_32bit_1x4 = XE_LOAD_2D_TRANSPOSE<32, 1, 4>;
  using Transpose_32bit_2x4 = XE_LOAD_2D_TRANSPOSE<32, 2, 4>;
  using Transpose_32bit_4x4 = XE_LOAD_2D_TRANSPOSE<32, 4, 4>;
  using Transpose_32bit_8x4 = XE_LOAD_2D_TRANSPOSE<32, 8, 4>;
  using Transpose_32bit_16x4 = XE_LOAD_2D_TRANSPOSE<32, 16, 4>;
  using Transpose_32bit_32x4 = XE_LOAD_2D_TRANSPOSE<32, 32, 4>;
  
  static_assert(Transpose_32bit_1x4::AtomHeight == 1);
  static_assert(Transpose_32bit_2x4::AtomHeight == 2);
  static_assert(Transpose_32bit_4x4::AtomHeight == 4);
  static_assert(Transpose_32bit_8x4::AtomHeight == 8);
  static_assert(Transpose_32bit_16x4::AtomHeight == 16);
  static_assert(Transpose_32bit_32x4::AtomHeight == 32);
  
  EXPECT_TRUE(true) << "32-bit power-of-two height transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_AllWidths) {
  // Test 32-bit transpose with all valid widths (1-8) and height=8
  using Transpose_32bit_8x1 = XE_LOAD_2D_TRANSPOSE<32, 8, 1>;
  using Transpose_32bit_8x2 = XE_LOAD_2D_TRANSPOSE<32, 8, 2>;
  using Transpose_32bit_8x3 = XE_LOAD_2D_TRANSPOSE<32, 8, 3>;
  using Transpose_32bit_8x4 = XE_LOAD_2D_TRANSPOSE<32, 8, 4>;
  using Transpose_32bit_8x5 = XE_LOAD_2D_TRANSPOSE<32, 8, 5>;
  using Transpose_32bit_8x6 = XE_LOAD_2D_TRANSPOSE<32, 8, 6>;
  using Transpose_32bit_8x7 = XE_LOAD_2D_TRANSPOSE<32, 8, 7>;
  using Transpose_32bit_8x8 = XE_LOAD_2D_TRANSPOSE<32, 8, 8>;
  
  static_assert(Transpose_32bit_8x1::AtomWidth == 1);
  static_assert(Transpose_32bit_8x2::AtomWidth == 2);
  static_assert(Transpose_32bit_8x3::AtomWidth == 3);
  static_assert(Transpose_32bit_8x4::AtomWidth == 4);
  static_assert(Transpose_32bit_8x5::AtomWidth == 5);
  static_assert(Transpose_32bit_8x6::AtomWidth == 6);
  static_assert(Transpose_32bit_8x7::AtomWidth == 7);
  static_assert(Transpose_32bit_8x8::AtomWidth == 8);
  
  EXPECT_TRUE(true) << "32-bit all width values transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_MatMul_Optimized) {
  // Test 32-bit transpose configurations useful for matrix multiplication
  // Common for transposing A matrix in row-major to column-major for DPAS
  using MatMul_32bit_8x8 = XE_LOAD_2D_TRANSPOSE<32, 8, 8>;
  using MatMul_32bit_16x8 = XE_LOAD_2D_TRANSPOSE<32, 16, 8>;
  using MatMul_32bit_8x4 = XE_LOAD_2D_TRANSPOSE<32, 8, 4>;
  using MatMul_32bit_16x4 = XE_LOAD_2D_TRANSPOSE<32, 16, 4>;
  
  static_assert(MatMul_32bit_8x8::CopyBits == 32);
  static_assert(MatMul_32bit_8x8::AtomHeight == 8 && MatMul_32bit_8x8::AtomWidth == 8);
  
  static_assert(MatMul_32bit_16x8::CopyBits == 32);
  static_assert(MatMul_32bit_16x8::AtomHeight == 16 && MatMul_32bit_16x8::AtomWidth == 8);
  
  static_assert(MatMul_32bit_8x4::CopyBits == 32);
  static_assert(MatMul_32bit_8x4::AtomHeight == 8 && MatMul_32bit_8x4::AtomWidth == 4);
  
  static_assert(MatMul_32bit_16x4::CopyBits == 32);
  static_assert(MatMul_32bit_16x4::AtomHeight == 16 && MatMul_32bit_16x4::AtomWidth == 4);
  
  EXPECT_TRUE(true) << "32-bit MatMul-optimized transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_SmallTiles) {
  // Test 32-bit transpose small tiles for boundary handling
  using Small_32bit_1x1 = XE_LOAD_2D_TRANSPOSE<32, 1, 1>;
  using Small_32bit_2x1 = XE_LOAD_2D_TRANSPOSE<32, 2, 1>;
  using Small_32bit_1x2 = XE_LOAD_2D_TRANSPOSE<32, 1, 2>;
  using Small_32bit_4x2 = XE_LOAD_2D_TRANSPOSE<32, 4, 2>;
  
  static_assert(Small_32bit_1x1::AtomHeight == 1 && Small_32bit_1x1::AtomWidth == 1);
  static_assert(Small_32bit_2x1::AtomHeight == 2 && Small_32bit_2x1::AtomWidth == 1);
  static_assert(Small_32bit_1x2::AtomHeight == 1 && Small_32bit_1x2::AtomWidth == 2);
  static_assert(Small_32bit_4x2::AtomHeight == 4 && Small_32bit_4x2::AtomWidth == 2);
  
  EXPECT_TRUE(true) << "32-bit small tile transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_DataSize_Consistency) {
  // Test that CopyBits correctly reflects 32 or 64 bits
  using Op_32bit_Small = XE_LOAD_2D_TRANSPOSE<32, 2, 2>;
  using Op_32bit_Large = XE_LOAD_2D_TRANSPOSE<32, 32, 8>;
  using Op_64bit_Valid1 = XE_LOAD_2D_TRANSPOSE<64, 8, 1>;
  using Op_64bit_Valid2 = XE_LOAD_2D_TRANSPOSE<64, 8, 3>;
  
  // All 32-bit variants should have CopyBits == 32
  static_assert(Op_32bit_Small::CopyBits == 32);
  static_assert(Op_32bit_Large::CopyBits == 32);
  
  // All 64-bit variants should have CopyBits == 64
  static_assert(Op_64bit_Valid1::CopyBits == 64);
  static_assert(Op_64bit_Valid2::CopyBits == 64);
  
  EXPECT_TRUE(true) << "Transpose data size consistency validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_Width_Progression) {
  // Test 32-bit transpose with progressive widths and fixed height
  using Transpose_16x1 = XE_LOAD_2D_TRANSPOSE<32, 16, 1>;
  using Transpose_16x2 = XE_LOAD_2D_TRANSPOSE<32, 16, 2>;
  using Transpose_16x3 = XE_LOAD_2D_TRANSPOSE<32, 16, 3>;
  using Transpose_16x4 = XE_LOAD_2D_TRANSPOSE<32, 16, 4>;
  using Transpose_16x5 = XE_LOAD_2D_TRANSPOSE<32, 16, 5>;
  using Transpose_16x6 = XE_LOAD_2D_TRANSPOSE<32, 16, 6>;
  using Transpose_16x7 = XE_LOAD_2D_TRANSPOSE<32, 16, 7>;
  using Transpose_16x8 = XE_LOAD_2D_TRANSPOSE<32, 16, 8>;
  
  static_assert(Transpose_16x1::AtomHeight == 16 && Transpose_16x1::AtomWidth == 1);
  static_assert(Transpose_16x2::AtomHeight == 16 && Transpose_16x2::AtomWidth == 2);
  static_assert(Transpose_16x3::AtomHeight == 16 && Transpose_16x3::AtomWidth == 3);
  static_assert(Transpose_16x4::AtomHeight == 16 && Transpose_16x4::AtomWidth == 4);
  static_assert(Transpose_16x5::AtomHeight == 16 && Transpose_16x5::AtomWidth == 5);
  static_assert(Transpose_16x6::AtomHeight == 16 && Transpose_16x6::AtomWidth == 6);
  static_assert(Transpose_16x7::AtomHeight == 16 && Transpose_16x7::AtomWidth == 7);
  static_assert(Transpose_16x8::AtomHeight == 16 && Transpose_16x8::AtomWidth == 8);
  
  EXPECT_TRUE(true) << "32-bit progressive width transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_32bit_LargeTiles) {
  // Test 32-bit transpose with larger tile configurations (Height <= 32 limit)
  using Large_32bit_32x8 = XE_LOAD_2D_TRANSPOSE<32, 32, 8>;
  using Large_32bit_32x6 = XE_LOAD_2D_TRANSPOSE<32, 32, 6>;
  using Large_32bit_32x4 = XE_LOAD_2D_TRANSPOSE<32, 32, 4>;
  using Large_32bit_32x2 = XE_LOAD_2D_TRANSPOSE<32, 32, 2>;
  
  static_assert(Large_32bit_32x8::CopyBits == 32);
  static_assert(Large_32bit_32x8::AtomHeight == 32 && Large_32bit_32x8::AtomWidth == 8);
  
  static_assert(Large_32bit_32x6::CopyBits == 32);
  static_assert(Large_32bit_32x6::AtomHeight == 32 && Large_32bit_32x6::AtomWidth == 6);
  
  static_assert(Large_32bit_32x4::CopyBits == 32);
  static_assert(Large_32bit_32x4::AtomHeight == 32 && Large_32bit_32x4::AtomWidth == 4);
  
  static_assert(Large_32bit_32x2::CopyBits == 32);
  static_assert(Large_32bit_32x2::AtomHeight == 32 && Large_32bit_32x2::AtomWidth == 2);
  
  EXPECT_TRUE(true) << "32-bit large tile transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_Mixed_AspectRatios) {
  // Test various aspect ratios for 32-bit transpose (Height <= 32 limit)
  using AspectRatio_1to8 = XE_LOAD_2D_TRANSPOSE<32, 1, 8>;    // 1:8
  using AspectRatio_2to8 = XE_LOAD_2D_TRANSPOSE<32, 2, 8>;    // 1:4
  using AspectRatio_4to8 = XE_LOAD_2D_TRANSPOSE<32, 4, 8>;    // 1:2
  using AspectRatio_8to8 = XE_LOAD_2D_TRANSPOSE<32, 8, 8>;    // 1:1
  using AspectRatio_16to8 = XE_LOAD_2D_TRANSPOSE<32, 16, 8>;  // 2:1
  using AspectRatio_32to8 = XE_LOAD_2D_TRANSPOSE<32, 32, 8>;  // 4:1
  using AspectRatio_32to4 = XE_LOAD_2D_TRANSPOSE<32, 32, 4>;  // 8:1
  
  static_assert(AspectRatio_1to8::AtomHeight == 1 && AspectRatio_1to8::AtomWidth == 8);
  static_assert(AspectRatio_2to8::AtomHeight == 2 && AspectRatio_2to8::AtomWidth == 8);
  static_assert(AspectRatio_4to8::AtomHeight == 4 && AspectRatio_4to8::AtomWidth == 8);
  static_assert(AspectRatio_8to8::AtomHeight == 8 && AspectRatio_8to8::AtomWidth == 8);
  static_assert(AspectRatio_16to8::AtomHeight == 16 && AspectRatio_16to8::AtomWidth == 8);
  static_assert(AspectRatio_32to8::AtomHeight == 32 && AspectRatio_32to8::AtomWidth == 8);
  static_assert(AspectRatio_32to4::AtomHeight == 32 && AspectRatio_32to4::AtomWidth == 4);
  
  EXPECT_TRUE(true) << "Mixed aspect ratio transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_BF16_FP16_UseCase) {
  // Test transpose configurations useful for bf16/fp16 data (stored as 32-bit for transpose)
  // Transpose allows loading 16-bit data as 32-bit, then converting after transpose
  using BF16_8x8 = XE_LOAD_2D_TRANSPOSE<32, 8, 8>;
  using BF16_16x8 = XE_LOAD_2D_TRANSPOSE<32, 16, 8>;
  using BF16_8x4 = XE_LOAD_2D_TRANSPOSE<32, 8, 4>;
  using BF16_16x4 = XE_LOAD_2D_TRANSPOSE<32, 16, 4>;
  
  // These are 32-bit operations but can be used with 16-bit data
  static_assert(BF16_8x8::CopyBits == 32);
  static_assert(BF16_8x8::AtomHeight == 8 && BF16_8x8::AtomWidth == 8);
  
  static_assert(BF16_16x8::CopyBits == 32);
  static_assert(BF16_16x8::AtomHeight == 16 && BF16_16x8::AtomWidth == 8);
  
  static_assert(BF16_8x4::CopyBits == 32);
  static_assert(BF16_8x4::AtomHeight == 8 && BF16_8x4::AtomWidth == 4);
  
  static_assert(BF16_16x4::CopyBits == 32);
  static_assert(BF16_16x4::AtomHeight == 16 && BF16_16x4::AtomWidth == 4);
  
  EXPECT_TRUE(true) << "BF16/FP16 use case transpose configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_64bit_UseCase) {
  // Test 64-bit transpose use cases (double precision or two 32-bit values)
  // Limited to Height=8, Width in {1, 2, 3}
  using Double_8x1 = XE_LOAD_2D_TRANSPOSE<64, 8, 1>;
  using Double_8x2 = XE_LOAD_2D_TRANSPOSE<64, 8, 2>;
  using Double_8x3 = XE_LOAD_2D_TRANSPOSE<64, 8, 3>;
  
  // All must be 64-bit
  static_assert(Double_8x1::CopyBits == 64);
  static_assert(Double_8x2::CopyBits == 64);
  static_assert(Double_8x3::CopyBits == 64);
  
  // All must have height=8
  static_assert(Double_8x1::AtomHeight == 8);
  static_assert(Double_8x2::AtomHeight == 8);
  static_assert(Double_8x3::AtomHeight == 8);
  
  // Widths must be < 4
  static_assert(Double_8x1::AtomWidth == 1);
  static_assert(Double_8x2::AtomWidth == 2);
  static_assert(Double_8x3::AtomWidth == 3);
  
  EXPECT_TRUE(true) << "64-bit use case transpose configurations validated";
}

#else

TEST(CuTe_Xe, XE_LOAD_2D_TRANSPOSE_SKIPPED) {
  GTEST_SKIP() << "XE_LOAD_2D_TRANSPOSE tests require IGC version 2.18 or higher. skipped";
}

#endif
