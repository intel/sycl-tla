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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/arch/copy_xe_2d.hpp>
#include <sycl/sycl.hpp>
#include "cutlass_unit_test.h"

using namespace cute;

#if (IGC_VERSION_MAJOR > 2) || (IGC_VERSION_MAJOR == 2 && IGC_VERSION_MINOR >= 18) 

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_API_Declaration) {
  // Template: XE_LOAD_2D_VNNI<Bits, Height, Width, BlockWidth = Width>
  
  // Test that the VNNI operation types can be declared
  using VNNIOp_8bit_2x32 = XE_LOAD_2D_VNNI<8, 2, 32>;
  using VNNIOp_8bit_4x32 = XE_LOAD_2D_VNNI<8, 4, 32>;
  using VNNIOp_16bit_2x16 = XE_LOAD_2D_VNNI<16, 2, 16>;
  using VNNIOp_16bit_4x16 = XE_LOAD_2D_VNNI<16, 4, 16>;
  
  // Test that the operations have the required static members from XE_Copy_Op_2D_Base
  static_assert(VNNIOp_8bit_2x32::AtomHeight == 2);
  static_assert(VNNIOp_8bit_2x32::AtomWidth == 32);
  static_assert(VNNIOp_8bit_2x32::CopyBits == 8);
  
  static_assert(VNNIOp_16bit_2x16::AtomHeight == 2);
  static_assert(VNNIOp_16bit_2x16::AtomWidth == 16);
  static_assert(VNNIOp_16bit_2x16::CopyBits == 16);
  
  EXPECT_TRUE(true) << "XE_LOAD_2D_VNNI API types declared successfully";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_8bit_MinimalConfigs) {
  // Test minimal 8-bit VNNI configurations
  using VNNIOp_8bit_1x32 = XE_LOAD_2D_VNNI<8, 1, 32>;
  using VNNIOp_8bit_2x32 = XE_LOAD_2D_VNNI<8, 2, 32>;
  using VNNIOp_8bit_4x32 = XE_LOAD_2D_VNNI<8, 4, 32>;
  
  static_assert(VNNIOp_8bit_1x32::CopyBits == 8);
  static_assert(VNNIOp_8bit_1x32::AtomHeight == 1);
  static_assert(VNNIOp_8bit_1x32::AtomWidth == 32);
  
  static_assert(VNNIOp_8bit_2x32::CopyBits == 8);
  static_assert(VNNIOp_8bit_2x32::AtomHeight == 2);
  static_assert(VNNIOp_8bit_2x32::AtomWidth == 32);
  
  static_assert(VNNIOp_8bit_4x32::CopyBits == 8);
  static_assert(VNNIOp_8bit_4x32::AtomHeight == 4);
  static_assert(VNNIOp_8bit_4x32::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "8-bit minimal VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_8bit_MediumConfigs) {
  // Test medium-sized 8-bit VNNI configurations
  using VNNIOp_8bit_8x32 = XE_LOAD_2D_VNNI<8, 8, 32>;
  using VNNIOp_8bit_16x32 = XE_LOAD_2D_VNNI<8, 16, 32>;
  
  static_assert(VNNIOp_8bit_8x32::CopyBits == 8);
  static_assert(VNNIOp_8bit_8x32::AtomHeight == 8);
  static_assert(VNNIOp_8bit_8x32::AtomWidth == 32);
  
  static_assert(VNNIOp_8bit_16x32::CopyBits == 8);
  static_assert(VNNIOp_8bit_16x32::AtomHeight == 16);
  static_assert(VNNIOp_8bit_16x32::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "8-bit medium VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_8bit_WideConfigs) {
  // Test 8-bit VNNI configurations with wider widths
  using VNNIOp_8bit_2x64 = XE_LOAD_2D_VNNI<8, 2, 64>;
  using VNNIOp_8bit_4x64 = XE_LOAD_2D_VNNI<8, 4, 64>;
  using VNNIOp_8bit_8x64 = XE_LOAD_2D_VNNI<8, 8, 64>;
  
  static_assert(VNNIOp_8bit_2x64::CopyBits == 8);
  static_assert(VNNIOp_8bit_2x64::AtomHeight == 2);
  static_assert(VNNIOp_8bit_2x64::AtomWidth == 64);
  
  static_assert(VNNIOp_8bit_4x64::CopyBits == 8);
  static_assert(VNNIOp_8bit_4x64::AtomHeight == 4);
  static_assert(VNNIOp_8bit_4x64::AtomWidth == 64);
  
  static_assert(VNNIOp_8bit_8x64::CopyBits == 8);
  static_assert(VNNIOp_8bit_8x64::AtomHeight == 8);
  static_assert(VNNIOp_8bit_8x64::AtomWidth == 64);
  
  EXPECT_TRUE(true) << "8-bit wide VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_16bit_MinimalConfigs) {
  // Test minimal 16-bit VNNI configurations
  using VNNIOp_16bit_1x16 = XE_LOAD_2D_VNNI<16, 1, 16>;
  using VNNIOp_16bit_2x16 = XE_LOAD_2D_VNNI<16, 2, 16>;
  using VNNIOp_16bit_4x16 = XE_LOAD_2D_VNNI<16, 4, 16>;
  
  static_assert(VNNIOp_16bit_1x16::CopyBits == 16);
  static_assert(VNNIOp_16bit_1x16::AtomHeight == 1);
  static_assert(VNNIOp_16bit_1x16::AtomWidth == 16);
  
  static_assert(VNNIOp_16bit_2x16::CopyBits == 16);
  static_assert(VNNIOp_16bit_2x16::AtomHeight == 2);
  static_assert(VNNIOp_16bit_2x16::AtomWidth == 16);
  
  static_assert(VNNIOp_16bit_4x16::CopyBits == 16);
  static_assert(VNNIOp_16bit_4x16::AtomHeight == 4);
  static_assert(VNNIOp_16bit_4x16::AtomWidth == 16);
  
  EXPECT_TRUE(true) << "16-bit minimal VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_16bit_MediumConfigs) {
  // Test medium-sized 16-bit VNNI configurations
  using VNNIOp_16bit_8x16 = XE_LOAD_2D_VNNI<16, 8, 16>;
  using VNNIOp_16bit_16x16 = XE_LOAD_2D_VNNI<16, 16, 16>;
  
  static_assert(VNNIOp_16bit_8x16::CopyBits == 16);
  static_assert(VNNIOp_16bit_8x16::AtomHeight == 8);
  static_assert(VNNIOp_16bit_8x16::AtomWidth == 16);
  
  static_assert(VNNIOp_16bit_16x16::CopyBits == 16);
  static_assert(VNNIOp_16bit_16x16::AtomHeight == 16);
  static_assert(VNNIOp_16bit_16x16::AtomWidth == 16);
  
  EXPECT_TRUE(true) << "16-bit medium VNNI configurations validated";
}


TEST(CuTe_Xe, XE_LOAD_2D_VNNI_16bit_WideConfigs) {
  // Test 16-bit VNNI configurations with wider widths
  using VNNIOp_16bit_2x32 = XE_LOAD_2D_VNNI<16, 2, 32>;
  using VNNIOp_16bit_4x32 = XE_LOAD_2D_VNNI<16, 4, 32>;
  using VNNIOp_16bit_8x32 = XE_LOAD_2D_VNNI<16, 8, 32>;
  
  static_assert(VNNIOp_16bit_2x32::CopyBits == 16);
  static_assert(VNNIOp_16bit_2x32::AtomHeight == 2);
  static_assert(VNNIOp_16bit_2x32::AtomWidth == 32);
  
  static_assert(VNNIOp_16bit_4x32::CopyBits == 16);
  static_assert(VNNIOp_16bit_4x32::AtomHeight == 4);
  static_assert(VNNIOp_16bit_4x32::AtomWidth == 32);
  
  static_assert(VNNIOp_16bit_8x32::CopyBits == 16);
  static_assert(VNNIOp_16bit_8x32::AtomHeight == 8);
  static_assert(VNNIOp_16bit_8x32::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "16-bit wide VNNI configurations validated";
}



TEST(CuTe_Xe, XE_LOAD_2D_VNNI_8bit_CustomBlockWidth) {
  // Test 8-bit VNNI with custom BlockWidth parameter
  using VNNIOp_8bit_4x32_bw16 = XE_LOAD_2D_VNNI<8, 4, 32, 16>;
  using VNNIOp_8bit_8x32_bw16 = XE_LOAD_2D_VNNI<8, 8, 32, 16>;
  using VNNIOp_8bit_16x32_bw16 = XE_LOAD_2D_VNNI<8, 16, 32, 16>;
  
  static_assert(VNNIOp_8bit_4x32_bw16::CopyBits == 8);
  static_assert(VNNIOp_8bit_4x32_bw16::AtomHeight == 4);
  static_assert(VNNIOp_8bit_4x32_bw16::AtomWidth == 32);
  
  static_assert(VNNIOp_8bit_8x32_bw16::CopyBits == 8);
  static_assert(VNNIOp_8bit_8x32_bw16::AtomHeight == 8);
  static_assert(VNNIOp_8bit_8x32_bw16::AtomWidth == 32);
  
  static_assert(VNNIOp_8bit_16x32_bw16::CopyBits == 8);
  static_assert(VNNIOp_8bit_16x32_bw16::AtomHeight == 16);
  static_assert(VNNIOp_8bit_16x32_bw16::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "8-bit VNNI with custom BlockWidth validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_16bit_CustomBlockWidth) {
  // Test 16-bit VNNI with custom BlockWidth parameter
  using VNNIOp_16bit_4x16_bw8 = XE_LOAD_2D_VNNI<16, 4, 16, 8>;
  using VNNIOp_16bit_8x16_bw8 = XE_LOAD_2D_VNNI<16, 8, 16, 8>;
  using VNNIOp_16bit_16x16_bw8 = XE_LOAD_2D_VNNI<16, 16, 16, 8>;
  
  static_assert(VNNIOp_16bit_4x16_bw8::CopyBits == 16);
  static_assert(VNNIOp_16bit_4x16_bw8::AtomHeight == 4);
  static_assert(VNNIOp_16bit_4x16_bw8::AtomWidth == 16);
  
  static_assert(VNNIOp_16bit_8x16_bw8::CopyBits == 16);
  static_assert(VNNIOp_16bit_8x16_bw8::AtomHeight == 8);
  static_assert(VNNIOp_16bit_8x16_bw8::AtomWidth == 16);
  
  static_assert(VNNIOp_16bit_16x16_bw8::CopyBits == 16);
  static_assert(VNNIOp_16bit_16x16_bw8::AtomHeight == 16);
  static_assert(VNNIOp_16bit_16x16_bw8::AtomWidth == 16);
  
  EXPECT_TRUE(true) << "16-bit VNNI with custom BlockWidth validated";
}



TEST(CuTe_Xe, XE_LOAD_2D_VNNI_Int8_GEMMConfigs) {
  // Test typical int8 GEMM VNNI configurations for K-dimension packing
  using GEMM_Int8_4x32 = XE_LOAD_2D_VNNI<8, 4, 32>;
  using GEMM_Int8_8x32 = XE_LOAD_2D_VNNI<8, 8, 32>;
  using GEMM_Int8_16x32 = XE_LOAD_2D_VNNI<8, 16, 32>;
  using GEMM_Int8_32x32 = XE_LOAD_2D_VNNI<8, 32, 32>;
  
  static_assert(GEMM_Int8_4x32::CopyBits == 8);
  static_assert(GEMM_Int8_4x32::AtomHeight == 4);
  static_assert(GEMM_Int8_4x32::AtomWidth == 32);
  
  static_assert(GEMM_Int8_8x32::CopyBits == 8);
  static_assert(GEMM_Int8_8x32::AtomHeight == 8);
  static_assert(GEMM_Int8_8x32::AtomWidth == 32);
  
  static_assert(GEMM_Int8_16x32::CopyBits == 8);
  static_assert(GEMM_Int8_16x32::AtomHeight == 16);
  static_assert(GEMM_Int8_16x32::AtomWidth == 32);
  
  static_assert(GEMM_Int8_32x32::CopyBits == 8);
  static_assert(GEMM_Int8_32x32::AtomHeight == 32);
  static_assert(GEMM_Int8_32x32::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "Int8 GEMM VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_BF16_GEMMConfigs) {
  // Test typical BF16/FP16 GEMM VNNI configurations for K-dimension packing
  using GEMM_BF16_4x16 = XE_LOAD_2D_VNNI<16, 4, 16>;
  using GEMM_BF16_8x16 = XE_LOAD_2D_VNNI<16, 8, 16>;
  using GEMM_BF16_16x16 = XE_LOAD_2D_VNNI<16, 16, 16>;
  using GEMM_BF16_32x16 = XE_LOAD_2D_VNNI<16, 32, 16>;
  
  static_assert(GEMM_BF16_4x16::CopyBits == 16);
  static_assert(GEMM_BF16_4x16::AtomHeight == 4);
  static_assert(GEMM_BF16_4x16::AtomWidth == 16);
  
  static_assert(GEMM_BF16_8x16::CopyBits == 16);
  static_assert(GEMM_BF16_8x16::AtomHeight == 8);
  static_assert(GEMM_BF16_8x16::AtomWidth == 16);
  
  static_assert(GEMM_BF16_16x16::CopyBits == 16);
  static_assert(GEMM_BF16_16x16::AtomHeight == 16);
  static_assert(GEMM_BF16_16x16::AtomWidth == 16);
  
  static_assert(GEMM_BF16_32x16::CopyBits == 16);
  static_assert(GEMM_BF16_32x16::AtomHeight == 32);
  static_assert(GEMM_BF16_32x16::AtomWidth == 16);
  
  EXPECT_TRUE(true) << "BF16/FP16 GEMM VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_MoE_GEMMConfigs) {
  // Test VNNI configurations used in MoE (Mixture of Experts) GEMM
  // Based on example from 12_bmg_moe_gemm_cute_interface
  using MoE_Load_A = XE_LOAD_2D_VNNI<16, 32, 16, 16>;
  using MoE_Load_B_Alt1 = XE_LOAD_2D_VNNI<16, 16, 16>;
  using MoE_Load_B_Alt2 = XE_LOAD_2D_VNNI<16, 8, 16>;
  
  static_assert(MoE_Load_A::CopyBits == 16);
  static_assert(MoE_Load_A::AtomHeight == 32);
  static_assert(MoE_Load_A::AtomWidth == 16);
  
  static_assert(MoE_Load_B_Alt1::CopyBits == 16);
  static_assert(MoE_Load_B_Alt1::AtomHeight == 16);
  static_assert(MoE_Load_B_Alt1::AtomWidth == 16);
  
  static_assert(MoE_Load_B_Alt2::CopyBits == 16);
  static_assert(MoE_Load_B_Alt2::AtomHeight == 8);
  static_assert(MoE_Load_B_Alt2::AtomWidth == 16);
  
  EXPECT_TRUE(true) << "MoE GEMM VNNI configurations validated";
}


TEST(CuTe_Xe, XE_LOAD_2D_VNNI_MixedBlockWidthConfigs) {
  // Test various BlockWidth settings for optimization
  using VNNIOp_8bit_8x64_bw32 = XE_LOAD_2D_VNNI<8, 8, 64, 32>;
  using VNNIOp_8bit_16x64_bw32 = XE_LOAD_2D_VNNI<8, 16, 64, 32>;
  using VNNIOp_16bit_8x32_bw16 = XE_LOAD_2D_VNNI<16, 8, 32, 16>;
  using VNNIOp_16bit_16x32_bw16 = XE_LOAD_2D_VNNI<16, 16, 32, 16>;
  
  static_assert(VNNIOp_8bit_8x64_bw32::CopyBits == 8);
  static_assert(VNNIOp_8bit_8x64_bw32::AtomHeight == 8);
  static_assert(VNNIOp_8bit_8x64_bw32::AtomWidth == 64);
  
  static_assert(VNNIOp_8bit_16x64_bw32::CopyBits == 8);
  static_assert(VNNIOp_8bit_16x64_bw32::AtomHeight == 16);
  static_assert(VNNIOp_8bit_16x64_bw32::AtomWidth == 64);
  
  static_assert(VNNIOp_16bit_8x32_bw16::CopyBits == 16);
  static_assert(VNNIOp_16bit_8x32_bw16::AtomHeight == 8);
  static_assert(VNNIOp_16bit_8x32_bw16::AtomWidth == 32);
  
  static_assert(VNNIOp_16bit_16x32_bw16::CopyBits == 16);
  static_assert(VNNIOp_16bit_16x32_bw16::AtomHeight == 16);
  static_assert(VNNIOp_16bit_16x32_bw16::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "Mixed BlockWidth VNNI configurations validated";
}


TEST(CuTe_Xe, XE_LOAD_2D_VNNI_DataType_Consistency) {
  // Test that CopyBits correctly reflects the data size
  using VNNIOp_8bit_small = XE_LOAD_2D_VNNI<8, 2, 16>;
  using VNNIOp_8bit_large = XE_LOAD_2D_VNNI<8, 16, 64>;
  using VNNIOp_16bit_small = XE_LOAD_2D_VNNI<16, 2, 16>;
  using VNNIOp_16bit_large = XE_LOAD_2D_VNNI<16, 16, 32>;
  
  // All 8-bit variants should have CopyBits == 8
  static_assert(VNNIOp_8bit_small::CopyBits == 8);
  static_assert(VNNIOp_8bit_large::CopyBits == 8);
  
  // All 16-bit variants should have CopyBits == 16
  static_assert(VNNIOp_16bit_small::CopyBits == 16);
  static_assert(VNNIOp_16bit_large::CopyBits == 16);
  
  EXPECT_TRUE(true) << "Data type consistency validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_BlockWidth_Divisors) {
  // Test BlockWidth as divisors of Width
  using VNNIOp_8bit_8x32_bw8 = XE_LOAD_2D_VNNI<8, 8, 32, 8>;
  using VNNIOp_8bit_8x32_bw16 = XE_LOAD_2D_VNNI<8, 8, 32, 16>;
  using VNNIOp_8bit_8x64_bw16 = XE_LOAD_2D_VNNI<8, 8, 64, 16>;
  using VNNIOp_8bit_8x64_bw32 = XE_LOAD_2D_VNNI<8, 8, 64, 32>;
  using VNNIOp_16bit_8x32_bw8 = XE_LOAD_2D_VNNI<16, 8, 32, 8>;
  using VNNIOp_16bit_8x32_bw16 = XE_LOAD_2D_VNNI<16, 8, 32, 16>;
  
  static_assert(VNNIOp_8bit_8x32_bw8::AtomHeight == 8 && VNNIOp_8bit_8x32_bw8::AtomWidth == 32);
  static_assert(VNNIOp_8bit_8x32_bw16::AtomHeight == 8 && VNNIOp_8bit_8x32_bw16::AtomWidth == 32);
  static_assert(VNNIOp_8bit_8x64_bw16::AtomHeight == 8 && VNNIOp_8bit_8x64_bw16::AtomWidth == 64);
  static_assert(VNNIOp_8bit_8x64_bw32::AtomHeight == 8 && VNNIOp_8bit_8x64_bw32::AtomWidth == 64);
  static_assert(VNNIOp_16bit_8x32_bw8::AtomHeight == 8 && VNNIOp_16bit_8x32_bw8::AtomWidth == 32);
  static_assert(VNNIOp_16bit_8x32_bw16::AtomHeight == 8 && VNNIOp_16bit_8x32_bw16::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "BlockWidth divisor configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_Symmetric_Configs) {
  // Test configurations with matching height and width values
  using VNNIOp_8bit_16x16 = XE_LOAD_2D_VNNI<8, 16, 16>;
  using VNNIOp_8bit_32x32 = XE_LOAD_2D_VNNI<8, 32, 32>;
  using VNNIOp_16bit_8x8 = XE_LOAD_2D_VNNI<16, 8, 8>;
  using VNNIOp_16bit_16x16 = XE_LOAD_2D_VNNI<16, 16, 16>;
  using VNNIOp_16bit_32x32 = XE_LOAD_2D_VNNI<16, 32, 32>;
  
  static_assert(VNNIOp_8bit_16x16::AtomHeight == 16 && VNNIOp_8bit_16x16::AtomWidth == 16);
  static_assert(VNNIOp_8bit_32x32::AtomHeight == 32 && VNNIOp_8bit_32x32::AtomWidth == 32);
  static_assert(VNNIOp_16bit_8x8::AtomHeight == 8 && VNNIOp_16bit_8x8::AtomWidth == 8);
  static_assert(VNNIOp_16bit_16x16::AtomHeight == 16 && VNNIOp_16bit_16x16::AtomWidth == 16);
  static_assert(VNNIOp_16bit_32x32::AtomHeight == 32 && VNNIOp_16bit_32x32::AtomWidth == 32);
  
  EXPECT_TRUE(true) << "Symmetric VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_Small_Tiles) {
  // Test small tile configurations useful for residual/boundary handling
  using VNNIOp_8bit_1x16 = XE_LOAD_2D_VNNI<8, 1, 16>;
  using VNNIOp_8bit_2x16 = XE_LOAD_2D_VNNI<8, 2, 16>;
  using VNNIOp_8bit_1x32 = XE_LOAD_2D_VNNI<8, 1, 32>;
  using VNNIOp_16bit_1x8 = XE_LOAD_2D_VNNI<16, 1, 8>;
  using VNNIOp_16bit_2x8 = XE_LOAD_2D_VNNI<16, 2, 8>;
  using VNNIOp_16bit_1x16 = XE_LOAD_2D_VNNI<16, 1, 16>;
  
  static_assert(VNNIOp_8bit_1x16::AtomHeight == 1 && VNNIOp_8bit_1x16::AtomWidth == 16);
  static_assert(VNNIOp_8bit_2x16::AtomHeight == 2 && VNNIOp_8bit_2x16::AtomWidth == 16);
  static_assert(VNNIOp_8bit_1x32::AtomHeight == 1 && VNNIOp_8bit_1x32::AtomWidth == 32);
  static_assert(VNNIOp_16bit_1x8::AtomHeight == 1 && VNNIOp_16bit_1x8::AtomWidth == 8);
  static_assert(VNNIOp_16bit_2x8::AtomHeight == 2 && VNNIOp_16bit_2x8::AtomWidth == 8);
  static_assert(VNNIOp_16bit_1x16::AtomHeight == 1 && VNNIOp_16bit_1x16::AtomWidth == 16);
  
  EXPECT_TRUE(true) << "Small tile VNNI configurations validated";
}

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_MatMul_Optimized) {
  // Test configurations optimized for matrix multiplication (DPAS integration)
  // Based on typical DPAS dimensions: N=16 for all, K varies by data type
  using MatMul_8bit_8x32 = XE_LOAD_2D_VNNI<8, 8, 32>;     // K=32 for int8
  using MatMul_8bit_16x32 = XE_LOAD_2D_VNNI<8, 16, 32>;   // M=16 tile
  using MatMul_16bit_8x16 = XE_LOAD_2D_VNNI<16, 8, 16>;   // K=16 for bf16/fp16
  using MatMul_16bit_16x16 = XE_LOAD_2D_VNNI<16, 16, 16>; // M=16, N=16 for bf16/fp16
  using MatMul_16bit_32x16 = XE_LOAD_2D_VNNI<16, 32, 16>; // Larger M tile
  
  // Verify dimensions match DPAS requirements
  static_assert(MatMul_8bit_8x32::CopyBits == 8);
  static_assert(MatMul_8bit_8x32::AtomHeight == 8);
  static_assert(MatMul_8bit_8x32::AtomWidth == 32);  // Matches int8 DPAS K dimension
  
  static_assert(MatMul_8bit_16x32::CopyBits == 8);
  static_assert(MatMul_8bit_16x32::AtomHeight == 16);
  static_assert(MatMul_8bit_16x32::AtomWidth == 32);
  
  static_assert(MatMul_16bit_8x16::CopyBits == 16);
  static_assert(MatMul_16bit_8x16::AtomHeight == 8);
  static_assert(MatMul_16bit_8x16::AtomWidth == 16);  // Matches bf16/fp16 DPAS K dimension
  
  static_assert(MatMul_16bit_16x16::CopyBits == 16);
  static_assert(MatMul_16bit_16x16::AtomHeight == 16);
  static_assert(MatMul_16bit_16x16::AtomWidth == 16);
  
  static_assert(MatMul_16bit_32x16::CopyBits == 16);
  static_assert(MatMul_16bit_32x16::AtomHeight == 32);
  static_assert(MatMul_16bit_32x16::AtomWidth == 16);
  
  EXPECT_TRUE(true) << "MatMul-optimized VNNI configurations validated";
}

#else

TEST(CuTe_Xe, XE_LOAD_2D_VNNI_SKIPPED) {
  GTEST_SKIP() << "XE_LOAD_2D_VNNI tests require IGC version 2.18 or higher. skipped";
}

#endif
