/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include "gemm_common.hpp"
#include "utils.hpp"

// ================  bfloat16_t  ================

TEST(PVC_CuTe_Xe, gemm_bf16_RowMajor_RowMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, bfloat16_t, bfloat16_t, float,
      XE_2D_U16x16x32_LD_N, XE_2D_U16x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32BF16BF16F32_TT,
      cute::LayoutRight, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_bf16_RowMajor_ColumnMajor) {
  run<gemm_device_partition_fragment_abc<
      64, 128, 32, 64, 128, bfloat16_t, bfloat16_t, float,
      XE_2D_U16x16x32_LD_N, XE_2D_U16x16x16_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32BF16BF16F32_TT,
      cute::LayoutRight, cute::LayoutRight>>();
}

TEST(PVC_CuTe_Xe, gemm_bf16_ColumnMajor_RowMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, bfloat16_t, bfloat16_t, float,
      XE_2D_U16x16x16_LD_T, XE_2D_U16x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32BF16BF16F32_TT,
      cute::LayoutLeft, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_bf16_ColumnMajor_ColumnMajor) {
  run<gemm_device_partition_fragment_abc<
      128, 128, 32, 64, 128, bfloat16_t, bfloat16_t, float,
      XE_2D_U16x16x16_LD_T, XE_2D_U16x16x16_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32BF16BF16F32_TT,
      cute::LayoutLeft, cute::LayoutRight>>();
}



// ================  fp16  ================

TEST(PVC_CuTe_Xe, gemm_fp16_RowMajor_RowMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, half_t, half_t, float,
      XE_2D_U16x32x32_LD_N, XE_2D_U16x32x32_LD_N,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32F16F16F32_TT,
      cute::LayoutRight, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_fp16_RowMajor_ColumnMajor) {
  run<gemm_device_partition_fragment_abc<
      64, 128, 32, 64, 128, half_t, half_t, float,
      XE_2D_U16x32x32_LD_N, XE_2D_U16x16x16_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32F16F16F32_TT,
      cute::LayoutRight, cute::LayoutRight>>();
}

TEST(PVC_CuTe_Xe, gemm_fp16_ColumnMajor_RowMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, half_t, half_t, float,
      XE_2D_U16x16x16_LD_T, XE_2D_U16x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32F16F16F32_TT,
      cute::LayoutLeft, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_fp16_ColumnMajor_ColumnMajor) {
  run<gemm_device_partition_fragment_abc<
      128, 128, 32, 64, 128, half_t, half_t, float,
      XE_2D_U16x16x16_LD_T, XE_2D_U16x16x16_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x16_F32F16F16F32_TT,
      cute::LayoutLeft, cute::LayoutRight>>();
}



// ================  tfloat32_t  ================

TEST(PVC_CuTe_Xe, gemm_tf32_RowMajor_RowMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, tfloat32_t, tfloat32_t, float,
      XE_2D_TF32x32x16_LD_N, XE_2D_U32x32x16_LD_N,
      XE_2D_U32x8x16_ST_N, XE_8x16x8_F32TF32TF32F32_TT,
      cute::LayoutRight, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_tf32_RowMajor_ColumnMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, tfloat32_t, tfloat32_t, float,
      XE_2D_TF32x32x16_LD_N, XE_2D_U32x16x8_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x8_F32TF32TF32F32_TT,
      cute::LayoutRight, cute::LayoutRight>>();
}

TEST(PVC_CuTe_Xe, gemm_tf32_ColumnMajor_RowMajor) {
  GTEST_SKIP() << "Not support tf32 column major A now";
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, tfloat32_t, tfloat32_t, float,
      XE_2D_TF32x16x8_LD_T, XE_2D_U32x32x16_LD_N,
      XE_2D_U32x8x16_ST_N, XE_8x16x8_F32TF32TF32F32_TT,
      cute::LayoutLeft, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_tf32_ColumnMajor_ColumnMajor) {
  GTEST_SKIP() << "Not support tf32 column major A now";
  run<gemm_device_partition_fragment_abc<
      128, 128, 32, 64, 128, tfloat32_t, tfloat32_t, float,
      XE_2D_TF32x16x8_LD_T, XE_2D_U32x16x8_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x8_F32TF32TF32F32_TT,
      cute::LayoutLeft, cute::LayoutRight>>();
}



// ================  int8_t  ================

TEST(PVC_CuTe_Xe, gemm_int8_RowMajor_RowMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 256, 32, 64, 32, int8_t, int8_t, int32_t,
      XE_2D_U8x32x32_LD_N, XE_2D_U8x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32S8S8S32_TT,
      cute::LayoutRight, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_int8_RowMajor_ColumnMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 256, 32, 64, 32, int8_t, int8_t, int32_t,
      XE_2D_U8x32x32_LD_N, XE_2D_U8x32x16_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32S8S8S32_TT,
      cute::LayoutRight, cute::LayoutRight>>();
}

TEST(PVC_CuTe_Xe, gemm_int8_ColumnMajor_RowMajor) {
  GTEST_SKIP() << "Not support int8 column major A now";
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, int8_t, int8_t, int32_t,
      XE_2D_U8x32x32_LD_T, XE_2D_U8x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32S8S8S32_TT,
      cute::LayoutLeft, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_int8_ColumnMajor_ColumnMajor) {
  GTEST_SKIP() << "Not support int8 column major A now";
  run<gemm_device_partition_fragment_abc<
      128, 128, 32, 64, 128, int8_t, int8_t, int32_t,
      XE_2D_U8x32x32_LD_T, XE_2D_U8x32x16_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32S8S8S32_TT,
      cute::LayoutLeft, cute::LayoutRight>>();
}



// ================  uint8_t  ================

TEST(PVC_CuTe_Xe, gemm_uint8_RowMajor_RowMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 256, 32, 64, 32, uint8_t, uint8_t, int32_t,
      XE_2D_U8x32x32_LD_N, XE_2D_U8x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32U8U8S32_TT,
      cute::LayoutRight, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_uint8_RowMajor_ColumnMajor) {
  run<gemm_device_partition_fragment_abc<
      256, 256, 32, 64, 32, uint8_t, uint8_t, int32_t,
      XE_2D_U8x32x32_LD_N, XE_2D_U8x32x16_LD_T,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32U8U8S32_TT,
      cute::LayoutRight, cute::LayoutRight>>();
}

TEST(PVC_CuTe_Xe, gemm_uint8_ColumnMajor_RowMajor) {
  GTEST_SKIP() << "Not support uint8 column major A now";
  run<gemm_device_partition_fragment_abc<
      256, 128, 32, 64, 128, uint8_t, uint8_t, int32_t,
      XE_2D_U8x32x32_LD_T, XE_2D_U8x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32U8U8S32_TT,
      cute::LayoutLeft, cute::LayoutLeft>>();
}

TEST(PVC_CuTe_Xe, gemm_uint8_ColumnMajor_ColumnMajor) {
  GTEST_SKIP() << "Not support uint8 column major A now";
  run<gemm_device_partition_fragment_abc<
      128, 128, 32, 64, 128, uint8_t, uint8_t, int32_t,
      XE_2D_U8x32x32_LD_T, XE_2D_U8x32x32_LD_V,
      XE_2D_U32x8x16_ST_N, XE_8x16x32_S32U8U8S32_TT,
      cute::LayoutLeft, cute::LayoutRight>>();
}
