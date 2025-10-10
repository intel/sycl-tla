/***************************************************************************************************
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
/*! \file
    \brief Tests for device-wide GEMM interface

*/

>>>>>>> d6137b22 (Gemm Universal unit tests for MainloopIntelW8A8 API)
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "default_gemm_configuration.hpp"
#include "gemm_testbed_3x.hpp"

namespace cutlass {
namespace {

template <typename LayoutA, typename LayoutB>
struct XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative {
  using ElementA = float_e5m2_t;
  using ElementB = float_e5m2_t;

  using Config = gemm::device::DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    ElementA, LayoutA,
    ElementB, LayoutB,
    float, layout::RowMajor,
    float>;

  using DispatchPolicy = gemm::MainloopIntelW8A8<2, gemm::KernelXe>;

  using CollectiveMainloop = gemm::collective::CollectiveMma<
    DispatchPolicy, typename Config::TileShape,
    ElementA, detail::TagToStrideA_t<LayoutA>,
    ElementB, detail::TagToStrideB_t<LayoutB>,
    typename Config::TiledMma,
    typename Config::GmemTiledCopyA, void, void, cute::identity,
    typename Config::GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      CollectiveMainloop,
      typename Config::CollectiveEpilogue
      >;

  using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;
};

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, llama2_7b) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, gpt3) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, mistral_7b) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, tensor_parallel) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, model_parallel) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, micro_batch) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, large_batch) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8n_f32t_tensor_op_f32_cooperative, tensor_parallel) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::ColumnMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8n_fp8t_f32t_tensor_op_f32_cooperative, model_parallel) {
  using LayoutA = layout::ColumnMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8n_fp8n_f32t_tensor_op_f32_cooperative, model_parallel) {
  using LayoutA = layout::ColumnMajor;
  using LayoutB = layout::ColumnMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

} // namespace
} // namespace cutlass
 
