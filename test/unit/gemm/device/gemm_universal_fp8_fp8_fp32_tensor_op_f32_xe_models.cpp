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
#include <gtest/gtest.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "default_gemm_configuration.hpp"
#include "gemm_testbed_3x.hpp"

using namespace cutlass;

namespace {

template<typename LayoutA, typename LayoutB>
struct MainloopIntelW8A8_GemmConfig {
    using ElementA = float_e5m2_t;
    using ElementB = float_e5m2_t;
    using TileShape = Shape<_256, _256, _32>;
    constexpr static int PipelineStages = 2;
    using Schedule = gemm::KernelXe;
    using TiledMma = typename TiledMMAHelper<
        MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
        Layout<TileShape>,
        Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
    >::TiledMMA;
    using GmemTiledCopyA = XE_2D_U8x32x32_LD_N;
    using GmemTiledCopyB = XE_2D_U8x32x32_LD_V;

    using DispatchPolicy = gemm::MainloopIntelW8A8<PipelineStages, Schedule>;

    using CollectiveMainloop = gemm::collective::CollectiveMma<
        DispatchPolicy, TileShape,
        ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>,
        ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>,
        TiledMma,
        GmemTiledCopyA, void, void, cute::identity,  // A
        GmemTiledCopyB, void, void, cute::identity   // B
    >;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        float, float
    >;

    using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
        cutlass::epilogue::IntelXeXMX16,
        EpilogueOp,
        TileShape,
        decltype(tile_shape(TiledMma()))
    >;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
        cutlass::epilogue::IntelXeXMX16,
        TileShape,
        float, cutlass::gemm::TagToStrideC_t<layout::RowMajor>,
        float, cutlass::gemm::TagToStrideC_t<layout::RowMajor>,
        FusionCallBacks,
        XE_2D_U32x8x16_LD_N, void, void,
        XE_2D_U32x8x16_ST_N, void, void
    >;

    using GemmKernel = gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;
};

TEST(MainloopIntelW8A8_Special, LargeModel_LLaMA2_7B) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(4096, 4096, 11008, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, LargeModel_Mistral_7B) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(4096, 4096, 14336, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, TensorParallel) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(4096, 1024, 4096, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, ModelParallel) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1024, 4096, 4096, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, MicroBatch) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(128, 128, 8192, 4, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, LargeBatch) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(512, 512, 2048, 32, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, SquareSmall) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(64, 64, 64, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, SquareMedium) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(512, 512, 512, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, SquareLarge) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(2048, 2048, 2048, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, TallMatrix) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(4096, 512, 4096, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, WideMatrix) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(512, 4096, 4096, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, Batch8) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(512, 512, 512, 8, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, Batch16Large) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1024, 1024, 1024, 16, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, LargeK) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(64, 64, 8192, 1, 1.0, 0.0));
}

TEST(MainloopIntelW8A8_Special, LargeN) {
    using Gemm = typename MainloopIntelW8A8_GemmConfig<layout::RowMajor, layout::RowMajor>::Gemm;
    EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(64, 8192, 64, 1, 1.0, 0.0));
}

} // namespace
