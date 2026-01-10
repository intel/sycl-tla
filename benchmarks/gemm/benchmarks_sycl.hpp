/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

 #pragma once

#include "gemm_configuration_sycl.hpp"

using Scheduler = cutlass::gemm::device::Scheduler;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using BmgGemm_BF16FP32_TileShape_512_256_32 = Shape<_512, _256, _32>;
using BmgGemm_BF16FP32_Tile_512_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_512_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_512_256_32, BmgGemm_BF16FP32_Tile_512_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32);

static void register_gemm_benchmarks() {
// TODO: support sglang cases
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32);
}
