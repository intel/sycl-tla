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
using MMAAtom = MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>;

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

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using BmgGemm_BF16FP32_TileShape_8_128_32 = Shape<_8, _128, _32>;
using BmgTile_6 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_8_128_32>, Layout<Shape<_1, _4, _1>, Stride<_0, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RCR_5 = Gemm_Bench_BF16FP32_RCR<BmgGemm_BF16FP32_TileShape_8_128_32, BmgTile_6, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_5);

using BmgTile_7 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_8_128_32>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RCR_7 = Gemm_Bench_BF16FP32_RCR<BmgGemm_BF16FP32_TileShape_8_128_32, BmgTile_7, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_7);

using BmgGemm_BF16FP32_TileShape_8_64_32 = Shape<_8, _64, _32>;
using BmgTile_8 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_8_64_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RCR_9 = Gemm_Bench_BF16FP32_RCR<BmgGemm_BF16FP32_TileShape_8_64_32, BmgTile_8, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_9);

using BmgGemm_BF16FP32_TileShape_16_64_32 = Shape<_16, _64, _32>;
using BmgTile_9 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_16_64_32>, Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RCR_16 = Gemm_Bench_BF16FP32_RCR<BmgGemm_BF16FP32_TileShape_16_64_32, BmgTile_9, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_16);

using BmgTile_1 = TiledMMA<MMAAtom, Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>, Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>, Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;
using BmgGemmBF16BF16FP32_RCR_6 = Gemm_Bench_BF16FP32_RCR<Shape<_256, _256, _32>, BmgTile_1, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_6);

static void register_gemm_benchmarks() {
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_5);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_6);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_7);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_9);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_16);
}
