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

#define BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, M, N, K) \
using PREFIX##_StreamK_##M##x##N##x##K = \
    CONFIG<Shape<_##M, _##N, _##K>, Scheduler::GemmStreamK>; \
using PREFIX##_DataParallel_##M##x##N##x##K = \
    CONFIG<Shape<_##M, _##N, _##K>, Scheduler::GemmDataParallel>; \
using PREFIX##_SplitK_##M##x##N##x##K = \
    CONFIG<Shape<_##M, _##N, _##K>, Scheduler::GemmSplitK>; \
CUTLASS_CREATE_GEMM_BENCHMARK(PREFIX##_StreamK_##M##x##N##x##K); \
CUTLASS_CREATE_GEMM_BENCHMARK(PREFIX##_DataParallel_##M##x##N##x##K); \
CUTLASS_CREATE_GEMM_BENCHMARK(PREFIX##_SplitK_##M##x##N##x##K);

#define BMG_REGISTER_STREAMK_TILE(PREFIX, M, N, K) \
  CUTLASS_BENCHMARK(PREFIX##_StreamK_##M##x##N##x##K); \
  CUTLASS_BENCHMARK(PREFIX##_DataParallel_##M##x##N##x##K); \
  CUTLASS_BENCHMARK(PREFIX##_SplitK_##M##x##N##x##K);

#define BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, M, N, K) \
using PREFIX##_Gemm_TileShape_##M##x##N##x##K = Shape<_##M, _##N, _##K>; \
using PREFIX##_Gemm_Tile_##M##x##N##x##K##_SG8x4 = typename TiledMMAHelper< \
    MMA_ATOM, Layout<PREFIX##_Gemm_TileShape_##M##x##N##x##K>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA; \
using PREFIX##_Gemm_##M##x##N##x##K##_SG8x4 = CONFIG< \
    PREFIX##_Gemm_TileShape_##M##x##N##x##K, PREFIX##_Gemm_Tile_##M##x##N##x##K##_SG8x4, void, void>; \
CUTLASS_CREATE_GEMM_BENCHMARK(PREFIX##_Gemm_##M##x##N##x##K##_SG8x4);

#define BMG_REGISTER_GEMM_TILE(PREFIX, M, N, K) \
  CUTLASS_BENCHMARK(PREFIX##_Gemm_##M##x##N##x##K##_SG8x4);

#define BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE(PREFIX, CONFIG, MMA_ATOM, M, N, K, SGM, SGN, STAGES) \
using PREFIX##_GemmExhaustive_TileShape_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES = Shape<_##M, _##N, _##K>; \
using PREFIX##_GemmExhaustive_Tile_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES = typename TiledMMAHelper< \
    MMA_ATOM, Layout<PREFIX##_GemmExhaustive_TileShape_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES>, Layout<Shape<_##SGM, _##SGN, _1>, Stride<_##SGN, _1, _0>>>::TiledMMA; \
using PREFIX##_GemmExhaustive_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES = CONFIG< \
    PREFIX##_GemmExhaustive_TileShape_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES, PREFIX##_GemmExhaustive_Tile_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES, void, void, STAGES>; \
CUTLASS_CREATE_GEMM_BENCHMARK(PREFIX##_GemmExhaustive_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES);

#define BMG_REGISTER_EXHAUSTIVE_GEMM_TILE_STAGE(PREFIX, M, N, K, SGM, SGN, STAGES) \
  CUTLASS_BENCHMARK(PREFIX##_GemmExhaustive_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES);

#define BMG_DECLARE_GEMM_TILE_SG(PREFIX, CONFIG, MMA_ATOM, M, N, K, SGM, SGN) \
using PREFIX##_Gemm_TileShape_##M##x##N##x##K = Shape<_##M, _##N, _##K>; \
using PREFIX##_Gemm_Tile_##M##x##N##x##K##_SG##SGM##x##SGN = typename TiledMMAHelper< \
    MMA_ATOM, Layout<PREFIX##_Gemm_TileShape_##M##x##N##x##K>, Layout<Shape<_##SGM, _##SGN, _1>, Stride<_##SGN, _1, _0>>>::TiledMMA; \
using PREFIX##_Gemm_##M##x##N##x##K##_SG##SGM##x##SGN = CONFIG< \
    PREFIX##_Gemm_TileShape_##M##x##N##x##K, PREFIX##_Gemm_Tile_##M##x##N##x##K##_SG##SGM##x##SGN, void, void>; \
CUTLASS_CREATE_GEMM_BENCHMARK(PREFIX##_Gemm_##M##x##N##x##K##_SG##SGM##x##SGN);

#define BMG_REGISTER_GEMM_TILE_SG(PREFIX, M, N, K, SGM, SGN) \
  CUTLASS_BENCHMARK(PREFIX##_Gemm_##M##x##N##x##K##_SG##SGM##x##SGN);

#define BMG_DECLARE_EXPANDED_GEMM_TILES(PREFIX, CONFIG, MMA_ATOM) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 64, 64, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 64, 64, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 64, 128, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 64, 128, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 64, 256, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 64, 256, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 64, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 64, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 128, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 128, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 256, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 256, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 256, 16) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 128, 512, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 64, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 64, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 128, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 128, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 192, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 256, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 256, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 256, 256, 16) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 512, 64, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 512, 64, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 512, 128, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 512, 128, 64) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 512, 256, 32) \
BMG_DECLARE_GEMM_TILE(PREFIX, CONFIG, MMA_ATOM, 512, 256, 64)

#define BMG_REGISTER_EXPANDED_GEMM_TILES(PREFIX) \
BMG_REGISTER_GEMM_TILE(PREFIX, 64, 64, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 64, 64, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 64, 128, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 64, 128, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 64, 256, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 64, 256, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 64, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 64, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 128, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 128, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 256, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 256, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 256, 16) \
BMG_REGISTER_GEMM_TILE(PREFIX, 128, 512, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 64, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 64, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 128, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 128, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 192, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 256, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 256, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 256, 256, 16) \
BMG_REGISTER_GEMM_TILE(PREFIX, 512, 64, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 512, 64, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 512, 128, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 512, 128, 64) \
BMG_REGISTER_GEMM_TILE(PREFIX, 512, 256, 32) \
BMG_REGISTER_GEMM_TILE(PREFIX, 512, 256, 64)

#define BMG_DECLARE_EXPANDED_STREAMK_TILES(PREFIX, CONFIG) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 64, 64, 32) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 64, 64, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 64, 128, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 64, 256, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 128, 64, 32) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 128, 64, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 128, 128, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 128, 256, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 256, 64, 32) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 256, 64, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 256, 128, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 256, 256, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 512, 64, 32) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 512, 64, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 512, 128, 64) \
BMG_DECLARE_STREAMK_TILE(PREFIX, CONFIG, 512, 256, 64)

#define BMG_REGISTER_EXPANDED_STREAMK_TILES(PREFIX) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 64, 64, 32) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 64, 64, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 64, 128, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 64, 256, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 128, 64, 32) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 128, 64, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 128, 128, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 128, 256, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 256, 64, 32) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 256, 64, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 256, 128, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 256, 256, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 512, 64, 32) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 512, 64, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 512, 128, 64) \
BMG_REGISTER_STREAMK_TILE(PREFIX, 512, 256, 64)

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB,
  int PipelineStages = 2>
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<float, float, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    PipelineStages>;

using BmgGemm_BF16FP32_TileShape_512_256_32 = Shape<_512, _256, _32>;
using BmgGemm_BF16FP32_Tile_512_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_512_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_512_256_32, BmgGemm_BF16FP32_Tile_512_256_32, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32);
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_GEMM_TILES(BmgGemmBF16BF16FP32_RRR, Gemm_Bench_BF16FP32_RRR, MMAAtom)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_DECLARE_GEMM_TILE_SG(BmgGemmBF16BF16FP32_RRR, Gemm_Bench_BF16FP32_RRR, MMAAtom, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB,
  int PipelineStages = 2>
using Gemm_Bench_BF16FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<float, float, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    PipelineStages>;

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

using BmgGemm_BF16FP32_TileShape_64_128_32 = Shape<_64, _128, _32>;
using BmgTile_17 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_64_128_32>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RCR_17 = Gemm_Bench_BF16FP32_RCR<BmgGemm_BF16FP32_TileShape_64_128_32, BmgTile_17, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_17);

using BmgGemm_BF16FP32_TileShape_128_128_32 = Shape<_128, _128, _32>;
using BmgTile_18 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_128_128_32>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RCR_18 = Gemm_Bench_BF16FP32_RCR<BmgGemm_BF16FP32_TileShape_128_128_32, BmgTile_18, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_18);

using BmgGemm_BF16FP32_TileShape_128_256_32 = Shape<_128, _256, _32>;
using BmgTile_19 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_128_256_32>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RCR_19 = Gemm_Bench_BF16FP32_RCR<BmgGemm_BF16FP32_TileShape_128_256_32, BmgTile_19, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_19);

using BmgTile_1 = TiledMMA<MMAAtom, Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>, Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>, Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;
using BmgGemmBF16BF16FP32_RRR_6 = Gemm_Bench_BF16FP32_RRR<Shape<_256, _256, _32>, BmgTile_1, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RRR_6);
using BmgGemmBF16BF16FP32_RCR_6 = Gemm_Bench_BF16FP32_RCR<Shape<_256, _256, _32>, BmgTile_1, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_6);
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_GEMM_TILES(BmgGemmBF16BF16FP32_RCR, Gemm_Bench_BF16FP32_RCR, MMAAtom)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_DECLARE_GEMM_TILE_SG(BmgGemmBF16BF16FP32_RCR, Gemm_Bench_BF16FP32_RCR, MMAAtom, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif

template <typename TileShape>
using BmgGemm_BF16BF16FP32_StreamK_Tile =
    typename TiledMMAHelper<
        MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>,
        Layout<TileShape>,
        Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemm_BF16BF16FP32_StreamK_Epilogue =
    cutlass::epilogue::fusion::LinearCombination<
        float,
        float,
        float,
        float,
        cutlass::FloatRoundStyle::round_to_nearest>;

template <typename TileShape, Scheduler TileScheduler>
using Gemm_Bench_BF16BF16FP32_RCR_StreamK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape,
    TileScheduler,
    BmgGemm_BF16BF16FP32_StreamK_Tile<TileShape>,
    void,
    void,
    BmgGemm_BF16BF16FP32_StreamK_Epilogue,
    2,
    cutlass::gemm::KernelXeCooperative>;

using BmgGemm_BF16BF16FP32_StreamK_TileShape_128_128_32 = Shape<_128, _128, _32>;
using BmgGemm_BF16BF16FP32_StreamK_TileShape_128_256_32 = Shape<_128, _256, _32>;
using BmgGemm_BF16BF16FP32_StreamK_TileShape_256_128_32 = Shape<_256, _128, _32>;
using BmgGemm_BF16BF16FP32_StreamK_TileShape_256_256_32 = Shape<_256, _256, _32>;

using BmgGemmBF16BF16FP32_RCR_StreamK_128x128x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_128_128_32, Scheduler::GemmStreamK>;
using BmgGemmBF16BF16FP32_RCR_DataParallel_128x128x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_128_128_32, Scheduler::GemmDataParallel>;
using BmgGemmBF16BF16FP32_RCR_SplitK_128x128x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_128_128_32, Scheduler::GemmSplitK>;
using BmgGemmBF16BF16FP32_RCR_StreamK_128x256x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_128_256_32, Scheduler::GemmStreamK>;
using BmgGemmBF16BF16FP32_RCR_DataParallel_128x256x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_128_256_32, Scheduler::GemmDataParallel>;
using BmgGemmBF16BF16FP32_RCR_SplitK_128x256x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_128_256_32, Scheduler::GemmSplitK>;
using BmgGemmBF16BF16FP32_RCR_StreamK_256x128x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_256_128_32, Scheduler::GemmStreamK>;
using BmgGemmBF16BF16FP32_RCR_DataParallel_256x128x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_256_128_32, Scheduler::GemmDataParallel>;
using BmgGemmBF16BF16FP32_RCR_SplitK_256x128x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_256_128_32, Scheduler::GemmSplitK>;
using BmgGemmBF16BF16FP32_RCR_StreamK_256x256x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_256_256_32, Scheduler::GemmStreamK>;
using BmgGemmBF16BF16FP32_RCR_DataParallel_256x256x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_256_256_32, Scheduler::GemmDataParallel>;
using BmgGemmBF16BF16FP32_RCR_SplitK_256x256x32 =
    Gemm_Bench_BF16BF16FP32_RCR_StreamK<BmgGemm_BF16BF16FP32_StreamK_TileShape_256_256_32, Scheduler::GemmSplitK>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_256x256x32);
BMG_DECLARE_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, Gemm_Bench_BF16BF16FP32_RCR_StreamK, 64, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, Gemm_Bench_BF16BF16FP32_RCR_StreamK, 64, 256, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, Gemm_Bench_BF16BF16FP32_RCR_StreamK, 512, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, Gemm_Bench_BF16BF16FP32_RCR_StreamK, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_STREAMK_TILES(BmgGemmBF16BF16FP32_RCR, Gemm_Bench_BF16BF16FP32_RCR_StreamK)
#endif

using MMAAtomTF32 = MMA_Atom<XE_DPAS_TT<8, float, tfloat32_t, tfloat32_t, float>>;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB,
  int PipelineStages = 2>
using Gemm_Bench_TF32FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<float, float, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    PipelineStages>;

#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_GEMM_TILES(BmgGemmTF32TF32FP32_RCR, Gemm_Bench_TF32FP32_RCR, MMAAtomTF32)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_DECLARE_GEMM_TILE_SG(BmgGemmTF32TF32FP32_RCR, Gemm_Bench_TF32FP32_RCR, MMAAtomTF32, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif

template <typename TileShape>
using BmgGemm_TF32TF32FP32_StreamK_Tile =
    typename TiledMMAHelper<
        MMAAtomTF32,
        Layout<TileShape>,
        Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemm_TF32TF32FP32_StreamK_Epilogue =
    cutlass::epilogue::fusion::LinearCombination<
        float,
        float,
        float,
        float,
        cutlass::FloatRoundStyle::round_to_nearest>;

template <typename TileShape, Scheduler TileScheduler>
using Gemm_Bench_TF32TF32FP32_RCR_StreamK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape,
    TileScheduler,
    BmgGemm_TF32TF32FP32_StreamK_Tile<TileShape>,
    void,
    void,
    BmgGemm_TF32TF32FP32_StreamK_Epilogue,
    2,
    cutlass::gemm::KernelXeCooperative>;

using BmgGemm_TF32TF32FP32_StreamK_TileShape_128_128_32 = Shape<_128, _128, _32>;
using BmgGemm_TF32TF32FP32_StreamK_TileShape_128_256_32 = Shape<_128, _256, _32>;
using BmgGemm_TF32TF32FP32_StreamK_TileShape_256_128_32 = Shape<_256, _128, _32>;
using BmgGemm_TF32TF32FP32_StreamK_TileShape_256_256_32 = Shape<_256, _256, _32>;

using BmgGemmTF32TF32FP32_RCR_StreamK_128x128x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_128_128_32, Scheduler::GemmStreamK>;
using BmgGemmTF32TF32FP32_RCR_DataParallel_128x128x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_128_128_32, Scheduler::GemmDataParallel>;
using BmgGemmTF32TF32FP32_RCR_SplitK_128x128x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_128_128_32, Scheduler::GemmSplitK>;
using BmgGemmTF32TF32FP32_RCR_StreamK_128x256x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_128_256_32, Scheduler::GemmStreamK>;
using BmgGemmTF32TF32FP32_RCR_DataParallel_128x256x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_128_256_32, Scheduler::GemmDataParallel>;
using BmgGemmTF32TF32FP32_RCR_SplitK_128x256x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_128_256_32, Scheduler::GemmSplitK>;
using BmgGemmTF32TF32FP32_RCR_StreamK_256x128x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_256_128_32, Scheduler::GemmStreamK>;
using BmgGemmTF32TF32FP32_RCR_DataParallel_256x128x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_256_128_32, Scheduler::GemmDataParallel>;
using BmgGemmTF32TF32FP32_RCR_SplitK_256x128x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_256_128_32, Scheduler::GemmSplitK>;
using BmgGemmTF32TF32FP32_RCR_StreamK_256x256x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_256_256_32, Scheduler::GemmStreamK>;
using BmgGemmTF32TF32FP32_RCR_DataParallel_256x256x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_256_256_32, Scheduler::GemmDataParallel>;
using BmgGemmTF32TF32FP32_RCR_SplitK_256x256x32 =
    Gemm_Bench_TF32TF32FP32_RCR_StreamK<BmgGemm_TF32TF32FP32_StreamK_TileShape_256_256_32, Scheduler::GemmSplitK>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_256x256x32);
BMG_DECLARE_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, Gemm_Bench_TF32TF32FP32_RCR_StreamK, 64, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, Gemm_Bench_TF32TF32FP32_RCR_StreamK, 64, 256, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, Gemm_Bench_TF32TF32FP32_RCR_StreamK, 512, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, Gemm_Bench_TF32TF32FP32_RCR_StreamK, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_STREAMK_TILES(BmgGemmTF32TF32FP32_RCR, Gemm_Bench_TF32TF32FP32_RCR_StreamK)
#endif

using MMAAtomF16 = MMA_Atom<XE_8x16x16_F32F16F16F32_TT>;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB,
  int PipelineStages = 2>
using Gemm_Bench_F16FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<float, float, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    PipelineStages>;

using BmgF16Tile_6 = TiledMMAHelper<MMAAtomF16, Layout<BmgGemm_BF16FP32_TileShape_8_128_32>, Layout<Shape<_1, _4, _1>, Stride<_0, _1, _0>>>::TiledMMA;
using BmgGemmFP16FP16FP32_RCR_5 = Gemm_Bench_F16FP32_RCR<BmgGemm_BF16FP32_TileShape_8_128_32, BmgF16Tile_6, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_5);

using BmgF16Tile_7 = TiledMMAHelper<MMAAtomF16, Layout<BmgGemm_BF16FP32_TileShape_8_128_32>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using BmgGemmFP16FP16FP32_RCR_7 = Gemm_Bench_F16FP32_RCR<BmgGemm_BF16FP32_TileShape_8_128_32, BmgF16Tile_7, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_7);

using BmgF16Tile_8 = TiledMMAHelper<MMAAtomF16, Layout<BmgGemm_BF16FP32_TileShape_8_64_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmFP16FP16FP32_RCR_9 = Gemm_Bench_F16FP32_RCR<BmgGemm_BF16FP32_TileShape_8_64_32, BmgF16Tile_8, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_9);

using BmgF16Tile_9 = TiledMMAHelper<MMAAtomF16, Layout<BmgGemm_BF16FP32_TileShape_16_64_32>, Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmFP16FP16FP32_RCR_16 = Gemm_Bench_F16FP32_RCR<BmgGemm_BF16FP32_TileShape_16_64_32, BmgF16Tile_9, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_16);

using BmgF16Tile_17 = TiledMMAHelper<MMAAtomF16, Layout<BmgGemm_BF16FP32_TileShape_64_128_32>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmFP16FP16FP32_RCR_17 = Gemm_Bench_F16FP32_RCR<BmgGemm_BF16FP32_TileShape_64_128_32, BmgF16Tile_17, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_17);

using BmgF16Tile_18 = TiledMMAHelper<MMAAtomF16, Layout<BmgGemm_BF16FP32_TileShape_128_128_32>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmFP16FP16FP32_RCR_18 = Gemm_Bench_F16FP32_RCR<BmgGemm_BF16FP32_TileShape_128_128_32, BmgF16Tile_18, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_18);

using BmgF16Tile_19 = TiledMMAHelper<MMAAtomF16, Layout<BmgGemm_BF16FP32_TileShape_128_256_32>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmFP16FP16FP32_RCR_19 = Gemm_Bench_F16FP32_RCR<BmgGemm_BF16FP32_TileShape_128_256_32, BmgF16Tile_19, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_19);

using BmgF16Tile_1 = TiledMMA<MMAAtomF16, Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>, Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>, Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;
using BmgGemmFP16FP16FP32_RCR_6 = Gemm_Bench_F16FP32_RCR<Shape<_256, _256, _32>, BmgF16Tile_1, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmFP16FP16FP32_RCR_6);
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_GEMM_TILES(BmgGemmFP16FP16FP32_RCR, Gemm_Bench_F16FP32_RCR, MMAAtomF16)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_DECLARE_GEMM_TILE_SG(BmgGemmFP16FP16FP32_RCR, Gemm_Bench_F16FP32_RCR, MMAAtomF16, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif

template <typename TileShape>
using BmgGemm_F16F16FP32_StreamK_Tile =
    typename TiledMMAHelper<
        MMA_Atom<XE_DPAS_TT<8, float, half_t, half_t, float>>,
        Layout<TileShape>,
        Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemm_F16F16FP32_StreamK_Epilogue =
    cutlass::epilogue::fusion::LinearCombination<
        float,
        float,
        float,
        float,
        cutlass::FloatRoundStyle::round_to_nearest>;

template <typename TileShape, Scheduler TileScheduler>
using Gemm_Bench_F16F16FP32_RCR_StreamK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape,
    TileScheduler,
    BmgGemm_F16F16FP32_StreamK_Tile<TileShape>,
    void,
    void,
    BmgGemm_F16F16FP32_StreamK_Epilogue,
    2,
    cutlass::gemm::KernelXeCooperative>;

using BmgGemm_F16F16FP32_StreamK_TileShape_128_128_32 = Shape<_128, _128, _32>;
using BmgGemm_F16F16FP32_StreamK_TileShape_128_256_32 = Shape<_128, _256, _32>;
using BmgGemm_F16F16FP32_StreamK_TileShape_256_128_32 = Shape<_256, _128, _32>;
using BmgGemm_F16F16FP32_StreamK_TileShape_256_256_32 = Shape<_256, _256, _32>;

using BmgGemmF16F16FP32_RCR_StreamK_128x128x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_128_128_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16FP32_RCR_DataParallel_128x128x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_128_128_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16FP32_RCR_SplitK_128x128x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_128_128_32, Scheduler::GemmSplitK>;
using BmgGemmF16F16FP32_RCR_StreamK_128x256x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_128_256_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16FP32_RCR_DataParallel_128x256x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_128_256_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16FP32_RCR_SplitK_128x256x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_128_256_32, Scheduler::GemmSplitK>;
using BmgGemmF16F16FP32_RCR_StreamK_256x128x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_256_128_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16FP32_RCR_DataParallel_256x128x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_256_128_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16FP32_RCR_SplitK_256x128x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_256_128_32, Scheduler::GemmSplitK>;
using BmgGemmF16F16FP32_RCR_StreamK_256x256x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_256_256_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16FP32_RCR_DataParallel_256x256x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_256_256_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16FP32_RCR_SplitK_256x256x32 =
    Gemm_Bench_F16F16FP32_RCR_StreamK<BmgGemm_F16F16FP32_StreamK_TileShape_256_256_32, Scheduler::GemmSplitK>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_256x256x32);
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16FP32_RCR, Gemm_Bench_F16F16FP32_RCR_StreamK, 64, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16FP32_RCR, Gemm_Bench_F16F16FP32_RCR_StreamK, 64, 256, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16FP32_RCR, Gemm_Bench_F16F16FP32_RCR_StreamK, 512, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16FP32_RCR, Gemm_Bench_F16F16FP32_RCR_StreamK, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_STREAMK_TILES(BmgGemmF16F16FP32_RCR, Gemm_Bench_F16F16FP32_RCR_StreamK)
#endif

using MMAAtomF16AccF16 = MMA_Atom<XE_DPAS_TT<8, half_t, half_t, half_t, half_t>>;
using BmgGemm_F16F16F16_Epilogue =
    cutlass::epilogue::fusion::LinearCombination<
        cutlass::half_t,
        cutlass::half_t,
        cutlass::half_t,
        cutlass::half_t,
        cutlass::FloatRoundStyle::round_to_nearest>;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB,
  int PipelineStages = 2>
using Gemm_Bench_F16F16F16_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    BmgGemm_F16F16F16_Epilogue,
    PipelineStages>;

#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_GEMM_TILES(BmgGemmF16F16F16_RCR, Gemm_Bench_F16F16F16_RCR, MMAAtomF16AccF16)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_DECLARE_GEMM_TILE_SG(BmgGemmF16F16F16_RCR, Gemm_Bench_F16F16F16_RCR, MMAAtomF16AccF16, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif

template <typename TileShape>
using BmgGemm_F16F16F16_StreamK_Tile =
    typename TiledMMAHelper<
        MMAAtomF16AccF16,
        Layout<TileShape>,
        Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemm_F16F16F16_StreamK_Epilogue = BmgGemm_F16F16F16_Epilogue;

template <typename TileShape, Scheduler TileScheduler>
using Gemm_Bench_F16F16F16_RCR_StreamK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    TileShape,
    TileScheduler,
    BmgGemm_F16F16F16_StreamK_Tile<TileShape>,
    void,
    void,
    BmgGemm_F16F16F16_StreamK_Epilogue,
    2,
    cutlass::gemm::KernelXeCooperative>;

using BmgGemm_F16F16F16_StreamK_TileShape_128_128_32 = Shape<_128, _128, _32>;
using BmgGemm_F16F16F16_StreamK_TileShape_128_256_32 = Shape<_128, _256, _32>;
using BmgGemm_F16F16F16_StreamK_TileShape_256_128_32 = Shape<_256, _128, _32>;
using BmgGemm_F16F16F16_StreamK_TileShape_256_256_32 = Shape<_256, _256, _32>;

using BmgGemmF16F16F16_RCR_StreamK_128x128x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_128_128_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16F16_RCR_DataParallel_128x128x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_128_128_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16F16_RCR_SplitK_128x128x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_128_128_32, Scheduler::GemmSplitK>;
using BmgGemmF16F16F16_RCR_StreamK_128x256x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_128_256_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16F16_RCR_DataParallel_128x256x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_128_256_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16F16_RCR_SplitK_128x256x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_128_256_32, Scheduler::GemmSplitK>;
using BmgGemmF16F16F16_RCR_StreamK_256x128x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_256_128_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16F16_RCR_DataParallel_256x128x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_256_128_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16F16_RCR_SplitK_256x128x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_256_128_32, Scheduler::GemmSplitK>;
using BmgGemmF16F16F16_RCR_StreamK_256x256x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_256_256_32, Scheduler::GemmStreamK>;
using BmgGemmF16F16F16_RCR_DataParallel_256x256x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_256_256_32, Scheduler::GemmDataParallel>;
using BmgGemmF16F16F16_RCR_SplitK_256x256x32 =
    Gemm_Bench_F16F16F16_RCR_StreamK<BmgGemm_F16F16F16_StreamK_TileShape_256_256_32, Scheduler::GemmSplitK>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_128x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_128x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_256x128x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_256x256x32);
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_256x256x32);
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16F16_RCR, Gemm_Bench_F16F16F16_RCR_StreamK, 64, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16F16_RCR, Gemm_Bench_F16F16F16_RCR_StreamK, 64, 256, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16F16_RCR, Gemm_Bench_F16F16F16_RCR_StreamK, 512, 128, 32)
BMG_DECLARE_STREAMK_TILE(BmgGemmF16F16F16_RCR, Gemm_Bench_F16F16F16_RCR_StreamK, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
BMG_DECLARE_EXPANDED_STREAMK_TILES(BmgGemmF16F16F16_RCR, Gemm_Bench_F16F16F16_RCR_StreamK)
#endif

#if defined(CUTLASS_BENCHMARK_EXHAUSTIVE_GEMM)
#include "cutlass_benchmark_exhaustive_gemm_declare.hpp"
#endif

static void register_gemm_benchmarks() {
#if defined(CUTLASS_BENCHMARK_ENABLE_LIBRARY_GEMM)
  cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::GEMMOptions>::Register(
      "cutlass_library_gemm",
      &cutlass::benchmark::cutlass_library_gemm_func);
#endif
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32);
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  BMG_REGISTER_EXPANDED_GEMM_TILES(BmgGemmBF16BF16FP32_RRR)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_REGISTER_GEMM_TILE_SG(BmgGemmBF16BF16FP32_RRR, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_5);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_6);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RRR_6);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_7);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_9);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_16);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_17);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_18);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_19);
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  BMG_REGISTER_EXPANDED_GEMM_TILES(BmgGemmBF16BF16FP32_RCR)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_REGISTER_GEMM_TILE_SG(BmgGemmBF16BF16FP32_RCR, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_DataParallel_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_SplitK_256x256x32);
  BMG_REGISTER_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, 64, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, 64, 256, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, 512, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmBF16BF16FP32_RCR, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  BMG_REGISTER_EXPANDED_STREAMK_TILES(BmgGemmBF16BF16FP32_RCR)
#endif
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_StreamK_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_DataParallel_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmTF32TF32FP32_RCR_SplitK_256x256x32);
  BMG_REGISTER_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, 64, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, 64, 256, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, 512, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmTF32TF32FP32_RCR, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  BMG_REGISTER_EXPANDED_GEMM_TILES(BmgGemmTF32TF32FP32_RCR)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_REGISTER_GEMM_TILE_SG(BmgGemmTF32TF32FP32_RCR, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
  BMG_REGISTER_EXPANDED_STREAMK_TILES(BmgGemmTF32TF32FP32_RCR)
#endif
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_5);
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_6);
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_7);
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_9);
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_16);
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_17);
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_18);
  CUTLASS_BENCHMARK(BmgGemmFP16FP16FP32_RCR_19);
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  BMG_REGISTER_EXPANDED_GEMM_TILES(BmgGemmFP16FP16FP32_RCR)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_REGISTER_GEMM_TILE_SG(BmgGemmFP16FP16FP32_RCR, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
#endif
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_StreamK_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_DataParallel_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16FP32_RCR_SplitK_256x256x32);
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16FP32_RCR, 64, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16FP32_RCR, 64, 256, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16FP32_RCR, 512, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16FP32_RCR, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  BMG_REGISTER_EXPANDED_STREAMK_TILES(BmgGemmF16F16FP32_RCR)
#endif
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_128x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_128x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_256x128x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_StreamK_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_DataParallel_256x256x32);
  CUTLASS_BENCHMARK(BmgGemmF16F16F16_RCR_SplitK_256x256x32);
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16F16_RCR, 64, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16F16_RCR, 64, 256, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16F16_RCR, 512, 128, 32)
  BMG_REGISTER_STREAMK_TILE(BmgGemmF16F16F16_RCR, 512, 256, 32)
#if defined(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  BMG_REGISTER_EXPANDED_GEMM_TILES(BmgGemmF16F16F16_RCR)
#define BMG_SOURCE_GEMM_TILE_SG(M, N, K, SGM, SGN) BMG_REGISTER_GEMM_TILE_SG(BmgGemmF16F16F16_RCR, M, N, K, SGM, SGN)
#include "bmg_gemm_source_tile_sg.def"
#undef BMG_SOURCE_GEMM_TILE_SG
  BMG_REGISTER_EXPANDED_STREAMK_TILES(BmgGemmF16F16F16_RCR)
#endif
#if defined(CUTLASS_BENCHMARK_EXHAUSTIVE_GEMM)
#include "cutlass_benchmark_exhaustive_gemm_register.hpp"
#endif
}
