#pragma once
#include "../../../benchmarks/gemm/gemm_configuration_sycl.hpp"
using Scheduler = cutlass::gemm::device::Scheduler;

// MMA atom (modern — matches example 00)
using MMAAtom = MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>;

// Gemm_Bench_BF16FP32_RCR template (compact — for RCR types)
template <typename TileShape, typename Tiler, typename GmemCA, typename GmemCB, int PS=2>
using Gemm_Bench_BF16FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemCA, GmemCB,
    cutlass::epilogue::fusion::LinearCombination<float,float,float,float,cutlass::FloatRoundStyle::round_to_nearest>,
    PS>;

// Gemm_Bench_BF16FP32_RRR template (compact — for RRR types)
template <typename TileShape, typename Tiler, typename GmemCA, typename GmemCB, int PS=2>
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemCA, GmemCB,
    cutlass::epilogue::fusion::LinearCombination<float,float,float,float,cutlass::FloatRoundStyle::round_to_nearest>,
    PS>;

using BmgTile_19 = TiledMMAHelper<MMAAtom, Layout<BmgGemm_BF16FP32_TileShape_128_256_32>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgTile_1 = TiledMMA<MMAAtom, Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>, Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>, Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;
using BmgGemmBF16BF16FP32_RCR_6 = Gemm_Bench_BF16FP32_RCR<Shape<_256, _256, _32>, BmgTile_1, void, void>;
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_6);

static void register_gemm_benchmarks() {
  CUTLASS_BENCHMARK(BmgGemmBF16BF16FP32_RCR_6);
}
