// perf_analysis_case.cpp — Self-contained kernel performance analysis
// =====================================================================
// Compares profiler (GemmConfiguration → GemmUniversalAdapter) vs
// example (GemmUniversalAdapter direct) for the SAME RRR 256x256x32 SG8x4 kernel.
//
// Build (from sycl-tla root with oneAPI 2025.3):
//   source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
//   export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
//   export IGC_VectorAliasBBThreshold=10000
//   export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
//   icpx -fsycl -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
//     -I. -Iinclude -Itools/util/include \
//     tools/perf_analysis_case.cpp -o /tmp/perf_case
//
// Run:
//   ZE_AFFINITY_MASK=7 /tmp/perf_case --method=profiler --m=8192 --n=4096 --k=1536
//   ZE_AFFINITY_MASK=7 /tmp/perf_case --method=example --m=8192 --n=4096 --k=1536
//
// This code directly replicates the EXACT kernel types used in both paths,
// enabling precise A/B comparison.

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/xe_gemm.hpp"
#include "sycl_common.hpp"
#include "helper.h"

#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace cute;

// ── Method 1: GemmConfiguration (profiler path) ──────────────────────────
// This is the EXACT template instantiation that BenchmarkRunnerGemm uses.

template <typename TileShape, typename Tiler, typename GmemTiledCopyA, typename GmemTiledCopyB, int PipelineStages = 2>
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, cutlass::gemm::device::Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<float, float, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    PipelineStages>;

// ── Method 2: Direct GemmUniversalAdapter (example path) ─────────────────

template <typename TileShape, typename TiledMma>
struct ExampleKernel {
  using MMAAtom = typename TiledMma::Atom;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementOutput = float;
  using ElementComputeEpilogue = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue,
      ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape,
      decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy, TileShape,
      void, ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks, void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape,
      ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      void, void, void, cute::identity,
      void, void, void, cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ── Common measurement ───────────────────────────────────────────────────

template <typename GemmType>
double measure(int m, int n, int k, int warmup, int measure, cutlass::KernelHardwareInfo& hw_info) {
  using GemmKernel = typename GemmType::GemmKernel;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = float;
  using ElementAccumulator = float;

  auto problem_shape_MNKL = cute::make_shape(m, n, k, 1);
  auto [M, N, K, L] = problem_shape_MNKL;

  using StrideA = typename GemmType::StrideA;
  using StrideB = typename GemmType::StrideB;
  using StrideC = typename GemmType::StrideC;
  using StrideD = typename GemmType::StrideD;
  // For the example path, Stride types come from GemmKernel
  // We use the kernel's stride types directly

  auto stride_A = cutlass::make_cute_packed_stride(
      cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>{}, cute::make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(
      cutlass::gemm::TagToStrideB_t<cutlass::layout::RowMajor>{}, cute::make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>{}, cute::make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>{}, cute::make_shape(M, N, L));

  cutlass::DeviceAllocation<ElementA> block_A(M * K * L);
  cutlass::DeviceAllocation<ElementB> block_B(K * N * L);
  cutlass::DeviceAllocation<ElementC> block_C(M * N * L);
  cutlass::DeviceAllocation<ElementC> block_D(M * N * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_C, 2021);

  typename GemmKernel::Arguments arguments{};
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = cute::make_shape(m, n, k, 1);
  arguments.mainloop = {block_A.get(), stride_A, block_B.get(), stride_B};
  arguments.epilogue = {{ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
                        block_C.get(), stride_C, block_D.get(), stride_D};
  arguments.hw_info = hw_info;

  GemmType gemm_op;
  size_t ws = GemmType::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
    std::cerr << "Cannot implement" << std::endl;
    return 0.0;
  }
  gemm_op.initialize(arguments, workspace.get());

  // Initial run (discarded)
  gemm_op.run();
  compat::wait();

  // Warmup (discarded)
  for (int w = 0; w < warmup; ++w) gemm_op.run();
  compat::wait();

  // Measurement
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < measure; ++i) gemm_op.run();
  compat::wait();

  double avg_sec = timer.seconds() / measure;
  double gflop = 2.0 * m * n * k * l * 1e-9;
  return gflop / avg_sec;
}

// ── Main ─────────────────────────────────────────────────────────────────

int main(int argc, const char** argv) {
  cutlass::CommandLine cmd(argc, argv);

  std::string method = "profiler";
  cmd.get_cmd_line_argument("method", method, std::string("profiler"));

  int m = 8192, n = 4096, k = 1536;
  cmd.get_cmd_line_argument("m", m, 8192);
  cmd.get_cmd_line_argument("n", n, 4096);
  cmd.get_cmd_line_argument("k", k, 1536);

  int warmup = 100, measure = 100;
  cmd.get_cmd_line_argument("warmup", warmup, 100);
  cmd.get_cmd_line_argument("measure", measure, 100);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
      hw_info.device_id);

  // Define shared types
  using TileShape = Shape<_256, _256, _32>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
      Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  double tflops = 0.0;

  if (method == "profiler") {
    // Exact profiler kernel type
    using ProfilerGemm = Gemm_Bench_BF16FP32_RRR<TileShape, TiledMma, void, void, 2>;
    // GemmConfiguration::Gemm = GemmUniversalAdapter<GemmKernel>
    using ProfilerAdapter = typename ProfilerGemm::Gemm;
    tflops = measure<ProfilerAdapter>(m, n, k, warmup, measure, hw_info);
    std::cout << "METHOD=profiler(GemmConfiguration)" << std::endl;

  } else if (method == "example") {
    // Exact example kernel type
    using ExampleConfig = ExampleKernel<TileShape, TiledMma>;
    using ExampleAdapter = typename ExampleConfig::Gemm;
    tflops = measure<ExampleAdapter>(m, n, k, warmup, measure, hw_info);
    std::cout << "METHOD=example(GemmUniversalAdapter)" << std::endl;

  } else {
    std::cerr << "Unknown method: " << method << " (use 'profiler' or 'example')" << std::endl;
    return 1;
  }

  std::cout << "m=" << m << " n=" << n << " k=" << k << std::endl;
  std::cout << "warmup=" << warmup << " measure=" << measure << std::endl;
  std::cout << "tflops(GFLOPS)=" << tflops << std::endl;
  std::cout << "eu_count=" << hw_info.sm_count << std::endl;

  // Convert to TFLOPS for comparison with previous results
  std::cout << "TFLOPS=" << (tflops / 1000.0) << std::endl;

  return 0;
}
