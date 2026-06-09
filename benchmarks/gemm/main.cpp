#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include <fstream>
#include <iostream>
#include "benchmark_runner.hpp"

#if defined(SYCL_NVIDIA_TARGET) || !defined(CUTLASS_ENABLE_SYCL)
#include "benchmarks_cuda.hpp"
#elif defined(SYCL_INTEL_TARGET)
#if defined(CUTLASS_BENCHMARK_USE_FILTERED_HEADER)
#include "benchmarks_sycl.filtered.hpp"
#else
#include "benchmarks_sycl.hpp"
#endif
#endif

template <class Result>
int print_direct_result(std::string const& kernel, Result const& result) {
  std::cout << "median_tflops=" << result.tflops
            << " avg_runtime_ms=" << result.avg_runtime_ms
            << " total_runtime_ms=" << result.total_runtime_ms
            << " input_bytes_per_buffer=" << result.input_bytes_per_buffer
            << " input_pool_target_bytes=" << result.input_pool_target_bytes
            << " pool_buffers=" << result.pool_buffers
            << " warmup_iters=" << result.warmup_iters
            << " measure_iters=" << result.measure_iters
            << " KERNEL=" << kernel
            << " STATUS=OK" << std::endl;
  return 0;
}

// ── Dual-mode profiler main ──
// --config_file=PATH  → Google Benchmark path (legacy)
// --kernel=NAME        → GB-free direct profiling (new)

int main(int argc, const char** argv) {
  cutlass::CommandLine cmd(argc, argv);

  // Legacy GB mode
  std::string config_file;
  cmd.get_cmd_line_argument("config_file", config_file, std::string(""));
  if (!config_file.empty()) {
    BenchmarkOptions options;
    options.parse(argc, argv);
    if (options.error) return -1;
    std::ifstream file(options.config_file);
    if (!file.is_open()) { std::cerr << "Cannot open config" << std::endl; return 1; }
    register_gemm_benchmarks();
    std::string line;
    while (std::getline(file, line))
      if (!line.empty() && line[0] != '#') register_benchmarks<cutlass::benchmark::GEMMOptions>(line);
    file.close();
    ::benchmark::Initialize(nullptr, nullptr);
    ::benchmark::SetDefaultTimeUnit(::benchmark::kMillisecond);
    ::benchmark::RunSpecifiedBenchmarks();
    compat::wait();
    ::benchmark::Shutdown();
    return 0;
  }

  // ── Direct profiling (GB-free, matches NVIDIA CUTLASS profiler pattern) ──
  std::string kernel;
  cmd.get_cmd_line_argument("kernel", kernel, std::string(""));
  if (kernel.empty()) { std::cerr << "--kernel=NAME [--m=8192 --n=4096 --k=1536]" << std::endl; return 1; }

  register_gemm_benchmarks();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  cutlass::benchmark::GEMMOptions opts;
  cmd.get_cmd_line_argument("m", opts.m, 8192); cmd.get_cmd_line_argument("n", opts.n, 4096);
  cmd.get_cmd_line_argument("k", opts.k, 1536); cmd.get_cmd_line_argument("l", opts.l, 1);
  cmd.get_cmd_line_argument("alpha", opts.alpha, 1.0f); cmd.get_cmd_line_argument("beta", opts.beta, 0.0f);
  opts.verify_library = 0; opts.split_k_slices = 0;

#define RUN(K) if (kernel == #K) { auto result = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct_result(opts, hw); return print_direct_result(kernel, result); }
  RUN(BmgGemmBF16BF16FP32_RRR_Gemm_256x256x32_SG8x4)
  RUN(BmgGemmBF16BF16FP32_RRR_Gemm_256x256x64_SG8x4)
  RUN(BmgGemmBF16BF16FP32_RCR_6)
  RUN(BmgGemmBF16BF16FP32_RCR_17)
  RUN(BmgGemmBF16BF16FP32_RCR_18)
  RUN(BmgGemmBF16BF16FP32_RCR_19)
  RUN(BmgGemmBF16BF16FP32_RRR_6)
#undef RUN
  std::cerr << "not found: " << kernel << std::endl;
  return 1;
}
