#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include <iostream>
#include "benchmark_runner.hpp"

#if defined(SYCL_NVIDIA_TARGET) || !defined(CUTLASS_ENABLE_SYCL)
#include "benchmarks_cuda.hpp"
#elif defined(SYCL_INTEL_TARGET)
#include "benchmarks_sycl.hpp"
#endif

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

  double tflops = 0; bool ok = false;
#define RUN(K) if (kernel == #K) { tflops = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct(opts, hw); ok = true; }
  RUN(BmgGemmBF16BF16FP32_RRR_Gemm_256x256x32_SG8x4)
  RUN(BmgGemmBF16BF16FP32_RRR_Gemm_256x256x64_SG8x4)
  RUN(BmgGemmBF16BF16FP32_RCR_6)
  RUN(BmgGemmBF16BF16FP32_RCR_17)
  RUN(BmgGemmBF16BF16FP32_RCR_18)
  RUN(BmgGemmBF16BF16FP32_RCR_19)
#undef RUN
  if (!ok) { std::cerr << "not found: " << kernel << std::endl; return 1; }
  std::cout << "median_tflops=" << tflops << " KERNEL=" << kernel << " STATUS=OK" << std::endl;
  return 0;
}
