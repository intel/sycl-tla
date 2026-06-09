import sys

manifest = sys.argv[1]
output = sys.argv[2]
kernels = []
with open(manifest) as f:
    for l in f:
        if l.strip():
            kernels.append(l.strip())

runs = '\n'.join(f'  RUN({k})' for k in kernels)
main = f'''#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include <iomanip>
#include <fstream>
#include <iostream>
#include "benchmark_runner.hpp"
#if defined(SYCL_INTEL_TARGET)
#if defined(CUTLASS_BENCHMARK_USE_FILTERED_HEADER)
#include "benchmarks_sycl.filtered.hpp"
#else
#include "benchmarks_sycl.hpp"
#endif
#endif

template <class Result>
int print_direct_result(std::string const& kernel, Result const& result) {{
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT kernel=" << kernel
            << " median_tflops=" << result.tflops
            << " avg_runtime_ms=" << result.avg_runtime_ms
            << " total_runtime_ms=" << result.total_runtime_ms
            << " input_bytes_per_buffer=" << result.input_bytes_per_buffer
            << " input_pool_target_bytes=" << result.input_pool_target_bytes
            << " pool_buffers=" << result.pool_buffers
            << " measure_iters=" << result.measure_iters
            << " warmup_iters=" << result.warmup_iters
            << std::endl;
  return 0;
}}

int main(int argc, const char** argv) {{
  cutlass::CommandLine cmd(argc, argv);
  std::string config_file;
  cmd.get_cmd_line_argument("config_file", config_file, std::string(""));
  if (!config_file.empty()) {{
    BenchmarkOptions options;
    options.parse(argc, argv);
    if (options.error) return -1;
    std::ifstream file(options.config_file);
    if (!file.is_open()) {{ std::cerr << "Cannot open config" << std::endl; return 1; }}
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
  }}

  std::string kernel; cmd.get_cmd_line_argument("kernel", kernel, std::string(""));
  if (kernel.empty()) {{ std::cerr << "--kernel=NAME [--m=8192 --n=4096 --k=1536]" << std::endl; return 1; }}
  register_gemm_benchmarks();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  cutlass::benchmark::GEMMOptions opts;
  cmd.get_cmd_line_argument("m", opts.m, 8192); cmd.get_cmd_line_argument("n", opts.n, 4096);
  cmd.get_cmd_line_argument("k", opts.k, 1536); cmd.get_cmd_line_argument("l", opts.l, 1);
  cmd.get_cmd_line_argument("alpha", opts.alpha, 1.0f); cmd.get_cmd_line_argument("beta", opts.beta, 0.0f);
  opts.verify_library = 0; opts.split_k_slices = 0;
#define RUN(K) if (kernel == #K) {{ auto result = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct_result(opts, hw); return print_direct_result(kernel, result); }}
{runs}
#undef RUN
  std::cerr << "NOT_FOUND" << std::endl;
  return 1;
}}
'''
with open(output, 'w') as f:
    f.write(main)
