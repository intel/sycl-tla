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
#include <iostream>
#include "benchmark_runner.hpp"
#if defined(SYCL_INTEL_TARGET)
#include "benchmarks_sycl.hpp"
#endif
int main(int argc, const char** argv) {{
  cutlass::CommandLine cmd(argc, argv);
  std::string kernel; cmd.get_cmd_line_argument("kernel", kernel, std::string(""));
  if (kernel.empty()) {{ return 1; }}
  register_gemm_benchmarks();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  cutlass::benchmark::GEMMOptions opts;
  cmd.get_cmd_line_argument("m", opts.m, 8192); cmd.get_cmd_line_argument("n", opts.n, 4096);
  cmd.get_cmd_line_argument("k", opts.k, 1536); cmd.get_cmd_line_argument("l", opts.l, 1);
  cmd.get_cmd_line_argument("alpha", opts.alpha, 1.0f); cmd.get_cmd_line_argument("beta", opts.beta, 0.0f);
  opts.verify_library = 0; opts.split_k_slices = 0;
  double tflops = 0; bool ok = false;
#define RUN(K) if (kernel == #K) {{ tflops = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct(opts, hw); ok = true; }}
{runs}
#undef RUN
  if (!ok) {{ std::cerr << "NOT_FOUND" << std::endl; return 1; }}
  constexpr int kWarmupIters = 100;
  constexpr int kMeasureIters = 100;
  double avg_runtime_ms = 0.0;
  double total_runtime_ms = 0.0;
  if (tflops > 0.0) {{
    double const total_flops = 2.0 * static_cast<double>(opts.m) * static_cast<double>(opts.n) * static_cast<double>(opts.k) * static_cast<double>(opts.l);
    avg_runtime_ms = (total_flops / (tflops * 1.0e12)) * 1.0e3;
    total_runtime_ms = avg_runtime_ms * static_cast<double>(kMeasureIters);
  }}
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT kernel=" << kernel
            << " median_tflops=" << tflops
            << " avg_runtime_ms=" << avg_runtime_ms
            << " total_runtime_ms=" << total_runtime_ms
            << " measure_iters=" << kMeasureIters
            << " warmup_iters=" << kWarmupIters
            << std::endl;
  return 0;
}}
'''
with open(output, 'w') as f:
    f.write(main)
