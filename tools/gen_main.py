import sys
manifest = sys.argv[1]
output = sys.argv[2]
kernels = []
with open(manifest) as f:
    for l in f:
        if l.strip(): kernels.append(l.strip())
runs = '\n'.join(f'  RUN({k})' for k in kernels)
main = f'''#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
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
  cmd.get_cmd_line_argument("k", opts.k, 1536); opts.verify_library = 0;
  double tflops = 0; bool ok = false;
#define RUN(K) if (kernel == #K) {{ tflops = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct(opts, hw); ok = true; }}
{runs}
#undef RUN
  if (!ok) {{ std::cerr << "NOT_FOUND" << std::endl; return 1; }}
  std::cout << "RESULT kernel=" << kernel << " median_tflops=" << tflops << std::endl;
  return 0;
}}
'''
with open(output, 'w') as f: f.write(main)
