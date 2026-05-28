#!/usr/bin/env python3
"""Build one batch of GEMM kernels."""
import sys, os, subprocess, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--build-dir", required=True)
    p.add_argument("--manifest", required=True, help="Kernel list file (one per line)")
    p.add_argument("--target", default="intel_gpu_bmg_g31")
    p.add_argument("--cores", default=0, type=int, help="0=50% CPU")
    args = p.parse_args()

    os.makedirs(f"{args.build_dir}/benchmarks/gemm", exist_ok=True)

    # Write filter file in cmake ^Name$ format
    with open(args.manifest) as f:
        kernels = [l.strip() for l in f if l.strip()]
    filter_path = f"{args.build_dir}/benchmarks/gemm/cutlass_benchmark_filter.hpp"
    with open(filter_path, "w") as f:
        for k in kernels:
            f.write(f"^{k}$\n")
    print(f"Filter: {len(kernels)} kernels → {filter_path}")

    # Shared deps: use a common dir
    deps = f"{os.path.dirname(args.build_dir)}/shared_deps"
    os.makedirs(deps, exist_ok=True)

    # cmake configure
    cmake_cmd = [
        "cmake", "-S", args.repo_root, "-B", args.build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=icpx",
        "-DDPCPP_SYCL_TARGET=" + args.target,
        "-DDPCPP_HOST_COMPILER=g++-13",
        "-DCUTLASS_ENABLE_SYCL=ON", "-DCUTLASS_NVCC_ARCHS=",
        "-DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON",
        "-DCUTLASS_BENCHMARK_EXHAUSTIVE_GEMM=ON",
        f"-DCUTLASS_KERNEL_FILTER_FILE={filter_path}",
        f"-DGOOGLETEST_DIR={os.path.dirname(args.build_dir)}/../../shared_deps/googletest-src",
        f"-DGOOGLEBENCHMARK_DIR={os.path.dirname(args.build_dir)}/../../shared_deps/googlebenchmark-src",
    ]
    subprocess.run(cmake_cmd, check=True, capture_output=True, text=True)

    # Build
    cores = args.cores if args.cores > 0 else max(1, os.cpu_count() // 2)
    make_cmd = ["make", "-C", args.build_dir, "cutlass_benchmarks_gemm_sycl", f"-j{cores}"]
    result = subprocess.run(make_cmd, capture_output=True, text=True)
    
    binary = f"{args.build_dir}/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
    if os.path.isfile(binary):
        size = os.path.getsize(binary)
        print(f"BUILD_OK: {size} bytes")
        return 0
    else:
        print("BUILD_FAILED")
        for line in result.stdout.split("\n")[-5:] + result.stderr.split("\n")[-5:]:
            if "error:" in line:
                print(f"  {line}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
