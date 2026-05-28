#!/usr/bin/env python3
"""
Parallel batch build + screen pipeline using multi-core parallelism.

Architecture:
  1. Generate kernel list from catalog
  2. Split into N batches (1-2 kernels each)
  3. For each batch, in parallel (P processes):
     a. gen_mini_hpp.py → small benchmarks_sycl.hpp
     b. cmake configure → build directory
     c. make -jC (C cores per build)
  4. Screen kernels one-by-one on GPU 5 and GPU 7

Usage:
  DRY_RUN=true python3 tools/parallel_build.py  # test 4 batches
  BATCH_SIZE=2 PARALLEL=8 python3 tools/parallel_build.py  # full run
"""
import os, sys, json, subprocess, threading, time, argparse, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[1]
ONEAPI_SH = "/opt/intel/oneapi/compiler/2025.3/env/vars.sh"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class BuildManager:
    def __init__(self, args):
        self.ws = Path(args.workspace)
        self.dtype = args.dtype
        self.batch_size = args.batch_size
        self.parallel = args.parallel
        self.cores = args.cores
        self.gpus = [int(g) for g in args.gpus.split(",")]
        self.shape = (args.m, args.n, args.k)
        self.timeout = args.timeout
        self.dry_run = args.dry_run
        self.target = args.target
        self.lock = threading.Lock()
        self.stats = {"built": 0, "failed": 0, "screened": 0, "passed": 0}
        self.sem = threading.BoundedSemaphore(self.parallel)
        
        # Find dependency source directory
        self.deps = self._find_deps()
    
    def _find_deps(self):
        """Find a working cmake build with cached deps."""
        candidates = [
            "/root/cutlass_profile_device7_b70_2500mhz/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001",
        ]
        for cand in candidates:
            gtest = Path(cand) / "_deps/googletest-src"
            if gtest.is_dir():
                return {"gtest": str(gtest)}
        log("WARNING: no working deps found — cmake may try to download")
        return {"gtest": ""}
    
    def generate_kernels(self):
        """Phase 1: Generate kernel list from catalog."""
        log("Phase 1: Generating kernel list...")
        sys.path.insert(0, str(REPO / "test/benchmarks"))
        from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
        from intel_gemm_profiler.constraints import default_constraints
        
        cons = default_constraints()
        cat = generated_layered_bmg_kernel_catalog(constraints=cons)
        df = {"bf16": "bf16", "f16": "f16"}.get(self.dtype, "bf16")
        all_k = sorted(set(k["kernel_name"] for k in cat["kernels"] if k.get("dtype_family") == df))
        # Exclude streamk_example runners
        all_k = [k for k in all_k if not k.startswith("03_bmg") and "streamk_example" not in k]
        
        batches = [all_k[i:i+self.batch_size] for i in range(0, len(all_k), self.batch_size)]
        
        self.ws.mkdir(parents=True, exist_ok=True)
        (self.ws / "builds").mkdir(exist_ok=True)
        (self.ws / "results").mkdir(exist_ok=True)
        (self.ws / "logs").mkdir(exist_ok=True)
        
        self.manifest = {"total": len(all_k), "batch_size": self.batch_size, "batches": []}
        for i, batch in enumerate(batches):
            bid = f"batch_{i:04d}"
            mani_f = str(self.ws / "builds" / f"{bid}.txt")
            with open(mani_f, "w") as f:
                for k in batch: f.write(k + "\n")
            self.manifest["batches"].append({
                "id": bid, "count": len(batch), "gpu": self.gpus[i % len(self.gpus)],
                "manifest": mani_f, "kernels": batch,
            })
        
        with open(str(self.ws / "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=2)
        
        log(f"Generated {len(batches)} batches ({len(all_k)} kernels)")
        return self.manifest
    
    def setup_env(self):
        """Setup CPU governors + GPU freq."""
        log("Setting up environment...")
        # CPU performance
        for gov_path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
            try: gov_path.write_text("performance")
            except: pass
        
        # GPU freq
        for gpu in self.gpus:
            fpath = Path(f"/sys/class/drm/card{gpu}/gt_max_freq_mhz")
            if fpath.exists():
                try: fpath.write_text("2500")
                except: pass

    def build_batch(self, batch):
        """Build one batch: gen mini HPP → cmake → make."""
        bid = batch["id"]
        bdir = self.ws / "builds" / bid
        
        # Clean and recreate
        if bdir.exists():
            subprocess.run(["rm", "-rf", str(bdir)], capture_output=True)
        (bdir / "benchmarks/gemm").mkdir(parents=True, exist_ok=True)
        
        # Generate mini benchmarks_sycl.hpp
        mini_hpp = bdir / "benchmarks/gemm/benchmarks_sycl.hpp"
        result = subprocess.run([
            sys.executable, str(REPO / "tools/gen_mini_hpp.py"),
            "--manifest", batch["manifest"],
            "--output", str(mini_hpp),
        ], capture_output=True, text=True)
        if result.returncode != 0:
            return f"GEN_HPP_FAIL: {bid}"
        
        # Write main.cpp (minimal, just kernel dispatch)
        main_cpp = REPO / "benchmarks/gemm/main.cpp"
        with open(main_cpp, "w") as f:
            f.write('''#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include <iostream>
#include "benchmark_runner.hpp"
#if defined(SYCL_INTEL_TARGET)
#include "benchmarks_sycl.hpp"
#endif
int main(int argc, const char** argv) {
  cutlass::CommandLine cmd(argc, argv);
  std::string kernel;
  cmd.get_cmd_line_argument("kernel", kernel, std::string(""));
  if (kernel.empty()) { std::cerr << "--kernel=NAME" << std::endl; return 1; }
  register_gemm_benchmarks();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  cutlass::benchmark::GEMMOptions opts;
  cmd.get_cmd_line_argument("m", opts.m, ''' + str(self.shape[0]) + '''); cmd.get_cmd_line_argument("n", opts.n, ''' + str(self.shape[1]) + ''');
  cmd.get_cmd_line_argument("k", opts.k, ''' + str(self.shape[2]) + '''); opts.verify_library = 0;
  double tflops = 0; bool ok = false;
''')
            for k in batch["kernels"]:
                f.write(f'#define RUN_{k.replace("BmgGemm", "").replace("::", "_")} 0\n')
            f.write('''
#define RUN(K) if (kernel == #K) { tflops = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct(opts, hw); ok = true; }
''')
            for k in batch["kernels"]:
                f.write(f'  RUN({k})\n')
            f.write('''#undef RUN
  if (!ok) { std::cerr << "NOT_FOUND" << std::endl; return 1; }
  std::cout << "RESULT: kernel=" << kernel << " median_tflops=" << tflops << " STATUS=OK" << std::endl;
  return 0;
}
''')
        
        # Write cmake filter
        filter_f = bdir / "benchmarks/gemm/cutlass_benchmark_filter.hpp"
        with open(filter_f, "w") as f:
            for k in batch["kernels"]:
                f.write(f"^{k}$\n")
        
        # cmake configure
        cmake_result = subprocess.run([
            "cmake", "-S", str(REPO), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=icpx",
            "-DDPCPP_SYCL_TARGET=" + self.target, "-DDPCPP_HOST_COMPILER=g++-13",
            "-DCUTLASS_ENABLE_SYCL=ON", "-DCUTLASS_NVCC_ARCHS=",
            "-DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON",
            f"-DCUTLASS_KERNEL_FILTER_FILE={filter_f}",
            f"-DGOOGLETEST_DIR={self.deps['gtest']}",
        ], capture_output=True, text=True, timeout=120)
        if cmake_result.returncode != 0:
            return f"CMAKE_FAIL: {bid}"
        
        # make
        t0 = time.time()
        make_result = subprocess.run(
            ["make", "-C", str(bdir), "cutlass_benchmarks_gemm_sycl", f"-j{self.cores}"],
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0
        
        binpath = bdir / "benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
        if binpath.is_file():
            size = binpath.stat().st_size
            return f"BUILD_OK: {bid} ({size} bytes, {elapsed:.0f}s)"
        
        # Show errors
        for line in make_result.stderr.split("\n")[-3:] + make_result.stdout.split("\n")[-3:]:
            if "error:" in line:
                return f"BUILD_FAIL: {bid} — {line.strip()[:120]}"
        return f"BUILD_FAIL: {bid}"
    
    def run(self):
        self.setup_env()
        manifest = self.generate_kernels()
        
        batches = manifest["batches"]
        if self.dry_run:
            batches = batches[:4]
            log(f"DRY RUN: {len(batches)} batches")
        else:
            log(f"FULL RUN: {len(batches)} batches")
        
        log(f"Building with {self.parallel} parallel × {self.cores} cores")
        
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.parallel) as ex:
            futures = {ex.submit(self.build_batch, b): b for b in batches}
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = f"EXCEPTION: {batch['id']} — {e}"
                with self.lock:
                    if "BUILD_OK" in result:
                        self.stats["built"] += 1
                    else:
                        self.stats["failed"] += 1
                    log(result)
        
        elapsed = time.time() - t0
        log(f"Build complete: {self.stats['built']} OK, {self.stats['failed']} FAIL in {elapsed:.0f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", default="/root/cutlass_profile_device7_b70_2500mhz/screen_ws")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--gpus", default="5,7")
    p.add_argument("--m", type=int, default=8192)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=1536)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "true") == "true")
    p.add_argument("--target", default="intel_gpu_bmg_g31")
    args = p.parse_args()
    
    mgr = BuildManager(args)
    mgr.run()
