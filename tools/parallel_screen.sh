#!/bin/bash
#=============================================================================
# Parallel Batch Build + Screen Pipeline
# Uses multi-core: N independent cmake builds in parallel, each with -j16
#=============================================================================
set -euo pipefail

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SELF_DIR/.." && pwd)
ONEAPI_SH="/opt/intel/oneapi/compiler/2025.3/env/vars.sh"
WORKSPACE="${WORKSPACE:-/root/cutlass_profile_device7_b70_2500mhz/screen_ws}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-2}"          # kernels per batch
PARALLEL_BUILDS="${PARALLEL_BUILDS:-12}" # parallel cmake builds
BUILD_CORES="${BUILD_CORES:-16}"       # cores per cmake build
SHAPE_M="${SHAPE_M:-8192}"
SHAPE_N="${SHAPE_N:-4096}"
SHAPE_K="${SHAPE_K:-1536}"
GPUS="${GPUS:-5,7}"
TIMEOUT="${TIMEOUT:-120}"
DRY_RUN="${DRY_RUN:-true}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

main() {
  source "$ONEAPI_SH" 2>/dev/null || true
  export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
  export IGC_VectorAliasBBThreshold=10000
  export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"

  rm -rf "$WORKSPACE"
  mkdir -p "$WORKSPACE"/{builds,results,logs}

  # CPU freq
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$gov" 2>/dev/null; done

  # --- Phase 1: Generate kernel list ---
  log "Phase 1: Generating kernel list from catalog..."
  cd "$REPO/test"
  python3 << PYEOF
import sys; sys.path.insert(0, "benchmarks")
from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints
import json, os, re

cons = default_constraints()
cat = generated_layered_bmg_kernel_catalog(constraints=cons)
all_k = sorted(set(k["kernel_name"] for k in cat["kernels"] if k.get("dtype_family") == "${DTYPE}"))
# Exclude streamk_example runners
all_k = [k for k in all_k if not k.startswith("03_bmg") and "streamk_example" not in k]
print(f"Total kernels: {len(all_k)}", file=sys.stderr)

batch_size = ${BATCH_SIZE}
batches = [all_k[i:i+batch_size] for i in range(0, len(all_k), batch_size)]

os.makedirs("${WORKSPACE}/builds", exist_ok=True)
manifest = {"dtype": "${DTYPE}", "total_kernels": len(all_k), "batch_count": len(batches)}
for i, batch in enumerate(batches):
    bid = f"batch_{i:04d}"
    mani_f = f"${WORKSPACE}/builds/{bid}.txt"
    with open(mani_f, "w") as f:
        for k in batch: f.write(k + "\n")
    manifest[bid] = {"idx": i, "count": len(batch), "gpu": [5,7][i%2], "manifest": mani_f}
with open("${WORKSPACE}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Generated {len(batches)} batches × ${BATCH_SIZE} kernels", file=sys.stderr)
PYEOF

  log "Phase 1 done: $(python3 -c "import json;m=json.load(open('${WORKSPACE}/manifest.json'));print(m['batch_count'],'batches,',m['total_kernels'],'kernels')")"

  # --- Phase 2: Build batches in parallel ---
  log "Phase 2: Building batches ($PARALLEL_BUILDS parallel × $BUILD_CORES cores each)..."
  
  python3 << PYEOF
import json, subprocess, os, sys, threading, time, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

manifest = json.load(open("${WORKSPACE}/manifest.json"))
batch_ids = sorted([k for k in manifest.keys() if k.startswith("batch_")])

# Find a good cmake template for deps
GOOD_DEPS = None
for cand in ["/root/cutlass_profile_device7_b70_2500mhz/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001"]:
    if os.path.isdir(f"{cand}/_deps/googletest-src"):
        GOOD_DEPS = cand
        break

if GOOD_DEPS is None:
    print("FATAL: no good deps found", file=sys.stderr)
    sys.exit(1)

GTEST = f"{GOOD_DEPS}/_deps/googletest-src"
GB = f"{GOOD_DEPS}/_deps/googlebenchmark-src"

lock = threading.Lock()
completed = {"ok": 0, "fail": 0}
semaphore = threading.Semaphore(${PARALLEL_BUILDS})

# Pre-warm: use a shared build dir for deps (avoid rebuilding GB per batch)
SHARED_DEPS = "${WORKSPACE}/builds/shared"
os.makedirs(f"{SHARED_DEPS}/benchmarks/gemm", exist_ok=True)

def build_batch(bid):
    info = manifest[bid]
    bdir = f"${WORKSPACE}/builds/{bid}"
    subprocess.run(["rm", "-rf", bdir], capture_output=True)
    os.makedirs(f"{bdir}/benchmarks/gemm", exist_ok=True)
    
    # Generate slim benchmarks_sycl.hpp for this batch
    with open(info["manifest"]) as f:
        kernels = [l.strip() for l in f if l.strip()]
    
    hpp_path = f"{bdir}/benchmarks/gemm/benchmarks_sycl.hpp"
    with open(hpp_path, "w") as f:
        f.write("""#pragma once
#include "${REPO}/benchmarks/gemm/gemm_configuration_sycl.hpp"
using Scheduler = cutlass::gemm::device::Scheduler;
using MMAAtom = MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>;

template <typename TS, typename T, typename C, typename CB, int PS=2>
using Gemm_Bench_BF16FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe, cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
    TS, Scheduler::Gemm, T, C, CB,
    cutlass::epilogue::fusion::LinearCombination<float,float,float,float,cutlass::FloatRoundStyle::round_to_nearest>, PS>;

template <typename TS, typename T, typename C, typename CB, int PS=2>
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe, cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
    TS, Scheduler::Gemm, T, C, CB,
    cutlass::epilogue::fusion::LinearCombination<float,float,float,float,cutlass::FloatRoundStyle::round_to_nearest>, PS>;

static void register_gemm_benchmarks() {
    for k in kernels:
        f.write(f'  CUTLASS_BENCHMARK({k});  // declared externally\n')
    f.write("}\n")

    # Write filter
    with open(f"{bdir}/benchmarks/gemm/cutlass_benchmark_filter.hpp", "w") as f:
        for k in kernels:
            f.write(f"^{k}$\n")

    # cmake
    result = subprocess.run([
        "cmake", "-S", "${REPO}", "-B", bdir,
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=icpx",
        "-DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31", "-DDPCPP_HOST_COMPILER=g++-13",
        "-DCUTLASS_ENABLE_SYCL=ON", "-DCUTLASS_NVCC_ARCHS=",
        "-DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON",
        f"-DCUTLASS_KERNEL_FILTER_FILE={bdir}/benchmarks/gemm/cutlass_benchmark_filter.hpp",
        f"-DGOOGLETEST_DIR={GTEST}", f"-DGOOGLEBENCHMARK_DIR={GB}",
    ], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        with lock: print(f"CMAKE_FAIL: {bid}", flush=True); completed["fail"] += 1
        return

    # Build
    sem = threading.Semaphore(0)
    def do_build():
        subprocess.run(["make", "-C", bdir, "cutlass_benchmarks_gemm_sycl", f"-j${BUILD_CORES}"], 
                      capture_output=True, text=True)
        sem.release()
    
    build_thread = threading.Thread(target=do_build)
    build_thread.start()
    # Wait at most 600s
    build_thread.join(timeout=600)
    if build_thread.is_alive():
        with lock: print(f"BUILD_TIMEOUT: {bid}", flush=True); completed["fail"] += 1
        return

    binpath = f"{bdir}/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
    if os.path.isfile(binpath):
        with lock: print(f"BUILD_OK: {bid} ({os.path.getsize(binpath)} bytes)", flush=True); completed["ok"] += 1
    else:
        with lock: print(f"BUILD_FAIL: {bid}", flush=True); completed["fail"] += 1

# Process in parallel with limited concurrency
semaphore = threading.BoundedSemaphore(${PARALLEL_BUILDS})
def worker(bid):
    semaphore.acquire()
    try:
        build_batch(bid)
    finally:
        semaphore.release()

if "${DRY_RUN}" == "true":
    ids = batch_ids[:4]  # Only first 4 batches for dry run
    print(f"DRY RUN: building {len(ids)} batches", file=sys.stderr)
else:
    ids = batch_ids

with ThreadPoolExecutor(max_workers=${PARALLEL_BUILDS}) as ex:
    list(ex.map(worker, ids))

print(f"DONE: {completed['ok']} OK, {completed['fail']} FAIL", file=sys.stderr)
PYEOF

  log "Pipeline complete. Results: $WORKSPACE/results/"
}

main "$@"
