#!/bin/bash
#=============================================================================
# Per-Batch Build + Screen Pipeline
# 
# For each batch:
#   1. Backup source, generate mini HPP + main.cpp
#   2. make main.cpp.o (baked perf flags, 33s)
#   3. icpx manual link (fixed link command)  
#   4. Screen kernels one-by-one on assigned GPU
#   5. Restore source files
#
# Usage:
#   BATCH_SIZE=2 PARALLEL=8 DRY_RUN=true bash tools/build_screen.sh
#   BATCH_SIZE=2 PARALLEL=8 bash tools/build_screen.sh  # full run
#=============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
ONEAPI="/opt/intel/oneapi/compiler/2025.3"
WORKSPACE="${WORKSPACE:-/root/cutlass_profile_device7_b70_2500mhz/screen_ws}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PARALLEL="${PARALLEL:-8}"
BUILD_CORES="${BUILD_CORES:-128}"
SHAPE="${SHAPE:-8192 4096 1536}"
GPUS="${GPUS:-5,7}"
TIMEOUT="${TIMEOUT:-60}"
DRY_RUN="${DRY_RUN:-true}"
CATALOG="${CATALOG:-layered_bmg}"
GOOD_WS=/root/cutlass_profile_device7_b70_2500mhz/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws
GOOD_BATCH=$GOOD_WS/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001

read M N K <<< "$SHAPE"

log() { echo "[$(date +%H:%M:%S)] $*"; }

setup() {
  source "$ONEAPI/env/vars.sh" 2>/dev/null || true
  export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
  export IGC_VectorAliasBBThreshold=10000
  export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
  
  rm -rf "$WORKSPACE"
  mkdir -p "$WORKSPACE"/{builds,results,logs}
  
  # CPU performance
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$gov" 2>/dev/null || true
  done
  
  # GPU freq
  for gpu in ${GPUS//,/ }; do
    f="/sys/class/drm/card${gpu}/gt_max_freq_mhz"
    [ -f "$f" ] && echo 2500 > "$f" 2>/dev/null || true
  done
  
  # Restore deps
  mkdir -p "$REPO/_deps"/{googlebenchmark-src/include/benchmark,googletest-src}
  cp -r "$GOOD_BATCH/_deps/googlebenchmark-src/include/benchmark/"* "$REPO/_deps/googlebenchmark-src/include/benchmark/" 2>/dev/null || true
  cp -r "$GOOD_BATCH/_deps/googletest-src/googletest" "$REPO/_deps/googletest-src/" 2>/dev/null || true
  cp -r "$GOOD_BATCH/_deps/googletest-src/googlemock" "$REPO/_deps/googletest-src/" 2>/dev/null || true
  
  # Pre-built libs
  GB_LIB=$(find "$GOOD_BATCH" -name "libbenchmark.a" | head -1)
  CUTLASS_LIB=$(find "$GOOD_BATCH" -name "libcutlass.a" | head -1)
  
  log "Setup done. GB_LIB=$GB_LIB CUTLASS_LIB=$CUTLASS_LIB"
}

gen_manifest() {
  log "Phase 1: Generating kernel list..."
  cd "$REPO/test"
  python3 << PYEOF
import sys; sys.path.insert(0, "benchmarks")
from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints
import json, os

cons = default_constraints()
cat = generated_layered_bmg_kernel_catalog(constraints=cons)
df = {"bf16": "bf16"}.get("${DTYPE}", "bf16")
all_k = sorted(set(k["kernel_name"] for k in cat["kernels"] if k.get("dtype_family") == df))
all_k = [k for k in all_k if not k.startswith("03_bmg") and "streamk_example" not in k]

batch_sz = ${BATCH_SIZE}
batches = [all_k[i:i+batch_sz] for i in range(0, len(all_k), batch_sz)]

os.makedirs("${WORKSPACE}/builds", exist_ok=True)
manifest = {"total": len(all_k), "batch_size": batch_sz, "batches": []}
for i, batch in enumerate(batches):
    bid = f"batch_{i:04d}"
    mani_f = f"${WORKSPACE}/builds/{bid}.txt"
    with open(mani_f, "w") as f:
        for k in batch: f.write(k + "\n")
    manifest["batches"].append({"id": bid, "count": len(batch), "gpu": [${GPUS//,/ }][i%2], "manifest": mani_f})
with open("${WORKSPACE}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Generated {len(batches)} batches ({len(all_k)} kernels)", file=sys.stderr)
PYEOF
  log "Manifest: $WORKSPACE/manifest.json"
}

build_and_screen_batch() {
  local bid=$1 manifest=$2 gpu=$3
  
  # Copy/reuse cmake infra
  local bdir="$WORKSPACE/builds/$bid"
  if [ ! -d "$bdir" ]; then
    cp -a "$GOOD_BATCH" "$bdir"
    find "$bdir" -name "*.cmake" -o -name "CMakeCache.txt" | xargs sed -i "s|selected_kernel_batch_001|${bid}|g" 2>/dev/null
  fi
  
  # Backup + gen mini
  cp "$REPO/benchmarks/gemm/benchmarks_sycl.hpp" "$REPO/benchmarks/gemm/benchmarks_sycl.hpp.bak" 2>/dev/null
  cp "$REPO/benchmarks/gemm/main.cpp" "$REPO/benchmarks/gemm/main.cpp.bak" 2>/dev/null
  
  python3 "$REPO/tools/gen_mini_hpp.py" --manifest "$manifest" --output /tmp/${bid}.hpp > /dev/null 2>&1
  cp /tmp/${bid}.hpp "$REPO/benchmarks/gemm/benchmarks_sycl.hpp"
  
  # Generate main.cpp with kernels
  local runs=""
  while IFS= read -r k; do
    [ -n "$k" ] && runs="$runs  RUN($k)\\n"
  done < "$manifest"
  sed "s/__M__/${M}/; s/__N__/${N}/; s/__K__/${K}/; s|  __RUN_LIST__|${runs}|" "$REPO/tools/main_template.cpp.in" > "$REPO/benchmarks/gemm/main.cpp"
  
  # Compile
  rm -f "$bdir/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o"
  rm -f "$bdir/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
  make -C "$bdir" cutlass_benchmarks_gemm_sycl -j${BUILD_CORES} 2>/dev/null || true
  
  # Manual link
  local obj="$bdir/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o"
  local bin="$bdir/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
  icpx -fsycl -fsycl-targets=spir64_gen \
    -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    -O3 -DNDEBUG \
    "$obj" -o "$bin" "$GB_LIB" \
    -L/lib64/stubs -Wl,-rpath,/lib64/stubs: "$CUTLASS_LIB" \
    -Wl,-rpath=/opt/intel/oneapi/mkl/2025.3/lib \
    /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_ilp64.so \
    /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_thread.so \
    /opt/intel/oneapi/mkl/2025.3/lib/libmkl_core.so \
    /opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so \
    -lm -ldl -lpthread /opt/intel/oneapi/compiler/2025.3/lib/libsycl.so \
    2>/dev/null
  
  if [ ! -f "$bin" ]; then
    log "BUILD_FAIL: $bid"
    return 1
  fi
  log "BUILD_OK: $bid ($(stat -c%s "$bin") bytes)"
  
  # Screen
  local result="$WORKSPACE/results/${bid}_gpu${gpu}.csv"
  echo "kernel,tflops,status,gpu" > "$result"
  local count=0 pass=0
  export ZE_AFFINITY_MASK="$gpu"
  
  while IFS= read -r kernel; do
    [ -z "$kernel" ] && continue
    count=$((count + 1))
    local out tflops status
    out=$(timeout ${TIMEOUT} "$bin" --kernel="$kernel" --m="$M" --n="$N" --k="$K" 2>&1) || true
    tflops=$(echo "$out" | grep -oP 'median_tflops=\K[0-9.]+' || echo "0")
    status=$(echo "$out" | grep -oP 'STATUS=\K[A-Z]+' || echo "TIMEOUT")
    echo "$kernel,$tflops,$status,$gpu" >> "$result"
    [ "$status" = "OK" ] && pass=$((pass + 1))
    if [ $((count % 5)) -eq 0 ]; then
      log "  GPU$gpu $bid: $count/$pass (last: ${tflops} TFLOPS)"
    fi
  done < "$manifest"
  
  log "GPU$gpu $bid DONE: $pass/$count passed"
  
  # Restore
  mv "$REPO/benchmarks/gemm/benchmarks_sycl.hpp.bak" "$REPO/benchmarks/gemm/benchmarks_sycl.hpp" 2>/dev/null || true
  mv "$REPO/benchmarks/gemm/main.cpp.bak" "$REPO/benchmarks/gemm/main.cpp" 2>/dev/null || true
  
  return 0
}

main() {
  setup
  gen_manifest
  
  local batches
  batches=$(python3 -c "import json; m=json.load(open('$WORKSPACE/manifest.json')); print(len(m['batches']))")
  
  if [ "$DRY_RUN" = "true" ]; then
    batches=4
    log "DRY RUN: $batches batches"
  else
    log "FULL RUN: $batches batches, $PARALLEL parallel"
  fi
  
  # Process batches in parallel
  python3 << PYEOF
import json, subprocess, sys, os, threading, time

manifest = json.load(open("${WORKSPACE}/manifest.json"))
batches = manifest["batches"][:${batches}] if ${DRY_RUN} == "true" else manifest["batches"]

sem = threading.BoundedSemaphore(${PARALLEL})
lock = threading.Lock()
results = {"ok": 0, "fail": 0}

def process(batch):
    sem.acquire()
    try:
        cmd = [
            "bash", "${SCRIPT_DIR}/build_screen.sh", "process_one",
            batch["id"], batch["manifest"], str(batch["gpu"])
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        with lock:
            for line in result.stderr.split("\\n") + result.stdout.split("\\n"):
                if "BUILD_OK" in line or "DONE" in line:
                    print(line.strip(), flush=True)
            if result.returncode == 0:
                results["ok"] += 1
            else:
                results["fail"] += 1
    except Exception as e:
        with lock:
            results["fail"] += 1
            print(f"EXCEPTION: {batch['id']}: {e}", flush=True)
    finally:
        sem.release()

threads = []
for batch in batches:
    t = threading.Thread(target=process, args=(batch,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()

print(f"COMPLETE: {results['ok']} OK, {results['fail']} FAIL", flush=True)
PYEOF
}

if [ "${1:-}" = "process_one" ]; then
  build_and_screen_batch "${2:-}" "${3:-}" "${4:-}"
else
  main
fi
