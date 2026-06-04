#!/bin/bash
# Production screening pipeline — fully self-contained.
# Usage: BATCHES=886 RESULTS_DIR=/path/to/results bash run_seq.sh
set -eo pipefail

# --- Paths (set via env or auto-detect) ---
REPO="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
S="${REPO_ROOT:-$REPO}"
WS="${SCREEN_WS:-$REPO/../screen_ws}"
RESULTS_DIR="${RESULTS_DIR:-$WS/results}"
BATCHES="${BATCHES:-4}"
# Build directory with pre-built dependencies
BDIR="${BUILD_DIR:-$REPO/../build_candidate_preflight}"
GPU_COUNT="${GPU_COUNT:-4}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Env setup ---
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh 2>/dev/null || true
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000 IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $gov 2>/dev/null; done
mkdir -p "$RESULTS_DIR" "$WS/builds"

cd "$S"
git checkout -- benchmarks/gemm/benchmarks_sycl.hpp benchmarks/gemm/main.cpp 2>/dev/null
rm -f benchmarks/gemm/benchmarks_sycl.hpp.cache

# --- Auto-generate manifest if missing or stale ---
_regenerate_manifest() {
    log "Regenerating kernel manifest..."
    python3 -c "
import sys, json, os
sys.path.insert(0, '$S/test/benchmarks')
sys.path.insert(0, '$S/python')
from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints
cons = default_constraints()
cat = generated_layered_bmg_kernel_catalog(constraints=cons)
all_k = sorted(set(k['kernel_name'] for k in cat['kernels'] if k.get('dtype_family', '') in ('bf16', '16b')))
all_k = [k for k in all_k if not k.startswith('03_bmg') and 'streamk_example' not in k]
b_size = 2
batches = [all_k[i:i+b_size] for i in range(0, len(all_k), b_size)]
m = {'total': len(all_k), 'batch_size': b_size, 'batches': []}
for i, batch in enumerate(batches):
    bid = f'batch_{i:04d}'
    mf = f'$WS/builds/{bid}.txt'
    with open(mf, 'w') as f:
        for k in batch: f.write(k + '\n')
    m['batches'].append({'id': bid, 'count': len(batch), 'gpu': i % $GPU_COUNT, 'manifest': mf})
with open('$WS/manifest.json', 'w') as f:
    json.dump(m, f, indent=2)
print(f'{len(batches)} batches ({len(all_k)} kernels)')
"
}

_manifest_total() {
    python3 -c "import json; m=json.load(open('$WS/manifest.json')); print(len(m.get('batches', [])))" 2>/dev/null || echo 0
}

TOTAL=$(_manifest_total)
if [ "$TOTAL" -eq 0 ] || [ "$TOTAL" = "0" ]; then
    _regenerate_manifest
    TOTAL=$(_manifest_total)
fi
[ "$BATCHES" = "all" ] && BATCHES=$TOTAL
log "Processing $BATCHES of $TOTAL batches"

for i in $(seq 0 $((BATCHES-1))); do
  bid=$(printf "batch_%04d" $i)
  bf="$WS/builds/${bid}.txt"
  [ ! -f "$bf" ] && continue
  gpu=$(( i % GPU_COUNT ))
  
  cp $S/benchmarks/gemm/benchmarks_sycl.hpp /tmp/bak_hpp
  cp $S/benchmarks/gemm/main.cpp /tmp/bak_main
  rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
  
  python3 $S/tools/gen_mini_hpp.py --manifest "$bf" --output /tmp/${bid}.hpp > /dev/null 2>&1
  cp /tmp/${bid}.hpp $S/benchmarks/gemm/benchmarks_sycl.hpp
  python3 $S/tools/gen_main.py "$bf" "$S/benchmarks/gemm/main.cpp"
  
  rm -f $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o $BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
  touch $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts
  touch $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.make
  
  # Build failures are expected for bad batches; keep screening instead of
  # exiting the whole run under set -e.
  BUILD_OK=1
  if ! make -C $BDIR cutlass_benchmarks_gemm_sycl -j128 > /tmp/mk_${bid}.log 2>&1; then
    BUILD_OK=0
  fi
  BIN=$BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
  
  if [ "$BUILD_OK" -ne 1 ] || [ ! -x "$BIN" ]; then
    if grep -q "1 error generated" /tmp/mk_${bid}.log 2>/dev/null; then
      log "[$bid] COMPILE FAIL"
    else
      log "[$bid] LINK FAIL"
    fi
    cp /tmp/bak_hpp $S/benchmarks/gemm/benchmarks_sycl.hpp
    cp /tmp/bak_main $S/benchmarks/gemm/main.cpp
    rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
    continue
  fi
  
  export ZE_AFFINITY_MASK=$gpu
  R="$RESULTS_DIR/${bid}_gpu${gpu}.csv"
  echo "kernel,tflops,status,gpu" > $R
  while read kernel; do
    [ -z "$kernel" ] && continue
    set +e
    out=$(timeout 120 $BIN --kernel=$kernel --m=8192 --n=4096 --k=1536 2>&1)
    rc=$?
    set -e
    tf=$(echo "$out" | grep -oP "median_tflops=\K[0-9.]+" || echo "0")
    if echo "$out" | grep -q "RESULT kernel="; then
      st="OK"
    elif [ "$rc" -eq 124 ]; then
      st="TIMEOUT"
    elif echo "$out" | grep -q "NOT_FOUND"; then
      st="NOT_FOUND"
    else
      st=$(echo "$out" | grep -oP "STATUS=\K[A-Z]+" | head -1)
      [ -z "$st" ] && st="FAIL"
    fi
    echo "$kernel,$tf,$st,$gpu" >> $R
  done < "$bf"
  log "[$((i+1))/$BATCHES] GPU$gpu $bid done"
  
  cp /tmp/bak_hpp $S/benchmarks/gemm/benchmarks_sycl.hpp
  cp /tmp/bak_main $S/benchmarks/gemm/main.cpp
  rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
done
log "Done. Results: $RESULTS_DIR"
