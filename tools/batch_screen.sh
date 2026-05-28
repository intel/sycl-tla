#!/bin/bash
#=============================================================================
# SYCL-TLA GEMM Profiler — Batch Build + Screen Pipeline
# 
# Usage:
#   bash tools/batch_screen.sh \
#     --workspace /path/to/ws \
#     --shape "8192 4096 1536" \
#     --dtype bf16 \
#     --batch-size 8 \
#     --gpus "5,7" \
#     --gpu-freq 2500 \
#     --cpu-mode performance \
#     --dry-run
#
# Full clean pipeline:
#   1. Generate kernel list from catalog
#   2. Split into batches (each batch → one small binary)
#   3. cmake configure + build each batch
#   4. Screen kernels one-by-one on specified GPUs
#   5. Aggregate results to CSV
#=============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# --- Defaults ---
WORKSPACE=""
SHAPE_M=8192; SHAPE_N=4096; SHAPE_K=1536
DTYPE="bf16"
BATCH_SIZE=8
GPUS="5,7"
GPU_FREQ=2500
CPU_MODE="performance"
DRY_RUN=false
TIMEOUT=120
ONEAPI_DIR="/opt/intel/oneapi/compiler/2025.3"
CATALOG_SOURCE="layered_bmg"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --shape) read M N K <<< "$2"; SHAPE_M=$M; SHAPE_N=$N; SHAPE_K=$K; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpu-freq) GPU_FREQ="$2"; shift 2 ;;
    --cpu-mode) CPU_MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --catalog-source) CATALOG_SOURCE="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [ -z "$WORKSPACE" ]; then
  echo "ERROR: --workspace required"
  exit 1
fi

rm -rf "$WORKSPACE"
mkdir -p "$WORKSPACE"/{builds,results,logs}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_LOG="$WORKSPACE/logs/screen_${TIMESTAMP}.log"
RESULT_CSV="$WORKSPACE/results/all_results.csv"

# --- Logging ---
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SCREEN_LOG"; }

# --- Setup ---
setup_environment() {
  log "Setting up environment..."
  
  # oneAPI
  source "$ONEAPI_DIR/env/vars.sh" 2>/dev/null || true
  
  # CPU performance mode
  if [ "$CPU_MODE" = "performance" ]; then
    log "Setting CPU governor to performance..."
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      echo performance > "$gov" 2>/dev/null || true
    done
  fi
  
  # GPU frequency lock
  log "Locking GPU frequency to ${GPU_FREQ}MHz..."
  for gpu_id in ${GPUS//,/ }; do
    fpath="/sys/class/drm/card${gpu_id}/gt_max_freq_mhz"
    if [ -f "$fpath" ]; then
      echo "$GPU_FREQ" > "$fpath" 2>/dev/null && log "  GPU $gpu_id: locked to ${GPU_FREQ}MHz" || log "  GPU $gpu_id: freq lock FAILED (need root?)"
    else
      log "  GPU $gpu_id: freq control not available"
    fi
  done

  # Perf flags (baked at compile time via env vars)
  export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
  export IGC_VectorAliasBBThreshold=10000
  export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
  log "Perf flags exported for compile time"
}

# --- Phase 1: Generate kernel list ---
generate_kernel_list() {
  log "Phase 1: Generating kernel list from catalog..."
  
  python3 "$REPO_ROOT/test/benchmarks/intel_gemm_profiler.py" \
    --workspace "$WORKSPACE/tmp_gen" \
    --dtype "$DTYPE" \
    --kernel-catalog-source "$CATALOG_SOURCE" \
    --max-shapes 1 \
    --skip-run \
    --probe-mode off \
    2>/dev/null || true
  
  # Extract kernel names from phase_b_summary
  if [ -f "$WORKSPACE/tmp_gen/reports/phase_b_summary.json" ]; then
    python3 -c "
import json
with open('$WORKSPACE/tmp_gen/reports/phase_b_summary.json') as f:
    s = json.load(f)
kernels = s.get('selected_kernel_filter_list', [])
if not kernels:
    # Fallback: read accepted candidates
    accepted = s.get('accepted_candidate_count', 0)
    print(f'INFO: {accepted} accepted candidates but no filter list', file=__import__('sys').stderr)
" 2>/dev/null
  fi
  
  # Use Python catalog API directly
  python3 << PYEOF
import sys, json
sys.path.insert(0, "$REPO_ROOT/test/benchmarks")
from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints

cons = default_constraints()
cat = generated_layered_bmg_kernel_catalog(constraints=cons)
dtype_map = {"bf16": "bf16", "f16": "f16"}
dtype_family = dtype_map.get("$DTYPE", "bf16")
all_kernels = sorted(set(k["kernel_name"] for k in cat["kernels"] if k.get("dtype_family") == dtype_family))

with open("$WORKSPACE/all_kernels.txt", "w") as f:
    for k in all_kernels:
        f.write(k + "\n")
print(f"Generated {len(all_kernels)} kernels", file=sys.stderr)

# Split into batches
batch_size = $BATCH_SIZE
batches = [all_kernels[i:i+batch_size] for i in range(0, len(all_kernels), batch_size)]
manifest = {"total_kernels": len(all_kernels), "batch_size": batch_size, "batches": []}
for i, batch in enumerate(batches):
    batch_id = f"batch_{i:04d}"
    batch_dir = f"$WORKSPACE/builds/{batch_id}"
    manifest["batches"].append({
        "batch_id": batch_id, "build_dir": batch_dir, "kernel_count": len(batch),
        "kernels": batch, "gpu": $((i % $(echo $GPUS | tr ',' '\n' | wc -l)))
    })

with open("$WORKSPACE/batch_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Split into {len(batches)} batches of {batch_size}", file=sys.stderr)
PYEOF

  TOTAL_KERNELS=$(wc -l < "$WORKSPACE/all_kernels.txt")
  NUM_BATCHES=$(python3 -c "import json; print(len(json.load(open('$WORKSPACE/batch_manifest.json'))['batches']))")
  log "Total kernels: $TOTAL_KERNELS, Batches: $NUM_BATCHES"
}

# --- Phase 2: Build one batch ---
build_batch() {
  local batch_id=$1 build_dir=$2 filter_file=$3
  log "  Building $batch_id..."
  
  rm -rf "$build_dir"
  mkdir -p "$build_dir/benchmarks/gemm"
  
  # Write filter file (^Name$ format for cmake)
  python3 -c "
with open('$filter_file') as f:
    kernels = [l.strip() for l in f if l.strip()]
with open('$build_dir/benchmarks/gemm/cutlass_benchmark_filter.hpp', 'w') as f:
    for k in kernels:
        f.write(f'^{k}$\n')
" 2>/dev/null

  # cmake configure
  cmake -S "$REPO_ROOT" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=icpx \
    -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
    -DDPCPP_HOST_COMPILER=g++-13 \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DCUTLASS_NVCC_ARCHS= \
    -DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON \
    -DCUTLASS_BENCHMARK_EXHAUSTIVE_GEMM=ON \
    -DCUTLASS_KERNEL_FILTER_FILE="$build_dir/benchmarks/gemm/cutlass_benchmark_filter.hpp" \
    -DGOOGLETEST_DIR="$REPO_ROOT/_deps/googletest-src" \
    -DGOOGLEBENCHMARK_DIR="$WORKSPACE/builds/shared_deps/googlebenchmark-src" \
    2>&1 | tail -3
  
  # Build (50% CPU)
  CORES=$(($(nproc) / 2))
  make -C "$build_dir" cutlass_benchmarks_gemm_sycl -j"$CORES" 2>&1 | grep -E "error:|Built target" | head -5
  
  if [ -f "$build_dir/benchmarks/gemm/cutlass_benchmarks_gemm_sycl" ]; then
    log "  $batch_id: BUILD_OK ($(stat -c%s "$build_dir/benchmarks/gemm/cutlass_benchmarks_gemm_sycl") bytes)"
    return 0
  else
    log "  $batch_id: BUILD_FAILED"
    return 1
  fi
}

# --- Phase 3: Screen kernels on one GPU ---
screen_kernels() {
  local gpu_id=$1 batch_dir=$2 kernel_file=$3 result_file=$4
  log "  Screening on GPU $gpu_id..."
  
  local bin="$batch_dir/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
  if [ ! -f "$bin" ]; then
    log "  GPU $gpu_id: binary not found, skipping"
    return 1
  fi
  
  export ZE_AFFINITY_MASK="$gpu_id"
  local count=0 pass=0
  
  while IFS= read -r kernel; do
    [ -z "$kernel" ] && continue
    count=$((count + 1))
    
    local out
    out=$(timeout "$TIMEOUT" "$bin" --kernel="$kernel" --m="$SHAPE_M" --n="$SHAPE_N" --k="$SHAPE_K" 2>&1) || true
    
    local tflops="$(echo "$out" | grep -oP 'median_tflops=\K[0-9.]+' || echo "0")"
    local status="$(echo "$out" | grep -oP 'STATUS=\K[A-Z]+' || echo "TIMEOUT")"
    
    echo "$kernel,$tflops,$status,GPU$gpu_id" >> "$result_file"
    
    if [ "$status" = "OK" ]; then pass=$((pass + 1)); fi
    
    if [ $((count % 10)) -eq 0 ]; then
      log "    GPU$gpu_id: $count done, $pass passed (last: ${tflops} TFLOPS)"
    fi
  done < "$kernel_file"
  
  log "  GPU $gpu_id DONE: $pass/$count passed"
}

# --- Main ---
main() {
  log "=== SYCL-TLA Batch Screen Pipeline ==="
  log "Shape: ${SHAPE_M}x${SHAPE_N}x${SHAPE_K}, Dtype: $DTYPE, GPUs: $GPUS"
  
  setup_environment
  generate_kernel_list
  
  if [ "$DRY_RUN" = true ]; then
    # Dry run: test first 2 batches only
    log "DRY RUN: testing first batch only"
    python3 -c "
import json
with open('$WORKSPACE/batch_manifest.json') as f:
    m = json.load(f)
m['batches'] = m['batches'][:2]
with open('$WORKSPACE/batch_manifest.json', 'w') as f:
    json.dump(m, f)
"
  fi
  
  # Phase 2: Build all batches
  log "Phase 2: Building batches..."
  BUILD_PASS=0; BUILD_FAIL=0
  
  python3 -c "
import json, subprocess, sys
with open('$WORKSPACE/batch_manifest.json') as f:
    m = json.load(f)

# First batch: also builds shared deps (googlebenchmark)
for i, batch in enumerate(m['batches']):
    bid = batch['batch_id']
    bdir = batch['build_dir']
    # Write kernel list
    kfile = f'$WORKSPACE/builds/{bid}_kernels.txt'
    with open(kfile, 'w') as f:
        for k in batch['kernels']:
            f.write(k + '\n')
    print(f'{bid}: {len(batch[\"kernels\"])} kernels written to {kfile}')
" 2>/dev/null
  
  # Build sequentially (first batch takes longest, rest reuse deps)
  local i=0
  python3 -c "
import json, subprocess, sys, os
with open('$WORKSPACE/batch_manifest.json') as f:
    m = json.load(f)

for batch in m['batches'][:1]:  # Just first batch for now
    bid = batch['batch_id']
    bdir = batch['build_dir']
    kfile = f'$WORKSPACE/builds/{bid}_kernels.txt'
    
    sys.stderr.write(f'Building {bid}...\n')
    result = subprocess.run([
        'bash', '$SCRIPT_DIR/batch_screen.sh', 'build_one',
        bdir, kfile, '$REPO_ROOT', '$WORKSPACE'
    ], capture_output=True, text=True)
    sys.stderr.write(result.stderr)
" 2>/dev/null
  
  log "Pipeline complete. Results: $RESULT_CSV"
}

# Allow sub-command invocation
if [ "${1:-}" = "build_one" ]; then
  build_batch "$2" "$3" "$4" "$5"
elif [ "${1:-}" = "screen_one" ]; then
  screen_kernels "$2" "$3" "$4" "$5"
else
  main
fi
