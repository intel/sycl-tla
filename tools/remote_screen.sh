#!/bin/bash
#=============================================================================
# Remote Screening Launcher — runs batch build + screen on remote B70
#=============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# --- Config ---
REMOTE_HOST="${REMOTE_HOST:-10.239.11.149}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_REPO="${REMOTE_REPO:-/root/cutlass_profile_device7_b70_2500mhz/sycl-tla}"
WORKSPACE="${WORKSPACE:-/root/cutlass_profile_device7_b70_2500mhz/screen_ws}"
SHAPE="${SHAPE:-8192 4096 1536}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPUS="${GPUS:-5,7}"
GPU_FREQ="${GPU_FREQ:-2500}"
DRY_RUN="${DRY_RUN:-true}"
DTYPE="${DTYPE:-bf16}"
ONEAPI_DIR="/opt/intel/oneapi/compiler/2025.3"

read M N K <<< "$SHAPE"

echo "=== Remote Screen Pipeline ==="
echo "Host: $REMOTE_HOST, Shape: ${M}x${N}x${K}, GPUs: $GPUS, Dry-run: $DRY_RUN"
echo ""

# Step 1: Sync code
echo "--- Syncing code ---"
ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "
  cd $REMOTE_REPO && git fetch origin && git reset --hard origin/main && git clean -fd
  echo 'HEAD:' \$(git log --oneline -1)
"

# Step 2: Copy scripts (in case they're not in repo yet)
scp -o StrictHostKeyChecking=no "$REPO_ROOT/tools/gen_batches.py" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_REPO/tools/"
scp -o StrictHostKeyChecking=no "$REPO_ROOT/tools/build_batch.py" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_REPO/tools/"
scp -o StrictHostKeyChecking=no "$REPO_ROOT/tools/screen_batch.py" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_REPO/tools/"

# Step 3: Generate batches
echo "--- Generating batches ---"
ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "
  source $ONEAPI_DIR/env/vars.sh 2>/dev/null || true
  rm -rf $WORKSPACE
  cd $REMOTE_REPO
  python3 tools/gen_batches.py --workspace $WORKSPACE --dtype $DTYPE --batch-size $BATCH_SIZE
  echo 'Batch manifest:'
  python3 -c \"import json; m=json.load(open('$WORKSPACE/batch_manifest.json')); print(f'Batches: {len(m[\"batches\"])}, Total kernels: {sum(b[\"count\"] for b in m[\"batches\"])}')\"
"

# Step 4: Setup environment (CPU + GPU)
echo "--- Setting up environment ---"
ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "
  # CPU performance mode
  echo 'Setting CPU governors to performance...'
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > \$gov 2>/dev/null || true
  done
  # GPU freq lock
  for gpu in ${GPUS//,/ }; do
    fpath=\"/sys/class/drm/card\${gpu}/gt_max_freq_mhz\"
    if [ -f \"\$fpath\" ]; then
      echo $GPU_FREQ > \"\$fpath\" 2>/dev/null && echo \"GPU \$gpu: $GPU_FREQ MHz\" || echo \"GPU \$gpu: lock failed\"
    fi
  done
  # Verify
  sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
"

# Step 5: Build + Screen (dry run or full)
if [ "$DRY_RUN" = "true" ]; then
  echo "--- DRY RUN: Building first batch ---"
  BATCH_JSON=$(ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" \
    "python3 -c \"import json; m=json.load(open('$WORKSPACE/batch_manifest.json')); print(json.dumps(m['batches'][0]))\"")
  BID=$(echo "$BATCH_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
  MANIFEST=$(echo "$BATCH_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['manifest'])")
  
  ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "
    source $ONEAPI_DIR/env/vars.sh 2>/dev/null || true
    export SYCL_PROGRAM_COMPILE_OPTIONS='-ze-opt-large-register-file -gline-tables-only'
    export IGC_VectorAliasBBThreshold=10000
    export IGC_ExtraOCLOptions='-cl-intel-256-GRF-per-thread'
    cd $REMOTE_REPO
    python3 tools/build_batch.py \
      --repo-root $REMOTE_REPO \
      --build-dir $WORKSPACE/builds/$BID \
      --manifest $MANIFEST
  "
  
  echo "--- DRY RUN: Screening on GPU 5 ---"
  ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "
    source $ONEAPI_DIR/env/vars.sh 2>/dev/null || true
    cd $REMOTE_REPO
    python3 tools/screen_batch.py \
      --binary $WORKSPACE/builds/$BID/benchmarks/gemm/cutlass_benchmarks_gemm_sycl \
      --manifest $MANIFEST --gpu 5 --m $M --n $N --k $K \
      --output $WORKSPACE/results/${BID}_gpu5.csv --batch-id $BID
  "
  
  echo "--- DRY RUN COMPLETE ---"
  ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "head -15 $WORKSPACE/results/*.csv"
else
  echo "--- FULL SCREENING ---"
  # Launch background screen on GPU 5 and GPU 7
  ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "
    source $ONEAPI_DIR/env/vars.sh 2>/dev/null || true
    export SYCL_PROGRAM_COMPILE_OPTIONS='-ze-opt-large-register-file -gline-tables-only'
    export IGC_VectorAliasBBThreshold=10000
    export IGC_ExtraOCLOptions='-cl-intel-256-GRF-per-thread'
    
    cd $REMOTE_REPO
    
    # Read manifest
    python3 -c \"
import json, subprocess, os, sys
with open('$WORKSPACE/batch_manifest.json') as f:
    m = json.load(f)

batches = m['batches']
print(f'Total: {len(batches)} batches', file=sys.stderr)

# Screen each batch on its assigned GPU (one at a time per GPU)
from concurrent.futures import ThreadPoolExecutor
import threading
lock = threading.Lock()

results = {'gpu5': [], 'gpu7': []}

def process(batch):
    bid = batch['id']
    gpu = batch['gpu']
    # Build first
    br = subprocess.run([
        'python3', 'tools/build_batch.py',
        '--repo-root', '$REMOTE_REPO',
        '--build-dir', f'$WORKSPACE/builds/{bid}',
        '--manifest', batch['manifest'],
    ], capture_output=True, text=True)
    
    if 'BUILD_OK' not in br.stdout:
        with lock: print(f'BUILD_FAIL: {bid}', flush=True)
        return
    
    # Screen
    sr = subprocess.run([
        'python3', 'tools/screen_batch.py',
        '--binary', f'$WORKSPACE/builds/{bid}/benchmarks/gemm/cutlass_benchmarks_gemm_sycl',
        '--manifest', batch['manifest'],
        '--gpu', str(gpu), '--m', '$M', '--n', '$N', '--k', '$K',
        '--output', f'$WORKSPACE/results/{bid}_gpu{gpu}.csv',
        '--batch-id', bid,
    ], capture_output=True, text=True)
    
    with lock:
        for line in sr.stdout.split('\\n'):
            if 'DONE' in line:
                print(line, flush=True)

# Process sequentially (build uses all cores anyway)
for b in batches:
    process(b)

print('ALL BATCHES COMPLETE', flush=True)
\"
  "
fi

echo "=== Done ==="
