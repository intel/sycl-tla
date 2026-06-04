#!/bin/bash
# Clean remote full retest launcher for Maginfra2/B70.
# Usage on remote:
#   bash tools/remote_clean_full_retest.sh
# Optional overrides:
#   RUN_ID=my_run BUILD_DIR=/path/to/build GPU_COUNT=4 BATCHES=all bash tools/remote_clean_full_retest.sh

set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/root/cutlass_profile_device7_b70_2500mhz}"
REPO_ROOT="${REPO_ROOT:-$ROOT_DIR/sycl-tla}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/screen_runs}"
RUN_ID="${RUN_ID:-full_retest_$(date +%Y%m%d_%H%M%S)}"
SCREEN_WS="${SCREEN_WS:-$RUNS_DIR/$RUN_ID}"
RESULTS_DIR="${RESULTS_DIR:-$SCREEN_WS/results}"
LOG_FILE="${LOG_FILE:-/tmp/${RUN_ID}.log}"
GIT_REF="${GIT_REF:-origin/main}"
GPU_COUNT="${GPU_COUNT:-4}"
BATCHES="${BATCHES:-all}"
GPU_MAX_FREQ_MHZ="${GPU_MAX_FREQ_MHZ:-2500}"

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

setup_env() {
  source /opt/intel/oneapi/compiler/2025.3/env/vars.sh 2>/dev/null || true
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
  export SYCL_PROGRAM_COMPILE_OPTIONS="${SYCL_PROGRAM_COMPILE_OPTIONS:--ze-opt-large-register-file -gline-tables-only}"
  export IGC_VectorAliasBBThreshold="${IGC_VectorAliasBBThreshold:-10000}"
  export IGC_ExtraOCLOptions="${IGC_ExtraOCLOptions:--cl-intel-256-GRF-per-thread}"

  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$gov" 2>/dev/null || true
  done

  for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    freq_path="/sys/class/drm/card${gpu}/gt_max_freq_mhz"
    if [ -f "$freq_path" ]; then
      echo "$GPU_MAX_FREQ_MHZ" > "$freq_path" 2>/dev/null || true
    fi
  done
}

kill_existing_runs() {
  mapfile -t pids < <(python3 <<'PY'
import subprocess
from collections import defaultdict

lines = subprocess.check_output(
    ["ps", "-eo", "pid=,ppid=,stat=,cmd="],
    text=True,
).splitlines()

procs = {}
children = defaultdict(list)
roots = []

for line in lines:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid, ppid, stat, cmd = parts
    pid = int(pid)
    ppid = int(ppid)
    procs[pid] = (ppid, stat, cmd)
    children[ppid].append(pid)

for pid, (_, stat, cmd) in procs.items():
    if stat.startswith("Z"):
        continue
    if "tools/run_seq.sh" in cmd and "remote_clean_full_retest.sh" not in cmd:
        roots.append(pid)

seen = set()
order = []

def visit(pid):
    for child in children.get(pid, []):
        visit(child)
    if pid in seen:
        return
    seen.add(pid)
    stat = procs.get(pid, ("", "Z", ""))[1]
    if not stat.startswith("Z"):
        order.append(pid)

for root in roots:
    visit(root)

for pid in order:
    print(pid)
PY
)

  if [ "${#pids[@]}" -eq 0 ]; then
    log "No active run_seq.sh processes found."
    return
  fi

  log "Stopping existing screening PIDs: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 3
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

sync_repo() {
  cd "$REPO_ROOT"
  git fetch origin
  git reset --hard "$GIT_REF"
  git clean -fd -e _deps
  git checkout -- benchmarks/gemm/benchmarks_sycl.hpp benchmarks/gemm/main.cpp 2>/dev/null || true
  rm -f benchmarks/gemm/benchmarks_sycl.hpp.cache
  log "Repo synced to $(git log --oneline -1)"
}

prepare_workspace() {
  mkdir -p "$RUNS_DIR"
  rm -rf "$SCREEN_WS"
  mkdir -p "$SCREEN_WS/results" "$SCREEN_WS/builds"
  rm -f "$LOG_FILE"

  cat > "$SCREEN_WS/run_meta.txt" <<EOF
run_id=$RUN_ID
screen_ws=$SCREEN_WS
results_dir=$RESULTS_DIR
log_file=$LOG_FILE
build_dir=$BUILD_DIR
git_ref=$GIT_REF
git_head=$(cd "$REPO_ROOT" && git rev-parse HEAD)
started_at=$(date -Iseconds)
gpu_count=$GPU_COUNT
batches=$BATCHES
EOF
}

reconfigure_build() {
  if [ ! -d "$BUILD_DIR" ]; then
    echo "BUILD_DIR not found: $BUILD_DIR" >&2
    exit 1
  fi

  log "Reconfiguring build tree..."
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" > "/tmp/${RUN_ID}_cmake.log" 2>&1
  tail -10 "/tmp/${RUN_ID}_cmake.log" || true

  rm -f \
    "$BUILD_DIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o" \
    "$BUILD_DIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
}

launch_run() {
  log "Launching full retest..."
  nohup env \
    REPO_ROOT="$REPO_ROOT" \
    SCREEN_WS="$SCREEN_WS" \
    BUILD_DIR="$BUILD_DIR" \
    RESULTS_DIR="$RESULTS_DIR" \
    GPU_COUNT="$GPU_COUNT" \
    BATCHES="$BATCHES" \
    bash "$REPO_ROOT/tools/run_seq.sh" >> "$LOG_FILE" 2>&1 < /dev/null &
  RUN_PID=$!

  echo "pid=$RUN_PID" >> "$SCREEN_WS/run_meta.txt"
  sleep 5
  if ! kill -0 "$RUN_PID" 2>/dev/null; then
    echo "run_seq.sh failed to stay alive; see $LOG_FILE" >&2
    exit 1
  fi

  log "Started PID $RUN_PID"
  ps -p "$RUN_PID" -o pid,etime,cmd
  sed -n '1,12p' "$LOG_FILE" || true
  log "Workspace: $SCREEN_WS"
  log "Results:   $RESULTS_DIR"
  log "Log file:  $LOG_FILE"
}

main() {
  setup_env
  kill_existing_runs
  sync_repo
  prepare_workspace
  reconfigure_build
  launch_run
}

main "$@"
