#!/bin/bash
# Remote exact-shape search launcher for B70/BMG.
# Runs shapes serially, while distributing batches across multiple GPUs.
#
# Example:
#   bash tools/remote_exact_shape_search.sh
#   RUN_ID=my_run GPU_IDS=0,1,2,3,4 SHAPES="2048x384x3584;8192x384x3584" bash tools/remote_exact_shape_search.sh
#   SKIP_SYNC=1 bash tools/remote_exact_shape_search.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/cutlass_profile_device7_b70_2500mhz}"
REPO_ROOT="${REPO_ROOT:-$ROOT_DIR/sycl-tla}"
GOOD_BUILD="${GOOD_BUILD:-$ROOT_DIR/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/screen_runs}"
RUN_ID="${RUN_ID:-shape_search_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RUNS_DIR/$RUN_ID}"
RESULTS_DIR="${RESULTS_DIR:-$RUN_DIR/results}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
MANIFEST_DIR="${MANIFEST_DIR:-$RUN_DIR/manifests}"
WORKERS_DIR="${WORKERS_DIR:-$RUN_DIR/workers}"
STATUS_DIR="${STATUS_DIR:-$RUN_DIR/status}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-$RUNS_DIR/exact_shape_search_latest.txt}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4}"
SHAPES="${SHAPES:-2048x384x3584;8192x384x3584}"
LAYOUTS="${LAYOUTS:-rcr,rrr}"
KERNEL_CATALOG_SOURCE="${KERNEL_CATALOG_SOURCE:-layered_bmg}"

DTYPE_A="${DTYPE_A:-bf16}"
DTYPE_B="${DTYPE_B:-bf16}"
DTYPE_C="${DTYPE_C:-f32}"
DTYPE_D="${DTYPE_D:-f32}"
DTYPE_ACC="${DTYPE_ACC:-f32}"

BATCH_SIZE="${BATCH_SIZE:-1}"
BUILD_JOBS="${BUILD_JOBS:-32}"
GPU_MAX_FREQ_MHZ="${GPU_MAX_FREQ_MHZ:-2500}"
TIMEOUT="${TIMEOUT:-120}"
GIT_REF="${GIT_REF:-origin/main}"
SKIP_SYNC="${SKIP_SYNC:-0}"
STOP_EXISTING="${STOP_EXISTING:-1}"
ACTIVE_SHAPE_TAG=""
WORKER_SYNC_FILES=(
  benchmarks/common.hpp
  benchmarks/gemm/CMakeLists.txt
  benchmarks/gemm/benchmark_runner.hpp
  benchmarks/gemm/benchmarks_sycl.hpp
  benchmarks/gemm/bmg_streamk_seed_tile.def
  benchmarks/gemm/bmg_streamk_expanded_tile.def
  benchmarks/gemm/bmg_streamk_exhaustive_missing_tile.def
  test/benchmarks/intel_gemm_profiler/catalog.py
  tools/gen_main.py
  tools/gen_mini_hpp.py
  tools/remote_exact_shape_search.sh
  tools/exact_shape_search_report.py
)

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

mark_active_shape_failed() {
  [ -n "${ACTIVE_SHAPE_TAG:-}" ] || return 0
  [ -n "${STATUS_DIR:-}" ] || return 0
  mkdir -p "$STATUS_DIR" 2>/dev/null || true
  echo "failed" > "$STATUS_DIR/${ACTIVE_SHAPE_TAG}.status" 2>/dev/null || true
  touch "$STATUS_DIR/${ACTIVE_SHAPE_TAG}.failed" 2>/dev/null || true
  rm -f "$STATUS_DIR/current_shape" 2>/dev/null || true
}

on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    mark_active_shape_failed
  fi
}

trap on_exit EXIT

setup_env() {
  local had_nounset=0
  case $- in
    *u*) had_nounset=1; set +u ;;
  esac
  source /opt/intel/oneapi/compiler/2025.3/env/vars.sh 2>/dev/null || true
  [ "$had_nounset" -eq 1 ] && set -u
  apply_perf_env

  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$gov" 2>/dev/null || true
  done
}

apply_perf_env() {
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
  export SYCL_PROGRAM_COMPILE_OPTIONS="${SYCL_PROGRAM_COMPILE_OPTIONS:--ze-opt-large-register-file -gline-tables-only}"
  export IGC_VectorAliasBBThreshold="${IGC_VectorAliasBBThreshold:-10000}"
  export IGC_ExtraOCLOptions="${IGC_ExtraOCLOptions:--cl-intel-256-GRF-per-thread}"
}

parse_gpu_ids() {
  IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
  [ "${#GPU_IDS_ARR[@]}" -gt 0 ] || fail "GPU_IDS must not be empty"
}

parse_shapes() {
  IFS=';' read -r -a SHAPE_SPECS <<< "$SHAPES"
  [ "${#SHAPE_SPECS[@]}" -gt 0 ] || fail "SHAPES must not be empty"

  SHAPE_TAGS_ARR=()
  SHAPE_M_ARR=()
  SHAPE_N_ARR=()
  SHAPE_K_ARR=()

  local spec m n k
  for spec in "${SHAPE_SPECS[@]}"; do
    IFS='x' read -r m n k <<< "$spec"
    [[ -n "$m" && -n "$n" && -n "$k" ]] || fail "Invalid shape spec: $spec"
    SHAPE_TAGS_ARR+=("${m}_${n}_${k}")
    SHAPE_M_ARR+=("$m")
    SHAPE_N_ARR+=("$n")
    SHAPE_K_ARR+=("$k")
  done
}

lock_gpu_frequency() {
  local gpu freq_path xpu_discovery=""
  for gpu in "${GPU_IDS_ARR[@]}"; do
    [ -e "/sys/class/drm/card${gpu}" ] || fail "GPU card${gpu} not found"
    freq_path="/sys/class/drm/card${gpu}/gt_max_freq_mhz"
    if [ -f "$freq_path" ]; then
      echo "$GPU_MAX_FREQ_MHZ" > "$freq_path" 2>/dev/null || true
      continue
    fi

    command -v xpu-smi >/dev/null 2>&1 || fail "GPU card${gpu} has no gt_max_freq_mhz and xpu-smi is unavailable"
    if [ -z "$xpu_discovery" ]; then
      xpu_discovery=$(xpu-smi discovery 2>/dev/null) || fail "xpu-smi discovery failed while locking GPU frequency"
    fi

    local xpu_device_id
    xpu_device_id=$(
      XPU_DISCOVERY="$xpu_discovery" GPU_CARD="$gpu" python3 <<'PY'
import os
import re
import sys

card = os.environ["GPU_CARD"]
text = os.environ.get("XPU_DISCOVERY", "")
device_id = ""
current_id = ""
for raw_line in text.splitlines():
    line = raw_line.strip()
    match = re.match(r"^\|\s*([0-9]+)\s*\|", line)
    if match:
        current_id = match.group(1)
        continue
    drm_match = re.search(r"DRM Device:\s*/dev/dri/card([0-9]+)", line)
    if drm_match and drm_match.group(1) == card:
        device_id = current_id
        break

sys.stdout.write(device_id)
PY
    )
    [ -n "$xpu_device_id" ] || fail "Unable to map card${gpu} to an xpu-smi device id"
    xpu-smi config -d "$xpu_device_id" -t 0 --frequencyrange "$GPU_MAX_FREQ_MHZ,$GPU_MAX_FREQ_MHZ" > /dev/null \
      || fail "xpu-smi frequency lock failed for card${gpu} (device ${xpu_device_id})"
  done
}

list_descendant_pids() {
  local root_pid="$1"
  ROOT_PID="$root_pid" python3 <<'PY'
import os
import subprocess
from collections import defaultdict

root_pid = int(os.environ["ROOT_PID"])
lines = subprocess.check_output(["ps", "-eo", "pid=,ppid=,stat="], text=True).splitlines()
children = defaultdict(list)
stats = {}
for line in lines:
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
        continue
    pid, ppid, stat = parts
    pid = int(pid)
    ppid = int(ppid)
    children[ppid].append(pid)
    stats[pid] = stat

ordered = []

def visit(pid):
    for child in children.get(pid, []):
        visit(child)
        if not stats.get(child, "").startswith("Z"):
            ordered.append(child)

visit(root_pid)
seen = set()
for pid in ordered:
    if pid not in seen:
        seen.add(pid)
        print(pid)
PY
}

cleanup_worker_descendants() {
  local worker_pid="$1"
  local shape_tag="$2"
  local gpu="$3"
  local reason="$4"
  local -a stale_pids=()
  local pid

  mapfile -t stale_pids < <(list_descendant_pids "$worker_pid")
  [ "${#stale_pids[@]}" -gt 0 ] || return 0

  log "shape=$shape_tag gpu=$gpu cleanup[$reason] stale_pids=${stale_pids[*]}"
  for pid in "${stale_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1

  mapfile -t stale_pids < <(list_descendant_pids "$worker_pid")
  [ "${#stale_pids[@]}" -gt 0 ] || return 0

  log "shape=$shape_tag gpu=$gpu cleanup[$reason] forcing stale_pids=${stale_pids[*]}"
  for pid in "${stale_pids[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
}

kill_existing_runs() {
  [ "$STOP_EXISTING" = "1" ] || return 0

  local self_pid="$$"
  mapfile -t pids < <(SELF_PID="$self_pid" python3 <<'PY'
import os
import subprocess
from collections import defaultdict

lines = subprocess.check_output(["ps", "-eo", "pid=,ppid=,stat=,cmd="], text=True).splitlines()
procs = {}
children = defaultdict(list)
self_pid = int(os.environ["SELF_PID"])
for line in lines:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid, ppid, stat, cmd = parts
    pid = int(pid)
    ppid = int(ppid)
    procs[pid] = (ppid, stat, cmd)
    children[ppid].append(pid)

protected = {self_pid}
cursor = self_pid
while cursor in procs:
    parent = procs[cursor][0]
    if parent <= 1 or parent in protected:
        break
    protected.add(parent)
    cursor = parent

roots = []
for pid, (_, stat, cmd) in procs.items():
    if pid in protected:
        continue
    if stat.startswith("Z"):
        continue
    if "remote_exact_shape_search.sh" in cmd or "launch_shape_" in cmd:
        roots.append(pid)

seen = set()
order = []
def visit(pid):
    for child in children.get(pid, []):
        visit(child)
    if pid in seen:
        return
    seen.add(pid)
    if pid in protected:
        return
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
    return 0
  fi

  log "Stopping existing exact-shape search PIDs: ${pids[*]}"
  local pid
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
  [ "$SKIP_SYNC" = "1" ] && return 0
  cd "$REPO_ROOT"
  git fetch origin
  git reset --hard "$GIT_REF"
  git clean -fd -e _deps
  git checkout -- benchmarks/gemm/benchmarks_sycl.hpp benchmarks/gemm/main.cpp 2>/dev/null || true
  rm -f benchmarks/gemm/benchmarks_sycl.hpp.cache
  log "Repo synced to $(git log --oneline -1)"
}

validate_template_build() {
  [ -d "$GOOD_BUILD" ] || fail "GOOD_BUILD not found: $GOOD_BUILD"
  [ -f "$GOOD_BUILD/_deps/googlebenchmark-build/src/libbenchmark.a" ] || fail "Missing libbenchmark.a in $GOOD_BUILD"
  [ -d "$REPO_ROOT/_deps/googlebenchmark-src" ] || fail "Missing googlebenchmark source tree in $REPO_ROOT/_deps"
  [ -d "$REPO_ROOT/_deps/googletest-src" ] || fail "Missing googletest source tree in $REPO_ROOT/_deps"
}

prepare_run_dir() {
  mkdir -p "$RUN_DIR" "$RUNS_DIR"
  rm -rf "$RESULTS_DIR" "$LOG_DIR" "$MANIFEST_DIR" "$WORKERS_DIR" "$STATUS_DIR"
  rm -f "$RUN_DIR/launcher.pid" "$RUN_DIR/run_meta.txt" "$RUN_DIR/manifest.json" "$RUN_DIR/requested_shapes.json"
  rm -f "$RUN_DIR"/failed_*.txt
  mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$MANIFEST_DIR" "$WORKERS_DIR" "$STATUS_DIR"
  printf '%s\n' "$RUN_DIR" > "$LATEST_RUN_FILE"
  printf '%s\n' "$$" > "$RUN_DIR/launcher.pid"
  : > "$STATUS_DIR/completed_shapes.txt"
  : > "$STATUS_DIR/worker_pids.txt"
  rm -f "$STATUS_DIR/current_shape"
}

write_metadata() {
  local head_commit
  head_commit=$(cd "$REPO_ROOT" && git rev-parse HEAD)

  python3 - <<PY
import json
from pathlib import Path

run_dir = Path("${RUN_DIR}")
shapes = []
shape_specs = "${SHAPES}".split(";")
for spec in shape_specs:
    m, n, k = spec.split("x")
    shapes.append({"m": int(m), "n": int(n), "k": int(k)})

doc = {
    "dtype_a": "${DTYPE_A}",
    "dtype_b": "${DTYPE_B}",
    "dtype_c": "${DTYPE_C}",
    "dtype_d": "${DTYPE_D}",
    "dtype_acc": "${DTYPE_ACC}",
    "layouts": "${LAYOUTS}".split(","),
    "gpu_ids": [int(x) for x in "${GPU_IDS_CSV}".split(",") if x],
    "execution_mode": "shape_serial_multi_gpu",
    "shapes": shapes,
}
(run_dir / "requested_shapes.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY

  cat > "$RUN_DIR/run_meta.txt" <<EOF
run_id=$RUN_ID
repo_root=$REPO_ROOT
good_build=$GOOD_BUILD
git_head=$head_commit
dtype_a=$DTYPE_A
dtype_b=$DTYPE_B
dtype_c=$DTYPE_C
dtype_d=$DTYPE_D
dtype_acc=$DTYPE_ACC
layouts=$LAYOUTS
kernel_catalog_source=$KERNEL_CATALOG_SOURCE
gpu_ids=$GPU_IDS_CSV
perf_env_ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR
perf_env_SYCL_PROGRAM_COMPILE_OPTIONS=$SYCL_PROGRAM_COMPILE_OPTIONS
perf_env_IGC_VectorAliasBBThreshold=$IGC_VectorAliasBBThreshold
perf_env_IGC_ExtraOCLOptions=$IGC_ExtraOCLOptions
batch_size=$BATCH_SIZE
build_jobs=$BUILD_JOBS
timeout=$TIMEOUT
execution_mode=shape_serial_multi_gpu
shape_order=$SHAPES
started_at=$(date -Iseconds)
EOF
}

generate_manifests() {
  export REPO_ROOT RUN_DIR MANIFEST_DIR DTYPE_A DTYPE_B DTYPE_C DTYPE_D DTYPE_ACC LAYOUTS BATCH_SIZE KERNEL_CATALOG_SOURCE
  export GPU_IDS_CSV
  python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
run_dir = Path(os.environ["RUN_DIR"])
manifest_dir = Path(os.environ["MANIFEST_DIR"])
sys.path.insert(0, str(repo_root / "test/benchmarks"))
sys.path.insert(0, str(repo_root / "python"))

from intel_gemm_profiler.catalog import (
    generated_layered_bmg_kernel_catalog,
    generated_layered_bmg_scheduler_expanded_kernel_catalog,
)
from intel_gemm_profiler.constraints import default_constraints

layouts = tuple(x for x in os.environ["LAYOUTS"].split(",") if x)
gpu_ids = [int(x) for x in os.environ["GPU_IDS_CSV"].split(",") if x]
batch_size = int(os.environ["BATCH_SIZE"])

catalog_source = os.environ["KERNEL_CATALOG_SOURCE"]
catalog_factories = {
    "layered_bmg": generated_layered_bmg_kernel_catalog,
    "layered_bmg_scheduler_expanded": generated_layered_bmg_scheduler_expanded_kernel_catalog,
}
if catalog_source not in catalog_factories:
    raise SystemExit(f"Unsupported KERNEL_CATALOG_SOURCE: {catalog_source}")

catalog = catalog_factories[catalog_source](constraints=default_constraints())
selected_entries = [
    entry
    for entry in catalog["kernels"]
    if entry.get("layout") in layouts
    and entry.get("dtype_a") == os.environ["DTYPE_A"]
    and entry.get("dtype_b") == os.environ["DTYPE_B"]
    and entry.get("dtype_c") == os.environ["DTYPE_C"]
    and entry.get("dtype_d", entry.get("dtype_c")) == os.environ["DTYPE_D"]
    and entry.get("dtype_acc") == os.environ["DTYPE_ACC"]
    and entry.get("runner") != "streamk_example"
]
selected_by_kernel = {entry["kernel_name"]: entry for entry in selected_entries}
kernels = sorted(selected_by_kernel)

if not kernels:
    raise SystemExit("No kernels matched the requested dtype/layout filters.")

batches = [kernels[i:i + batch_size] for i in range(0, len(kernels), batch_size)]
manifest = {
    "total_kernels": len(kernels),
    "batch_size": batch_size,
    "batch_count": len(batches),
    "gpu_count": len(gpu_ids),
    "kernel_catalog_source": catalog_source,
    "catalog_version": catalog.get("catalog_version", ""),
    "kernel_metadata": str(run_dir / "kernel_metadata.json"),
}

manifest_dir.mkdir(parents=True, exist_ok=True)
kernel_metadata = {}
for kernel_name in kernels:
    entry = selected_by_kernel[kernel_name]
    kernel_metadata[kernel_name] = {
        "kernel_name": kernel_name,
        "layout": entry.get("layout"),
        "runner": entry.get("runner"),
        "scheduler_family": entry.get("scheduler_family"),
        "decomposition_mode": entry.get("decomposition_mode"),
        "streamk_mode": entry.get("streamk_mode"),
        "reduction_mode": entry.get("reduction_mode"),
        "tile_m": entry.get("tile_m"),
        "tile_n": entry.get("tile_n"),
        "tile_k": entry.get("tile_k"),
        "sg_m": entry.get("sg_m"),
        "sg_n": entry.get("sg_n"),
        "stages": entry.get("stages"),
        "split_k": entry.get("split_k"),
        "dtype_a": entry.get("dtype_a"),
        "dtype_b": entry.get("dtype_b"),
        "dtype_c": entry.get("dtype_c"),
        "dtype_d": entry.get("dtype_d", entry.get("dtype_c")),
        "dtype_acc": entry.get("dtype_acc"),
    }
(run_dir / "kernel_metadata.json").write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
for gpu_slot in range(len(gpu_ids)):
    (manifest_dir / f"gpu{gpu_ids[gpu_slot]}_batches.txt").write_text("", encoding="utf-8")

for index, batch in enumerate(batches):
    batch_id = f"batch_{index:04d}"
    batch_manifest = manifest_dir / f"{batch_id}.txt"
    batch_manifest.write_text("\n".join(batch) + "\n", encoding="utf-8")
    assigned_gpu = gpu_ids[index % len(gpu_ids)]
    with (manifest_dir / f"gpu{assigned_gpu}_batches.txt").open("a", encoding="utf-8") as handle:
        handle.write(batch_id + "\n")
    manifest[batch_id] = {
        "manifest": str(batch_manifest),
        "count": len(batch),
        "gpu": assigned_gpu,
    }

(run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "total_kernels": len(kernels),
            "batch_count": len(batches),
            "gpu_count": len(gpu_ids),
            "kernel_catalog_source": catalog_source,
        }
    )
)
PY
}

prepare_worker() {
  local gpu="$1"
  local worker_root="$WORKERS_DIR/gpu${gpu}"
  local worker_repo="$worker_root/repo"
  local worker_build="$worker_root/build"
  local relpath

  apply_perf_env

  rm -rf "$worker_root"
  mkdir -p "$worker_root"

  git worktree add --force --detach "$worker_repo" HEAD > "$LOG_DIR/worktree_gpu${gpu}.log" 2>&1
  ln -sfn "$REPO_ROOT/_deps" "$worker_repo/_deps"

  # Worker repos are created from git HEAD, but the remote root repo is often
  # updated by file sync without a matching commit. Mirror the current working
  # tree for the exact-search-critical files so workers run the synced code.
  for relpath in "${WORKER_SYNC_FILES[@]}"; do
    mkdir -p "$worker_repo/$(dirname "$relpath")"
    cp "$REPO_ROOT/$relpath" "$worker_repo/$relpath"
  done

  # The benchmark executable does not require CUTLASS unit tests. Leaving
  # tests enabled here pulls in GTest::gtest during fresh worker configure.
  cmake -S "$worker_repo" -B "$worker_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=icpx \
    -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
    -DDPCPP_HOST_COMPILER=g++-13 \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_NVCC_ARCHS= \
    -DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON \
    -DCUTLASS_BENCHMARK_EXHAUSTIVE_GEMM=ON \
    -DCUTLASS_BENCHMARK_EXHAUSTIVE_STREAMK=ON \
    -DGOOGLETEST_DIR="$REPO_ROOT/_deps/googletest-src" \
    -DGOOGLEBENCHMARK_DIR="$REPO_ROOT/_deps/googlebenchmark-src" \
    > "$LOG_DIR/cmake_gpu${gpu}.log" 2>&1

  mkdir -p "$worker_build/_deps"
  ln -sfn "$GOOD_BUILD/_deps/googlebenchmark-build" "$worker_build/_deps/googlebenchmark-build"
  if [ -d "$GOOD_BUILD/_deps/googletest-build" ]; then
    ln -sfn "$GOOD_BUILD/_deps/googletest-build" "$worker_build/_deps/googletest-build"
  fi

  [ -f "$worker_build/_deps/googlebenchmark-build/src/libbenchmark.a" ] || fail "gpu${gpu}: missing shared libbenchmark.a after worker setup"
}

prepare_workers() {
  local gpu
  git -C "$REPO_ROOT" worktree prune > /dev/null 2>&1 || true
  for gpu in "${GPU_IDS_ARR[@]}"; do
    prepare_worker "$gpu"
  done
}

run_shape_worker() {
  local gpu="$1"
  local shape_tag="$2"
  local shape_m="$3"
  local shape_n="$4"
  local shape_k="$5"
  local worker_pid="${BASHPID:-$$}"

  local worker_root="$WORKERS_DIR/gpu${gpu}"
  local worker_repo="$worker_root/repo"
  local worker_build="$worker_root/build"
  local batch_list="$MANIFEST_DIR/gpu${gpu}_batches.txt"
  local shape_dir="$RESULTS_DIR/$shape_tag"
  local fail_file="$RUN_DIR/failed_${shape_tag}_gpu${gpu}.txt"
  local orig_hpp="/tmp/${RUN_ID}_gpu${gpu}_benchmarks_sycl.hpp"
  local orig_main="/tmp/${RUN_ID}_gpu${gpu}_main.cpp"
  local -a assigned_batches=()
  local -a kernels=()

  mkdir -p "$shape_dir"
  : > "$fail_file"

  cp "$worker_repo/benchmarks/gemm/benchmarks_sycl.hpp" "$orig_hpp"
  cp "$worker_repo/benchmarks/gemm/main.cpp" "$orig_main"
  apply_perf_env
  export ZE_AFFINITY_MASK="$gpu"

  mapfile -t assigned_batches < "$batch_list"
  for batch_id in "${assigned_batches[@]}"; do
    [ -n "$batch_id" ] || continue
    local manifest_path="$MANIFEST_DIR/${batch_id}.txt"
    [ -f "$manifest_path" ] || continue
    cleanup_worker_descendants "$worker_pid" "$shape_tag" "$gpu" "pre_${batch_id}"

    cp "$orig_hpp" "$worker_repo/benchmarks/gemm/benchmarks_sycl.hpp"
    cp "$orig_main" "$worker_repo/benchmarks/gemm/main.cpp"
    rm -f "$worker_repo/benchmarks/gemm/benchmarks_sycl.hpp.cache"

    python3 "$worker_repo/tools/gen_mini_hpp.py" --manifest "$manifest_path" --output "/tmp/${RUN_ID}_${shape_tag}_${batch_id}_gpu${gpu}.hpp" > /dev/null 2>&1
    cp "/tmp/${RUN_ID}_${shape_tag}_${batch_id}_gpu${gpu}.hpp" "$worker_repo/benchmarks/gemm/benchmarks_sycl.hpp"
    python3 "$worker_repo/tools/gen_main.py" "$manifest_path" "$worker_repo/benchmarks/gemm/main.cpp"

    rm -f "$worker_build/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o" \
          "$worker_build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
    touch "$worker_build/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts"
    touch "$worker_build/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.make"

    local build_log="$LOG_DIR/${shape_tag}_${batch_id}_gpu${gpu}.make.log"
    local build_ok=1
    if ! make -C "$worker_build" cutlass_benchmarks_gemm_sycl -j"$BUILD_JOBS" > "$build_log" 2>&1; then
      build_ok=0
    fi

    local bin="$worker_build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
    if [ "$build_ok" -ne 1 ] || [ ! -x "$bin" ]; then
      cleanup_worker_descendants "$worker_pid" "$shape_tag" "$gpu" "build_fail_${batch_id}"
      echo "$batch_id" >> "$fail_file"
      if grep -q "error:" "$build_log" 2>/dev/null; then
        log "shape=$shape_tag gpu=$gpu [$batch_id] COMPILE FAIL"
      else
        log "shape=$shape_tag gpu=$gpu [$batch_id] LINK FAIL"
      fi
      continue
    fi

    local result_csv="$shape_dir/${batch_id}_gpu${gpu}.csv"
    echo "kernel,tflops,avg_runtime_ms,total_runtime_ms,measure_iters,warmup_iters,latency_source,status,gpu,m,n,k" > "$result_csv"
    mapfile -t kernels < "$manifest_path"
    for kernel in "${kernels[@]}"; do
      [ -n "$kernel" ] || continue

      set +e
      out=$(timeout "$TIMEOUT" "$bin" --kernel="$kernel" --m="$shape_m" --n="$shape_n" --k="$shape_k" 2>&1)
      rc=$?
      set -e

      tf=$(echo "$out" | grep -oP 'median_tflops=\K[0-9.]+' || echo "0")
      avg_runtime_ms=$(echo "$out" | grep -oP 'avg_runtime_ms=\K[0-9.]+' || echo "")
      total_runtime_ms=$(echo "$out" | grep -oP 'total_runtime_ms=\K[0-9.]+' || echo "")
      measure_iters=$(echo "$out" | grep -oP 'measure_iters=\K[0-9]+' || echo "")
      warmup_iters=$(echo "$out" | grep -oP 'warmup_iters=\K[0-9]+' || echo "")
      if echo "$out" | grep -q 'RESULT kernel='; then
        status="OK"
      elif [ "$rc" -eq 124 ]; then
        status="TIMEOUT"
      elif echo "$out" | grep -q 'NOT_FOUND'; then
        status="NOT_FOUND"
      else
        status=$(echo "$out" | grep -oP 'STATUS=\K[A-Z]+' | head -1)
        [ -n "$status" ] || status="FAIL"
      fi
      cleanup_worker_descendants "$worker_pid" "$shape_tag" "$gpu" "kernel_${batch_id}"

      latency_source=""
      if [ -n "$total_runtime_ms" ] || [ -n "$avg_runtime_ms" ]; then
        latency_source="reported"
      fi

      echo "$kernel,$tf,$avg_runtime_ms,$total_runtime_ms,$measure_iters,$warmup_iters,$latency_source,$status,$gpu,$shape_m,$shape_n,$shape_k" >> "$result_csv"
    done

    cleanup_worker_descendants "$worker_pid" "$shape_tag" "$gpu" "post_${batch_id}"
    log "shape=$shape_tag gpu=$gpu $batch_id done"
  done

  cleanup_worker_descendants "$worker_pid" "$shape_tag" "$gpu" "worker_exit"
  cp "$orig_hpp" "$worker_repo/benchmarks/gemm/benchmarks_sycl.hpp" 2>/dev/null || true
  cp "$orig_main" "$worker_repo/benchmarks/gemm/main.cpp" 2>/dev/null || true
  rm -f "$worker_repo/benchmarks/gemm/benchmarks_sycl.hpp.cache"
  rm -f "$orig_hpp" "$orig_main"
}

run_shapes() {
  local total_batches total_kernels shape_index shape_tag count gpu
  total_batches=$(python3 -c "import json; print(json.load(open('$RUN_DIR/manifest.json'))['batch_count'])")
  total_kernels=$(python3 -c "import json; print(json.load(open('$RUN_DIR/manifest.json'))['total_kernels'])")
  log "Starting shape-serial search: $total_kernels kernels, $total_batches batches, GPUs $GPU_IDS_CSV"

  for shape_index in "${!SHAPE_TAGS_ARR[@]}"; do
    shape_tag="${SHAPE_TAGS_ARR[$shape_index]}"
    ACTIVE_SHAPE_TAG="$shape_tag"
    mkdir -p "$RESULTS_DIR/$shape_tag"
    echo "$shape_tag" > "$STATUS_DIR/current_shape"
    echo "running" > "$STATUS_DIR/${shape_tag}.status"
    rm -f "$STATUS_DIR/${shape_tag}.done" "$STATUS_DIR/${shape_tag}.failed"
    log "=== Starting shape $shape_tag on GPUs $GPU_IDS_CSV ==="

    pids=()
    for gpu in "${GPU_IDS_ARR[@]}"; do
      run_shape_worker \
        "$gpu" \
        "$shape_tag" \
        "${SHAPE_M_ARR[$shape_index]}" \
        "${SHAPE_N_ARR[$shape_index]}" \
        "${SHAPE_K_ARR[$shape_index]}" \
        > "$LOG_DIR/worker_${shape_tag}_gpu${gpu}.log" 2>&1 &
      pids+=("$!")
      printf '%s\n' "$!" >> "$STATUS_DIR/worker_pids.txt"
    done

    local pid
    local wait_failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        wait_failed=1
      fi
    done
    if [ "$wait_failed" -ne 0 ]; then
      mark_active_shape_failed
      fail "shape $shape_tag failed"
    fi

    local failed_batches=0
    local fail_path
    local fail_count
    for fail_path in "$RUN_DIR"/failed_"${shape_tag}"_gpu*.txt; do
      [ -f "$fail_path" ] || continue
      fail_count=$(awk 'NF {count++} END {print count + 0}' "$fail_path")
      failed_batches=$((failed_batches + fail_count))
    done

    count=$(find "$RESULTS_DIR/$shape_tag" -maxdepth 1 -name '*.csv' | wc -l)
    if [ "$failed_batches" -ne 0 ]; then
      mark_active_shape_failed
      fail "shape $shape_tag has $failed_batches failed batches"
    fi
    if [ "$count" -eq 0 ]; then
      mark_active_shape_failed
      fail "shape $shape_tag produced no CSV files"
    fi
    if [ "$count" -ne "$total_batches" ]; then
      mark_active_shape_failed
      fail "shape $shape_tag produced $count CSV files, expected $total_batches"
    fi
    echo "completed" > "$STATUS_DIR/${shape_tag}.status"
    touch "$STATUS_DIR/${shape_tag}.done"
    printf '%s\n' "$shape_tag" >> "$STATUS_DIR/completed_shapes.txt"
    log "=== Finished shape $shape_tag, csv_count=$count ==="
    ACTIVE_SHAPE_TAG=""
  done

  rm -f "$STATUS_DIR/current_shape"
  log "Done. Results: $RESULTS_DIR"
}

main() {
  parse_gpu_ids
  parse_shapes
  setup_env
  lock_gpu_frequency
  kill_existing_runs
  sync_repo
  validate_template_build
  prepare_run_dir
  write_metadata
  generate_manifests
  prepare_workers
  run_shapes
}

main "$@"
