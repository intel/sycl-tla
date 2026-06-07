#!/bin/bash
# Remote scheduler sample validation: exercise the widened bruteforce_scheduler
# path with a small hand-picked kernel filter on a B70 node.

set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/root/cutlass_profile_device7_b70_2500mhz}"
REPO_ROOT="${REPO_ROOT:-$ROOT_DIR/sycl-tla}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/screen_runs}"
SAMPLE_RUN_ID="${SAMPLE_RUN_ID:-scheduler_sample_validation_$(date +%Y%m%d_%H%M%S)}"
WORKSPACE="${WORKSPACE:-$RUNS_DIR/$SAMPLE_RUN_ID}"
BUILD_DIR="${BUILD_DIR:-$WORKSPACE/build/candidate_benchmarks}"
GPU_ID="${GPU_ID:-0}"
RUN_SYNC="${RUN_SYNC:-1}"
GIT_REF="${GIT_REF:-origin/main}"
ONEAPI_DIR="${ONEAPI_DIR:-/opt/intel/oneapi/compiler/2025.3}"
GOOGLEBENCHMARK_DIR="${GOOGLEBENCHMARK_DIR:-$REPO_ROOT/_deps/googlebenchmark-src}"
GOOGLEBENCHMARK_BUILD_DIR="${GOOGLEBENCHMARK_BUILD_DIR:-}"
CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-icpx}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
BUILD_TIMEOUT_SECONDS="${BUILD_TIMEOUT_SECONDS:-1800}"
SKIP_RUN="${SKIP_RUN:-0}"
M_DIM="${M_DIM:-8192}"
N_DIM="${N_DIM:-384}"
K_DIM="${K_DIM:-3584}"

SHAPES_JSON="$WORKSPACE/scheduler_sample_shapes.json"
KERNEL_FILTER="$WORKSPACE/compiled_kernels.txt"
SELECTED_KERNELS_JSON="$WORKSPACE/selected_kernels.json"
LOG_FILE="$WORKSPACE/profiler.log"

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

setup_env() {
  # shellcheck source=/dev/null
  source "$ONEAPI_DIR/env/vars.sh" 2>/dev/null || true
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
  export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-$GPU_ID}"
  export SYCL_PROGRAM_COMPILE_OPTIONS="${SYCL_PROGRAM_COMPILE_OPTIONS:--ze-opt-large-register-file -gline-tables-only}"
  export IGC_VectorAliasBBThreshold="${IGC_VectorAliasBBThreshold:-10000}"
  export IGC_ExtraOCLOptions="${IGC_ExtraOCLOptions:--cl-intel-256-GRF-per-thread}"
}

sync_repo() {
  cd "$REPO_ROOT"
  git fetch origin
  git reset --hard "$GIT_REF"
  git clean -fd -e _deps
  git checkout -- benchmarks/gemm/benchmarks_sycl.hpp benchmarks/gemm/main.cpp 2>/dev/null || true
  rm -f benchmarks/gemm/benchmarks_sycl.hpp.cache
}

prepare_workspace() {
  rm -rf "$WORKSPACE"
  mkdir -p "$WORKSPACE"
}

prepare_googlebenchmark_link() {
  if [ -d "$GOOGLEBENCHMARK_DIR" ] && [ ! -e "$REPO_ROOT/_deps/googlebenchmark-src" ]; then
    mkdir -p "$REPO_ROOT/_deps"
    ln -s "$GOOGLEBENCHMARK_DIR" "$REPO_ROOT/_deps/googlebenchmark-src"
    log "linked _deps/googlebenchmark-src -> $GOOGLEBENCHMARK_DIR"
  fi

  if [ -z "$GOOGLEBENCHMARK_BUILD_DIR" ]; then
    local candidate
    for candidate in \
      "$ROOT_DIR"/build/_deps/googlebenchmark-build \
      /root/jyli/docker_root/sycl-tla/build/_deps/googlebenchmark-build \
      /root/jyli/docker_root/libraries.ai.cutlass.internal/build/_deps/googlebenchmark-build \
      "$ROOT_DIR"/ali_one_8192_4096_1536_expanded_bmg_batched8_20260519_2238_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001/_deps/googlebenchmark-build
    do
      if [ -f "$candidate/src/libbenchmark.a" ]; then
        GOOGLEBENCHMARK_BUILD_DIR="$candidate"
        export GOOGLEBENCHMARK_BUILD_DIR
        log "using GOOGLEBENCHMARK_BUILD_DIR=$GOOGLEBENCHMARK_BUILD_DIR"
        break
      fi
    done
  fi
}

write_shapes_json() {
  python3 - "$SHAPES_JSON" "$M_DIM" "$N_DIM" "$K_DIM" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
m_dim, n_dim, k_dim = map(int, sys.argv[2:5])
payload = {
    "schema_version": "1.0",
    "generated_at": "",
    "shape_set_id": f"scheduler_sample_{m_dim}_{n_dim}_{k_dim}",
    "source": "remote_scheduler_sample_validation",
    "shapes": [
        {
            "shape_id": f"rcr_bf16_{m_dim}_{n_dim}_{k_dim}",
            "layout": "rcr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_d": "f32",
            "dtype_acc": "f32",
            "m": m_dim,
            "n": n_dim,
            "k": k_dim,
            "batch_count": 1,
            "runtime_defaults": {},
        },
        {
            "shape_id": f"rrr_bf16_{m_dim}_{n_dim}_{k_dim}",
            "layout": "rrr",
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_c": "f32",
            "dtype_d": "f32",
            "dtype_acc": "f32",
            "m": m_dim,
            "n": n_dim,
            "k": k_dim,
            "batch_count": 1,
            "runtime_defaults": {},
        },
    ],
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_sample_kernel_filter() {
  python3 - "$REPO_ROOT" "$KERNEL_FILTER" "$SELECTED_KERNELS_JSON" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
filter_path = Path(sys.argv[2])
selected_json = Path(sys.argv[3])
sys.path.insert(0, str(repo_root / "test/benchmarks"))
sys.path.insert(0, str(repo_root / "python"))

from intel_gemm_profiler.catalog import generated_layered_bmg_scheduler_expanded_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints

catalog = generated_layered_bmg_scheduler_expanded_kernel_catalog(
    constraints=default_constraints()
)

scheduler_kernels = [
    kernel
    for kernel in catalog["kernels"]
    if kernel.get("dtype_family") == "bf16" and kernel.get("decomposition_mode") in {"StreamK", "DataParallel", "SplitK"}
]
scheduler_kernels.sort(
    key=lambda kernel: (
        kernel["layout"],
        kernel["decomposition_mode"],
        kernel["sg_m"],
        kernel["sg_n"],
        kernel["stages"],
        kernel["tile_m"],
        kernel["tile_n"],
        kernel["tile_k"],
        kernel["kernel_name"],
    )
)

preferred_tiles = [
    (128, 128, 32),
    (128, 256, 32),
    (256, 128, 32),
    (256, 256, 32),
    (64, 128, 32),
    (64, 256, 32),
    (512, 128, 32),
    (512, 256, 32),
]
preferred_tile_rank = {tile: index for index, tile in enumerate(preferred_tiles)}

requests = [
    ("rcr", "StreamK", 4, 4, 1),
    ("rcr", "DataParallel", 8, 2, 3),
    ("rcr", "SplitK", 4, 8, 2),
    ("rrr", "StreamK", 2, 8, 1),
    ("rrr", "DataParallel", 4, 4, 3),
    ("rrr", "SplitK", 8, 2, 1),
    ("rcr", "StreamK", 8, 4, 2),
]

selected = []
used_kernel_names = set()
for layout, decomposition_mode, sg_m, sg_n, stages in requests:
    matches = [
        kernel
        for kernel in scheduler_kernels
        if kernel["layout"] == layout
        and kernel["decomposition_mode"] == decomposition_mode
        and kernel["sg_m"] == sg_m
        and kernel["sg_n"] == sg_n
        and kernel["stages"] == stages
        and kernel["kernel_name"] not in used_kernel_names
    ]
    if not matches:
        raise SystemExit(
            f"missing sample kernel for layout={layout} mode={decomposition_mode} sg={sg_m}x{sg_n} st={stages}"
        )
    preferred_matches = [
        kernel
        for kernel in matches
        if (kernel["tile_m"], kernel["tile_n"], kernel["tile_k"]) in preferred_tile_rank
    ]
    if preferred_matches:
        preferred_matches.sort(
            key=lambda kernel: (
                preferred_tile_rank[(kernel["tile_m"], kernel["tile_n"], kernel["tile_k"])],
                kernel["kernel_name"],
            )
        )
        kernel = preferred_matches[0]
        selection_reason = "preferred_validated_tile"
    else:
        matches.sort(
            key=lambda kernel: (
                -(kernel["tile_m"] * kernel["tile_n"]),
                -kernel["tile_k"],
                -kernel["tile_m"],
                -kernel["tile_n"],
                kernel["kernel_name"],
            )
        )
        kernel = matches[0]
        selection_reason = "largest_available_tile"
    used_kernel_names.add(kernel["kernel_name"])
    selected.append(
        {
            "kernel_name": kernel["kernel_name"],
            "layout": kernel["layout"],
            "scheduler_family": kernel["scheduler_family"],
            "decomposition_mode": kernel["decomposition_mode"],
            "tile_m": kernel["tile_m"],
            "tile_n": kernel["tile_n"],
            "tile_k": kernel["tile_k"],
            "sg_m": kernel["sg_m"],
            "sg_n": kernel["sg_n"],
            "stages": kernel["stages"],
            "selection_reason": selection_reason,
        }
    )

filter_path.write_text(
    "".join(f"^{item['kernel_name']}$\n" for item in selected),
    encoding="utf-8",
)
selected_json.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
print(json.dumps(selected, indent=2))
PY
}

run_profiler() {
  local -a cmd=(
    python3 "$REPO_ROOT/test/benchmarks/intel_gemm_profiler.py"
    --workspace "$WORKSPACE"
    --cwd "$REPO_ROOT"
    --shell-init "source $ONEAPI_DIR/env/vars.sh 2>/dev/null || true"
    --dtype bf16
    --probe-mode off
    --shapes-json "$SHAPES_JSON"
    --search-strategy bruteforce_scheduler
    --compiled-kernel-list "$KERNEL_FILTER"
    --build-candidate-benchmark
    --benchmark-build-dir "$BUILD_DIR"
    --cmake-source-dir "$REPO_ROOT"
    --cmake-cxx-compiler "$CMAKE_CXX_COMPILER"
    --candidate-build-batch-size 1
    --run-candidate-build-preflight
    --use-candidate-build-preflight-benchmarks
    --candidate-build-parallelism 1
    --benchmark-entry-chunk-size 1
    --top-k 1
    --confirm-runs 1
    --build-timeout "$BUILD_TIMEOUT_SECONDS"
    --timeout "$TIMEOUT_SECONDS"
  )

  if [ -d "$GOOGLEBENCHMARK_DIR" ]; then
    cmd+=(--googlebenchmark-dir "$GOOGLEBENCHMARK_DIR")
  fi
  if [ -d "$GOOGLEBENCHMARK_BUILD_DIR" ]; then
    cmd+=(--googlebenchmark-build-dir "$GOOGLEBENCHMARK_BUILD_DIR")
  fi

  log "running profiler sample validation"
  "${cmd[@]}" | tee "$LOG_FILE"
}

print_summary() {
  python3 - "$WORKSPACE" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

workspace = Path(sys.argv[1])
reports = workspace / "reports"
phase_b = reports / "phase_b_summary.json"
results_csv = reports / "gemm_profile_results.csv"

if phase_b.exists():
    payload = json.loads(phase_b.read_text(encoding="utf-8"))
    print("phase_b_done:", payload.get("done"))
    print("phase_b_selected_shapes:", payload.get("selected_shape_count"))
    print("phase_b_total_rows:", payload.get("phase_b_result_count"))

if results_csv.exists():
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    print("csv_rows:", len(rows))
    print("status_counts:", dict(Counter(row.get("status", "") for row in rows)))
    print("kernel_status_counts:", dict(Counter(row.get("kernel_id", "") for row in rows)))

print("workspace:", workspace)
print("selected_kernels_json:", workspace / "selected_kernels.json")
print("kernel_filter:", workspace / "compiled_kernels.txt")
print("log_file:", workspace / "profiler.log")
PY
}

main() {
  log "scheduler sample run id: $SAMPLE_RUN_ID"
  setup_env
  if [ "$RUN_SYNC" = "1" ]; then
    log "syncing repo to $GIT_REF"
    sync_repo
  fi
  prepare_workspace
  prepare_googlebenchmark_link
  write_shapes_json
  log "selected expanded scheduler kernels:"
  write_sample_kernel_filter
  if [ "$SKIP_RUN" = "1" ]; then
    log "SKIP_RUN=1; generated sample artifacts only"
    print_summary
    exit 0
  fi
  run_profiler
  print_summary
}

main "$@"
