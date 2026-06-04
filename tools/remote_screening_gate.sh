#!/bin/bash
# Remote screening gate: cover known failure classes before launching a full run.
# Intended to run directly on the B70 node.

set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/root/cutlass_profile_device7_b70_2500mhz}"
REPO_ROOT="${REPO_ROOT:-$ROOT_DIR/sycl-tla}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/screen_runs}"
GATE_RUN_ID="${GATE_RUN_ID:-screening_gate_$(date +%Y%m%d_%H%M%S)}"
GATE_WS="${GATE_WS:-$RUNS_DIR/$GATE_RUN_ID}"
GATE_RESULTS_DIR="${GATE_RESULTS_DIR:-$GATE_WS/results}"
GPU_ID="${GPU_ID:-0}"
RUN_SYNC="${RUN_SYNC:-1}"
GIT_REF="${GIT_REF:-origin/main}"

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

setup_env() {
  source /opt/intel/oneapi/compiler/2025.3/env/vars.sh 2>/dev/null || true
  export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
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
  rm -rf "$GATE_WS"
  mkdir -p "$GATE_RESULTS_DIR" "$GATE_WS/manifests"
}

reconfigure_build() {
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" > "/tmp/${GATE_RUN_ID}_cmake.log" 2>&1
}

ORIG_HPP=""
ORIG_MAIN=""

cleanup() {
  if [ -n "$ORIG_HPP" ] && [ -f "$ORIG_HPP" ]; then
    cp "$ORIG_HPP" "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp" 2>/dev/null || true
  fi
  if [ -n "$ORIG_MAIN" ] && [ -f "$ORIG_MAIN" ]; then
    cp "$ORIG_MAIN" "$REPO_ROOT/benchmarks/gemm/main.cpp" 2>/dev/null || true
  fi
  rm -f "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp.cache"
  [ -n "$ORIG_HPP" ] && rm -f "$ORIG_HPP"
  [ -n "$ORIG_MAIN" ] && rm -f "$ORIG_MAIN"
}

create_backups() {
  ORIG_HPP=$(mktemp /tmp/screen_gate_hpp.XXXXXX)
  ORIG_MAIN=$(mktemp /tmp/screen_gate_main.XXXXXX)
  cp "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp" "$ORIG_HPP"
  cp "$REPO_ROOT/benchmarks/gemm/main.cpp" "$ORIG_MAIN"
}

write_case_manifest() {
  local case_name=$1
  local manifest="$GATE_WS/manifests/${case_name}.txt"
  python3 - "$case_name" "$manifest" "$REPO_ROOT" <<'PY'
import re
import sys
from pathlib import Path

case_name, manifest, repo_root = sys.argv[1:4]
repo = Path(repo_root)
sys.path.insert(0, str(repo / "test/benchmarks"))
sys.path.insert(0, str(repo / "python"))

from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints

cat = generated_layered_bmg_kernel_catalog(constraints=default_constraints())
kernels = sorted(set(
    k["kernel_name"]
    for k in cat["kernels"]
    if k.get("dtype_family", "") in ("bf16", "16b")
))
kernels = [k for k in kernels if not k.startswith("03_bmg") and "streamk_example" not in k]

def require(names):
    missing = [name for name in names if name not in kernels]
    if missing:
        raise SystemExit(f"{case_name}: missing kernels: {missing}")
    return names

def take(pattern, count):
    picked = [k for k in kernels if re.match(pattern, k)]
    if len(picked) < count:
        raise SystemExit(f"{case_name}: need {count} matches for {pattern}, got {len(picked)}")
    return picked[:count]

cases = {
    "seed": lambda: require([
        "BmgGemmBF16BF16FP32_RCR_16",
        "BmgGemmBF16BF16FP32_RCR_17",
    ]),
    "direct_rrr": lambda: require([
        "BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32",
    ]),
    "streamk": lambda: require([
        "BmgGemmBF16BF16FP32_RCR_StreamK_128x128x32",
        "BmgGemmBF16BF16FP32_RCR_DataParallel_128x256x32",
        "BmgGemmBF16BF16FP32_RCR_SplitK_256x128x32",
    ]),
    "rcr_exhaustive": lambda: take(r"^BmgGemmBF16BF16FP32_RCR_GemmExhaustive_.*$", 2),
    "rrr_exhaustive": lambda: require([
        "BmgGemmBF16BF16FP32_RRR_GemmExhaustive_512x128x16_SG8x4_ST2",
        "BmgGemmBF16BF16FP32_RRR_GemmExhaustive_512x128x16_SG8x4_ST3",
    ]),
    "mixed_types": lambda: require([
        "BmgGemmBF16BF16FP32_RCR_16",
        "BmgGemmBF16BF16FP32_RCR_StreamK_128x128x32",
        "BmgGemmBF16BF16FP32_RCR_GemmExhaustive_32x128x64_SG4x4_ST1",
        "BmgGemmBF16BF16FP32_RRR_GemmExhaustive_512x128x16_SG8x4_ST2",
    ]),
}

selected = cases[case_name]()
Path(manifest).write_text("\n".join(selected) + "\n")
print("\n".join(selected))
PY
}

run_case() {
  local case_name=$1
  local manifest="$GATE_WS/manifests/${case_name}.txt"
  local hpp="/tmp/${case_name}.hpp"
  local mk_log="/tmp/${case_name}.mk.log"
  local result_csv="$GATE_RESULTS_DIR/${case_name}.csv"
  local bin="$BUILD_DIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"

  log "CASE ${case_name}: generating manifest"
  write_case_manifest "$case_name" > "/tmp/${case_name}.manifest.log"

  cp "$ORIG_HPP" "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp"
  cp "$ORIG_MAIN" "$REPO_ROOT/benchmarks/gemm/main.cpp"
  rm -f "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp.cache"

  python3 "$REPO_ROOT/tools/gen_mini_hpp.py" --manifest "$manifest" --output "$hpp" > /tmp/${case_name}.gen.log
  cp "$hpp" "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp"
  python3 "$REPO_ROOT/tools/gen_main.py" "$manifest" "$REPO_ROOT/benchmarks/gemm/main.cpp"

  case "$case_name" in
    seed)
      grep -q 'CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_16);' "$hpp"
      ;;
    rrr_exhaustive)
      grep -q 'CUTLASS_CREATE_GEMM_BENCHMARK(PREFIX##_GemmExhaustive_##M##x##N##x##K##_SG##SGM##x##SGN##_ST##STAGES);' "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp"
      grep -q 'BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE(BmgGemmBF16BF16FP32_RRR, Gemm_Bench_BF16FP32_RRR, MMAAtom, 512, 128, 16, 8, 4, 2)' "$hpp"
      ;;
  esac

  rm -f \
    "$BUILD_DIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o" \
    "$BUILD_DIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
  touch \
    "$BUILD_DIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts" \
    "$BUILD_DIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.make"

  if ! make -C "$BUILD_DIR" cutlass_benchmarks_gemm_sycl -j128 > "$mk_log" 2>&1; then
    echo "CASE ${case_name}: compile failed" >&2
    tail -80 "$mk_log" >&2 || true
    exit 1
  fi

  if [ ! -x "$bin" ]; then
    echo "CASE ${case_name}: binary missing after build" >&2
    exit 1
  fi

  export ZE_AFFINITY_MASK="$GPU_ID"
  echo "kernel,tflops,status,gpu" > "$result_csv"
  while read -r kernel; do
    [ -z "$kernel" ] && continue
    set +e
    out=$(timeout 120 "$bin" --kernel="$kernel" --m=8192 --n=4096 --k=1536 2>&1)
    rc=$?
    set -e
    tf=$(echo "$out" | grep -oP "median_tflops=\\K[0-9.]+" || echo "0")
    if echo "$out" | grep -q "RESULT kernel="; then
      st="OK"
    elif [ "$rc" -eq 124 ]; then
      st="TIMEOUT"
    elif echo "$out" | grep -q "NOT_FOUND"; then
      st="NOT_FOUND"
    else
      st=$(echo "$out" | grep -oP "STATUS=\\K[A-Z]+" | head -1)
      [ -z "$st" ] && st="FAIL"
    fi
    echo "$kernel,$tf,$st,$GPU_ID" >> "$result_csv"
    if [ "$st" != "OK" ]; then
      echo "CASE ${case_name}: runtime failure for $kernel ($st)" >&2
      echo "$out" >&2
      exit 1
    fi
  done < "$manifest"

  cp "$ORIG_HPP" "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp"
  cp "$ORIG_MAIN" "$REPO_ROOT/benchmarks/gemm/main.cpp"
  rm -f "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp.cache"
  cmp -s "$ORIG_HPP" "$REPO_ROOT/benchmarks/gemm/benchmarks_sycl.hpp"
  cmp -s "$ORIG_MAIN" "$REPO_ROOT/benchmarks/gemm/main.cpp"
  log "CASE ${case_name}: PASS"
}

main() {
  setup_env
  if [ "$RUN_SYNC" = "1" ]; then
    sync_repo
  fi
  prepare_workspace
  reconfigure_build
  create_backups
  trap cleanup EXIT INT TERM

  for case_name in seed direct_rrr streamk rcr_exhaustive rrr_exhaustive mixed_types; do
    run_case "$case_name"
  done

  log "Screening gate passed. Results: $GATE_RESULTS_DIR"
}

main "$@"
