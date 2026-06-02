#!/bin/bash
# =============================================================================
# Remote Full Retest — after all compile-fail fixes applied
# 
# Strategy: 
#   Phase 0: Local validation — gen_mini_hpp all 886 batches, fast syntax check
#   Phase 1: Smoke test — 20 batches, expect 0 COMPILE FAIL
#   Phase 2: Full 886-batch run — if Phase 1 passes
#
# Usage on remote:
#   cd /root/cutlass_profile_device7_b70_2500mhz
#   bash sycl-tla/tools/remote_full_retest.sh phase0   # validate locally
#   bash sycl-tla/tools/remote_full_retest.sh phase1   # 20-batch smoke
#   bash sycl-tla/tools/remote_full_retest.sh phase2   # full 886-batch
# =============================================================================

set -eo pipefail

PHASE="${1:-phase0}"
REPO_ROOT="/root/cutlass_profile_device7_b70_2500mhz"
S="$REPO_ROOT/sycl-tla"
WS="$REPO_ROOT/screen_ws_v3"          # v3 workspace — fresh results
RESULTS_DIR="$WS/results"
BDIR="/root/cutlass_profile_device7_b70_2500mhz/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001"
GB_LIB="$BDIR/_deps/googlebenchmark-build/src/libbenchmark.a"
CUTLASS_LIB="$BDIR/tools/library/libcutlass.a"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- Env setup ----
setup_env() {
    source /opt/intel/oneapi/compiler/2025.3/env/vars.sh 2>/dev/null || true
    export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
    export IGC_VectorAliasBBThreshold=10000
    export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$gov" 2>/dev/null || true
    done
}

# ---- Phase 0: Validate all gen_mini outputs syntactically (no compilation) ----
phase0_validate() {
    log "=== Phase 0: gen_mini_hpp validation on all batches ==="
    setup_env
    cd "$S"

    # Update code to latest
    git fetch origin main
    git reset --hard origin/main
    log "Code updated to: $(git log --oneline -1)"

    # Regenerate manifest
    log "Regenerating manifest..."
    mkdir -p "$WS/builds"
    python3 -c "
import sys, json
sys.path.insert(0, '$S/test/benchmarks')
sys.path.insert(0, '$S/python')
from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints
cons = default_constraints()
cat = generated_layered_bmg_kernel_catalog(constraints=cons)
df = 'bf16'
all_k = sorted(set(k['kernel_name'] for k in cat['kernels'] if k.get('dtype_family') == df))
all_k = [k for k in all_k if not k.startswith('03_bmg') and 'streamk_example' not in k]
batch_size = 2
batches = [all_k[i:i+batch_size] for i in range(0, len(all_k), batch_size)]
tot = len(batches)
manifest = {'total': len(all_k), 'batch_size': batch_size, 'batches': []}
for i, batch in enumerate(batches):
    bid = f'batch_{i:04d}'
    mani_f = '$WS/builds/' + bid + '.txt'
    with open(mani_f, 'w') as f:
        for k in batch: f.write(k + '\n')
    manifest['batches'].append({'id': bid, 'count': len(batch), 'gpu': i % 4, 'manifest': mani_f})
with open('$WS/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Generated {tot} batches ({len(all_k)} kernels)')
"

    TOTAL=$(python3 -c "import json; print(len(json.load(open('$WS/manifest.json'))['batches']))")

    # Validate all batches: run gen_mini_hpp and verify no preamble leftovers
    log "Validating gen_mini_hpp on $TOTAL batches (expect ~3 min)..."
    python3 -c "
import sys, os, subprocess, shutil, re, time, json
sys.path.insert(0, '$S/test/benchmarks')
sys.path.insert(0, '$S/python')

# Cache original benchmarks_sycl.hpp
SRC = '$S/benchmarks/gemm/benchmarks_sycl.hpp'
CACHE = '$S/benchmarks/gemm/benchmarks_sycl.hpp.cache'
ORIG = '/tmp/orig_benchmarks_sycl.hpp'
shutil.copy2(SRC, ORIG)

with open('$WS/manifest.json') as f:
    manifest = json.load(f)
batches = manifest['batches']

def classify(name):
    import re
    if 'GemmExhaustive' in name: return 'ge'
    m = re.match(r'^(\w+)_Gemm_\d', name)
    if m: return 'gs'
    if 'StreamK' in name or 'DataParallel' in name or 'SplitK' in name: return 'sk'
    return 'hw'

fails = 0
t0 = time.time()
for i, b in enumerate(batches):
    bid = b['id']
    bf = b['manifest']
    shutil.copy2(ORIG, SRC)
    if os.path.exists(CACHE): os.unlink(CACHE)
    
    result = subprocess.run([
        sys.executable, '$S/tools/gen_mini_hpp.py',
        '--manifest', bf, '--output', f'/tmp/{bid}_check.hpp'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f'FAIL {bid}: exit={result.returncode} err={result.stderr[:100]}')
        fails += 1
        continue
    
    with open(f'/tmp/{bid}_check.hpp') as f: content = f.read()
    pos = content.find('static void register_gemm_benchmarks()')
    preamble = content[:pos] if pos > 0 else ''
    declares = len([l for l in preamble.split(chr(10))
                    if l.strip().startswith('BMG_DECLARE_') or l.strip().startswith('BMG_REGISTER_')])
    
    expected = sum(1 for k in b['kernels'] if classify(k) in ('ge', 'gs'))
    if pos < 0 or declares != expected:
        print(f'FAIL {bid}: {declares} declares expected {expected}')
        fails += 1
    
    os.unlink(f'/tmp/{bid}_check.hpp')
    if (i+1) % 100 == 0:
        print(f'  [{i+1}/{len(batches)}] {fails} fails, {time.time()-t0:.0f}s', flush=True)

shutil.copy2(ORIG, SRC)
print(f'DONE: {fails}/{len(batches)} failures')
exit(fails)
"
    STATUS=$?
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ "$STATUS" -eq 0 ]; then
        log "✅ ALL $TOTAL BATCHES gen_mini VALID — 0 leftover declarations"
    else
        log "❌ $STATUS gen_mini failures — abort!"
        exit 1
    fi
}

# ---- Phase 1: Smoke test (20 batches) ----
phase1_smoke() {
    log "=== Phase 1: Smoke test — 20 batches ==="
    setup_env
    mkdir -p "$RESULTS_DIR"
    cd "$S"

    # Verify code is latest
    git fetch origin main
    git checkout -- benchmarks/gemm/benchmarks_sycl.hpp benchmarks/gemm/main.cpp 2>/dev/null || true

    TOTAL=$(python3 -c "import json; print(len(json.load(open('$WS/manifest.json'))['batches']))")
    log "Testing first 20 of $TOTAL batches"

    COMPILE_FAIL=0
    LINK_FAIL=0
    PASS=0
    START_TIME=$(date +%s)

    for i in $(seq 0 19); do
        bid=$(printf "batch_%04d" $i)
        bf="$WS/builds/${bid}.txt"
        [ ! -f "$bf" ] && continue
        gpu=$(( i % 4 ))

        cp "$S/benchmarks/gemm/benchmarks_sycl.hpp" /tmp/bak_hpp
        cp "$S/benchmarks/gemm/main.cpp" /tmp/bak_main
        rm -f "$S/benchmarks/gemm/benchmarks_sycl.hpp.cache"

        python3 "$S/tools/gen_mini_hpp.py" --manifest "$bf" --output "/tmp/${bid}.hpp"
        cp "/tmp/${bid}.hpp" "$S/benchmarks/gemm/benchmarks_sycl.hpp"
        python3 "$S/tools/gen_main.py" "$bf" "$S/benchmarks/gemm/main.cpp"

        rm -f "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o" \
              "$BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
        touch "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts" \
              "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.make"
        sed -i "/^\.DELETE_ON_ERROR/d" "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/build.make" 2>/dev/null || true

        make -C "$BDIR" cutlass_benchmarks_gemm_sycl -j128 > "/tmp/mk_${bid}.log" 2>&1
        OBJ="$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o"

        if [ ! -s "$OBJ" ]; then
            log "[$bid] ❌ COMPILE FAIL"
            cp /tmp/bak_hpp "$S/benchmarks/gemm/benchmarks_sycl.hpp"
            cp /tmp/bak_main "$S/benchmarks/gemm/main.cpp"
            COMPILE_FAIL=$((COMPILE_FAIL+1))
            continue
        fi

        BIN="$BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
        icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
            -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
            -O3 "$OBJ" -o "$BIN" "$GB_LIB" -L/lib64/stubs -Wl,-rpath,/lib64/stubs: "$CUTLASS_LIB" \
            -Wl,-rpath=/opt/intel/oneapi/mkl/2025.3/lib \
            /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_ilp64.so \
            /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_thread.so \
            /opt/intel/oneapi/mkl/2025.3/lib/libmkl_core.so \
            /opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so \
            -lm -ldl -lpthread /opt/intel/oneapi/compiler/2025.3/lib/libsycl.so \
            > "/tmp/lnk_${bid}.log" 2>&1

        if [ ! -x "$BIN" ]; then
            log "[$bid] ❌ LINK FAIL"
            cp /tmp/bak_hpp "$S/benchmarks/gemm/benchmarks_sycl.hpp"
            cp /tmp/bak_main "$S/benchmarks/gemm/main.cpp"
            LINK_FAIL=$((LINK_FAIL+1))
            continue
        fi

        # Quick screen (just verify kernel runs)
        export ZE_AFFINITY_MASK=$gpu
        R="$RESULTS_DIR/${bid}_gpu${gpu}.csv"
        echo "kernel,tflops,status,gpu" > "$R"
        while read -r kernel; do
            [ -z "$kernel" ] && continue
            out=$(timeout 120 "$BIN" --kernel="$kernel" --m=8192 --n=4096 --k=1536 2>&1) || true
            tf=$(echo "$out" | grep -oP "median_tflops=\K[0-9.]+" || echo "0")
            st=$(echo "$out" | grep -oP "STATUS=\K[A-Z]+" || echo "TIMEOUT")
            echo "$kernel,$tf,$st,$gpu" >> "$R"
        done < "$bf"
        PASS=$((PASS+1))
        log "[$((i+1))/20] GPU$gpu $bid ✅ OK"

        cp /tmp/bak_hpp "$S/benchmarks/gemm/benchmarks_sycl.hpp"
        cp /tmp/bak_main "$S/benchmarks/gemm/main.cpp"
        rm -f "$S/benchmarks/gemm/benchmarks_sycl.hpp.cache"
    done

    ELAPSED=$(($(date +%s) - START_TIME))
    log "Phase 1 DONE: $PASS pass, $COMPILE_FAIL compile fail, $LINK_FAIL link fail in ${ELAPSED}s"
    if [ "$COMPILE_FAIL" -eq 0 ] && [ "$LINK_FAIL" -eq 0 ]; then
        log "✅ SMOKE TEST PASSED — 0 failures! Ready for Phase 2"
    else
        log "❌ SMOKE TEST FAILED — abort, debug before Phase 2"
        exit 1
    fi
}

# ---- Phase 2: Full 886-batch run ----
phase2_full() {
    log "=== Phase 2: Full 886-batch screening ==="
    setup_env
    mkdir -p "$RESULTS_DIR"
    cd "$S"

    git checkout -- benchmarks/gemm/benchmarks_sycl.hpp benchmarks/gemm/main.cpp 2>/dev/null || true
    rm -f benchmarks/gemm/benchmarks_sycl.hpp.cache

    TOTAL=$(python3 -c "import json; print(len(json.load(open('$WS/manifest.json'))['batches']))")
    log "Processing $TOTAL batches on GPUs 0,1,2,3 (round-robin)"

    COMPILE_FAIL=0
    LINK_FAIL=0
    SCREENED=0
    START_TIME=$(date +%s)

    for i in $(seq 0 $((TOTAL-1))); do
        bid=$(printf "batch_%04d" $i)
        bf="$WS/builds/${bid}.txt"
        [ ! -f "$bf" ] && continue
        gpu=$(( i % 4 ))

        cp "$S/benchmarks/gemm/benchmarks_sycl.hpp" /tmp/bak_hpp
        cp "$S/benchmarks/gemm/main.cpp" /tmp/bak_main
        rm -f "$S/benchmarks/gemm/benchmarks_sycl.hpp.cache"

        python3 "$S/tools/gen_mini_hpp.py" --manifest "$bf" --output "/tmp/${bid}.hpp" > /dev/null 2>&1
        cp "/tmp/${bid}.hpp" "$S/benchmarks/gemm/benchmarks_sycl.hpp"
        python3 "$S/tools/gen_main.py" "$bf" "$S/benchmarks/gemm/main.cpp"

        rm -f "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o" \
              "$BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
        touch "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts" \
              "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.make"
        sed -i "/^\.DELETE_ON_ERROR/d" "$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/build.make" 2>/dev/null || true

        make -C "$BDIR" cutlass_benchmarks_gemm_sycl -j128 > "/tmp/mk_${bid}.log" 2>&1
        OBJ="$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o"

        if [ ! -s "$OBJ" ]; then
            log "[$bid] ❌ COMPILE FAIL"
            cp /tmp/bak_hpp "$S/benchmarks/gemm/benchmarks_sycl.hpp"
            cp /tmp/bak_main "$S/benchmarks/gemm/main.cpp"
            rm -f "$S/benchmarks/gemm/benchmarks_sycl.hpp.cache"
            COMPILE_FAIL=$((COMPILE_FAIL+1))
            continue
        fi

        BIN="$BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl"
        icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
            -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
            -O3 "$OBJ" -o "$BIN" "$GB_LIB" -L/lib64/stubs -Wl,-rpath,/lib64/stubs: "$CUTLASS_LIB" \
            -Wl,-rpath=/opt/intel/oneapi/mkl/2025.3/lib \
            /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_ilp64.so \
            /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_thread.so \
            /opt/intel/oneapi/mkl/2025.3/lib/libmkl_core.so \
            /opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so \
            -lm -ldl -lpthread /opt/intel/oneapi/compiler/2025.3/lib/libsycl.so \
            > "/tmp/lnk_${bid}.log" 2>&1

        if [ ! -x "$BIN" ]; then
            log "[$bid] ❌ LINK FAIL"
            cp /tmp/bak_hpp "$S/benchmarks/gemm/benchmarks_sycl.hpp"
            cp /tmp/bak_main "$S/benchmarks/gemm/main.cpp"
            rm -f "$S/benchmarks/gemm/benchmarks_sycl.hpp.cache"
            LINK_FAIL=$((LINK_FAIL+1))
            continue
        fi

        export ZE_AFFINITY_MASK=$gpu
        R="$RESULTS_DIR/${bid}_gpu${gpu}.csv"
        echo "kernel,tflops,status,gpu" > "$R"
        while read -r kernel; do
            [ -z "$kernel" ] && continue
            out=$(timeout 120 "$BIN" --kernel="$kernel" --m=8192 --n=4096 --k=1536 2>&1) || true
            tf=$(echo "$out" | grep -oP "median_tflops=\K[0-9.]+" || echo "0")
            st=$(echo "$out" | grep -oP "STATUS=\K[A-Z]+" || echo "TIMEOUT")
            echo "$kernel,$tf,$st,$gpu" >> "$R"
        done < "$bf"

        SCREENED=$((SCREENED+1))
        ELAPSED=$(($(date +%s) - START_TIME))
        ETA=$(echo "scale=0; $ELAPSED / max($SCREENED, 1) * ($TOTAL - $SCREENED) / 60" | bc 2>/dev/null || echo "?")
        log "[$SCREENED/$TOTAL] GPU$gpu $bid ✅ (compile_fail=$COMPILE_FAIL link_fail=$LINK_FAIL ETA=${ETA}m)"

        cp /tmp/bak_hpp "$S/benchmarks/gemm/benchmarks_sycl.hpp"
        cp /tmp/bak_main "$S/benchmarks/gemm/main.cpp"
        rm -f "$S/benchmarks/gemm/benchmarks_sycl.hpp.cache"
    done

    ELAPSED=$(($(date +%s) - START_TIME))
    log "=== Phase 2 DONE ==="
    log "Screened: $SCREENED, Compile fail: $COMPILE_FAIL, Link fail: $LINK_FAIL"
    log "Results: $RESULTS_DIR"
    log "Total time: $((ELAPSED/60))m ${ELAPSED}s"

    # Quick stats
    python3 -c "
import os, glob
csvs = glob.glob('$RESULTS_DIR/*.csv')
print(f'CSV files: {len(csvs)}')
tfs = []
for f in csvs:
    with open(f) as fh:
        for line in fh:
            if 'tflops' in line: continue
            p = line.strip().split(',')
            if len(p) >= 2:
                try: tfs.append(float(p[1]))
                except: pass
if tfs:
    print(f'Max TFLOPS: {max(tfs):.1f}')
    print(f'Mean TFLOPS: {sum(tfs)/len(tfs):.1f}'))
"
}

# ---- Main ----
case "$PHASE" in
    phase0) phase0_validate ;;
    phase1) phase1_smoke ;;
    phase2) phase2_full ;;
    all)
        phase0_validate
        phase1_smoke
        phase2_full
        ;;
    *)
        echo "Usage: $0 {phase0|phase1|phase2|all}"
        echo "  phase0 — validate gen_mini_hpp on all 886 batches (fast, no compile)"
        echo "  phase1 — smoke test first 20 batches (compile + screen)"
        echo "  phase2 — full 886-batch run (after phase0+phase1 pass)"
        echo "  all    — run phases 0, 1, 2 sequentially"
        exit 1
        ;;
esac
