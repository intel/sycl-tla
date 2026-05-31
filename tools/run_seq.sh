#!/bin/bash
S=/root/cutlass_profile_device7_b70_2500mhz/sycl-tla
WS=/root/cutlass_profile_device7_b70_2500mhz/screen_ws
BDIR=/root/cutlass_profile_device7_b70_2500mhz/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001
GB_LIB=$BDIR/_deps/googlebenchmark-build/src/libbenchmark.a
CUTLASS_LIB=$BDIR/tools/library/libcutlass.a
RESULTS_DIR=${RESULTS_DIR:-$WS/results_full_fixed}
BATCHES=${BATCHES:-4}

log() { echo "[$(date +%H:%M:%S)] $*"; }

source /opt/intel/oneapi/compiler/2025.3/env/vars.sh 2>/dev/null || true
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000 IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $gov 2>/dev/null; done

mkdir -p "$RESULTS_DIR"
cd $S
git checkout -- benchmarks/gemm/benchmarks_sycl.hpp benchmarks/gemm/main.cpp 2>/dev/null
rm -f benchmarks/gemm/benchmarks_sycl.hpp.cache

TOTAL=$(python3 -c "import json; print(json.load(open('$WS/manifest.json'))['batches'])")
[ "$BATCHES" = "all" ] && BATCHES=$TOTAL
log "Processing $BATCHES of $TOTAL batches"

for i in $(seq 0 $((BATCHES-1))); do
  bid=$(printf "batch_%04d" $i)
  bf="$WS/builds/${bid}.txt"
  [ ! -f "$bf" ] && continue
  gpu=$(( 5 + (i % 2) ))
  
  cp $S/benchmarks/gemm/benchmarks_sycl.hpp /tmp/bak_hpp
  cp $S/benchmarks/gemm/main.cpp /tmp/bak_main
  rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
  
  python3 $S/tools/gen_mini_hpp.py --manifest "$bf" --output /tmp/${bid}.hpp > /dev/null 2>&1
  cp /tmp/${bid}.hpp $S/benchmarks/gemm/benchmarks_sycl.hpp
  python3 $S/tools/gen_main.py "$bf" "$S/benchmarks/gemm/main.cpp"
  
  rm -f $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o $BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
  touch $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts
  touch $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.make
  
  make -C $BDIR cutlass_benchmarks_gemm_sycl -j128 > /tmp/mk_${bid}.log 2>&1
  OBJ=$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o
  
  if [ ! -s "$OBJ" ]; then
    log "[$bid] COMPILE FAIL"
    cp /tmp/bak_hpp $S/benchmarks/gemm/benchmarks_sycl.hpp; cp /tmp/bak_main $S/benchmarks/gemm/main.cpp
    rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache; continue
  fi
  
  BIN=$BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
  icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    -O3 $OBJ -o $BIN $GB_LIB -L/lib64/stubs -Wl,-rpath,/lib64/stubs: $CUTLASS_LIB \
    -Wl,-rpath=/opt/intel/oneapi/mkl/2025.3/lib \
    /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_ilp64.so /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_thread.so \
    /opt/intel/oneapi/mkl/2025.3/lib/libmkl_core.so /opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so \
    -lm -ldl -lpthread /opt/intel/oneapi/compiler/2025.3/lib/libsycl.so > /tmp/lnk_${bid}.log 2>&1
  
  if [ ! -x "$BIN" ]; then
    log "[$bid] LINK FAIL"
    cp /tmp/bak_hpp $S/benchmarks/gemm/benchmarks_sycl.hpp; cp /tmp/bak_main $S/benchmarks/gemm/main.cpp
    rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache; continue
  fi
  
  export ZE_AFFINITY_MASK=$gpu
  R="$RESULTS_DIR/${bid}_gpu${gpu}.csv"
  echo "kernel,tflops,status,gpu" > $R
  while read kernel; do
    [ -z "$kernel" ] && continue
    out=$(timeout 120 $BIN --kernel=$kernel --m=8192 --n=4096 --k=1536 2>&1) || true
    tf=$(echo "$out" | grep -oP "median_tflops=\K[0-9.]+" || echo "0")
    st=$(echo "$out" | grep -oP "STATUS=\K[A-Z]+" || echo "TIMEOUT")
    echo "$kernel,$tf,$st,$gpu" >> $R
  done < "$bf"
  log "[$((i+1))/$BATCHES] GPU$gpu $bid done"
  
  cp /tmp/bak_hpp $S/benchmarks/gemm/benchmarks_sycl.hpp
  cp /tmp/bak_main $S/benchmarks/gemm/main.cpp
  rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
done
log "Done. Results: $RESULTS_DIR"
