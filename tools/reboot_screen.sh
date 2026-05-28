#!/bin/bash
# Reboot-safe batch build + screen pipeline
# Run after machine reboot: bash tools/reboot_screen.sh
# Usage:
#   BATCHES=4 bash tools/reboot_screen.sh        # dry-run: first 4 batches
#   BATCHES=all bash tools/reboot_screen.sh      # full screening

S=$(cd "$(dirname "$0")/.." && pwd)
WS=/root/cutlass_profile_device7_b70_2500mhz/screen_ws
BDIR=/root/cutlass_profile_device7_b70_2500mhz/ali_one_8192_4096_1536_layered_bmg_final_flagsfixed_20260522_0425_ws/build/candidate_benchmarks/candidate_batch_preflight/selected_kernel_batch_001
GOOD_D=$BDIR/_deps
GB_LIB=$BDIR/_deps/googlebenchmark-build/src/libbenchmark.a
CUTLASS_LIB=$BDIR/tools/library/libcutlass.a

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- Setup ----
log "Restoring deps..."
mkdir -p $S/_deps/googlebenchmark-src/include/benchmark
mkdir -p $S/_deps/googletest-src
cp -r $GOOD_D/googlebenchmark-src/include/benchmark/* $S/_deps/googlebenchmark-src/include/benchmark/ 2>/dev/null || true
cp -r $GOOD_D/googletest-src/googletest $S/_deps/googletest-src/ 2>/dev/null || true
cp -r $GOOD_D/googletest-src/googlemock $S/_deps/googletest-src/ 2>/dev/null || true

source /opt/intel/oneapi/compiler/2025.3/env/vars.sh 2>/dev/null || true
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000 IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"

for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $gov 2>/dev/null; done
log "CPU=performance, perf flags exported"

# Touch compiler_depend.ts to ensure it exists (hard dependency in build.make)
touch $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.ts
touch $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/compiler_depend.make

# ---- Generate manifest ----
if [ ! -f "$WS/manifest.json" ]; then
  log "Creating kernel batches..."
  rm -rf $WS; mkdir -p $WS/{builds,results}
  cd $S/test && python3 -c "
import sys,json,os; sys.path.insert(0,'benchmarks')
from intel_gemm_profiler.catalog import generated_layered_bmg_kernel_catalog
from intel_gemm_profiler.constraints import default_constraints
cons=default_constraints(); cat=generated_layered_bmg_kernel_catalog(constraints=cons)
all_k=sorted(set(k['kernel_name'] for k in cat['kernels'] if k.get('dtype_family')=='bf16'))
all_k=[k for k in all_k if not k.startswith('03_bmg') and 'streamk_example' not in k]
batches=[all_k[i:i+2] for i in range(0,len(all_k),2)]
for i,b in enumerate(batches):
    with open(f'$WS/builds/batch_{i:04d}.txt','w') as f:
        for k in b: f.write(k+'\n')
with open('$WS/manifest.json','w') as f: json.dump({'total':len(all_k),'batches':len(batches)},f)
print(f'{len(all_k)} kernels in {len(batches)} batches',file=sys.stderr)
" 2>&1
fi

BATCHES=${BATCHES:-4}
TOTAL=$(python3 -c "import json; print(json.load(open('$WS/manifest.json'))['batches'])")
[ "$BATCHES" = "all" ] && BATCHES=$TOTAL

log "Processing $BATCHES of $TOTAL batches..."

# ---- Batch loop ----
for i in $(seq 0 $((BATCHES-1))); do
  bid=$(printf "batch_%04d" $i)
  bf="$WS/builds/${bid}.txt"
  [ ! -f "$bf" ] && continue
  gpu=$(( 5 + (i % 2) ))
  
  # Backup source
  cp $S/benchmarks/gemm/benchmarks_sycl.hpp /tmp/bak_hpp_$$
  cp $S/benchmarks/gemm/main.cpp /tmp/bak_main_$$
  rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
  
  # Mini HPP
  python3 $S/tools/gen_mini_hpp.py --manifest "$bf" --output /tmp/${bid}.hpp 2>&1 | tail -1
  cp /tmp/${bid}.hpp $S/benchmarks/gemm/benchmarks_sycl.hpp
  
  # Mini main.cpp
  python3 -c "
kernels=[]; 
with open('$bf') as f:
    for l in f:
        if l.strip(): kernels.append(l.strip())
runs='\n'.join(f'  RUN({k})' for k in kernels)
m=f'''#include \"cutlass/cutlass.h\"
#include \"cutlass/kernel_hardware_info.h\"
#include \"cutlass/util/command_line.h\"
#include <iostream>
#include \"benchmark_runner.hpp\"
#if defined(SYCL_INTEL_TARGET)
#include \"benchmarks_sycl.hpp\"
#endif
int main(int argc, const char** argv) {{
  cutlass::CommandLine cmd(argc, argv);
  std::string kernel; cmd.get_cmd_line_argument(\"kernel\", kernel, std::string(\"\"));
  if (kernel.empty()) {{ std::cerr << \"--kernel=NAME\" << std::endl; return 1; }}
  register_gemm_benchmarks();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  cutlass::benchmark::GEMMOptions opts;
  cmd.get_cmd_line_argument(\"m\", opts.m, 8192); cmd.get_cmd_line_argument(\"n\", opts.n, 4096);
  cmd.get_cmd_line_argument(\"k\", opts.k, 1536); opts.verify_library = 0;
  double tflops = 0; bool ok = false;
#define RUN(K) if (kernel == #K) {{ tflops = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct(opts, hw); ok = true; }}
{runs}
#undef RUN
  if (!ok) {{ std::cerr << \"NOT_FOUND\" << std::endl; return 1; }}
  std::cout << \"RESULT: kernel=\" << kernel << \" median_tflops=\" << tflops << \" STATUS=OK\" << std::endl;
  return 0;
}}
'''
with open('$S/benchmarks/gemm/main.cpp','w') as f: f.write(m)
" 2>&1
  
  # ---- Compile .o via cmake make ----
  # make will compile .o successfully but fail at link (expected — GB stub)
  # We catch the link error and do manual link below
  rm -f $BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o
  rm -f $BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
  
  log "[$bid] Compiling..."
  make -C $BDIR cutlass_benchmarks_gemm_sycl -j128 2>&1 | grep -E "Built target|error:" | head -3 || true
  
  OBJ=$BDIR/benchmarks/gemm/CMakeFiles/cutlass_benchmarks_gemm_sycl.dir/main.cpp.o
  if [ ! -s "$OBJ" ]; then
    log "[$bid] COMPILE FAILED"
    cp /tmp/bak_hpp_$$ $S/benchmarks/gemm/benchmarks_sycl.hpp
    cp /tmp/bak_main_$$ $S/benchmarks/gemm/main.cpp
    rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
    continue
  fi
  
  # ---- Manual link ----
  BIN=$BDIR/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
  icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    -O3 $OBJ -o $BIN $GB_LIB -L/lib64/stubs -Wl,-rpath,/lib64/stubs: $CUTLASS_LIB \
    -Wl,-rpath=/opt/intel/oneapi/mkl/2025.3/lib \
    /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_ilp64.so /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_thread.so \
    /opt/intel/oneapi/mkl/2025.3/lib/libmkl_core.so /opt/intel/oneapi/compiler/2025.3/lib/libiomp5.so \
    -lm -ldl -lpthread /opt/intel/oneapi/compiler/2025.3/lib/libsycl.so 2>/dev/null
  
  if [ ! -x "$BIN" ]; then
    log "[$bid] LINK FAILED"
    cp /tmp/bak_hpp_$$ $S/benchmarks/gemm/benchmarks_sycl.hpp
    cp /tmp/bak_main_$$ $S/benchmarks/gemm/main.cpp
    rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
    continue
  fi
  log "[$bid] BUILD OK ($(stat -c%s $BIN) bytes)"
  
  # ---- Screen ----
  export ZE_AFFINITY_MASK=$gpu
  R="$WS/results/${bid}_gpu${gpu}.csv"
  echo "kernel,tflops,status,gpu" > $R
  kp=0; kc=0
  while read kernel; do
    [ -z "$kernel" ] && continue
    kc=$((kc+1))
    out=$(timeout 120 $BIN --kernel=$kernel --m=8192 --n=4096 --k=1536 2>&1) || true
    tf=$(echo "$out" | grep -oP "median_tflops=\K[0-9.]+" || echo "0")
    st=$(echo "$out" | grep -oP "STATUS=\K[A-Z]+" || echo "TIMEOUT")
    echo "$kernel,$tf,$st,$gpu" >> $R
    [ "$st" = "OK" ] && kp=$((kp+1))
  done < "$bf"
  log "  GPU$gpu $bid: $kp/$kc passed"
  
  # Restore source
  cp /tmp/bak_hpp_$$ $S/benchmarks/gemm/benchmarks_sycl.hpp
  cp /tmp/bak_main_$$ $S/benchmarks/gemm/main.cpp
  rm -f $S/benchmarks/gemm/benchmarks_sycl.hpp.cache
done

log "Done. Results: $WS/results/"
