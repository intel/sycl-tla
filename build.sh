script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf ${script_dir}/.git/hooks/pre-commit
rm -rf ${script_dir}/build && mkdir ${script_dir}/build && cd ${script_dir}/build

clear


# ================= unset envs =================
unset do_validation
unset epilogue
unset SYCL_PROGRAM_COMPILE_OPTIONS
unset sycl_compiler_path
unset gpu_driver_path
unset enable_prefetch
unset divide_b
unset disable_gemm
unset IGC_VISAPreSchedCtrl
unset IGC_EnableVISANoSchedule
unset IGC_ShaderDumpEnable
unset IGC_DumpToCustomDir
unset IGC_VISAOptions
unset IGC_DisableLoopUnroll
unset IGC_VectorAliasBBThreshold
unset IGC_VISAOptions
unset IGC_ExtraOCLOptions
unset SYCL_PROGRAM_COMPILE_OPTIONS
unset ONEAPI_DEVICE_SELECTOR
unset IGC_VISAOptions
unset IGC_VectorAliasBBThreshold


# ================= compiler / driver =================
# https://github.com/intel/llvm/releases/tag/nightly-2024-05-16
# https://ubit-gfx.intel.com/build/19168301/artifacts
sycl_compiler_path=/opt/cutlass/compiler/20250515/
gpu_driver_path=/opt/cutlass/gpu_driver/gfx-driver-ci-comp_igc-29375/extract/
export CPATH=$sycl_compiler_path:$sycl_compiler_path/include/:$sycl_compiler_path/include/sycl/
export LIBRARY_PATH=$gpu_driver_path/usr/lib/x86_64-linux-gnu/:$sycl_compiler_path/lib/
export LD_LIBRARY_PATH=$LIBRARY_PATH
export clang_path=${sycl_compiler_path}/bin/clang++


# ================= JIT / AOT =================
output=intel_gpu_bmg_g21
#output=spir64


# ================= IGC options =================
#export IGC_VISAPreSchedCtrl=6
#export IGC_EnableVISANoSchedule=0
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=${script_dir}/build/mm_dumps
export CUTLASS_SYCL_PROFILING_ENABLED=ON

export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
#export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
#export IGC_VISAOptions="-perfmodel"
#export IGC_VectorAliasBBThreshold=100000000000


export ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export OCL_ICD_VENDORS=$HOME


# ================= target =================
#target=./test/unit/cute/intel_xe/cutlass_test_unit_cute_intel_xe
target=./examples/sycl/02_bmg_gemm_mixed_dtype/02_bmg_gemm_bf16_s8_bf16
target=./examples/sycl/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_f16
#target=./examples/sycl/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_s8_f16_tensorwise
#target=./examples/sycl/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_s8

cmake .. -G Ninja -DCUTLASS_SYCL_PROFILING_ENABLED=ON -DCMAKE_CUDA_HOST_COMPILER=$clang_path -DCMAKE_CXX_FLAGS_RELEASE=$1 \
-DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$output -DCMAKE_CXX_COMPILER=$clang_path \
-DCMAKE_CXX_FLAGS=" -DCOMPILER_VERSION -ftarget-register-alloc-mode=pvc:auto -DSYCL_INTEL_TARGET -gline-tables-only $1 $2 $3" \
&& ninja -v $target && \
$target --m=32 --n=14336 --k=4096 --l=1 --iterations=20

# -gline-tables-only

# ================= unitrace command =================
#unitrace -k -i 20 --chrome-kernel-logging -o cutlass_pvc_gemm.csv $target

#print kernel calls
#unitrace --device-timing --kernel-submission --device-timeline --chrome-kernel-logging --chrome-device-logging --chrome-no-thread-on-device --chrome-no-engine-on-device -i 20 $target -o cutlass.csv
#unitrace -k --chrome-kernel-logging --chrome-device-logging --chrome-no-thread-on-device --chrome-no-engine-on-device -i 20 $target -o cutlass.csv

#check stalls
#unitrace --chrome-kernel-logging --stall-sampling -i 20 -o cutlass_pvc_gemm.csv $target

#unitrace --metric-list
#unitrace -k -g <group> -i 20 --chrome-kernel-logging $target -o

perf_py=/home/zt/workspace/cutlass/unitrace/tools/unitrace/scripts/metrics/analyzeperfmetrics.py
#csv_file=cutlass_pvc_gemm.metrics.3584319.csv
#python3 $perf_py -l $csv_file
#python3 $perf_py -m "XVE_STALL[%],XVE_THREADS_OCCUPANCY_ALL[%],XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%],XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%],XVE_INST_EXECUTED_SEND_ALL_UTILIZATION[%],XVE_INST_EXECUTED_CONTROL_ALL_UTILIZATION[%],XVE_INST_EXECUTED_XMX_ALL_UTILIZATION[%]" -y "Occupancy, Stalls and Function Unit Utilizations" -m "AvgGpuSliceFrequencyMHz[MHz]" -y "Frequency" -m "L3_BYTE_READ[bytes],L3_BYTE_WRITE[bytes],GPU_MEMORY_BYTE_READ[bytes],GPU_MEMORY_BYTE_WRITE[bytes]" -y "L3 and Memory" -m "XVE_ACTIVE[%],XVE_STALL[%]" -y "Active and Stalls" -b "L3_BYTE_READ[bytes],L3_BYTE_WRITE[bytes],GPU_MEMORY_BYTE_READ[bytes],GPU_MEMORY_BYTE_WRITE[bytes]" -t "Hardware Metrics" ${csv_file}
#python3 $perf_py -s $IGC_DumpToCustomDir -t "XVE Stalls by Instruction" $csv_file -o ${csv_file}.pdf
#python3 $perf_py -k "main::{lambda(auto:1)#3}" -s $IGC_DumpToCustomDir $csv_file -o ${csv_file}.pdf
