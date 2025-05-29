script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf ${script_dir}/.git/hooks/pre-commit
rm -rf ${script_dir}/build && mkdir ${script_dir}/build && cd ${script_dir}/build

clear
unset SYCL_PROGRAM_COMPILE_OPTIONS
unset IGC_VISAPreSchedCtrl
unset IGC_EnableVISANoSchedule
unset IGC_ShaderDumpEnable
unset IGC_DumpToCustomDir
unset IGC_VISAOptions
unset IGC_DisableLoopUnroll
unset IGC_VectorAliasBBThreshold
unset IGC_ExtraOCLOptions
unset ONEAPI_DEVICE_SELECTOR
unset OCL_ICD_VENDORS

#. /opt/intel/oneapi/2025.1/oneapi-vars.sh
. /opt/intel/oneapi/setvars.sh

export CC=icx
export CXX=icpx
#export CC=clang
#export CXX=clang++

#export LD_LIBRARY_PATH=$LIBRARY_PATH

#export ZE_AFFINITY_MASK=0
#export CUTLASS_ENABLE_SYCL=ON
# ON is GPU time
export CUTLASS_SYCL_PROFILING_ENABLED=ON
# OFF is wall time
#export CUTLASS_SYCL_PROFILING_ENABLED=OFF
#export DPCPP_SYCL_TARGET=intel_gpu_bmg_g21 
#export CUTLASS_ENABLE_BENCHMARKS=ON 

export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=${script_dir}/build/mm_dumps

#export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only" 
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
#export CMAKE_BUILD_TYPE=Release
#export IGC_VISAOptions="-perfmodel"
#export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export OCL_ICD_VENDORS=$HOME

# GPU time
cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DCUTLASS_SYCL_PROFILING_ENABLED=ON -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_FLAGS="-gline-tables-only $1 $2 $3"
# wall time
#cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DCUTLASS_SYCL_PROFILING_ENABLED=OFF -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release

target=./examples/sycl/11_pvc_gemm_int4_streamk/11_pvc_gemm_int4_streamk
ninja $target && \
$target --m=32 --n=14336 --k=4096 --l=1 --iterations=20 --flush_cache=1 --warmup=10 --l3_cache_size=32


#ninja benchmarks
#./benchmarks/benchmarks  --config_file=../benchmarks/pvc/input_files/pytorch_2.in

#./benchmarks/benchmarks  --config_file=../benchmarks/pvc/input_files/input_pytorch_1.in 
