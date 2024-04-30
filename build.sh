sycl_compiler_path=/opt/cutlass/compiler/0327/
gpu_driver_path=/opt/cutlass/gpu_driver/hotfix_agama-ci-devel-803.25/extract/
cuda_path=/usr/local/cuda-12.3/
mkl_path=/opt/intel/oneapi/mkl/2024.1

# AOT compile
output=intel_gpu_pvc

# jit compile
#output=spir64


export ZE_AFFINITY_MASK=0
export CPATH=$sycl_compiler_path:$sycl_compiler_path/include/:$sycl_compiler_path/include/sycl/:$mkl_path/include/
export LIBRARY_PATH=$gpu_driver_path/usr/lib/x86_64-linux-gnu/:$mkl_path/lib/:$sycl_compiler_path/lib/
export LD_LIBRARY_PATH=$LIBRARY_PATH
export IGC_EnableVISANoSchedule=1
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=./mm_dumps
export IGC_VATemp=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

target=./examples/cute/tutorial/pvc_sycl
rm -rf $target

cmake .. -G Ninja -DCMAKE_CUDA_HOST_COMPILER=${sycl_compiler_path}/bin/clang++ -DCMAKE_CUDA_COMPILER=$cuda_path/bin/nvcc \
-DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$output -DCMAKE_CXX_COMPILER=${sycl_compiler_path}/bin/clang++ \
-DCMAKE_CXX_FLAGS=" -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -DPREFETCH_DEFAULT" && ninja -v $target && $target
