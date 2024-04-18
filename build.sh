sycl_compiler_path=/opt/cutlass_compiler/
target=./examples/cute/tutorial/pvc_sycl
cuda_path=/usr/local/cuda-12.3/
rm -rf $target
export CPATH=$sycl_compiler_path:$sycl_compiler_path/include/:$sycl_compiler_path/include/sycl/:/opt/intel/oneapi/mkl/2024.1/include/
export LIBRARY_PATH=/opt/intel/oneapi/mkl/2024.1/lib/
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2024.1/lib/:${sycl_compiler_path}/lib/
#export IGC_EnableVISANoSchedule=1
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=./mm_dumps_prefetch_coop
#export IGC_VATemp=1
cmake .. -G Ninja -DCMAKE_CUDA_HOST_COMPILER=${sycl_compiler_path}/bin/clang++ -DCMAKE_CUDA_COMPILER=$cuda_path/bin/nvcc -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_pvc -DCMAKE_CXX_COMPILER=${sycl_compiler_path}/bin/clang++ -DCMAKE_CXX_FLAGS=" -DITEM_SIZE_X=4 -DITEM_SIZE_Y=32 -DSG_SIZE_X=64 -DSG_SIZE_Y=ITEM_SIZE_Y -DWG_SIZE_X=256 -DWG_SIZE_Y=256 -DKK=2 -DPREFETCH_DEFAULT -lmkl_intel_lp64 -lmkl_sequential -lmkl_core" && ninja -v $target && ONEAPI_DEVICE_SELECTOR=level_zero:gpu $target

