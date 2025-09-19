
[README](../../../README.md#documentation) > **CUTLASS 3: Building with SYCL support**

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/codeplaysoftware/cutlass-fork/badge)](https://scorecard.dev/viewer/?uri=github.com/codeplaysoftware/cutlass-fork)

SYCL[1] is a royalty-free, cross-platform abstraction layer that enables
code for heterogeneous and offload processors to be written with modern 
ISO C++, and provides API and abstractions to find devices and manage 
resources for GPUs. 

## Support for Intel GPUs

The CUTLASS-SYCL supports running on Intel GPUs.
Currently, Intel Data Center Max 1550 and 1100 (a.k.a Ponte Vecchio - PVC) along with Intel Arc B580 (a.k.a BattleMage - BMG) are supported.

The `examples` directory shows a number of GEMM algorithms and examples of 
CUTLASS-SYCL running on PVC and BMG, including flash attention V2.

Only Linux platforms are supported.

### Requirements (SYCL for Intel GPU)

To build CUTLASS SYCL support for Intel GPUs, you need the DPC++ compiler;
you can use the latest [[nightly build](https://github.com/intel/llvm/releases)] 
or a oneAPI toolkit from 2025.1 onwards. Intel Compute Runtime 25.13 (with Intel Graphics Compiler 2.10.10) is required. At the time of the release it can be installed from [intel-graphics-staging](https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics-staging). Installation from [intel-graphics](https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics) is recommended when it is available there.

Building the tests and the examples requires oneMKL for random number generation.

### Building with SYCL for Intel GPU support

The following instructions show how to use the nightly build to build the cutlass examples


```
# Download the nightly of DPCPP compiler
$ wget https://github.com/intel/llvm/releases/tag/nightly-2025-01-31
# Setup the environment variables
$ export PATH_TO_DPCPP=/path/to/your/dpcpp/installation
$ export PATH=${PATH_TO_DPCPP}/bin/:$PATH
$ export LD_LIBRARY_PATH=${PATH_TO_DPCPP}/lib/:$LD_LIBRARY_PATH
$ export RPATH=${PATH_TO_DPCPP}/lib/:$RPATH
# Create the build directory and configure CMake
# mkdir build_intel; cd build_intel
$ CC=clang CXX=clang++ cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_pvc \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_FLAGS="-ftemplate-backtrace-limit=0 -fdiagnostics-color=always"  
```

CMake will check that DPC++ compiler is available in the system,
and it will download the MKL library if it cannot find it.
To get better performance result we require the following combinations of the environment variables flags to provide better performance hints for generating optimised code. For ahead of time (AOT) compilation, the following options have to be set during compilation and for Just in time (JIT) Compilation  when running

```
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file" 
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"  
```
To build and run a simple PVC gemm example run the commands below.

```
$ ninja 00_pvc_gemm
$ ./examples/sycl/00_pvc_gemm/00_pvc_gemm
Disposition: Passed
Problem Size: 5120x4096x4096x1
Cutlass GEMM Performance:     [247.159]TFlop/s  (0.6951)ms
```

## Support for NVIDIA GPUs (validation only)

The SYCL backend supports compilation for NVIDIA GPUs using the 
oneAPI NVIDIA plugin. This support is only for testing and validation
purposes and not intended for production.

### Requirements 
 
To build CUTLASS SYCL support you need the latest version of DPC++ compiler. You can either use a recent [nightly build](https://github.com/intel/llvm/releases)
or build the compiler from source as described in [oneAPI DPC++ guideline]((https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda)).

### Building with SYCL for NVIDIA support
Once you have your compiler installed, you need to point the
`CMAKE_CUDA_HOST_COMPILER` flag to the clang++ provided by it.
This enables the compilation of SYCL sources without altering the current NVCC path. For example, to build SYCL support for SM 80
GPUs, you can use the following command:

```
cmake -G Ninja  \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=nvptx64-nvidia-cuda \
  -DDPCPP_SYCL_ARCH=sm_80
```

### Running the example

#### CuTe 
Currently, you can build the CuTe Tutorial using the following command: 

```
ninja [EXAMPLE_NAME]_sycl
```

You can run it like this from your build directory

```
LD_LIBRARY_PATH=/path/to/sycl/install/lib ./examples/cute/tutorial/[EXAMPLE_NAME]_sycl
```

#### CUTLASS Example
 Currently, the example `14_amper_tf32_tensorop_gemm` has been implemented for SYCL on Nvidia Ampere architecture. You can build this from your build directory by running :
 ```
  ninja 14_ampere_tf32_tensorop_gemm_cute
 ```
 You can run it like this from your build directory
 ```
  LD_LIBRARY_PATH=/path/to/sycl/install/lib ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm_cute
 ```

# References

[1] https://www.khronos.org/sycl/
