cmake_minimum_required(VERSION 3.26)

# Set Intel SYCL compiler before project() call
find_program(ICX_COMPILER icx)
find_program(ICPX_COMPILER icpx)
if(ICX_COMPILER AND ICPX_COMPILER)
    execute_process(
      COMMAND ${ICPX_COMPILER} --version
      OUTPUT_VARIABLE ICPX_VERSION_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "[0-9]+\\.[0-9]+" DPCPP_VERSION "${ICPX_VERSION_OUTPUT}")
    set(DPCPP_VERSION "${DPCPP_VERSION}" CACHE STRING "DPCPP major.minor version")
    set(CMAKE_C_COMPILER ${ICX_COMPILER})
    set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
    message(STATUS "Using Intel SYCL C++ compiler: ${ICPX_COMPILER} and C compiler: ${ICX_COMPILER} Version: ${DPCPP_VERSION}")
else()
    message(FATAL_ERROR "Intel SYCL C++ compiler (icpx) and/or C compiler (icx) not found. Please install Intel oneAPI toolkit.")
endif()

project(flash_attn)

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

include("cmake/utils.cmake")

# Find Python with all necessary components for building extensions
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(Torch REQUIRED)

# Intel XPU backend detection and setup
if(NOT TORCH_VERSION)
  run_python(TORCH_VERSION "import torch; print(torch.__version__)" "Failed to get Torch version")
endif()

# Check for Intel XPU support in PyTorch
run_python(XPU_AVAILABLE
  "import torch; print('true' if hasattr(torch, 'xpu') else 'false')"
  "Failed to check XPU availability")

if(NOT XPU_AVAILABLE STREQUAL "true")
  message(WARNING "Intel XPU is not available in this PyTorch installation. XPU kernels will be skipped.")
  return()
endif()

# Set up XPU compilation flags
set(GPU_LANG "SYCL")
add_compile_definitions(XPU_KERNEL)
add_compile_definitions(USE_XPU)

# Set SYCL-specific flags
# Set comprehensive SYCL compilation and linking flags
set(sycl_link_flags "-fsycl;--offload-compress;-fsycl-targets=spir64_gen,spir64;-Xs;-device pvc,xe-lpg,ats-m150 -options ' -cl-intel-enable-auto-large-GRF-mode -cl-poison-unsupported-fp64-kernels -cl-intel-greater-than-4GB-buffer-required';")
set(sycl_flags "-fsycl;-fhonor-nans;-fhonor-infinities;-fno-associative-math;-fno-approx-func;-fno-sycl-instrument-device-code;--offload-compress;-fsycl-targets=spir64_gen,spir64;")
message(STATUS "Configuring for Intel XPU backend using SYCL")

find_package(CutlassSycl)

if(DPCPP_VERSION STREQUAL "2025.2")
  set(CUTLASS_SYCL_REVISION "v0.5" CACHE STRING "CUTLASS revision to use")
elseif(DPCPP_VERSION STREQUAL "2025.1")
  set(CUTLASS_SYCL_REVISION "v3.9-0.3" CACHE STRING "CUTLASS revision to use")
elseif(DPCPP_VERSION STREQUAL "2025.0")
  set(CUTLASS_SYCL_REVISION "v3.9-0.2" CACHE STRING "CUTLASS revision to use")
else()
  message(FATAL_ERROR "Unknown DPCPP_VERSION: ${DPCPP_VERSION}")
endif()

if (NOT CutlassSycl_FOUND)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
  set(CUTLASS_ENABLE_BENCHMARKS OFF CACHE BOOL "Disable CUTLASS Benchmarks")
# Use the specified CUTLASS source directory for compilation if CUTLASS_SYCL_SRC_DIR is provided
  if (DEFINED ENV{CUTLASS_SYCL_SRC_DIR})
    set(CUTLASS_SYCL_SRC_DIR $ENV{CUTLASS_SYCL_SRC_DIR})
  endif()

  if(CUTLASS_SYCL_SRC_DIR)
    if(NOT IS_ABSOLUTE CUTLASS_SYCL_SRC_DIR)
      get_filename_component(CUTLASS_SYCL_SRC_DIR "${CUTLASS_SYCL_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The CUTLASS_SYCL_SRC_DIR is set, using ${CUTLASS_SYCL_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${CUTLASS_SYCL_SRC_DIR})
  else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/intel/cutlass-sycl.git
        GIT_TAG ${CUTLASS_SYCL_REVISION}
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW TRUE
    )
  endif()

  # Set Intel backend env
  message(STATUS "Setting Intel GPU optimization env vars for Cutlass-SYCL")
  set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL for CUTLASS")
  add_compile_definitions(CUTLASS_ENABLE_SYCL=1)
  set(DPCPP_SYCL_TARGET "intel_gpu_pvc" CACHE STRING "SYCL target for Intel GPU")
  add_compile_definitions(DPCPP_SYCL_TARGET=intel_gpu_pvc)
  set(SYCL_INTEL_TARGET ON CACHE BOOL "Enable SYCL for INTEL")
  add_compile_definitions(SYCL_INTEL_TARGET=1)

  set(ENV{SYCL_PROGRAM_COMPILE_OPTIONS} "-ze-opt-large-register-file")
  set(ENV{IGC_VISAOptions} "-perfmodel")
  set(ENV{IGC_VectorAliasBBThreshold} "10000")
  set(ENV{IGC_ExtraOCLOptions} "-cl-intel-256-GRF-per-thread")

  FetchContent_MakeAvailable(cutlass)

  include_directories(${CUTLASS_INCLUDE_DIR})
  include_directories(${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})
else()
  include_directories(${CUTLASS_INCLUDE_DIR})
  include_directories(${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})
endif(NOT CutlassSycl_FOUND)

string(REPLACE "-fsycl-targets=spir64_gen,spir64" "-fsycl-targets=intel_gpu_pvc" sycl_link_flags "${sycl_link_flags}")
string(REPLACE "-device pvc,xe-lpg,ats-m150" "" sycl_link_flags "${sycl_link_flags}")
string(APPEND sycl_link_flags "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier")
if(CUTLASS_SYCL_REVISION STREQUAL "v0.5")
  string(APPEND sycl_link_flags ",+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
endif()
string(REPLACE "-fsycl-targets=spir64_gen,spir64" "-fsycl-targets=intel_gpu_pvc" sycl_flags "${sycl_flags}")


set(TORCH_flash_attn_SRC
  torch-ext/torch_binding.cpp torch-ext/torch_binding.h
)


list(APPEND SRC "${TORCH_flash_attn_SRC}")
set(flash_attn_xpu_SRC
  "flash_attn_xpu/flash_api.cpp"
"flash_attn_xpu/src/prefill.hpp"
"flash_attn_xpu/src/fmha_utils.hpp"
"flash_attn_xpu/src/compat_wrapper.hpp"
"flash_attn_xpu/src/collective/fmha_fusion.hpp"
"flash_attn_xpu/src/collective/xe_flash_attn_prefill_epilogue.hpp"
"flash_attn_xpu/src/collective/xe_flash_attn_prefill_mma.hpp"
"flash_attn_xpu/src/collective/xe_flash_attn_prefill_softmax_epilogue.hpp"
"flash_attn_xpu/src/kernel/tile_scheduler.hpp"
"flash_attn_xpu/src/kernel/xe_flash_attn_prefill.hpp"
)



# Add SYCL-specific compilation flags for XPU sources
# Use default SYCL flags
foreach(_KERNEL_SRC ${flash_attn_xpu_SRC})
  if(_KERNEL_SRC MATCHES ".*\\.(cpp|cxx|cc)$")
    set_property(
      SOURCE ${_KERNEL_SRC}
      APPEND PROPERTY
      COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${sycl_flags}>"
    )
  endif()
endforeach()

list(APPEND SRC "${flash_attn_xpu_SRC}")
define_gpu_extension_target(
  _flash_attn_test
  DESTINATION _flash_attn_test
  INCLUDE_DIRECTORIES "${ONEDNN_XPU_INCLUDE_DIR}"
  LANGUAGE ${GPU_LANG}
  SOURCES ${SRC}
  COMPILE_FLAGS ${sycl_flags}
  USE_SABI 3
  WITH_SOABI)

# Add XPU/SYCL specific linker flags
target_link_options(_flash_attn_test PRIVATE ${sycl_link_flags})
target_link_libraries(_flash_attn_test PRIVATE dnnl)
