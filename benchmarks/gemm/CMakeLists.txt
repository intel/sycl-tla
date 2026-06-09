# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Pass these configuration files for the CI
set(CONFIG_FILE_INTEL_SMALL --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/input_small_gemm.in)
set(CONFIG_FILE_CUDA --config_file=${CUTLASS_DIR}/benchmarks/device/ampere/input_files/input_gemm.in)

cutlass_benchmark_add_suite(cutlass_benchmarks_gemm)

if(CUTLASS_ENABLE_SYCL)

if(SYCL_INTEL_TARGET)

option(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK "Enable expanded BMG StreamK/DataParallel/SplitK benchmark registrations." OFF)
option(CUTLASS_BENCHMARK_EXHAUSTIVE_GEMM "Enable filter-generated BMG regular GEMM exhaustive registrations." OFF)
option(CUTLASS_BENCHMARK_EXHAUSTIVE_STREAMK "Enable filter-generated BMG scheduler subgroup/stage registrations." OFF)

set(CUTLASS_GEMM_BENCHMARK_LIBRARIES)
if(KERNEL_FILTER_FILE AND TARGET cutlass_lib_static)
  list(APPEND CUTLASS_GEMM_BENCHMARK_LIBRARIES cutlass_lib_static)
  set(CUTLASS_GEMM_BENCHMARK_ENABLE_LIBRARY_GEMM ON)
else()
  set(CUTLASS_GEMM_BENCHMARK_ENABLE_LIBRARY_GEMM OFF)
endif()

set(CUTLASS_GEMM_BENCHMARK_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

if(KERNEL_FILTER_FILE)
  set(CUTLASS_BENCHMARK_FILTER_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/cutlass_benchmark_filter_manifest.list")
  set(CUTLASS_BENCHMARK_FILTERED_HEADER "${CMAKE_CURRENT_BINARY_DIR}/benchmarks_sycl.filtered.hpp")
  set(CUTLASS_BENCHMARK_FILTER_HEADER "${CMAKE_CURRENT_BINARY_DIR}/cutlass_benchmark_filter.hpp")
  set(CUTLASS_BENCHMARK_FILTERED_MAIN "${CMAKE_CURRENT_BINARY_DIR}/main.filtered.cpp")
  file(STRINGS "${KERNEL_FILTER_FILE}" CUTLASS_BENCHMARK_FILTER_LINES REGEX "^\\^[A-Za-z0-9_]+\\$$")
  file(WRITE "${CUTLASS_BENCHMARK_FILTER_MANIFEST}" "")
  file(WRITE "${CUTLASS_BENCHMARK_FILTER_HEADER}" "#pragma once\n")
  foreach(CUTLASS_BENCHMARK_FILTER_LINE IN LISTS CUTLASS_BENCHMARK_FILTER_LINES)
    string(REGEX REPLACE "^\\^([A-Za-z0-9_]+)\\$$" "\\1" CUTLASS_BENCHMARK_FILTER_KERNEL "${CUTLASS_BENCHMARK_FILTER_LINE}")
    file(APPEND "${CUTLASS_BENCHMARK_FILTER_MANIFEST}" "${CUTLASS_BENCHMARK_FILTER_KERNEL}\n")
    file(APPEND "${CUTLASS_BENCHMARK_FILTER_HEADER}" "#define CUTLASS_BENCHMARK_ENABLE_${CUTLASS_BENCHMARK_FILTER_KERNEL} 1\n")
  endforeach()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CUTLASS_SOURCE_DIR}/tools/gen_mini_hpp.py"
            --manifest "${CUTLASS_BENCHMARK_FILTER_MANIFEST}"
            --output "${CUTLASS_BENCHMARK_FILTERED_HEADER}"
    WORKING_DIRECTORY "${CUTLASS_SOURCE_DIR}"
    RESULT_VARIABLE CUTLASS_BENCHMARK_FILTERED_HEADER_RESULT
  )
  if(NOT CUTLASS_BENCHMARK_FILTERED_HEADER_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate filtered benchmarks_sycl header")
  endif()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CUTLASS_SOURCE_DIR}/tools/gen_main.py"
            "${CUTLASS_BENCHMARK_FILTER_MANIFEST}"
            "${CUTLASS_BENCHMARK_FILTERED_MAIN}"
    WORKING_DIRECTORY "${CUTLASS_SOURCE_DIR}"
    RESULT_VARIABLE CUTLASS_BENCHMARK_FILTERED_MAIN_RESULT
  )
  if(NOT CUTLASS_BENCHMARK_FILTERED_MAIN_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate filtered benchmark main")
  endif()
  set(CUTLASS_GEMM_BENCHMARK_MAIN "${CUTLASS_BENCHMARK_FILTERED_MAIN}")
endif()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_sycl
    "${CUTLASS_GEMM_BENCHMARK_MAIN}"
    SUITE cutlass_benchmarks_gemm
    LIBRARIES ${CUTLASS_GEMM_BENCHMARK_LIBRARIES}
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_INTEL_SMALL
)

if(CUTLASS_GEMM_BENCHMARK_ENABLE_LIBRARY_GEMM)
  target_compile_definitions(cutlass_benchmarks_gemm_sycl PRIVATE CUTLASS_BENCHMARK_ENABLE_LIBRARY_GEMM=1)
endif()
target_compile_definitions(cutlass_benchmarks_gemm_sycl PRIVATE CUTLASS_SYCL_PROFILING_ENABLED COMPAT_PROFILING_ENABLED)
if(KERNEL_FILTER_FILE)
  target_compile_definitions(cutlass_benchmarks_gemm_sycl PRIVATE CUTLASS_BENCHMARK_FILTER_ENABLED=1)
  target_compile_definitions(cutlass_benchmarks_gemm_sycl PRIVATE CUTLASS_BENCHMARK_USE_FILTERED_HEADER=1)
  target_include_directories(
    cutlass_benchmarks_gemm_sycl
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()
if(CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK)
  target_compile_definitions(cutlass_benchmarks_gemm_sycl PRIVATE CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=1)
endif()

else()
cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_sycl
    main.cpp
    SUITE cutlass_benchmarks_gemm
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_CUDA
)
endif()

else()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_cuda
    main.cu
    SUITE cutlass_benchmarks_gemm
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_CUDA
)

endif()
