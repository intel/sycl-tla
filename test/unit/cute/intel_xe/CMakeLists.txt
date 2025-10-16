# Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
# Copyright (C) 2025 Intel Corporation, All rights reserved.
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

# Function to check IGC version
function(check_igc_version)
    execute_process(
        COMMAND bash -c "dpkg -l | grep igc | grep 'intel-igc-core' | awk '{print $3}' | cut -d'.' -f1,2"
        OUTPUT_VARIABLE IGC_VER
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    # TEMPORARY: Force lower version for testing
    # set(IGC_VER "2.11")  # Change this to test different versions

    if(IGC_VER)
        string(REPLACE "." ";" VER_LIST ${IGC_VER})
        list(GET VER_LIST 0 MAJOR)
        list(GET VER_LIST 1 MINOR)
        set(IGC_VERSION_MAJOR ${MAJOR} PARENT_SCOPE)
        set(IGC_VERSION_MINOR ${MINOR} PARENT_SCOPE)
        set(IGC_SUFFICIENT $<VERSION_GREATER_EQUAL:${IGC_VER},2.18> PARENT_SCOPE)
    else()
        set(IGC_VERSION_MAJOR 0 PARENT_SCOPE)
        set(IGC_VERSION_MINOR 0 PARENT_SCOPE)
        set(IGC_SUFFICIENT FALSE PARENT_SCOPE)
    endif()
endfunction()

check_igc_version()

if(SYCL_INTEL_TARGET)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_intel_xe
  copy_1d.cpp
  copy_subgroup_block.cpp
  copy_block.cpp
  copy_scatter.cpp
  mma.cpp
  tiled_mma.cpp
  xe_copy_2d_test.cpp
)

target_compile_definitions(cutlass_test_unit_cute_intel_xe PRIVATE
    IGC_VERSION_MAJOR=${IGC_VERSION_MAJOR}
    IGC_VERSION_MINOR=${IGC_VERSION_MINOR}
)

else()
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_intel_xe
)
endif()
