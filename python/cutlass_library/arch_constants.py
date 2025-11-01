#################################################################################################
#
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
#
#################################################################################################

"""
Architecture range constants for CUTLASS library generation.
Shared across manifest.py and gemm_operation.py to avoid circular imports.
"""

###################################################################################################
# Architecture range constants
# Intel Xe architectures use the range [INTEL_XE_ARCH_MIN, INTEL_XE_ARCH_MAX)
# CUDA architectures use values >= CUDA_ARCH_MIN
###################################################################################################
INTEL_XE_ARCH_MIN = 12  # Minimum Intel Xe architecture (PVC = 12, BMG = 20)
INTEL_XE_ARCH_MAX = 50  # Upper bound (exclusive) for Intel Xe range
CUDA_ARCH_MIN = 50      # Minimum CUDA architecture (sm_50, sm_60, etc.)

###################################################################################################
# Specific Intel Xe architecture constants
###################################################################################################
# Intel Xe12 - PVC (Ponte Vecchio) HPC architecture
INTEL_XE12 = 12

# Intel Xe20 - BMG (Battlemage) gaming architecture  
INTEL_XE20 = 20

# Intel Xe35 - Future architecture placeholder
INTEL_XE35 = 35

###################################################################################################
# Architecture validation helpers
###################################################################################################
def is_intel_xe_arch(arch):
    """Check if the given architecture is an Intel Xe architecture."""
    return INTEL_XE_ARCH_MIN <= arch < INTEL_XE_ARCH_MAX

def is_cuda_arch(arch):
    """Check if the given architecture is a CUDA architecture."""
    return arch >= CUDA_ARCH_MIN

def get_arch_name(arch):
    """Get a human-readable name for the architecture."""
    if arch == INTEL_XE12:
        return "Intel Xe12 (PVC)"
    elif arch == INTEL_XE20:
        return "Intel Xe20 (BMG)" 
    elif arch == INTEL_XE35:
        return "Intel Xe35 (CRI)"
    elif is_intel_xe_arch(arch):
        return f"Intel Xe{arch}"
    elif is_cuda_arch(arch):
        return f"CUDA SM{arch}"
    else:
        return f"Unknown({arch})"

###################################################################################################
