#################################################################################################
#
# Copyright (c) 2023 - 2025 Codeplay Software Limited. All rights reserved.
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
Low-level functionality tests for GEMM with BF16 operands on Xe20 (BMG)
"""

from functools import partial
import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend.utils.device import device_cc
from cutlass_library.arch_constants import ( INTEL_XE12_PVC, is_intel_xe_arch)

from utils import LayoutCombination, add_test_gemm


cutlass_cppgen.set_log_level(logging.WARNING)
cc = 20  # BMG architecture is 20 (Xe2)
dtype = cutlass_cppgen.DataType.bf16


@unittest.skipIf(not is_intel_xe_arch(device_cc()), 'Device compute capability is insufficient for Xe20 tests.')
@unittest.skipIf(cutlass_cppgen.utils.datatypes.torch_type(dtype) is None, f'Version of torch installed does not contain a datatype match for {dtype}')
class GemmBF16Xe20(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


add_test_xe20_bf16 = partial(add_test_gemm, cls=GemmBF16Xe20, cc=INTEL_XE20_BMG,
                            element=dtype,
                            compilation_modes=["dpcpp"],
                            opclass=cutlass_cppgen.OpcodeClass.TensorOp,
                            stages=0,
                            cluster_shape=[1, 1, 1])

add_test_f32_acc = partial(add_test_xe20_bf16, alignments=[2, 2, 4],
                           element_C=cutlass_cppgen.DataType.f32,
                           element_output=cutlass_cppgen.DataType.f32,
                           element_accumulator=cutlass_cppgen.DataType.f32)
add_test_bf16_acc = partial(add_test_xe20_bf16, alignments=[2, 2, 2],
                            element_C=cutlass_cppgen.DataType.bf16,
                            element_output=cutlass_cppgen.DataType.bf16,
                            element_accumulator=cutlass_cppgen.DataType.bf16)

add_test_f32_acc(layouts=LayoutCombination.TTT,
                 threadblock_shape=[256, 256, 32], warp_count=[8, 4, 1])

add_test_f32_acc(layouts=LayoutCombination.TTT,
                 threadblock_shape=[128, 512, 32], warp_count=[4, 8, 1])

add_test_f32_acc(layouts=LayoutCombination.TTT,
                 threadblock_shape=[256, 128, 32], warp_count=[8, 4, 1])

add_test_f32_acc(layouts=LayoutCombination.TTT,
                 threadblock_shape=[128, 256, 16], warp_count=[4, 8, 1])

add_test_bf16_acc(layouts=LayoutCombination.TTT,
                  threadblock_shape=[256, 256, 32], warp_count=[8, 4, 1])

add_test_bf16_acc(layouts=LayoutCombination.TTT,
                  threadblock_shape=[128, 512, 32], warp_count=[4, 8, 1])

add_test_bf16_acc(layouts=LayoutCombination.TTT,
                  threadblock_shape=[256, 128, 32], warp_count=[8, 4, 1])

add_test_bf16_acc(layouts=LayoutCombination.TTT,
                  threadblock_shape=[128, 256, 16], warp_count=[4, 8, 1])


# TODO: Test more configurations as soon as they're supported by the
# CollectiveBuilder

if __name__ == '__main__':
    unittest.main()
