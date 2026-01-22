################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
################################################################################

"""
Unit test for load nodes in SM90
"""

import logging
import unittest

import cutlass_cppgen
from cutlass_cppgen.backend import *
from cutlass_cppgen.epilogue import *

from utils.evt_testbed import EVTTestBed, EVTTestCaseBase

cutlass_cppgen.set_log_level(logging.WARNING)


@unittest.skipIf(device_cc() not in [12, 20, 80, 86, 89, 90], "This unittest is only supported on CC [12, 20, 80, 86, 89, 90]")
class TestEVTLoad(EVTTestCaseBase):

    def test_tensor_load(self):
        """
        Load extra tensor with shape [m, n]
        """
        def evt_tensor_load(accum, C, aux, aux_batch):
            D = accum + C + aux + aux_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux": self.fake_tensor(self.element, (m, n)),
                "aux_batch": self.fake_tensor(np.float32, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_tensor_load, example_inputs)
            input_keys = ["C", "aux", "aux_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_row_broadcast(self):
        """
        Load extra tensor with shape [1, n]
        """
        def evt_row_broadcast(accum, C, bias, bias_batch):
            D = accum + C + bias + bias_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "bias": self.fake_tensor(self.element, (n,)),
                "bias_batch": self.fake_tensor(np.float32, (l, 1, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_row_broadcast, example_inputs)
            input_keys = ["C", "bias", "bias_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)


    def test_column_broadcast(self):
        """
        Load extra tensor with shape [m, 1]
        """
        def evt_column_broadcast(accum, C, bias, bias_batch):
            D = accum + C + bias + bias_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "bias": self.fake_tensor(self.element, (m, 1)),
                "bias_batch": self.fake_tensor(np.float32, (l, m, 1)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_column_broadcast, example_inputs)
            input_keys = ["C", "bias", "bias_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_scalar_broadcast(self):
        """
        Load extra tensor with shape [1, 1]
        """
        def evt_scalar_broadcast(accum, C, alpha, alpha_batch):
            D = accum + C + alpha + alpha_batch
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 0.5,
                "alpha_batch": self.fake_tensor(np.float32, (l, 1, 1)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_scalar_broadcast, example_inputs)
            input_keys = ["C", "alpha", "alpha_batch"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_mixed_broadcast_combinations(self):
        """
        Test combinations of different broadcast patterns
        """
        def evt_mixed_broadcast(accum, C, row_bias, col_bias, scalar_alpha):
            D = accum + C + row_bias + col_bias + scalar_alpha
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "row_bias": self.fake_tensor(self.element, (n,)),
                "col_bias": self.fake_tensor(self.element, (m, 1)),
                "scalar_alpha": 0.25,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_mixed_broadcast, example_inputs)
            input_keys = ["C", "row_bias", "col_bias", "scalar_alpha"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_multiple_tensor_loads(self):
        """
        Test loading multiple full tensors with different operations
        """
        def evt_multiple_tensors(accum, C, aux1, aux2, aux3):
            temp = accum + C + aux1
            D = temp * aux2 - aux3
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux1": self.fake_tensor(self.element, (l, m, n)),
                "aux2": self.fake_tensor(self.element, (l, m, n)),
                "aux3": self.fake_tensor(self.element, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_multiple_tensors, example_inputs)
            input_keys = ["C", "aux1", "aux2", "aux3"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_batch_and_broadcast_combination(self):
        """
        Test combination of batched tensors with different broadcast patterns
        """
        def evt_batch_broadcast_combo(accum, C, batch_tensor, row_bias, scalar_alpha):
            D = accum + C + batch_tensor + row_bias + scalar_alpha
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "batch_tensor": self.fake_tensor(np.float32, (l, m, n)),
                "row_bias": self.fake_tensor(self.element, (n,)),
                "scalar_alpha": 1.5,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_batch_broadcast_combo, example_inputs)
            input_keys = ["C", "batch_tensor", "row_bias", "scalar_alpha"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_complex_arithmetic_combination(self):
        """
        Test complex arithmetic operations with multiple load patterns
        """
        def evt_complex_arithmetic(accum, C, row_scale, col_bias, batch_offset):
            scaled = accum * row_scale
            biased = scaled + col_bias
            D = biased + C + batch_offset
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "row_scale": self.fake_tensor(self.element, (n,)),
                "col_bias": self.fake_tensor(self.element, (m, 1)),
                "batch_offset": self.fake_tensor(np.float32, (l, 1, 1)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_complex_arithmetic, example_inputs)
            input_keys = ["C", "row_scale", "col_bias", "batch_offset"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_all_broadcast_patterns_combined(self):
        """
        Test all broadcast patterns in a single operation
        """
        def evt_all_broadcasts(accum, C, full_tensor, row_bias, col_bias, scalar_alpha, batch_tensor):
            D = accum + C + full_tensor + row_bias + col_bias + scalar_alpha + batch_tensor
            return D

        for m, n, k, l in self.get_problem_sizes(4):  # Reduced iterations due to complexity
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "full_tensor": self.fake_tensor(self.element, (m, n)),
                "row_bias": self.fake_tensor(self.element, (n,)),
                "col_bias": self.fake_tensor(self.element, (m, 1)),
                "scalar_alpha": 0.1,
                "batch_tensor": self.fake_tensor(np.float32, (l, m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_all_broadcasts, example_inputs)
            input_keys = ["C", "full_tensor", "row_bias", "col_bias", "scalar_alpha", "batch_tensor"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)
    def test_scalar_and_tensor_load_combination(self):
        """
        Combination of scalar broadcast and tensor load
        D = accum + C + aux_tensor + scalar_alpha
        """
        def evt_scalar_tensor_combo(accum, C, aux_tensor, scalar_alpha):
            D = accum + C + aux_tensor + scalar_alpha
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux_tensor": self.fake_tensor(self.element, (m, n)),
                "scalar_alpha": 0.5,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_scalar_tensor_combo, example_inputs)
            input_keys = ["C", "aux_tensor", "scalar_alpha"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_tensor_multiplication_with_scalar(self):
        """
        Tensor load with scalar multiplication
        D = (accum + C) * aux_tensor * scalar_scale
        """
        def evt_tensor_mult_scalar(accum, C, aux_tensor, scalar_scale):
            temp = accum + C
            D = temp * aux_tensor * scalar_scale
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux_tensor": self.fake_tensor(self.element, (m, n)),
                "scalar_scale": 2.0,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_tensor_mult_scalar, example_inputs)
            input_keys = ["C", "aux_tensor", "scalar_scale"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_batched_tensor_with_scalar_operations(self):
        """
        Batched tensor load combined with scalar operations
        D = (accum + aux_tensor) * alpha + aux_batch * beta
        """
        def evt_batched_tensor_scalar(accum, C, aux_tensor, aux_batch, alpha, beta):
            temp = accum + aux_tensor
            D = temp * alpha + C + aux_batch * beta
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux_tensor": self.fake_tensor(self.element, (m, n)),
                "aux_batch": self.fake_tensor(np.float32, (l, m, n)),
                "alpha": 1.3,
                "beta": 0.6,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_batched_tensor_scalar, example_inputs)
            input_keys = ["C", "aux_tensor", "aux_batch", "alpha", "beta"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_tensor_scalar_division_combination(self):
        """
        Division operations with tensor loads and scalars
        D = (accum + aux_tensor) / divisor + C * multiplier
        """
        def evt_tensor_scalar_division(accum, C, aux_tensor, divisor, multiplier):
            temp = accum + aux_tensor
            D = temp / divisor + C * multiplier
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux_tensor": self.fake_tensor(self.element, (m, n)),
                "divisor": 2.0,
                "multiplier": 1.5,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_tensor_scalar_division, example_inputs)
            input_keys = ["C", "aux_tensor", "divisor", "multiplier"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_complex_scalar_tensor_expression(self):
        """
        Complex expression combining multiple scalars and tensor loads
        D = (accum * scale1 + aux1) * scale2 + (C + aux2) * scale3
        """
        def evt_complex_scalar_tensor(accum, C, aux1, aux2, scale1, scale2, scale3):
            temp1 = accum * scale1 + aux1
            temp2 = C + aux2
            D = temp1 * scale2 + temp2 * scale3
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux1": self.fake_tensor(self.element, (m, n)),
                "aux2": self.fake_tensor(self.element, (m, n)),
                "scale1": 0.8,
                "scale2": 1.2,
                "scale3": 0.9,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_complex_scalar_tensor, example_inputs)
            input_keys = ["C", "aux1", "aux2", "scale1", "scale2", "scale3"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_multiple_scalars_with_tensor_load_corrected(self):
        """
        CORRECTED: Multiple scalar broadcasts combined with tensor load
        D = accum * alpha + C * beta + aux_tensor + gamma
        """
        def evt_multi_scalar_tensor(accum, C, aux_tensor, alpha, beta, gamma):
            temp1 = accum * alpha
            temp2 = C * beta
            temp3 = aux_tensor + gamma  # Add scalar to tensor first
            D = temp1 + temp2 + temp3
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux_tensor": self.fake_tensor(self.element, (m, n)),  # Full tensor, not broadcast
                "alpha": 1.2,
                "beta": 0.8,
                "gamma": 0.3,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_multi_scalar_tensor, example_inputs)
            input_keys = ["C", "aux_tensor", "alpha", "beta", "gamma"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_multiple_tensor_loads_with_scalars_corrected(self):
        """
        CORRECTED: Multiple tensor loads combined with scalar operations
        D = (aux1 * scalar1) + (aux2 * scalar2) + accum + C
        """
        def evt_multi_tensor_scalar(accum, C, aux1, aux2, scalar1, scalar2):
            scaled1 = aux1 * scalar1
            scaled2 = aux2 * scalar2
            D = scaled1 + scaled2 + accum + C
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux1": self.fake_tensor(self.element, (m, n)),  # Full tensor
                "aux2": self.fake_tensor(self.element, (m, n)),  # Full tensor
                "scalar1": 1.5,
                "scalar2": 0.7,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_multi_tensor_scalar, example_inputs)
            input_keys = ["C", "aux1", "aux2", "scalar1", "scalar2"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_scalar_tensor_arithmetic_chain_corrected(self):
        """
        CORRECTED: Chain of arithmetic operations with scalars and tensors
        temp1 = accum * alpha, temp2 = aux_tensor + beta, D = temp1 + C + temp2
        """
        def evt_scalar_tensor_chain(accum, C, aux_tensor, alpha, beta):
            temp1 = accum * alpha
            temp2 = aux_tensor + beta
            D = temp1 + C + temp2
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "aux_tensor": self.fake_tensor(self.element, (m, n)),  # Full tensor
                "alpha": 0.9,
                "beta": 0.1,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_scalar_tensor_chain, example_inputs)
            input_keys = ["C", "aux_tensor", "alpha", "beta"]
            result_keys = ["D"]  # ← FIXED: Added missing result_keys
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_simple_scalar_arithmetic(self):
        """
        NEW: Simple scalar arithmetic that should work
        D = accum * alpha + C + beta
        """
        def evt_simple_scalar_arithmetic(accum, C, alpha, beta):
            temp = accum * alpha
            D = temp + C + beta
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 1.5,
                "beta": 0.5,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_simple_scalar_arithmetic, example_inputs)
            input_keys = ["C", "alpha", "beta"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_scalar_only_operations(self):
        """
        NEW: Test with only scalar operations
        D = accum * scale + offset
        """
        def evt_scalar_only(accum, C, scale, offset):
            D = accum * scale + C + offset
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "scale": 2.0,
                "offset": 1.0,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_scalar_only, example_inputs)
            input_keys = ["C", "scale", "offset"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_multiple_aux_tensors_different_types(self):
            """
            Load multiple auxiliary tensors with different element types
            Tests type conversion and multiple tensor loads
            """
            def evt_multiple_aux_tensors(accum, C, aux_fp32, aux_batch, scalar_scale):
                D = accum * scalar_scale + C + aux_fp32 + aux_batch
                return D

            for m, n, k, l in self.get_problem_sizes(8):
                example_inputs = {
                    "accum": self.fake_tensor(self.element, (l, m, n)),
                    "C": self.fake_tensor(self.element, (l, m, n)),
                    "aux_fp32": self.fake_tensor(np.float32, (m, n)),
                    "aux_batch": self.fake_tensor(self.element, (l, m, n)),
                    "scalar_scale": 0.25,
                    "D": self.fake_tensor(self.element, (l, m, n)),
                }

                launcher = EVTTestBed(self.element, evt_multiple_aux_tensors, example_inputs)
                input_keys = ["C", "aux_fp32", "aux_batch", "scalar_scale"]
                result_keys = ["D"]
                launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_tensor_load_with_multiply(self):
        """
        Load full auxiliary tensor and use with multiplication operations
        Tests tensor load with element-wise multiply instead of just addition
        """
        def evt_tensor_load_multiply(accum, C, scale_matrix, offset_matrix):
            D = (accum + C) * scale_matrix + offset_matrix
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "scale_matrix": self.fake_tensor(self.element, (m, n)),
                "offset_matrix": self.fake_tensor(self.element, (m, n)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_tensor_load_multiply, example_inputs)
            input_keys = ["C", "scale_matrix", "offset_matrix"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_multiple_scalar_broadcast(self):
        """
        Load and broadcast multiple scalar values with different operations
        Tests scalar broadcast with multiplication, addition, and division
        """
        def evt_multiple_scalars(accum, C, alpha, beta, gamma):
            D = (accum * alpha + C * beta) / gamma
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "alpha": 1.5,
                "beta": 2.0,
                "gamma": 0.5,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_multiple_scalars, example_inputs)
            input_keys = ["C", "alpha", "beta", "gamma"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_tensor_load_fused_activation(self):
        """
        Load auxiliary tensor with activation-like operations
        Tests tensor load with max operation (ReLU-like pattern)
        """
        def evt_tensor_load_activation(accum, C, bias_matrix, threshold):
            temp = accum + C + bias_matrix
            D = maximum(temp, threshold)
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "bias_matrix": self.fake_tensor(self.element, (m, n)),
                "threshold": 0.0,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_tensor_load_activation, example_inputs)
            input_keys = ["C", "bias_matrix", "threshold"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

    def test_scalar_broadcast_complex_expression(self):
        """
        Scalar broadcast in complex nested expression
        Tests scalar values used in multiple sub-expressions
        """
        def evt_scalar_complex(accum, C, scale1, scale2, offset):
            temp1 = accum * scale1
            temp2 = C * scale2
            D = (temp1 + temp2) * offset
            return D

        for m, n, k, l in self.get_problem_sizes(8):
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "C": self.fake_tensor(self.element, (l, m, n)),
                "scale1": 0.8,
                "scale2": 1.2,
                "offset": 0.5,
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_scalar_complex, example_inputs)
            input_keys = ["C", "scale1", "scale2", "offset"]
            result_keys = ["D"]
            launcher.verify((m, n, k), input_keys, result_keys, l)

if __name__ == '__main__':
    unittest.main()
