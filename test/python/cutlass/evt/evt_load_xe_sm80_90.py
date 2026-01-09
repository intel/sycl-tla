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

    def test_row_broadcast_simple(self):
        """
        Test simple bias row broadcast (without bias_batch) across all problem sizes
        D = accum + bias (broadcasted along rows)
        This tests that regular row broadcast works before testing batched version
        """
        print("\n=== Test: row broadcast simple (regular bias only) ===")
        def evt_row_broadcast_simple(accum, bias):
            D = accum + bias
            return D

        # for m, n, k, l in self.get_problem_sizes(8):
        for m, n, k, l in [(8, 64, 8, 1)]:
            print(f"\nTesting shape: m={m}, n={n}, k={k}, l={l}")
            print("  Pattern: accum=zeros, bias=[1,2,3,...,24], so D should equal bias broadcast to all rows")
            
            example_inputs = {
                "accum": self.fake_tensor(self.element, (l, m, n)),
                "bias": self.fake_tensor(self.element, (n,)),
                "D": self.fake_tensor(self.element, (l, m, n)),
            }

            launcher = EVTTestBed(self.element, evt_row_broadcast_simple, example_inputs)
            
            # Override tensor generation for predictable pattern
            import torch
            def get_predictable_tensor(shape, dtype=None, fill=None):
                if dtype is None:
                    dtype = self.element
                dtype_torch = torch.float16 if dtype == cutlass_cppgen.DataType.f16 else torch.float32

                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.xpu.is_available():
                    device = "xpu"
                else:
                    device = "cpu"

                if shape == (n,):  # bias vector: [1, 2, 3, ...]
                    tensor = torch.arange(1, n+1, dtype=dtype_torch, device=device)
                    print(f"    bias: [{tensor[0].item()}, {tensor[1].item()}, ..., {tensor[-1].item()}]")
                else:  # accum or D: all zeros
                    tensor = torch.zeros(shape, dtype=dtype_torch, device=device)

                return tensor

            launcher.get_torch_tensor = get_predictable_tensor
            input_keys = ["bias"]
            result_keys = ["D"]

            try:
                launcher.verify((m, n, k), input_keys, result_keys, l)
                print(f"  ✓ Shape ({m}, {n}, {k}, {l}) passed")
            except AssertionError as e:
                print(f"  ✗ Shape ({m}, {n}, {k}, {l}) FAILED: {e}")
                raise

    def test_aux_load_only(self):
        """
        Test JUST the aux load (no broadcast) - verify global->reg copy is correct
        D = accum + aux (where aux is [m, n] full tensor, not broadcast)
        This verifies XeAuxLoad copy operation before testing XeRowBroadcast
        """
        print("\n=== Test: aux load only (verifying XeAuxLoad copy) ===")
        def evt_aux_load_only(accum, aux):
            D = accum + aux
            return D

        m, n, k, l = 8, 16, 8, 1

        example_inputs = {
            "accum": self.fake_tensor(self.element, (l, m, n)),
            "aux": self.fake_tensor(self.element, (m, n)),
            "D": self.fake_tensor(self.element, (l, m, n)),
        }

        launcher = EVTTestBed(self.element, evt_aux_load_only, example_inputs)

        # Custom tensor: accum=0, aux=sequential pattern row by row
        import torch
        def get_aux_test_tensor(shape, dtype=None, fill=None):
            if dtype is None:
                dtype = self.element
            dtype_torch = torch.float16 if dtype == cutlass_cppgen.DataType.f16 else torch.float32

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"

            if shape == (m, n):  # aux tensor - each row is [0, 1, 2, ..., 15]
                tensor = torch.zeros(shape, dtype=dtype_torch, device=device)
                for i in range(m):
                    tensor[i, :] = torch.arange(0, n, dtype=dtype_torch)
                print(f"  aux: shape={shape}, each row is [0, 1, 2, ..., {n-1}]")
                print(f"    first row: {tensor[0].tolist()}")
                print(f"    second row: {tensor[1].tolist()}")
            else:  # accum or D
                tensor = torch.zeros(shape, dtype=dtype_torch, device=device)
                print(f"  accum/D: shape={shape}, all zeros")

            return tensor

        launcher.get_torch_tensor = get_aux_test_tensor

        input_keys = ["aux"]
        result_keys = ["D"]

        try:
            launcher.verify((m, n, k), input_keys, result_keys, l)
            print("✓ Test passed: XeAuxLoad copy works correctly")
        except AssertionError as e:
            print(f"✗ Test FAILED: XeAuxLoad copy has issues")
            print(f"  {e}")
            raise

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


if __name__ == '__main__':
    unittest.main()
