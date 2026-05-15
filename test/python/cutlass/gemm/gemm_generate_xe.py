#################################################################################################
#
# Copyright (c) 2023 - 2026 Intel Corporation. All rights reserved.
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

import json
import os
from argparse import Namespace
import unittest
from unittest import mock
from torch.utils import _pytree as pytree

from cutlass_library.arch_constants import (INTEL_XE12, INTEL_XE20)
import cutlass_library.generator as cutlass_generator
import cutlass_library.manifest as cutlass_manifest
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class GenerateXe(unittest.TestCase):
    def _run_generate_xe(self, arch_name, arch_const, reference_file):
        args = {
            "operations": "all",
            "build_dir": "",
            "curr_build_dir": ".",
            "generator_target": "library",
            "architectures": arch_name,
            "kernels": "",
            "ignore_kernels": "",
            "exclude_kernels": "",
            "filter_by_cc": "True",
            "cuda_version": "11.0.0",
            "kernel_filter_file": None,
            "heuristics_configs_per_problem": 10,
            "heuristics_restrict_kernels": False,
            "disable_full_archs_compilation": False,
            "instantiation_level": "",
            "disable_cutlass_package_imports": False
        }
        manifest = cutlass_manifest.Manifest(Namespace(**args))
        try:
            cutlass_generator.GenerateIntelXe(manifest, cuda_version="_", arch=arch_const)
        except AttributeError as e:
            raise NotImplementedError(
                f"Arch {arch_name} is not supported by current cutlass lib."
            ) from e
        xe_ops = pytree.tree_flatten(manifest.operations)[0]

        # Verify BF16 and F16 configurations have same number of non-StreamK ops.
        # StreamK ops may differ because bf16 accumulator doesn't support SYCL atomics
        # needed by BlockStripedReduce, while f16 accumulator does.
        bf16_operations = []
        f16_operations = []
        bf16_non_sk = []
        f16_non_sk = []
        for op in xe_ops:
            if "_bf16_" in op._procedural_name:
                bf16_operations.append(op._procedural_name)
                if "stream_k" not in op._procedural_name:
                    bf16_non_sk.append(op._procedural_name)
            if "_f16_" in op._procedural_name:
                f16_operations.append(op._procedural_name)
                if "stream_k" not in op._procedural_name:
                    f16_non_sk.append(op._procedural_name)

        assert len(bf16_non_sk) == len(f16_non_sk), f"{arch_name.upper()}: Number of non-StreamK bf16 and f16 operations should be the same"

        # Verify all generated ops against reference
        with open(os.path.join(DIR_PATH, reference_file), "r") as f:
            reference_data = json.load(f)
        total_ops = bf16_operations + f16_operations
        assert reference_data == total_ops, f"{arch_name.upper()}: Generated operations do not match reference data"

    def test_generate_xe12(self):
        self._run_generate_xe("pvc", INTEL_XE12, "data/generated_xe12_ops.json")

    def test_generate_xe20(self):
        self._run_generate_xe("bmg", INTEL_XE20, "data/generated_xe20_ops.json")

    @mock.patch.dict(os.environ, {"SYCL_TLA_ADDITIONAL_TILE_SHAPES": os.path.join(DIR_PATH, "data/custom_tile_shape.json")})
    def test_generate_xe20_with_custom_shapes(self):
        self._run_generate_xe("bmg", INTEL_XE20, "data/custom_generated_xe20_ops.json")

if __name__ == "__main__":
    unittest.main()
