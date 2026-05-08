#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import re


SCHEMA_VERSION = "1.0"

SEARCH_RUNTIME_SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "search_space_version": "2026-04-29",
    "compile_time_dimensions": [
        "dtype_a",
        "dtype_b",
        "dtype_c",
        "dtype_d",
        "dtype_acc",
        "layout",
        "tile_m",
        "tile_n",
        "tile_k",
        "sg_m",
        "sg_n",
        "stages",
        "split_k",
        "streamk_mode",
        "grf_mode",
        "mma_atom",
        "gmem_copy_atom_a",
        "gmem_copy_atom_b",
        "epilogue_op",
        "epilogue_tile",
        "epilogue_copy_atom_c",
        "epilogue_copy_atom_d",
        "ilp_class",
        "instantiation_level",
        "runner",
        "benchmark_target",
    ],
    "runtime_dimensions": [
        "shape_id",
        "m",
        "n",
        "k",
        "batch_count",
    ],
    "pruning_inputs": [
        "dpas_alignment",
        "safe_search_constraints",
        "phase_a_probe_results",
        "slm_limit_kb",
        "split_k_support",
    ],
    "microbench_guided_defaults": {
        "grf_mode": 256,
    },
}

CSV_FIELDS = [
    "run_id",
    "stage",
    "attempt_index",
    "shape_id",
    "candidate_id",
    "compiler_profile_id",
    "status",
    "verify_status",
    "layout",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_d",
    "dtype_acc",
    "m",
    "n",
    "k",
    "batch_count",
    "split_k",
    "avg_runtime_ms",
    "best_runtime_ms",
    "worst_runtime_ms",
    "avg_tflops",
    "avg_throughput",
    "max_error",
    "close_call_group",
    "failure_reason",
    "stdout_log",
]

BENCHMARK_ERROR_RE = re.compile(r"(^ERROR\b|\bERROR OCCURRED\b|Disposition Failed)")
