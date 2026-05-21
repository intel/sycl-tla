#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import re


SCHEMA_VERSION = "1.1"

SCHEDULER_METADATA_FIELDS = [
    "operator_family",
    "mainloop_dispatch_policy",
    "kernel_schedule",
    "tile_scheduler",
    "decomposition_mode",
    "epilogue_dispatch_policy",
    "example_family",
    "padding_mode",
    "activation",
    "bias_mode",
    "quant_mode",
    "scale_mode",
]

DEFAULT_SCHEDULER_METADATA = {
    "operator_family": "gemm",
    "mainloop_dispatch_policy": "MainloopXeL1Staged",
    "kernel_schedule": "KernelXe",
    "tile_scheduler": "Gemm",
    "decomposition_mode": "Gemm",
    "epilogue_dispatch_policy": "IntelXeGeneric",
    "example_family": "",
    "padding_mode": "",
    "activation": "",
    "bias_mode": "",
    "quant_mode": "",
    "scale_mode": "",
}

EPILOGUE_METADATA_FIELDS = [
    "element_output_epilogue",
    "element_compute_epilogue",
    "element_source_epilogue",
    "element_scalar_epilogue",
    "epilogue_op",
    "epilogue_tile",
    "epilogue_copy_atom_c",
    "epilogue_copy_atom_d",
]

STREAMK_EXAMPLE_SCHEDULER_METADATA = {
    "kernel_schedule": "KernelXeCooperative",
    "tile_scheduler": "StreamKScheduler",
    "example_family": "03_bmg_gemm_streamk",
}

RESULT_METADATA_FIELDS = [
    "runner",
    "benchmark_target",
    "streamk_mode",
    "streamk_dtype_preset",
    *SCHEDULER_METADATA_FIELDS,
    "support_status",
    "support_reason",
    "mma_atom",
    "gmem_copy_atom_a",
    "gmem_copy_atom_b",
    *EPILOGUE_METADATA_FIELDS,
]

REPORT_TRACKED_DIMENSIONS = [
    "runner",
    "benchmark_target",
    "operator_family",
    "layout",
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_d",
    "dtype_acc",
    "streamk_mode",
    "streamk_dtype_preset",
    "support_status",
    "support_reason",
    "split_k",
    "mainloop_dispatch_policy",
    "kernel_schedule",
    "tile_scheduler",
    "decomposition_mode",
    "epilogue_dispatch_policy",
    "element_output_epilogue",
    "element_compute_epilogue",
    "element_source_epilogue",
    "element_scalar_epilogue",
    "epilogue_op",
    "activation",
    "bias_mode",
    "quant_mode",
    "scale_mode",
    "mma_atom",
    "gmem_copy_atom_a",
    "gmem_copy_atom_b",
    "epilogue_copy_atom_c",
    "epilogue_copy_atom_d",
]

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
        "streamk_dtype_preset",
        "operator_family",
        "mainloop_dispatch_policy",
        "kernel_schedule",
        "tile_scheduler",
        "decomposition_mode",
        "epilogue_dispatch_policy",
        "element_output_epilogue",
        "element_compute_epilogue",
        "element_source_epilogue",
        "element_scalar_epilogue",
        "example_family",
        "padding_mode",
        "activation",
        "bias_mode",
        "quant_mode",
        "scale_mode",
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
    "runner",
    "benchmark_target",
    "streamk_mode",
    "streamk_dtype_preset",
    "support_status",
    "support_reason",
    "operator_family",
    "mainloop_dispatch_policy",
    "kernel_schedule",
    "tile_scheduler",
    "decomposition_mode",
    "epilogue_dispatch_policy",
    "example_family",
    "padding_mode",
    "activation",
    "bias_mode",
    "quant_mode",
    "scale_mode",
    "mma_atom",
    "gmem_copy_atom_a",
    "gmem_copy_atom_b",
    "element_output_epilogue",
    "element_compute_epilogue",
    "element_source_epilogue",
    "element_scalar_epilogue",
    "epilogue_op",
    "epilogue_tile",
    "epilogue_copy_atom_c",
    "epilogue_copy_atom_d",
    "avg_runtime_ms",
    "best_runtime_ms",
    "worst_runtime_ms",
    "runtime_median_ms",
    "runtime_stddev_ms",
    "warmup_iters",
    "measure_iters",
    "avg_tflops",
    "median_tflops",
    "avg_throughput",
    "max_error",
    "close_call_group",
    "failure_reason",
    "stdout_log",
]

BENCHMARK_ERROR_RE = re.compile(r"(^ERROR\b|\bERROR OCCURRED\b|Disposition Failed)")


def streamk_decomposition_mode(streamk_mode):
    return {
        "streamk": "StreamK",
        "data_parallel": "DataParallel",
        "splitk": "SplitK",
    }.get(streamk_mode, "Gemm")


def infer_epilogue_metadata(entry):
    dtype_c = entry.get("dtype_c", "")
    dtype_d = entry.get("dtype_d", dtype_c)
    dtype_acc = entry.get("dtype_acc", "")
    return {
        "element_output_epilogue": dtype_d,
        "element_compute_epilogue": dtype_acc,
        "element_source_epilogue": dtype_c,
        "element_scalar_epilogue": dtype_acc,
    }
