#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
from pathlib import Path

from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
from .utils import now_iso, read_json


DEFAULT_KERNEL_CATALOG_PATH = Path(__file__).resolve().parents[1] / "intel_gemm_kernel_catalog_level0.json"

SEED_KERNELS = {
    "bf16": [
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_5", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_7", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 8, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_9", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_16", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 16, "tile_n": 64, "tile_k": 32, "sg_m": 2, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_17", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 64, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_18", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_19", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 256, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmBF16BF16FP32_RCR_6", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "03_bmg_gemm_streamk_streamk_bf16", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "streamk"},
        {"kernel_name": "03_bmg_gemm_streamk_dp_bf16", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "data_parallel"},
        {"kernel_name": "03_bmg_gemm_streamk_splitk_bf16", "layout": "rcr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 2, "runner": "streamk_example", "streamk_mode": "splitk"},
    ],
    "f16": [
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_5", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_7", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 128, "tile_k": 32, "sg_m": 1, "sg_n": 8, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_9", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 16, "tile_n": 64, "tile_k": 32, "sg_m": 2, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_17", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 64, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_18", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 128, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_19", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 128, "tile_n": 256, "tile_k": 32, "sg_m": 4, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "BmgGemmFP16FP16FP32_RCR_6", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1},
        {"kernel_name": "03_bmg_gemm_streamk_streamk_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "streamk"},
        {"kernel_name": "03_bmg_gemm_streamk_dp_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "data_parallel"},
        {"kernel_name": "03_bmg_gemm_streamk_splitk_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 8, "tile_n": 64, "tile_k": 32, "sg_m": 1, "sg_n": 4, "stages": 2, "split_k": 2, "runner": "streamk_example", "streamk_mode": "splitk"},
    ],
}


def ilp_class(seed):
    ilp = (seed["tile_m"] // max(seed["sg_m"], 1) // 8) * (seed["tile_n"] // max(seed["sg_n"], 1) // 16)
    if ilp >= 16:
        return "ilp16"
    if ilp >= 8:
        return "ilp8"
    return "ilp4"


def kernel_catalog_entry(dtype, seed):
    entry = copy.deepcopy(seed)
    entry.setdefault("runner", "benchmark")
    entry["kernel_id"] = seed["kernel_name"]
    entry["instantiation_level"] = 0
    entry["benchmark_target"] = "cutlass_benchmarks_gemm_sycl" if entry["runner"] == "benchmark" else "03_bmg_gemm_streamk"
    entry["grf_mode"] = 256
    entry["ilp_class"] = ilp_class(entry)
    entry["streamk_mode"] = entry.get("streamk_mode", "")
    entry["runtime_defaults"] = {}
    entry["allowed_runtime_sweeps"] = ["shape_id", "m", "n", "k"]
    entry["source"] = "seed_catalog_level0"
    entry["dtype_family"] = dtype
    return entry


def generated_level0_kernel_catalog():
    catalog = []
    for dtype in sorted(SEED_KERNELS.keys()):
        for seed in SEED_KERNELS.get(dtype, []):
            catalog.append(kernel_catalog_entry(dtype, seed))
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": "level0-seed-catalog",
        "instantiation_levels": {
            "0": "existing validated benchmark-backed kernels",
            "1": "expanded tile and subgroup layouts",
            "2": "full autotuning catalog including copy/epilogue variants",
        },
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "kernels": catalog,
    }


def load_persisted_kernel_catalog(path=DEFAULT_KERNEL_CATALOG_PATH):
    if path.exists():
        return read_json(path)
    return generated_level0_kernel_catalog()


def build_kernel_catalog(dtypes=None, allowed_runners=("benchmark",), catalog_path=DEFAULT_KERNEL_CATALOG_PATH):
    source_catalog = load_persisted_kernel_catalog(catalog_path)
    selected_dtypes = set(dtypes) if dtypes is not None else None
    catalog = []
    for entry in source_catalog["kernels"]:
        if selected_dtypes is not None and entry["dtype_a"] not in selected_dtypes:
            continue
        if entry["runner"] not in allowed_runners:
            continue
        catalog.append(copy.deepcopy(entry))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "catalog_version": source_catalog["catalog_version"],
        "instantiation_levels": source_catalog["instantiation_levels"],
        "search_runtime_schema": source_catalog.get("search_runtime_schema", SEARCH_RUNTIME_SCHEMA),
        "kernels": catalog,
    }
