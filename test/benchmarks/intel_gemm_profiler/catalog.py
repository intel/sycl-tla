#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import sys
from pathlib import Path

from .schemas import SCHEMA_VERSION, SEARCH_RUNTIME_SCHEMA
from .utils import now_iso, read_json


DEFAULT_KERNEL_CATALOG_PATH = Path(__file__).resolve().parents[1] / "intel_gemm_kernel_catalog_level0.json"

SEED_KERNELS = {
    "bf16": [
        {"kernel_name": "BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32", "layout": "rrr", "dtype_a": "bf16", "dtype_b": "bf16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 512, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1},
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
    entry.setdefault("dtype_d", entry["dtype_c"])
    entry.setdefault("runner", "benchmark")
    entry["kernel_id"] = seed["kernel_name"]
    entry["instantiation_level"] = 0
    entry["benchmark_target"] = "cutlass_benchmarks_gemm_sycl" if entry["runner"] == "benchmark" else "03_bmg_gemm_streamk"
    entry["grf_mode"] = 256
    entry["ilp_class"] = ilp_class(entry)
    entry["streamk_mode"] = entry.get("streamk_mode", "")
    entry["batch_count"] = 1
    entry["runtime_defaults"] = {}
    entry["allowed_runtime_sweeps"] = ["shape_id", "m", "n", "k", "batch_count"]
    entry["source"] = "seed_catalog_level0"
    entry["dtype_family"] = dtype
    entry.setdefault("mma_atom", "XE_DPAS_TT")
    entry.setdefault("gmem_copy_atom_a", "auto")
    entry.setdefault("gmem_copy_atom_b", "auto")
    entry.setdefault("epilogue_op", "LinearCombination")
    entry.setdefault("epilogue_tile", "auto")
    entry.setdefault("epilogue_copy_atom_c", "auto")
    entry.setdefault("epilogue_copy_atom_d", "auto")
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
    path = path or DEFAULT_KERNEL_CATALOG_PATH
    if path.exists():
        catalog = read_json(path)
        catalog["search_runtime_schema"] = SEARCH_RUNTIME_SCHEMA
        for entry in catalog.get("kernels", []):
            entry.setdefault("dtype_d", entry["dtype_c"])
            entry.setdefault("batch_count", 1)
            entry.setdefault("allowed_runtime_sweeps", ["shape_id", "m", "n", "k", "batch_count"])
            if "batch_count" not in entry["allowed_runtime_sweeps"]:
                entry["allowed_runtime_sweeps"].append("batch_count")
            entry.setdefault("mma_atom", "XE_DPAS_TT")
            entry.setdefault("gmem_copy_atom_a", "auto")
            entry.setdefault("gmem_copy_atom_b", "auto")
            entry.setdefault("epilogue_op", "LinearCombination")
            entry.setdefault("epilogue_tile", "auto")
            entry.setdefault("epilogue_copy_atom_c", "auto")
            entry.setdefault("epilogue_copy_atom_d", "auto")
        return catalog
    return generated_level0_kernel_catalog()


def _import_cutlass_generator_modules():
    try:
        from cutlass_library.generator import GenerateIntelXe
        from cutlass_library.library import DataTypeNames, LayoutType, TileSchedulerType
        from cutlass_library.manifest import Manifest, Options
    except ImportError:
        python_root = Path(__file__).resolve().parents[3] / "python"
        if str(python_root) not in sys.path:
            sys.path.insert(0, str(python_root))
        from cutlass_library.generator import GenerateIntelXe
        from cutlass_library.library import DataTypeNames, LayoutType, TileSchedulerType
        from cutlass_library.manifest import Manifest, Options
    return GenerateIntelXe, DataTypeNames, LayoutType, TileSchedulerType, Manifest, Options


def _generator_arch_details(generator_arch):
    arch_key = str(generator_arch).lower()
    arch_map = {
        "bmg": ("bmg", 20),
        "xe20": ("bmg", 20),
        "20": ("bmg", 20),
        "pvc": ("pvc", 12),
        "xe12": ("pvc", 12),
        "12": ("pvc", 12),
    }
    if arch_key not in arch_map:
        raise ValueError(f"Unsupported Intel Xe generator arch: {generator_arch}")
    return arch_map[arch_key]


def _generator_layout_name(layout_type):
    return "r" if layout_type.name.startswith("RowMajor") else "c"


def _generator_dtype_family(dtype_a):
    if dtype_a in {"bf16", "f16"}:
        return "16b"
    if dtype_a in {"e4m3", "e5m2"}:
        return "fp8"
    if dtype_a in {"s8"}:
        return "int8"
    return dtype_a


def _generator_kernel_catalog_entry(operation, data_type_names, tile_scheduler_type, instantiation_level):
    tile_shape = operation.tile_description.tile_shape
    sg_shape = operation.tile_description.warp_count
    streamk_mode = "streamk" if operation.tile_scheduler == tile_scheduler_type.StreamK else ""
    dtype_a = data_type_names[operation.A.element]
    dtype_b = data_type_names[operation.B.element]
    dtype_c = data_type_names[operation.C.element]
    dtype_d = data_type_names[getattr(operation, "D", operation.C).element]
    dtype_acc = data_type_names[operation.accumulator_type()]
    entry = {
        "kernel_name": operation.procedural_name(),
        "layout": "".join(
            _generator_layout_name(layout_type)
            for layout_type in (operation.A.layout, operation.B.layout, operation.C.layout)
        ),
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_c": dtype_c,
        "dtype_d": dtype_d,
        "dtype_acc": dtype_acc,
        "tile_m": tile_shape[0],
        "tile_n": tile_shape[1],
        "tile_k": tile_shape[2],
        "sg_m": sg_shape[0],
        "sg_n": sg_shape[1],
        "stages": int(operation.tile_description.stages),
        "split_k": 1,
        "runner": "benchmark",
        "kernel_id": operation.procedural_name(),
        "instantiation_level": instantiation_level,
        "benchmark_target": "cutlass_benchmarks_gemm_sycl",
        "grf_mode": 256,
        "streamk_mode": streamk_mode,
        "runtime_defaults": {},
        "allowed_runtime_sweeps": ["shape_id", "m", "n", "k", "batch_count"],
        "source": "generator_manifest",
        "mma_atom": "XE_DPAS_TT",
        "gmem_copy_atom_a": "auto",
        "gmem_copy_atom_b": "auto",
        "epilogue_op": "LinearCombination",
        "epilogue_tile": "auto",
        "epilogue_copy_atom_c": "auto",
        "epilogue_copy_atom_d": "auto",
    }
    entry["ilp_class"] = ilp_class(entry)
    entry["dtype_family"] = _generator_dtype_family(dtype_a)
    return entry


def generated_generator_kernel_catalog(generator_arch="bmg", generator_instantiation_level=0):
    GenerateIntelXe, DataTypeNames, _, TileSchedulerType, Manifest, Options = _import_cutlass_generator_modules()
    arch_name, arch_value = _generator_arch_details(generator_arch)
    args = Options()
    args.kernels = ""
    args.curr_build_dir = "."
    args.architectures = arch_name
    args.filter_by_cc = "true"
    args.operations = "gemm"
    args.ignore_kernels = ""
    args.exclude_kernels = ""
    args.kernel_filter_file = None
    args.disable_full_archs_compilation = False
    args.instantiation_level = str(generator_instantiation_level)
    manifest = Manifest(args)
    GenerateIntelXe(manifest, cuda_version=None, arch=arch_value)
    kernels = []
    for operation in sorted(manifest.operations_by_name.values(), key=lambda op: op.procedural_name()):
        kernels.append(
            _generator_kernel_catalog_entry(
                operation,
                DataTypeNames,
                TileSchedulerType,
                int(generator_instantiation_level),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": f"generator-{arch_name}-level{int(generator_instantiation_level)}",
        "instantiation_levels": {
            "0": "generator conservative Intel Xe catalog",
            "1": "expanded tile, stage, and scheduler Intel Xe catalog",
            "2": "generator-expanded Intel Xe catalog including fp8/int8 families",
        },
        "generator_arch": arch_name,
        "generator_instantiation_level": int(generator_instantiation_level),
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "kernels": kernels,
    }


def build_kernel_catalog(
    dtypes=None,
    allowed_runners=("benchmark",),
    catalog_path=DEFAULT_KERNEL_CATALOG_PATH,
    catalog_source="persisted",
    generator_arch="bmg",
    generator_instantiation_level=0,
):
    if catalog_source == "persisted":
        source_catalog = load_persisted_kernel_catalog(catalog_path)
    elif catalog_source == "generator":
        source_catalog = generated_generator_kernel_catalog(
            generator_arch=generator_arch,
            generator_instantiation_level=generator_instantiation_level,
        )
    else:
        raise ValueError(f"Unsupported kernel catalog source: {catalog_source}")
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
        "catalog_source": catalog_source,
        "generator_arch": source_catalog.get("generator_arch", ""),
        "generator_instantiation_level": source_catalog.get("generator_instantiation_level", 0),
        "search_runtime_schema": source_catalog.get("search_runtime_schema", SEARCH_RUNTIME_SCHEMA),
        "kernels": catalog,
    }
