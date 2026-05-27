#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import sys
from pathlib import Path

from .schemas import (
    DEFAULT_SCHEDULER_METADATA,
    SCHEMA_VERSION,
    SEARCH_RUNTIME_SCHEMA,
    STREAMK_EXAMPLE_SCHEDULER_METADATA,
    infer_epilogue_metadata,
    streamk_decomposition_mode,
)
from .source_templates import observed_bmg_template_space
from .source_templates import is_valid_xe2_tile_sg
from .utils import now_iso, read_json


DEFAULT_KERNEL_CATALOG_PATH = Path(__file__).resolve().parents[1] / "intel_gemm_kernel_catalog_level0.json"


STREAMK_TILE_SHAPES = [
    (64, 128, 32),
    (64, 256, 32),
    (128, 128, 32),
    (128, 256, 32),
    (256, 128, 32),
    (256, 256, 32),
    (512, 128, 32),
    (512, 256, 32),
]
EXPANDED_STREAMK_TILE_SHAPES = [
    (tile_m, tile_n, tile_k)
    for tile_m in (64, 128, 256, 512)
    for tile_n in (64, 128, 256)
    for tile_k in (32, 64)
]
# All SG=8×4 legal tiles (48, used by layered_bmg exhaustive StreamK)
EXHAUSTIVE_STREAMK_8X4_TILES = None  # computed lazily via _get_exhaustive_8x4_tiles()

def _get_exhaustive_8x4_tiles():
    from .constraints import default_constraints as _dc
    from .source_templates import is_valid_xe2_tile_sg as _valid
    global EXHAUSTIVE_STREAMK_8X4_TILES
    if EXHAUSTIVE_STREAMK_8X4_TILES is None:
        cons = _dc()["allowed_values"]
        EXHAUSTIVE_STREAMK_8X4_TILES = [
            (m, n, k) for m in cons["tile_m"] for n in cons["tile_n"] for k in cons["tile_k"]
            if _valid((m, n, k), (8, 4, 1))
        ]
    return EXHAUSTIVE_STREAMK_8X4_TILES
SOURCE_OBSERVED_SG8X4_GEMM_TILE_SHAPES = [
    (128, 256, 16),
    (128, 512, 32),
    (256, 192, 64),
    (256, 256, 16),
]
EXPANDED_GEMM_TILE_SHAPES = sorted(set(EXPANDED_STREAMK_TILE_SHAPES) | set(SOURCE_OBSERVED_SG8X4_GEMM_TILE_SHAPES))
STREAMK_SPLIT_SIZES = (2, 3, 4, 6)
EXHAUSTIVE_REGULAR_GEMM_STAGES = (1, 2, 3)

TRUE_BF16_STREAMK_UNSUPPORTED_REASON = "bf16_accumulate_streamk_not_practical_sycl_atomic_unsupported"
TRUE_BF16_STREAMK_UNSUPPORTED_DETAIL = (
    "True BF16 accumulator/output StreamK is not a practical search target: "
    "StreamK/SplitK reductions require atomic add on the accumulator type, "
    "SYCL atomic_ref does not support cutlass::bfloat16_t, and BF16 accumulate "
    "has poor numerical value for the intended GEMM workloads. Keep this as a "
    "disabled placeholder only."
)
TRUE_BF16_STREAMK_FUTURE_ENABLE_CONDITION = (
    "Enable only if a safe BF16 reduction path is implemented, or if the "
    "candidate is changed to use FP32 accumulation with BF16 output."
)


def benchmark_streamk_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    tile_shapes=STREAMK_TILE_SHAPES,
    source="seed_catalog_level0",
    instantiation_level=0,
):
    entries = []
    for tile_m, tile_n, tile_k in tile_shapes:
        for name_mode, streamk_mode, split_k in (
            ("StreamK", "streamk", 1),
            ("DataParallel", "data_parallel", 1),
            *[("SplitK", "splitk", split_k) for split_k in STREAMK_SPLIT_SIZES],
        ):
            entries.append(
                {
                    "kernel_name": f"{name_prefix}_RCR_{name_mode}_{tile_m}x{tile_n}x{tile_k}",
                    "layout": "rcr",
                    "dtype_a": dtype_a,
                    "dtype_b": dtype_b,
                    "dtype_c": dtype_c,
                    "dtype_d": dtype_d or dtype_c,
                    "dtype_acc": dtype_acc,
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "tile_k": tile_k,
                    "sg_m": 8,
                    "sg_n": 4,
                    "stages": 2,
                    "split_k": split_k,
                    "streamk_mode": streamk_mode,
                    "kernel_schedule": "KernelXeCooperative",
                    "tile_scheduler": "StreamKScheduler",
                    "source": source,
                    "instantiation_level": instantiation_level,
                }
            )
    return entries


def benchmark_gemm_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    layout="rcr",
    tile_shapes=EXPANDED_GEMM_TILE_SHAPES,
    source="expanded_gemm_catalog",
    instantiation_level=1,
):
    return [
        {
            "kernel_name": f"{name_prefix}_{layout.upper()}_Gemm_{tile_m}x{tile_n}x{tile_k}_SG8x4",
            "layout": layout,
            "dtype_a": dtype_a,
            "dtype_b": dtype_b,
            "dtype_c": dtype_c,
            "dtype_d": dtype_d or dtype_c,
            "dtype_acc": dtype_acc,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "sg_m": 8,
            "sg_n": 4,
            "stages": 2,
            "split_k": 1,
            "kernel_schedule": "KernelXe",
            "tile_scheduler": "Gemm",
            "source": source,
            "instantiation_level": instantiation_level,
        }
        for tile_m, tile_n, tile_k in tile_shapes
    ]


def source_template_gemm_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    layout="rcr",
    source_template_space=None,
    source="source_template_gemm_catalog",
    instantiation_level=2,
):
    source_template_space = source_template_space or observed_bmg_template_space()
    entries = []
    for pair in source_template_space["valid_tile_sg_pairs"]:
        tile_m, tile_n, tile_k = pair["tile_shape"]
        sg_m, sg_n, _ = pair["sg_layout"]
        entries.append(
            {
                "kernel_name": f"{name_prefix}_{layout.upper()}_Gemm_{tile_m}x{tile_n}x{tile_k}_SG{sg_m}x{sg_n}",
                "layout": layout,
                "dtype_a": dtype_a,
                "dtype_b": dtype_b,
                "dtype_c": dtype_c,
                "dtype_d": dtype_d or dtype_c,
                "dtype_acc": dtype_acc,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "sg_m": sg_m,
                "sg_n": sg_n,
                "stages": 2,
                "split_k": 1,
                "kernel_schedule": "KernelXe",
                "tile_scheduler": "Gemm",
                "source": source,
                "instantiation_level": instantiation_level,
            }
        )
    return entries


def exhaustive_regular_gemm_tile_candidates(
    name_prefix,
    dtype_a,
    dtype_b,
    dtype_c,
    dtype_acc,
    dtype_d=None,
    layout="rcr",
    constraints=None,
    stages=EXHAUSTIVE_REGULAR_GEMM_STAGES,
    source="exhaustive_regular_gemm_catalog",
    instantiation_level=3,
):
    allowed = (constraints or {}).get("allowed_values", {})
    tile_m_values = allowed.get("tile_m", [8, 16, 32, 64, 128, 256, 512])
    tile_n_values = allowed.get("tile_n", [32, 64, 96, 128, 192, 256, 512])
    tile_k_values = allowed.get("tile_k", [16, 32, 64])
    sg_m_values = allowed.get("sg_m", [1, 2, 4, 8])
    sg_n_values = allowed.get("sg_n", [2, 4, 8])
    stage_values = [stage for stage in stages if stage in allowed.get("stages", list(stages))]
    limits = (constraints or {}).get("limits", {})
    valid_sg_sizes = limits.get("valid_subgroup_sizes")  # None=B70 default, [32,64]=B60
    entries = []
    for tile_m in tile_m_values:
        for tile_n in tile_n_values:
            for tile_k in tile_k_values:
                for sg_m in sg_m_values:
                    for sg_n in sg_n_values:
                        if not is_valid_xe2_tile_sg((tile_m, tile_n, tile_k), (sg_m, sg_n, 1),
                                                    sg_product_set=valid_sg_sizes):
                            continue
                        for stage in stage_values:
                            entries.append(
                                {
                                    "kernel_name": (
                                        f"{name_prefix}_{layout.upper()}_GemmExhaustive_"
                                        f"{tile_m}x{tile_n}x{tile_k}_SG{sg_m}x{sg_n}_ST{stage}"
                                    ),
                                    "layout": layout,
                                    "dtype_a": dtype_a,
                                    "dtype_b": dtype_b,
                                    "dtype_c": dtype_c,
                                    "dtype_d": dtype_d or dtype_c,
                                    "dtype_acc": dtype_acc,
                                    "tile_m": tile_m,
                                    "tile_n": tile_n,
                                    "tile_k": tile_k,
                                    "sg_m": sg_m,
                                    "sg_n": sg_n,
                                    "stages": stage,
                                    "split_k": 1,
                                    "kernel_schedule": "KernelXe",
                                    "tile_scheduler": "Gemm",
                                    "source": source,
                                    "instantiation_level": instantiation_level,
                                }
                            )
    return entries


def unsupported_true_bf16_streamk_example(kernel_suffix, streamk_mode, split_k):
    return {
        "kernel_name": f"03_bmg_gemm_streamk_{kernel_suffix}_bf16_bf16",
        "layout": "rcr",
        "dtype_a": "bf16",
        "dtype_b": "bf16",
        "dtype_c": "bf16",
        "dtype_d": "bf16",
        "dtype_acc": "bf16",
        "tile_m": 256,
        "tile_n": 256,
        "tile_k": 32,
        "sg_m": 8,
        "sg_n": 4,
        "stages": 2,
        "split_k": split_k,
        "runner": "streamk_example",
        "streamk_mode": streamk_mode,
        "streamk_dtype_preset": "bf16_bf16",
        "support_status": "unsupported",
        "support_reason": TRUE_BF16_STREAMK_UNSUPPORTED_REASON,
        "support_detail": TRUE_BF16_STREAMK_UNSUPPORTED_DETAIL,
        "support_future_enable_condition": TRUE_BF16_STREAMK_FUTURE_ENABLE_CONDITION,
    }


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
        *benchmark_streamk_tile_candidates("BmgGemmBF16BF16FP32", "bf16", "bf16", "f32", "f32"),
        unsupported_true_bf16_streamk_example("streamk", "streamk", 1),
        unsupported_true_bf16_streamk_example("dp", "data_parallel", 1),
        unsupported_true_bf16_streamk_example("splitk", "splitk", 2),
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
        *benchmark_streamk_tile_candidates("BmgGemmF16F16FP32", "f16", "f16", "f32", "f32"),
        {"kernel_name": "03_bmg_gemm_streamk_streamk_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "streamk"},
        {"kernel_name": "03_bmg_gemm_streamk_dp_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 1, "runner": "streamk_example", "streamk_mode": "data_parallel"},
        {"kernel_name": "03_bmg_gemm_streamk_splitk_f16", "layout": "rcr", "dtype_a": "f16", "dtype_b": "f16", "dtype_c": "f32", "dtype_acc": "f32", "tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "stages": 2, "split_k": 2, "runner": "streamk_example", "streamk_mode": "splitk"},
        *benchmark_streamk_tile_candidates("BmgGemmF16F16F16", "f16", "f16", "f16", "f16", dtype_d="f16"),
    ],
    "tf32": [
        *benchmark_streamk_tile_candidates("BmgGemmTF32TF32FP32", "tf32", "tf32", "f32", "f32"),
    ],
}


def ilp_class(seed):
    ilp = (seed["tile_m"] // max(seed["sg_m"], 1) // 8) * (seed["tile_n"] // max(seed["sg_n"], 1) // 16)
    if ilp >= 16:
        return "ilp16"
    if ilp >= 8:
        return "ilp8"
    return "ilp4"


def apply_scheduler_metadata(entry):
    scheduler_metadata = dict(DEFAULT_SCHEDULER_METADATA)
    if entry.get("streamk_mode"):
        scheduler_metadata["tile_scheduler"] = "StreamKScheduler"
        scheduler_metadata["decomposition_mode"] = streamk_decomposition_mode(entry["streamk_mode"])
    if entry.get("runner") == "streamk_example":
        scheduler_metadata.update(STREAMK_EXAMPLE_SCHEDULER_METADATA)
    for key, value in scheduler_metadata.items():
        entry.setdefault(key, value)
    return entry


def apply_epilogue_metadata(entry):
    for key, value in infer_epilogue_metadata(entry).items():
        entry.setdefault(key, value)
    return entry


def kernel_catalog_entry(dtype, seed):
    entry = copy.deepcopy(seed)
    entry.setdefault("dtype_d", entry["dtype_c"])
    entry.setdefault("runner", "benchmark")
    entry["kernel_id"] = seed["kernel_name"]
    entry.setdefault("instantiation_level", 0)
    entry["benchmark_target"] = "cutlass_benchmarks_gemm_sycl" if entry["runner"] == "benchmark" else "03_bmg_gemm_streamk"
    entry["grf_mode"] = 256
    entry["ilp_class"] = ilp_class(entry)
    entry["streamk_mode"] = entry.get("streamk_mode", "")
    entry.setdefault("streamk_dtype_preset", entry["dtype_a"] if entry["runner"] == "streamk_example" else "")
    entry.setdefault("support_status", "supported")
    entry.setdefault("support_reason", "")
    entry["batch_count"] = 1
    entry["runtime_defaults"] = {}
    entry["allowed_runtime_sweeps"] = ["shape_id", "m", "n", "k", "batch_count"]
    entry.setdefault("source", "seed_catalog_level0")
    entry["dtype_family"] = dtype
    entry.setdefault("mma_atom", "XE_DPAS_TT")
    entry.setdefault("gmem_copy_atom_a", "auto")
    entry.setdefault("gmem_copy_atom_b", "auto")
    entry.setdefault("epilogue_op", "LinearCombination")
    entry.setdefault("epilogue_tile", "auto")
    entry.setdefault("epilogue_copy_atom_c", "auto")
    entry.setdefault("epilogue_copy_atom_d", "auto")
    apply_epilogue_metadata(entry)
    apply_scheduler_metadata(entry)
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


def generated_expanded_streamk_kernel_catalog():
    expanded = copy.deepcopy(SEED_KERNELS)
    source_template_space = observed_bmg_template_space()
    expanded_gemm = {
        "bf16": [
            *benchmark_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rcr",
            ),
            *benchmark_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rrr",
            ),
        ],
        "f16": [
            *benchmark_gemm_tile_candidates(
                "BmgGemmFP16FP16FP32",
                "f16",
                "f16",
                "f32",
                "f32",
            ),
            *benchmark_gemm_tile_candidates(
                "BmgGemmF16F16F16",
                "f16",
                "f16",
                "f16",
                "f16",
                dtype_d="f16",
            ),
        ],
        "tf32": benchmark_gemm_tile_candidates(
            "BmgGemmTF32TF32FP32",
            "tf32",
            "tf32",
            "f32",
            "f32",
        ),
    }

    expanded_streamk = {
        "bf16": benchmark_streamk_tile_candidates(
            "BmgGemmBF16BF16FP32",
            "bf16",
            "bf16",
            "f32",
            "f32",
            tile_shapes=_get_exhaustive_8x4_tiles(),
            source="expanded_streamk_catalog",
            instantiation_level=1,
        ),
        "f16": [
            *benchmark_streamk_tile_candidates(
                "BmgGemmF16F16FP32",
                "f16",
                "f16",
                "f32",
                "f32",
                tile_shapes=_get_exhaustive_8x4_tiles(),
                source="expanded_streamk_catalog",
                instantiation_level=1,
            ),
            *benchmark_streamk_tile_candidates(
                "BmgGemmF16F16F16",
                "f16",
                "f16",
                "f16",
                "f16",
                dtype_d="f16",
                tile_shapes=_get_exhaustive_8x4_tiles(),
                source="expanded_streamk_catalog",
                instantiation_level=1,
            ),
        ],
        "tf32": benchmark_streamk_tile_candidates(
            "BmgGemmTF32TF32FP32",
            "tf32",
            "tf32",
            "f32",
            "f32",
            tile_shapes=EXPANDED_STREAMK_TILE_SHAPES,
            source="expanded_streamk_catalog",
            instantiation_level=1,
        ),
    }
    source_template_gemm = {
        "bf16": [
            *source_template_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rcr",
                source_template_space=source_template_space,
            ),
            *source_template_gemm_tile_candidates(
                "BmgGemmBF16BF16FP32",
                "bf16",
                "bf16",
                "f32",
                "f32",
                layout="rrr",
                source_template_space=source_template_space,
            ),
        ],
        "f16": [
            *source_template_gemm_tile_candidates(
                "BmgGemmFP16FP16FP32",
                "f16",
                "f16",
                "f32",
                "f32",
                source_template_space=source_template_space,
            ),
            *source_template_gemm_tile_candidates(
                "BmgGemmF16F16F16",
                "f16",
                "f16",
                "f16",
                "f16",
                dtype_d="f16",
                source_template_space=source_template_space,
            ),
        ],
        "tf32": source_template_gemm_tile_candidates(
            "BmgGemmTF32TF32FP32",
            "tf32",
            "tf32",
            "f32",
            "f32",
            source_template_space=source_template_space,
        ),
    }
    for entries_by_dtype in (expanded_gemm, source_template_gemm, expanded_streamk):
        for dtype, entries in entries_by_dtype.items():
            existing = {entry["kernel_name"] for entry in expanded.get(dtype, [])}
            new_entries = [entry for entry in entries if entry["kernel_name"] not in existing]
            expanded.setdefault(dtype, []).extend(new_entries)

    catalog = []
    for dtype in sorted(expanded.keys()):
        for seed in expanded.get(dtype, []):
            catalog.append(kernel_catalog_entry(dtype, seed))
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": "expanded-bmg-level1",
        "instantiation_levels": {
            "0": "existing validated benchmark-backed kernels",
            "1": "expanded BMG Gemm/StreamK/DataParallel/SplitK tile shapes with fixed 8x4 subgroup layout",
            "2": "reserved for copy atom, epilogue, stage, and subgroup-layout expansion",
        },
        "generator_arch": "bmg",
        "generator_instantiation_level": 1,
        "search_runtime_schema": SEARCH_RUNTIME_SCHEMA,
        "source_template_space": source_template_space,
        "kernels": catalog,
    }


def generated_layered_bmg_kernel_catalog(constraints=None):
    catalog = generated_expanded_streamk_kernel_catalog()
    expanded = {}
    for entry in catalog["kernels"]:
        expanded.setdefault(entry["dtype_family"], []).append(entry)
    exhaustive_regular_gemm = {
        "bf16": exhaustive_regular_gemm_tile_candidates(
            "BmgGemmBF16BF16FP32",
            "bf16",
            "bf16",
            "f32",
            "f32",
            constraints=constraints,
        ),
        "f16": exhaustive_regular_gemm_tile_candidates(
            "BmgGemmFP16FP16FP32",
            "f16",
            "f16",
            "f32",
            "f32",
            constraints=constraints,
        ),
    }
    for dtype, entries in exhaustive_regular_gemm.items():
        existing = {entry["kernel_name"] for entry in expanded.get(dtype, [])}
        expanded.setdefault(dtype, []).extend(
            kernel_catalog_entry(dtype, entry)
            for entry in entries
            if entry["kernel_name"] not in existing
        )
    kernels = []
    for dtype in sorted(expanded.keys()):
        kernels.extend(expanded[dtype])
    catalog["catalog_version"] = "layered-bmg-regular-gemm-exhaustive"
    catalog["instantiation_levels"]["3"] = (
        "regular GEMM legal tile/subgroup/stage enumeration generated from default constraints"
    )
    catalog["regular_gemm_exhaustive_space"] = {
        "stages": list(EXHAUSTIVE_REGULAR_GEMM_STAGES),
        "validity_model": "is_valid_xe2_tile_sg plus selected stage values",
        "bf16_kernel_count": len(exhaustive_regular_gemm["bf16"]),
        "f16_kernel_count": len(exhaustive_regular_gemm["f16"]),
    }
    catalog["kernels"] = kernels
    return catalog


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
            entry.setdefault("streamk_dtype_preset", entry["dtype_a"] if entry.get("runner") == "streamk_example" else "")
            entry.setdefault("support_status", "supported")
            entry.setdefault("support_reason", "")
            apply_epilogue_metadata(entry)
            apply_scheduler_metadata(entry)
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
        "streamk_dtype_preset": "",
        "support_status": "supported",
        "support_reason": "",
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
    apply_epilogue_metadata(entry)
    apply_scheduler_metadata(entry)
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
    elif catalog_source in {"expanded_streamk", "expanded_bmg"}:
        source_catalog = generated_expanded_streamk_kernel_catalog()
    elif catalog_source == "layered_bmg":
        from .constraints import default_constraints

        source_catalog = generated_layered_bmg_kernel_catalog(default_constraints())
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
        "source_template_space": source_catalog.get("source_template_space", {}),
        "regular_gemm_exhaustive_space": source_catalog.get("regular_gemm_exhaustive_space", {}),
        "search_runtime_schema": source_catalog.get("search_runtime_schema", SEARCH_RUNTIME_SCHEMA),
        "kernels": catalog,
    }
