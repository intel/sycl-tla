#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import re
from pathlib import Path

from .schemas import SCHEMA_VERSION
from .utils import now_iso


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_SCAN_DIRS = ("benchmarks/gemm", "examples", "python/cutlass_library")
SOURCE_TEMPLATE_SG_LAYOUTS = (
    (1, 4, 1),
    (1, 8, 1),
    (2, 4, 1),
    (4, 4, 1),
    (4, 8, 1),
    (8, 2, 1),
    (8, 4, 1),
)


def _read_template_sources(source_root=DEFAULT_SOURCE_ROOT, scan_dirs=DEFAULT_TEMPLATE_SCAN_DIRS):
    source_root = Path(source_root)
    chunks = []
    for scan_dir in scan_dirs:
        root = source_root / scan_dir
        if not root.exists():
            continue
        for suffix in ("*.hpp", "*.cpp", "*.py"):
            for path in root.rglob(suffix):
                if 'legacy/' in str(path) or '/legacy/' in str(path):
                    continue
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _shape_literals(text):
    return sorted(
        {
            tuple(int(part) for part in match)
            for match in re.findall(r"Shape<_([0-9]+),\s*_([0-9]+),\s*_([0-9]+)>", text)
        }
    )


def _plausible_gemm_tile(shape):
    m, n, k = shape
    return m in {8, 16, 32, 64, 128, 256, 512} and n >= 32 and k in {16, 32, 64}


def _sg_layout_literals(text):
    return sorted(
        {
            tuple(int(part) for part in match)
            for match in re.findall(
                r"Layout<Shape<_([0-9]+),\s*_([0-9]+),\s*_([0-9]+)>,\s*Stride<",
                text,
            )
        }
    )


def _dpas_atoms(text):
    return sorted(set(re.findall(r"XE_DPAS_TT<([^>]+)>", text)))


def _schedulers(text):
    return sorted(set(re.findall(r"Scheduler::(Gemm\w*)", text)))


def is_valid_xe2_tile_sg(tile_shape, sg_layout, atom_shape=(8, 16, 16), grf_mode=256, sg_product_set=None):
    tile_m, tile_n, tile_k = tile_shape
    sg_m, sg_n, sg_k = sg_layout
    atom_m, atom_n, atom_k = atom_shape
    if tile_m % (sg_m * atom_m) != 0:
        return False
    if tile_n % (sg_n * atom_n) != 0:
        return False
    if tile_k % (sg_k * atom_k) != 0:
        return False
    if sg_product_set is not None and sg_m * sg_n * sg_k not in sg_product_set:
        return False
    reg_m = tile_m // sg_m // atom_m
    reg_n = tile_n // sg_n // atom_n
    return reg_m * reg_n * 16 <= grf_mode


def observed_bmg_template_space(source_root=DEFAULT_SOURCE_ROOT):
    text = _read_template_sources(source_root)
    shape_literals = _shape_literals(text)
    tile_shapes = [shape for shape in shape_literals if _plausible_gemm_tile(shape)]
    sg_layouts = [
        layout
        for layout in _sg_layout_literals(text)
        if layout in SOURCE_TEMPLATE_SG_LAYOUTS
    ]
    valid_tile_sg_pairs = [
        {"tile_shape": list(tile_shape), "sg_layout": list(sg_layout)}
        for tile_shape in tile_shapes
        for sg_layout in sg_layouts
        if is_valid_xe2_tile_sg(tile_shape, sg_layout)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "source": "sycl_tla_source_template_scan",
        "scan_dirs": list(DEFAULT_TEMPLATE_SCAN_DIRS),
        "tile_shapes": [list(shape) for shape in tile_shapes],
        "sg_layouts": [list(layout) for layout in sg_layouts],
        "dpas_atoms": _dpas_atoms(text),
        "schedulers": _schedulers(text),
        "validity_model": {
            "atom_shape": [8, 16, 16],
            "grf_mode": 256,
            "checks": [
                "tile_m % (sg_m * atom_m) == 0",
                "tile_n % (sg_n * atom_n) == 0",
                "tile_k % (sg_k * atom_k) == 0",
                "accumulator_grf = reg_m * reg_n * 16 <= grf_mode",
            ],
        },
        "valid_tile_sg_pairs": valid_tile_sg_pairs,
    }

