#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

"""Candidate prefiltering based on ILP, tile utilization, and shape-fit heuristics.

Purpose: reduce the candidate space BEFORE compilation by skipping configurations
that are almost certain to underperform for a given target shape.  This is meant to
complement (not replace) the legality filter `is_valid_xe2_tile_sg`.

Levels:
  none       — no prefiltering (keep all candidates)
  light      — remove physically incompatible configs (tile_k > problem_K, etc.)
  medium     — light + remove ILP=1 + large-SG + tiny-tile combos
  aggressive — medium + remove stage=1 plain GEMM + tiny-SG + very imbalanced tiles
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional


def compute_ilp(tile_m: int, tile_n: int, tile_k: int, sg_m: int, sg_n: int) -> int:
    """Inner Loop Product — DPAS instructions per work-item per iteration.

        ILP = (tile_m / sg_m / 8) × (tile_n / sg_n / 16)

    Higher ILP → more compute per memory access → better latency hiding.
    ILP=1 means the work-item executes exactly one DPAS instruction per iteration.
    """
    return (tile_m // max(sg_m, 1) // 8) * (tile_n // max(sg_n, 1) // 16)


def _physically_invalid(candidate: Dict, target_shape: Dict) -> bool:
    """Configurations that cannot run on the target problem shape at all."""
    tm, tn, tk = candidate["tile_m"], candidate["tile_n"], candidate["tile_k"]
    pm, pn, pk = target_shape.get("m", 0), target_shape.get("n", 0), target_shape.get("k", 0)

    if tk > pk > 0:
        return True  # tile_k larger than problem K — K dimension not splittable
    if tm > pm * 8:
        return True  # tile_m >> problem M
    if tn > pn * 8:
        return True  # tile_n >> problem N
    if tm > 0 and tn > 0 and max(tm, tn) / min(tm, tn) > 32:
        return True  # extremely skinny tile → poor cache/register utilization
    return False


def _ilp(candidate: Dict) -> int:
    return compute_ilp(
        candidate["tile_m"],
        candidate["tile_n"],
        candidate["tile_k"],
        candidate["sg_m"],
        candidate["sg_n"],
    )


_PREFILTER_RULES = {
    "light": {
        "physically_invalid": lambda c, s: _physically_invalid(c, s),
    },
    "medium": {
        "physically_invalid": lambda c, s: _physically_invalid(c, s),
        "low_ilp_large_sg_small_tile": lambda c, s: (
            not c.get("streamk_mode")
            and _ilp(c) <= 1
            and c["sg_m"] * c["sg_n"] >= 8
            and c["tile_m"] < 64
        ),
    },
    "aggressive": {
        "physically_invalid": lambda c, s: _physically_invalid(c, s),
        "low_ilp_large_sg_small_tile": lambda c, s: (
            not c.get("streamk_mode")
            and _ilp(c) <= 1
            and c["sg_m"] * c["sg_n"] >= 8
            and c["tile_m"] < 64
        ),
        "ilp_le_2_plain_gemm": lambda c, s: (
            not c.get("streamk_mode")
            and _ilp(c) <= 2
        ),
        "stage1_plain_gemm": lambda c, s: (
            not c.get("streamk_mode")
            and c.get("stages", 2) == 1
        ),
        "tiny_sg_large_tile": lambda c, s: (
            not c.get("streamk_mode")
            and c["sg_m"] * c["sg_n"] <= 2
            and c["tile_m"] >= 128
        ),
    },
}


def prefilter_candidates(
    candidates: List[Dict],
    shapes: List[Dict],
    strategy: str = "none",
) -> List[Dict]:
    """Return a filtered list of candidates.

    Parameters
    ----------
    candidates : list of candidate dicts (from generate_candidate_space)
    shapes : list of shape dicts (the target problem shapes)
    strategy : one of "none", "light", "medium", "aggressive"
    """
    if strategy == "none" or strategy not in _PREFILTER_RULES:
        return list(candidates)

    rules = _PREFILTER_RULES[strategy]
    # Use the largest shape as reference for tile-utilization checks
    reference_shape = max(
        shapes,
        key=lambda s: (s.get("m", 0) * s.get("n", 0) * s.get("k", 0)),
        default={"m": 0, "n": 0, "k": 0},
    )

    kept = []
    for candidate in candidates:
        skip = False
        for rule_name, rule_fn in rules.items():
            if rule_fn(candidate, reference_shape):
                skip = True
                break
        if not skip:
            kept.append(candidate)
    return kept


def priority_score(candidate: Dict, target_shape: Dict) -> int:
    """Compute a priority score for screening ordering.

    Higher score → run earlier.  This ensures high-ILP, well-fitting candidates
    are screened first, so a TFLOPS baseline can be established quickly.
    """
    i = _ilp(candidate)
    s = i * 10  # base: ILP × 10

    tm, tn, tk = candidate["tile_m"], candidate["tile_n"], candidate["tile_k"]
    sg_count = candidate["sg_m"] * candidate["sg_n"]
    mode = candidate.get("streamk_mode", "") or "gemm"
    pm, pn = target_shape.get("m", 0), target_shape.get("n", 0)

    # Tile utilization bonus: how well tile divides the problem
    m_tiles = pm / tm if tm > 0 else 999
    n_tiles = pn / tn if tn > 0 else 999
    if 4 <= m_tiles <= 64:
        s += 20
    if 4 <= n_tiles <= 64:
        s += 20

    # SG occupancy
    if 8 <= sg_count <= 16:
        s += 10
    elif sg_count > 16:
        s -= 5

    # Stage bonus
    s += (candidate.get("stages", 2) - 1) * 5

    # Mode bonus — StreamK/DP are known high-performers for medium+ shapes
    if mode in ("streamk", "data_parallel"):
        s += 30
    elif mode == "splitk":
        s += 10 if candidate.get("split_k", 1) <= 3 else -5

    # tile_k bonus
    if tk == 64:
        s += 10
    elif tk == 16:
        s -= 5

    return s


def sort_candidates_by_priority(
    candidates: List[Dict],
    shapes: List[Dict],
) -> List[Dict]:
    """Sort candidates by priority score (highest first) for optimal screening order."""
    # Use the largest shape as reference
    reference_shape = max(
        shapes,
        key=lambda s: (s.get("m", 0) * s.get("n", 0) * s.get("k", 0)),
        default={"m": 0, "n": 0, "k": 0},
    )
    return sorted(candidates, key=lambda c: priority_score(c, reference_shape), reverse=True)
