#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import re

from .schemas import SCHEMA_VERSION
from .utils import now_iso


def default_constraints():
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "constraint_source": "default_bmg",
        "device_arch": "bmg",
        "limits": {"max_slm_kb": 128, "subgroup_size": 16, "max_split_k": 2, "max_stages": 3},
        "allowed_values": {
            "tile_m": [8, 16, 32, 64, 128, 256],
            "tile_n": [64, 128, 256],
            "tile_k": [32, 64],
            "sg_m": [1, 2, 4, 8],
            "sg_n": [4, 8],
            "stages": [1, 2, 3],
            "split_k": [1, 2],
            "grf_mode": [256],
        },
        "blocked_rules": [],
    }


def default_compiler_profiles():
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "profiles": [
            {
                "compiler_profile_id": "bmg.small_tile.default",
                "candidate_class": "small_tile",
                "description": "Default BMG profile for small tiles.",
                "selector": {"tile_m_max": 16, "sg_count_max": 8},
                "env": {"ONEAPI_DEVICE_SELECTOR": "level_zero:gpu", "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread", "IGC_VectorAliasBBThreshold": "100000000000", "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only"},
                "cmake_flags": ["-DCMAKE_BUILD_TYPE=Release"],
            },
            {
                "compiler_profile_id": "bmg.medium_tile.default",
                "candidate_class": "medium_tile",
                "description": "Default BMG profile for medium tiles.",
                "selector": {"tile_m_min": 32, "tile_m_max": 64, "sg_count_max": 16},
                "env": {"ONEAPI_DEVICE_SELECTOR": "level_zero:gpu", "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread", "IGC_VectorAliasBBThreshold": "100000000000", "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only"},
                "cmake_flags": ["-DCMAKE_BUILD_TYPE=Release"],
            },
            {
                "compiler_profile_id": "bmg.large_tile.default",
                "candidate_class": "large_tile",
                "description": "Default BMG profile for large tiles.",
                "selector": {"tile_m_min": 128, "sg_count_min": 16},
                "env": {"ONEAPI_DEVICE_SELECTOR": "level_zero:gpu", "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread", "IGC_VectorAliasBBThreshold": "100000000000", "IGC_VISAOptions": "-perfmodel", "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only"},
                "cmake_flags": ["-DCMAKE_BUILD_TYPE=Release"],
            },
        ],
    }


def apply_static_probe_constraints(base_constraints, env_caps):
    constraints = copy.deepcopy(base_constraints)
    constraints["constraint_source"] = "phase_a_static_probe"
    if not env_caps["executables"]["streamk_example_available"]:
        constraints["limits"]["max_split_k"] = 1
        constraints["allowed_values"]["split_k"] = [1]
    return constraints


def blocked(seed, constraints):
    if seed["sg_m"] * seed["sg_n"] > 32:
        return True
    allowed = constraints["allowed_values"]
    for key in ("tile_m", "tile_n", "tile_k", "sg_m", "sg_n", "stages", "split_k", "grf_mode"):
        if seed.get(key) not in allowed.get(key, []):
            return True
    for rule in constraints.get("blocked_rules", []):
        match = rule.get("match", {})
        if all(seed.get(name) == value for name, value in match.items()):
            return True
    return False


def blocked_rule_for_row(row):
    return {
        "rule_id": f"probe.blocked.{row['candidate_id']}",
        "match": {
            "tile_m": int(re.search(r"_tm(\d+)_", row["candidate_id"]).group(1)),
            "tile_n": int(re.search(r"_tn(\d+)_", row["candidate_id"]).group(1)),
            "tile_k": int(re.search(r"_tk(\d+)_", row["candidate_id"]).group(1)),
            "sg_m": int(re.search(r"_sg(\d+)x", row["candidate_id"]).group(1)),
            "sg_n": int(re.search(r"x(\d+)_st", row["candidate_id"]).group(1)),
            "split_k": int(row["split_k"]),
        },
    }


def apply_run_probe_constraints(static_constraints, probe_rows):
    constraints = copy.deepcopy(static_constraints)
    constraints["constraint_source"] = "phase_a_run_probe"
    if not any(row["status"] == "pass" and int(row["split_k"]) > 1 for row in probe_rows):
        constraints["limits"]["max_split_k"] = 1
        constraints["allowed_values"]["split_k"] = [1]
    failures = [row for row in probe_rows if row["status"] != "pass"]
    existing_ids = {rule.get("rule_id") for rule in constraints.get("blocked_rules", [])}
    for row in failures:
        rule = blocked_rule_for_row(row)
        if rule["rule_id"] not in existing_ids:
            constraints["blocked_rules"].append(rule)
            existing_ids.add(rule["rule_id"])
    return constraints


def apply_probe_results_to_profiles(profiles, compiler_probe_summary):
    updated = copy.deepcopy(profiles)
    result_by_id = {item["compiler_profile_id"]: item for item in compiler_probe_summary.get("results", [])}
    selected_profile_ids = set(compiler_probe_summary.get("selected_profile_ids", {}).values())
    for profile in updated["profiles"]:
        result = result_by_id.get(profile["compiler_profile_id"])
        if result:
            profile["probe_status"] = result["status"]
            profile["probe_avg_tflops"] = result["avg_tflops"]
            profile["probe_avg_runtime_ms"] = result["avg_runtime_ms"]
        profile["probe_selected"] = profile["compiler_profile_id"] in selected_profile_ids
    return updated
