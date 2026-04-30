#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import re
from pathlib import Path

from .schemas import SCHEMA_VERSION
from .utils import now_iso, read_json


DEFAULT_BUILD_CONFIG_PATH = Path(__file__).resolve().parent.parent / "build_config_bmg_perf.json"
DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent.parent / "runtime_config_bmg_perf.json"


def _default_build_config():
    return {
        "schema_version": SCHEMA_VERSION,
        "device_arch": "bmg",
        "purpose": "optimal_performance_profiling",
        "cmake_vars": {
            "CUTLASS_ENABLE_SYCL": "ON",
            "DPCPP_SYCL_TARGET": "bmg",
            "CMAKE_BUILD_TYPE": "Release",
            "CUTLASS_SYCL_PROFILING_ENABLED": "OFF",
            "CUTLASS_ENABLE_BENCHMARKS": "ON",
            "CUTLASS_ENABLE_EXAMPLES": "OFF",
            "CUTLASS_ENABLE_TESTS": "OFF",
        },
        "compile_env": {
            "CC": "icx",
            "CXX": "icpx",
            "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
            "IGC_VectorAliasBBThreshold": "100000000000",
            "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file",
        },
        "compile_env_variants": {
            "perf_default": {
                "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
                "IGC_VectorAliasBBThreshold": "100000000000",
                "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file",
            },
        },
        "compile_env_variant_metadata": {
            "perf_default": {
                "status": "validated",
                "notes": "Current best-known validated BMG baseline. Use 256-GRF with large-register-file.",
            },
        },
        "selected_compile_variant": "perf_default",
    }


def _default_runtime_config():
    return {
        "schema_version": SCHEMA_VERSION,
        "device_arch": "bmg",
        "runtime_env": {
            "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
        },
        "runtime_env_variants": {
            "default": {
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
            },
            "ze_affinity_0": {
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                "ZE_AFFINITY_MASK": "0",
            },
            "ze_affinity_1": {
                "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
                "ZE_AFFINITY_MASK": "1",
            },
        },
        "selected_runtime_variant": "default",
    }


def load_persisted_build_config(path=DEFAULT_BUILD_CONFIG_PATH):
    return read_json(path) if path.exists() else _default_build_config()


def load_persisted_runtime_config(path=DEFAULT_RUNTIME_CONFIG_PATH):
    return read_json(path) if path.exists() else _default_runtime_config()


def selected_runtime_env(profiles, profile=None):
    runtime_config = profiles.get("runtime_config", {})
    runtime_env = dict(runtime_config.get("runtime_env", {}))
    selected_variant = runtime_config.get("selected_runtime_variant")
    variant_overrides = runtime_config.get("runtime_env_variants", {}).get(selected_variant, {})
    runtime_env.update(variant_overrides)
    if profile:
        runtime_env.update(profile.get("runtime_env_override", {}))
    return runtime_env


def default_constraints():
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "constraint_source": "default_bmg",
        "device_arch": "bmg",
        "limits": {"max_slm_kb": 64, "subgroup_size": 16, "max_split_k": 2, "max_stages": 3},
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
        "build_config": load_persisted_build_config(),
        "runtime_config": load_persisted_runtime_config(),
        "profiles": [
            {
                "compiler_profile_id": "bmg.small_tile.default",
                "candidate_class": "small_tile",
                "description": "Default BMG profile for small tiles.",
                "selector": {"tile_m_max": 16, "sg_count_max": 8},
                "runtime_env_override": {},
            },
            {
                "compiler_profile_id": "bmg.medium_tile.default",
                "candidate_class": "medium_tile",
                "description": "Default BMG profile for medium tiles.",
                "selector": {"tile_m_min": 32, "tile_m_max": 64, "sg_count_max": 16},
                "runtime_env_override": {},
            },
            {
                "compiler_profile_id": "bmg.large_tile.default",
                "candidate_class": "large_tile",
                "description": "Default BMG profile for large tiles.",
                "selector": {"tile_m_min": 128, "sg_count_min": 16},
                "runtime_env_override": {},
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


def apply_run_probe_constraints(static_constraints, probe_rows, anomaly_report=None):
    constraints = copy.deepcopy(static_constraints)
    constraints["constraint_source"] = "phase_a_run_probe"
    constraints["limits"]["max_slm_kb"] = min(
        constraints["limits"].get("max_slm_kb", 64),
        64,
    )
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
    for rule in (anomaly_report or {}).get("auto_block_rules", []):
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
