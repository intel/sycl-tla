#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

from pathlib import Path

from .schemas import SCHEMA_VERSION
from .utils import read_json


DISPATCH_KEY_FIELDS = ("layout", "dtype_a", "dtype_b", "dtype_c", "dtype_acc", "m", "n", "k")


def normalize_dispatch_shape_key(shape):
    missing = [field for field in DISPATCH_KEY_FIELDS if field not in shape]
    if missing:
        raise ValueError(f"dispatch shape key missing fields: {', '.join(missing)}")
    key = {}
    for field in DISPATCH_KEY_FIELDS:
        value = shape[field]
        key[field] = int(value) if field in {"m", "n", "k"} else str(value)
    return key


def dispatch_shape_key_tuple(shape):
    normalized = normalize_dispatch_shape_key(shape)
    return tuple(normalized[field] for field in DISPATCH_KEY_FIELDS)


def load_dispatch_table(dispatch_table_or_path):
    if isinstance(dispatch_table_or_path, (str, Path)):
        dispatch_table = read_json(dispatch_table_or_path)
    else:
        dispatch_table = dispatch_table_or_path
    validate_dispatch_table(dispatch_table)
    return dispatch_table


def validate_dispatch_table(dispatch_table):
    schema_version = dispatch_table.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported dispatch table schema_version: {schema_version}")
    if "entries" not in dispatch_table or not isinstance(dispatch_table["entries"], list):
        raise ValueError("dispatch table must contain an entries list")
    seen = {}
    for index, entry in enumerate(dispatch_table["entries"]):
        if "shape_key" not in entry:
            raise ValueError(f"dispatch entry {index} is missing shape_key")
        key = dispatch_shape_key_tuple(entry["shape_key"])
        if key in seen:
            raise ValueError(f"duplicate dispatch shape_key at entries {seen[key]} and {index}")
        seen[key] = index
        if "candidate_id" not in entry:
            raise ValueError(f"dispatch entry {index} is missing candidate_id")
    return True


def build_dispatch_index(dispatch_table_or_path):
    dispatch_table = load_dispatch_table(dispatch_table_or_path)
    return {
        dispatch_shape_key_tuple(entry["shape_key"]): entry
        for entry in dispatch_table["entries"]
    }


def lookup_dispatch_entry(dispatch_table_or_path, shape, *, fallback_candidate_id=""):
    dispatch_table = load_dispatch_table(dispatch_table_or_path)
    requested_key = normalize_dispatch_shape_key(shape)
    entry = build_dispatch_index(dispatch_table).get(dispatch_shape_key_tuple(requested_key))
    if entry is not None:
        return {
            "status": "found",
            "match": "exact",
            "shape_key": requested_key,
            "entry": entry,
            "fallback": {"used": False, "reason": ""},
        }
    fallback = {
        "used": bool(fallback_candidate_id),
        "reason": "shape_not_found",
        "candidate_id": fallback_candidate_id,
    }
    return {
        "status": "fallback" if fallback["used"] else "missing",
        "match": "none",
        "shape_key": requested_key,
        "entry": None,
        "fallback": fallback,
    }
