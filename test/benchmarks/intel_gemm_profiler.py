#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

"""Backward-compatible shim for the modular Intel GEMM profiler package."""

import importlib.util
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).with_name("intel_gemm_profiler")
# This shim is loaded by file path from tests and wrapper scripts. A plain
# `from intel_gemm_profiler import *` would require the package directory to
# already be on sys.path, so we bootstrap the package explicitly here.
SPEC = importlib.util.spec_from_file_location(
    "_intel_gemm_profiler_pkg",
    PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

__all__ = list(getattr(MODULE, "__all__", []))

for name in __all__:
    globals()[name] = getattr(MODULE, name)


if __name__ == "__main__":
    main()
