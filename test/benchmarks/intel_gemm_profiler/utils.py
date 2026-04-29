#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import json
import shlex
import shutil
from datetime import datetime
from pathlib import Path


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def shell_join(command):
    return " ".join(shlex.quote(part) for part in command)


def resolve_executable(path, cwd=None):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    if cwd:
        joined = Path(cwd) / candidate
        if joined.exists():
            return joined.resolve()
    found = shutil.which(path)
    return Path(found) if found else None


def shell_init_with_env(shell_init, env_map):
    exports = [f"export {name}={shlex.quote(str(value))}" for name, value in env_map.items()]
    if shell_init and exports:
        return f"{shell_init} && " + " && ".join(exports)
    if exports:
        return " && ".join(exports)
    return shell_init
