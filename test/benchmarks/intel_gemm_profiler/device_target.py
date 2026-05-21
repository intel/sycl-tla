#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import copy
import json
import os
import re
import subprocess

from .schemas import SCHEMA_VERSION
from .utils import now_iso, shell_join


DEFAULT_TARGET_DETECTION = {
    "mode": "auto",
    "cmake_var": "DPCPP_SYCL_TARGET",
    "fallback_target": "bmg",
    "strict": False,
    "selected_device_env": "ZE_AFFINITY_MASK",
}

DEVICE_TARGET_BY_PCI_ID = {
    # B70 / BMG G31 boards report this generic device name through xpu-smi.
    "0xe223": {
        "device_arch": "bmg",
        "hw_spec_id": "bmg_g31",
        "cmake_target": "intel_gpu_bmg_g31",
    },
    # Common BMG G21 IDs used by B580/B60-class cards. Name matching below is
    # still the preferred path when the driver exposes product names.
    "0xe20b": {
        "device_arch": "bmg",
        "hw_spec_id": "bmg_g21",
        "cmake_target": "intel_gpu_bmg_g21",
    },
    "0xe20c": {
        "device_arch": "bmg",
        "hw_spec_id": "bmg_g21",
        "cmake_target": "intel_gpu_bmg_g21",
    },
}


def _target(cmake_target, device_arch, hw_spec_id, reason):
    return {
        "cmake_target": cmake_target,
        "device_arch": device_arch,
        "hw_spec_id": hw_spec_id,
        "reason": reason,
    }


def target_from_device_info(device):
    text = " ".join(str(device.get(key, "")) for key in ("device_name", "name", "pci_device_id", "raw")).lower()
    pci_id = str(device.get("pci_device_id", "")).lower()
    if pci_id in DEVICE_TARGET_BY_PCI_ID:
        return _target(**DEVICE_TARGET_BY_PCI_ID[pci_id], reason=f"matched PCI device id {pci_id}")
    if "pvc" in text or "ponte" in text or "data center gpu max" in text:
        return _target("intel_gpu_pvc", "pvc", "pvc", "matched PVC device name")
    if "b70" in text or "g31" in text or "0xe223" in text:
        return _target("intel_gpu_bmg_g31", "bmg", "bmg_g31", "matched BMG G31 device name/id")
    if "b60" in text or "b580" in text or "g21" in text or "0xe20b" in text or "0xe20c" in text:
        return _target("intel_gpu_bmg_g21", "bmg", "bmg_g21", "matched BMG G21 device name/id")
    if "bmg" in text or "battlemage" in text:
        return _target("bmg", "bmg", "bmg_g21", "matched generic BMG device name")
    return None


def parse_xpu_smi_discovery(text):
    devices = []
    current = None
    for line in text.splitlines():
        row = line.strip()
        match = re.match(r"^\|\s*([0-9]+)\s*\|\s*(.+?)\s*\|?$", row)
        if match:
            if current:
                devices.append(current)
            current = {"device_id": match.group(1), "raw": match.group(2).strip()}
            row = match.group(2).strip()
        elif current:
            match = re.match(r"^\|\s*\|\s*(.+?)\s*\|?$", row)
            if match:
                row = match.group(1).strip()
            else:
                current["raw"] = f"{current.get('raw', '')} {row}".strip()
                continue
        else:
            continue

        if ":" in row:
            key, value = [part.strip() for part in row.split(":", 1)]
            normalized_key = key.lower().replace(" ", "_")
            current[normalized_key] = value
            current["raw"] = f"{current.get('raw', '')} {row}".strip()
            id_match = re.search(r"\[(0x[0-9a-fA-F]+)\]", value)
            if id_match:
                current["pci_device_id"] = id_match.group(1).lower()
    if current:
        devices.append(current)
    return devices


def parse_xpu_smi_json(text):
    payload = json.loads(text)
    if isinstance(payload, dict):
        items = payload.get("device_list") or payload.get("devices") or payload.get("DeviceList") or []
    else:
        items = payload
    devices = []
    for item in items:
        if not isinstance(item, dict):
            continue
        flat = {str(key).lower().replace(" ", "_"): value for key, value in item.items()}
        device_id = flat.get("device_id", flat.get("id", flat.get("deviceid", "")))
        name = flat.get("device_name", flat.get("name", ""))
        raw = " ".join(str(value) for value in flat.values())
        id_match = re.search(r"0x[0-9a-fA-F]+", raw)
        devices.append(
            {
                "device_id": str(device_id),
                "device_name": str(name),
                "pci_device_id": id_match.group(0).lower() if id_match else "",
                "raw": raw,
            }
        )
    return devices


def selected_device_id(runtime_config=None, environ=None):
    environ = environ or os.environ
    runtime_config = runtime_config or {}
    env_name = runtime_config.get("device_target_detection", {}).get("selected_device_env", "ZE_AFFINITY_MASK")
    candidates = [
        environ.get(env_name, ""),
        runtime_config.get("runtime_env", {}).get(env_name, ""),
    ]
    selected_variant = runtime_config.get("selected_runtime_variant")
    if selected_variant:
        variant_env = runtime_config.get("runtime_env_variants", {}).get(selected_variant, {})
        candidates.insert(0, variant_env.get(env_name, ""))
    for value in candidates:
        match = re.search(r"([0-9]+)", str(value))
        if match:
            return match.group(1)
    return ""


def _run_discovery_command(command, shell_init):
    payload = shell_join(command)
    if shell_init:
        payload = f"{shell_init} && {payload}"
    return subprocess.run(["bash", "-lc", payload], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30)


def discover_xpu_smi_devices(shell_init=""):
    errors = []
    for command, parser in (
        (["xpu-smi", "discovery", "-j"], parse_xpu_smi_json),
        (["xpu-smi", "discovery"], parse_xpu_smi_discovery),
    ):
        try:
            process = _run_discovery_command(command, shell_init)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{shell_join(command)}: {exc}")
            continue
        if process.returncode != 0:
            errors.append(f"{shell_join(command)} returned {process.returncode}: {process.stdout.strip()}")
            continue
        try:
            devices = parser(process.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{shell_join(command)} parse failed: {exc}")
            continue
        if devices:
            return devices, shell_join(command), errors
    return [], "", errors


def resolve_device_target(build_config, runtime_config=None, shell_init="", environ=None, discovery_devices=None):
    resolved = copy.deepcopy(build_config)
    detection_config = dict(DEFAULT_TARGET_DETECTION)
    detection_config.update(resolved.get("device_target_detection", {}))
    cmake_var = detection_config.get("cmake_var", "DPCPP_SYCL_TARGET")
    cmake_vars = resolved.setdefault("cmake_vars", {})
    requested_target = str(cmake_vars.get(cmake_var, "") or "")
    mode = detection_config.get("mode", "manual")
    should_detect = (mode == "auto" and requested_target.lower() in {"", "auto"}) or requested_target.lower() == "auto"
    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "mode": mode,
        "cmake_var": cmake_var,
        "requested_target": requested_target,
        "selected_device_id": selected_device_id(runtime_config, environ),
        "status": "manual",
        "resolved_target": requested_target,
        "detected_device": {},
        "discovery_command": "",
        "errors": [],
    }
    if not should_detect:
        return resolved, record

    devices = discovery_devices
    if devices is None:
        devices, command, errors = discover_xpu_smi_devices(shell_init=shell_init)
        record["discovery_command"] = command
        record["errors"].extend(errors)
    selected_id = record["selected_device_id"]
    selected = None
    if selected_id:
        selected = next((device for device in devices if str(device.get("device_id", "")) == selected_id), None)
    if selected is None and devices:
        selected = devices[0]
    if selected is not None:
        record["detected_device"] = selected
        target = target_from_device_info(selected)
        if target:
            cmake_vars[cmake_var] = target["cmake_target"]
            resolved["device_arch"] = target["device_arch"]
            record.update(
                {
                    "status": "detected",
                    "resolved_target": target["cmake_target"],
                    "resolved_device_arch": target["device_arch"],
                    "resolved_hw_spec_id": target["hw_spec_id"],
                    "reason": target["reason"],
                }
            )
            return resolved, record

    fallback = detection_config.get("fallback_target", "")
    if fallback and not detection_config.get("strict", False):
        cmake_vars[cmake_var] = fallback
        record.update({"status": "fallback", "resolved_target": fallback, "reason": "device target detection did not match a known device"})
        return resolved, record

    raise RuntimeError(
        "Unable to auto-detect DPCPP_SYCL_TARGET. "
        "Set build_config.cmake_vars.DPCPP_SYCL_TARGET explicitly or provide device_target_detection.fallback_target. "
        f"Detection errors: {'; '.join(record['errors'])}"
    )


def resolve_profiles_device_target(profiles, shell_init="", environ=None, discovery_devices=None):
    resolved = copy.deepcopy(profiles)
    build_config, record = resolve_device_target(
        resolved.get("build_config", {}),
        runtime_config=resolved.get("runtime_config", {}),
        shell_init=shell_init,
        environ=environ,
        discovery_devices=discovery_devices,
    )
    resolved["build_config"] = build_config
    resolved["device_target_detection"] = record
    return resolved, record
