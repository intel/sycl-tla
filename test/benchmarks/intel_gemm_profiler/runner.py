#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import os
import platform
import re
import socket
import subprocess
from pathlib import Path

from .schemas import BENCHMARK_ERROR_RE
from .utils import now_iso, resolve_executable, shell_join, write_json
from .candidates import write_config


def collect_environment_metadata(shell_init, benchmark_exe, streamk_example_exe, cwd=None):
    tracked_env = {}
    for name in ("ONEAPI_DEVICE_SELECTOR", "SYCL_PROGRAM_COMPILE_OPTIONS", "IGC_ExtraOCLOptions", "IGC_VectorAliasBBThreshold", "IGC_VISAOptions"):
        value = os.environ.get(name)
        if value:
            tracked_env[name] = value
    benchmark_path = resolve_executable(benchmark_exe, cwd=cwd)
    streamk_path = resolve_executable(streamk_example_exe, cwd=cwd)
    return {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "hostname": socket.gethostname(),
        "node_id": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": os.sys.version.split()[0],
        "proxy_bootstrap_method": shell_init or "inherited-environment",
        "executables": {
            "benchmark_exe": str(benchmark_path) if benchmark_path else benchmark_exe,
            "benchmark_available": bool(benchmark_path),
            "streamk_example_exe": str(streamk_path) if streamk_path else streamk_example_exe,
            "streamk_example_available": bool(streamk_path),
        },
        "effective_env": tracked_env,
    }


def run_benchmark(command, log_path, cwd=None, shell_init=None, timeout=None):
    timed_out = False
    timeout_reason = ""
    try:
        if shell_init:
            payload = f"{shell_init} && {shell_join(command)}"
            process = subprocess.run(["bash", "-lc", payload], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
        else:
            process = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_reason = f"timeout after {timeout}s"
        process = subprocess.CompletedProcess(exc.cmd, 124, exc.stdout or "", exc.stderr or "")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(process.stdout or "")
        if timed_out:
            handle.write(f"\nTIMEOUT: {timeout_reason}\n")
    return process, timed_out, timeout_reason


def parse_metric(line, key):
    match = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
    return match.group(1) if match else ""


def parse_streamk_example_log(log_path, metadata_by_bm_name, run_id):
    bm_name = next(iter(metadata_by_bm_name))
    metadata = metadata_by_bm_name[bm_name]
    text = Path(log_path).read_text(encoding="utf-8")
    status = "pass" if "Disposition: Passed" in text else "fail"
    failure_reason = "" if status == "pass" else text.strip().splitlines()[-1] if text.strip() else "missing output"
    perf_match = re.search(r"Cutlass GEMM Performance:\s+\[([0-9.]+)\]TFlop/s\s+\(([0-9.]+)\)ms", text)
    avg_tflops = perf_match.group(1) if perf_match else ""
    avg_runtime_ms = perf_match.group(2) if perf_match else ""
    return [{
        "run_id": run_id,
        "stage": metadata["stage"],
        "attempt_index": metadata["attempt_index"],
        "shape_id": metadata["shape_id"],
        "candidate_id": metadata["candidate_id"],
        "compiler_profile_id": metadata["compiler_profile_id"],
        "status": status,
        "verify_status": status,
        "layout": metadata["layout"],
        "dtype_a": metadata["dtype_a"],
        "dtype_b": metadata["dtype_b"],
        "dtype_c": metadata["dtype_c"],
        "dtype_acc": metadata["dtype_acc"],
        "m": metadata["m"],
        "n": metadata["n"],
        "k": metadata["k"],
        "split_k": metadata.get("split_k", 1),
        "avg_runtime_ms": avg_runtime_ms,
        "best_runtime_ms": avg_runtime_ms,
        "worst_runtime_ms": avg_runtime_ms,
        "avg_tflops": avg_tflops,
        "avg_throughput": "",
        "max_error": "",
        "close_call_group": "",
        "failure_reason": failure_reason,
        "stdout_log": str(log_path),
    }]


def parse_benchmark_log(log_path, metadata_by_bm_name, run_id):
    rows = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if "manual_time" not in line and not BENCHMARK_ERROR_RE.search(stripped):
                continue
            parts = stripped.split()
            if not parts:
                continue
            token = parts[0]
            segments = token.split("/")
            if len(segments) < 2:
                continue
            metadata = metadata_by_bm_name.get(segments[1])
            if not metadata:
                continue
            failure = bool(BENCHMARK_ERROR_RE.search(stripped))
            rows.append(
                {
                    "run_id": run_id,
                    "stage": metadata["stage"],
                    "attempt_index": metadata["attempt_index"],
                    "shape_id": metadata["shape_id"],
                    "candidate_id": metadata["candidate_id"],
                    "compiler_profile_id": metadata["compiler_profile_id"],
                    "status": "fail" if failure else "pass",
                    "verify_status": "fail" if failure else "pass",
                    "layout": metadata["layout"],
                    "dtype_a": metadata["dtype_a"],
                    "dtype_b": metadata["dtype_b"],
                    "dtype_c": metadata["dtype_c"],
                    "dtype_acc": metadata["dtype_acc"],
                    "m": metadata["m"],
                    "n": metadata["n"],
                    "k": metadata["k"],
                    "split_k": metadata.get("split_k", 1),
                    "avg_runtime_ms": parse_metric(stripped, "avg_runtime_ms"),
                    "best_runtime_ms": parse_metric(stripped, "best_runtime_ms"),
                    "worst_runtime_ms": parse_metric(stripped, "worst_runtime_ms"),
                    "avg_tflops": parse_metric(stripped, "avg_tflops"),
                    "avg_throughput": parse_metric(stripped, "avg_throughput"),
                    "max_error": "",
                    "close_call_group": "",
                    "failure_reason": stripped if failure else "",
                    "stdout_log": str(log_path),
                }
            )
    return rows


def timeout_rows(entries, log_path, reason):
    rows = []
    for entry in entries:
        candidate = entry["candidate"]
        shape = entry["shape"]
        rows.append(
            {
                "run_id": entry["stage"],
                "stage": entry["stage"],
                "attempt_index": entry["attempt_index"],
                "shape_id": shape["shape_id"],
                "candidate_id": candidate["candidate_id"],
                "compiler_profile_id": candidate["compiler_profile_id"],
                "status": "timeout",
                "verify_status": "fail",
                "layout": shape["layout"],
                "dtype_a": shape["dtype_a"],
                "dtype_b": shape["dtype_b"],
                "dtype_c": shape["dtype_c"],
                "dtype_acc": shape["dtype_acc"],
                "m": shape["m"],
                "n": shape["n"],
                "k": shape["k"],
                "split_k": candidate.get("split_k", 1),
                "avg_runtime_ms": "",
                "best_runtime_ms": "",
                "worst_runtime_ms": "",
                "avg_tflops": "",
                "avg_throughput": "",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": reason,
                "stdout_log": str(log_path),
            }
        )
    return rows


def run_entries_with_benchmark(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None):
    metadata = write_config(entries, config_path)
    write_json(manifest_path, metadata)
    command = [exe, f"--config_file={config_path}"]
    result, timed_out, timeout_reason = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init, timeout=timeout)
    rows = parse_benchmark_log(log_path, metadata, run_id=entries[0]["stage"]) if entries else []
    if timed_out and not rows:
        rows = timeout_rows(entries, log_path, timeout_reason)
    if result.returncode != 0 and not rows:
        raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {log_path}")
    return rows, command


def run_entries_with_streamk_example(entries, logs_dir, exe, cwd=None, shell_init=None, timeout=None):
    rows = []
    commands = []
    for entry in entries:
        candidate = entry["candidate"]
        shape = entry["shape"]
        bm_name = entry["bm_name"]
        metadata = {
            bm_name: {
                "shape_id": shape["shape_id"],
                "candidate_id": candidate["candidate_id"],
                "compiler_profile_id": candidate["compiler_profile_id"],
                "stage": entry["stage"],
                "attempt_index": entry["attempt_index"],
                "layout": shape["layout"],
                "dtype_a": shape["dtype_a"],
                "dtype_b": shape["dtype_b"],
                "dtype_c": shape["dtype_c"],
                "dtype_acc": shape["dtype_acc"],
                "m": shape["m"],
                "n": shape["n"],
                "k": shape["k"],
                "kernel_name": candidate["kernel_name"],
                "split_k": candidate["split_k"],
                "runner": candidate.get("runner", "streamk_example"),
            }
        }
        log_path = logs_dir / f"{bm_name}.log"
        command = [exe, "--splitk", f"--dtype={candidate['dtype_a']}", f"--splits={candidate['split_k']}", f"--m={shape['m']}", f"--n={shape['n']}", f"--k={shape['k']}", "--iterations=20", "--verify=1"]
        result, timed_out, timeout_reason = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init, timeout=timeout)
        parsed = timeout_rows([entry], log_path, timeout_reason) if timed_out else parse_streamk_example_log(log_path, metadata, run_id=entry["stage"])
        if result.returncode != 0 and not parsed:
            raise RuntimeError(f"StreamK example subprocess failed with return code {result.returncode}. See {log_path}")
        rows.extend(parsed)
        commands.append(shell_join(command))
    return rows, commands
