#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import os
import platform
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

from .schemas import BENCHMARK_ERROR_RE, DEFAULT_SCHEDULER_METADATA, RESULT_METADATA_FIELDS, SCHEMA_VERSION, infer_epilogue_metadata
from .utils import now_iso, resolve_executable, shell_join, write_json
from .candidates import write_config


def collect_environment_metadata(shell_init, benchmark_exe, streamk_example_exe, cwd=None):
    tracked_env = {}
    for name in ("ONEAPI_DEVICE_SELECTOR", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "ZE_FLAT_DEVICE_HIERARCHY", "SYCL_PROGRAM_COMPILE_OPTIONS", "IGC_ExtraOCLOptions", "IGC_VectorAliasBBThreshold", "IGC_VISAOptions"):
        value = os.environ.get(name)
        if value:
            tracked_env[name] = value
    benchmark_path = resolve_executable(benchmark_exe, cwd=cwd)
    streamk_path = resolve_executable(streamk_example_exe, cwd=cwd)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "hostname": socket.gethostname(),
        "node_id": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
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
    def output_text(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    if shell_init:
        payload = f"{shell_init} && {shell_join(command)}"
        popen_command = ["bash", "-lc", payload]
    else:
        popen_command = command

    # Build subprocess environment — runtime env vars are set via shell_init from the config
    subprocess_env = os.environ.copy()

    process = subprocess.Popen(
        popen_command,
        cwd=cwd,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=subprocess_env,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
        process = subprocess.CompletedProcess(popen_command, process.returncode, output_text(stdout), "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_reason = f"timeout after {timeout}s"
        try:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        process = subprocess.CompletedProcess(popen_command, 124, output_text(stdout or exc.stdout), output_text(exc.stderr))
        # Brief delay to allow GPU driver to recover from hung process
        import time
        time.sleep(2.0)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(output_text(process.stdout))
        if timed_out:
            handle.write(f"\nTIMEOUT: {timeout_reason}\n")
    return process, timed_out, timeout_reason


def parse_metric(line, key):
    match = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
    return match.group(1) if match else ""


def row_result_metadata(metadata):
    defaults = dict(DEFAULT_SCHEDULER_METADATA)
    defaults.update(
        {
            "runner": "benchmark",
            "benchmark_target": "",
            "streamk_mode": "",
            "support_status": "supported",
            "support_reason": "",
            "mma_atom": "XE_DPAS_TT",
            "gmem_copy_atom_a": "auto",
            "gmem_copy_atom_b": "auto",
            "epilogue_op": "LinearCombination",
            "epilogue_tile": "auto",
            "epilogue_copy_atom_c": "auto",
            "epilogue_copy_atom_d": "auto",
        }
    )
    defaults.update(infer_epilogue_metadata(metadata))
    return {field: metadata.get(field, defaults.get(field, "")) for field in RESULT_METADATA_FIELDS}


def with_result_metadata(row, metadata):
    row.update(row_result_metadata(metadata))
    return row


def parse_streamk_example_log(log_path, metadata_by_bm_name, run_id):
    bm_name = next(iter(metadata_by_bm_name))
    metadata = metadata_by_bm_name[bm_name]
    text = Path(log_path).read_text(encoding="utf-8")
    status = "pass" if "Disposition: Passed" in text else "fail"
    failure_reason = "" if status == "pass" else text.strip().splitlines()[-1] if text.strip() else "missing output"
    perf_match = re.search(r"Cutlass GEMM Performance:\s+\[([0-9.]+)\]TFlop/s\s+\(([0-9.]+)\)ms", text)
    avg_tflops = perf_match.group(1) if perf_match else ""
    avg_runtime_ms = perf_match.group(2) if perf_match else ""
    return [
        with_result_metadata(
            {
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
                "dtype_d": metadata.get("dtype_d", metadata["dtype_c"]),
                "dtype_acc": metadata["dtype_acc"],
                "m": metadata["m"],
                "n": metadata["n"],
                "k": metadata["k"],
                "batch_count": metadata.get("batch_count", 1),
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
            },
            metadata,
        )
    ]


def parse_benchmark_log(log_path, metadata_by_bm_name, run_id):
    rows = []
    text = Path(log_path).read_text(encoding="utf-8")
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
                with_result_metadata(
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
                        "dtype_d": metadata.get("dtype_d", metadata["dtype_c"]),
                        "dtype_acc": metadata["dtype_acc"],
                        "m": metadata["m"],
                        "n": metadata["n"],
                        "k": metadata["k"],
                        "batch_count": metadata.get("batch_count", 1),
                        "split_k": metadata.get("split_k", 1),
                        "avg_runtime_ms": parse_metric(stripped, "runtime_trimmed_mean_ms") or parse_metric(stripped, "avg_runtime_ms"),
                        "best_runtime_ms": parse_metric(stripped, "runtime_min_ms") or parse_metric(stripped, "best_runtime_ms"),
                        "worst_runtime_ms": parse_metric(stripped, "runtime_max_ms") or parse_metric(stripped, "worst_runtime_ms"),
                        "runtime_median_ms": parse_metric(stripped, "runtime_median_ms"),
                        "runtime_stddev_ms": parse_metric(stripped, "runtime_stddev_ms"),
                        "warmup_iters": parse_metric(stripped, "warmup_iters"),
                        "measure_iters": parse_metric(stripped, "measure_iters"),
                        "avg_tflops": parse_metric(stripped, "avg_tflops"),
                        "median_tflops": parse_metric(stripped, "median_tflops"),
                        "avg_throughput": parse_metric(stripped, "avg_throughput"),
                        "max_error": "",
                        "close_call_group": "",
                        "failure_reason": stripped if failure else "",
                        "stdout_log": str(log_path),
                    },
                    metadata,
                )
            )
    if not rows and "Benchmark not found" in text:
        for bm_name, metadata in metadata_by_bm_name.items():
            rows.append(
                with_result_metadata(
                    {
                        "run_id": run_id,
                        "stage": metadata["stage"],
                        "attempt_index": metadata["attempt_index"],
                        "shape_id": metadata["shape_id"],
                        "candidate_id": metadata["candidate_id"],
                        "compiler_profile_id": metadata["compiler_profile_id"],
                        "status": "fail",
                        "verify_status": "fail",
                        "layout": metadata["layout"],
                        "dtype_a": metadata["dtype_a"],
                        "dtype_b": metadata["dtype_b"],
                        "dtype_c": metadata["dtype_c"],
                        "dtype_d": metadata.get("dtype_d", metadata["dtype_c"]),
                        "dtype_acc": metadata["dtype_acc"],
                        "m": metadata["m"],
                        "n": metadata["n"],
                        "k": metadata["k"],
                        "batch_count": metadata.get("batch_count", 1),
                        "split_k": metadata.get("split_k", 1),
                        "avg_runtime_ms": "",
                        "best_runtime_ms": "",
                        "worst_runtime_ms": "",
                        "runtime_median_ms": "",
                        "runtime_stddev_ms": "",
                        "warmup_iters": "",
                        "measure_iters": "",
                        "avg_tflops": "",
                        "median_tflops": "",
                        "avg_throughput": "",
                        "max_error": "",
                        "close_call_group": "",
                        "failure_reason": "benchmark registry entry not found for generated kernel",
                        "stdout_log": str(log_path),
                    },
                    metadata,
                )
            )
    return rows


def timeout_rows(entries, log_path, reason):
    rows = []
    for entry in entries:
        candidate = entry["candidate"]
        shape = entry["shape"]
        rows.append(
            with_result_metadata(
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
                    "dtype_d": shape.get("dtype_d", shape["dtype_c"]),
                    "dtype_acc": shape["dtype_acc"],
                    "m": shape["m"],
                    "n": shape["n"],
                    "k": shape["k"],
                    "batch_count": shape.get("batch_count", 1),
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
                },
                candidate,
            )
        )
    return rows


def chunked_path(path, chunk_index):
    return path.with_name(f"{path.stem}_part{chunk_index:03d}{path.suffix}")


def suffixed_path(path, suffix):
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def rows_seen_keys(rows):
    return {
        (row["stage"], int(row["attempt_index"]), row["shape_id"], row["candidate_id"])
        for row in rows
    }


def entries_missing_rows(entries, rows):
    seen = rows_seen_keys(rows)
    return [
        entry for entry in entries
        if (
            entry["stage"],
            int(entry["attempt_index"]),
            entry["shape"]["shape_id"],
            entry["candidate"]["candidate_id"],
        ) not in seen
    ]


def run_entries_with_benchmark_attempt(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None):
    metadata = write_config(entries, config_path)
    write_json(manifest_path, metadata)
    command = [exe, f"--config_file={config_path}"]
    result, timed_out, timeout_reason = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init, timeout=timeout)
    rows = parse_benchmark_log(log_path, metadata, run_id=entries[0]["stage"]) if entries else []
    return rows, command, result, timed_out, timeout_reason


def run_entries_with_benchmark_once(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None):
    rows, command, result, timed_out, timeout_reason = run_entries_with_benchmark_attempt(
        entries,
        config_path,
        manifest_path,
        log_path,
        exe,
        cwd=cwd,
        shell_init=shell_init,
        timeout=timeout,
    )
    if timed_out:
        rows.extend(timeout_rows(entries_missing_rows(entries, rows), log_path, timeout_reason))
    if result.returncode != 0 and not rows:
        raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {log_path}")
    return rows, command


def run_entries_with_benchmark_retrying_timeouts(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None, depth=0):
    rows, command, result, timed_out, timeout_reason = run_entries_with_benchmark_attempt(
        entries,
        config_path,
        manifest_path,
        log_path,
        exe,
        cwd=cwd,
        shell_init=shell_init,
        timeout=timeout,
    )
    commands = [command]
    if timed_out:
        missing = entries_missing_rows(entries, rows)
        if missing and len(missing) < len(entries):
            split_size = max(1, (len(missing) + 1) // 2)
            for retry_index, start in enumerate(range(0, len(missing), split_size)):
                retry_entries = missing[start:start + split_size]
                suffix = f"retry{depth:02d}_{retry_index:03d}"
                retry_rows, retry_commands = run_entries_with_benchmark_retrying_timeouts(
                    retry_entries,
                    suffixed_path(config_path, suffix),
                    suffixed_path(manifest_path, suffix),
                    suffixed_path(log_path, suffix),
                    exe,
                    cwd=cwd,
                    shell_init=shell_init,
                    timeout=timeout,
                    depth=depth + 1,
                )
                rows.extend(retry_rows)
                commands.extend(retry_commands)
        elif missing:
            rows.extend(timeout_rows(missing, log_path, timeout_reason))
    if result.returncode != 0 and not rows:
        raise RuntimeError(f"Benchmark subprocess failed with return code {result.returncode}. See {log_path}")
    return rows, commands


def run_entries_with_benchmark(entries, config_path, manifest_path, log_path, exe, cwd=None, shell_init=None, timeout=None, chunk_size=0, retry_timeouts=True):
    if not entries or chunk_size <= 0 or len(entries) <= chunk_size:
        if retry_timeouts:
            rows, commands = run_entries_with_benchmark_retrying_timeouts(entries, config_path, manifest_path, log_path, exe, cwd=cwd, shell_init=shell_init, timeout=timeout)
            return rows, commands[0] if len(commands) == 1 else commands
        return run_entries_with_benchmark_once(entries, config_path, manifest_path, log_path, exe, cwd=cwd, shell_init=shell_init, timeout=timeout)
    rows = []
    commands = []
    for chunk_index, start in enumerate(range(0, len(entries), chunk_size)):
        chunk = entries[start:start + chunk_size]
        chunk_config_path = chunked_path(config_path, chunk_index)
        chunk_manifest_path = chunked_path(manifest_path, chunk_index)
        chunk_log_path = chunked_path(log_path, chunk_index)
        if retry_timeouts:
            chunk_rows, command = run_entries_with_benchmark_retrying_timeouts(
                chunk,
                chunk_config_path,
                chunk_manifest_path,
                chunk_log_path,
                exe,
                cwd=cwd,
                shell_init=shell_init,
                timeout=timeout,
            )
        else:
            chunk_rows, command = run_entries_with_benchmark_once(
                chunk,
                chunk_config_path,
                chunk_manifest_path,
                chunk_log_path,
                exe,
                cwd=cwd,
                shell_init=shell_init,
                timeout=timeout,
            )
            command = [command]
        rows.extend(chunk_rows)
        commands.extend(command)
    return rows, commands


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
            }
        }
        metadata[bm_name].update(row_result_metadata(candidate))
        log_path = logs_dir / f"{bm_name}.log"
        streamk_dtype = candidate.get("streamk_dtype_preset", candidate["dtype_a"])
        command = [exe, f"--dtype={streamk_dtype}", f"--m={shape['m']}", f"--n={shape['n']}", f"--k={shape['k']}", "--iterations=20", "--verify=1"]
        streamk_mode = candidate.get("streamk_mode", "")
        if streamk_mode == "splitk":
            command.extend(["--splitk", f"--splits={candidate['split_k']}"])
        elif streamk_mode == "data_parallel":
            command.append("--dp")
        result, timed_out, timeout_reason = run_benchmark(command, log_path, cwd=cwd, shell_init=shell_init, timeout=timeout)
        parsed = timeout_rows([entry], log_path, timeout_reason) if timed_out else parse_streamk_example_log(log_path, metadata, run_id=entry["stage"])
        if result.returncode != 0 and not parsed:
            raise RuntimeError(f"StreamK example subprocess failed with return code {result.returncode}. See {log_path}")
        rows.extend(parsed)
        commands.append(shell_join(command))
    return rows, commands
