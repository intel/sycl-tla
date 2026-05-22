#!/usr/bin/env python3
"""GPU Hang Monitor — detects GPU-hung benchmarks, blacklists offending kernels.

Usage:
  nohup python3 gpu_hang_monitor.py \
    --workspace /path/to/workspace \
    --blacklist /path/to/blacklist.txt \
    --check-interval 30 \
    --hang-threshold 300 &

Behavior:
  - Every --check-interval seconds, scans for cutlass_benchmarks_gemm_sycl processes
  - If a benchmark process runs > --hang-threshold seconds, extracts the kernel name
  - Appends the guilty kernel to --blacklist file
  - Kills all benchmark processes + optionally the profiler workflow
  - Exits (caller should restart the profiler with resume + exclude the blacklist)
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def find_benchmark_pids():
    """Return list of PIDs for cutlass_benchmarks_gemm_sycl processes."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,etime,args"],
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return []
    pids = []
    for line in out.splitlines():
        if "cutlass_benchmarks_gemm_sycl" in line and "grep" not in line:
            parts = line.strip().split(None, 2)
            if len(parts) >= 2:
                pids.append((int(parts[0]), parts[1], parts[2] if len(parts) > 2 else ""))
    return pids


def runtime_seconds(etime_str):
    """Convert ps etime (DD-HH:MM:SS or HH:MM:SS) to seconds."""
    etime_str = etime_str.strip()
    if "-" in etime_str:
        days, rest = etime_str.split("-")
        days = int(days)
    else:
        days = 0
        rest = etime_str
    parts = list(map(int, rest.split(":")))
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return 0
    return days * 86400 + h * 3600 + m * 60 + s


def process_state(pid):
    """Return (state, wchan) for a PID."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        state = ""
        for line in status.splitlines():
            if line.startswith("State:"):
                state = line.split(":")[1].strip().split()[0]
                break
        wchan = Path(f"/proc/{pid}/wchan").read_text().strip()
        return state, wchan
    except Exception:
        return "?", "?"


def extract_kernel_from_config(config_file_path):
    """Parse a benchmark config file to find the kernel name."""
    try:
        content = Path(config_file_path).read_text()
        import re
        # Look for BmgGemm* pattern (benchmark filter name)
        match = re.search(r"BmgGemm[A-Za-z0-9_]+", content)
        if match:
            return match.group(0)
    except Exception:
        pass
    return None


def blacklist_kernel(blacklist_path, kernel_name):
    """Add a kernel to the blacklist file."""
    path = Path(blacklist_path)
    existing = set()
    if path.exists():
        existing = set(path.read_text().strip().splitlines())
    existing.add(kernel_name)
    path.write_text("\n".join(sorted(existing)) + "\n")
    return len(existing)


def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="GPU Hang Monitor")
    parser.add_argument("--workspace", required=True, help="Profiler workspace directory")
    parser.add_argument("--blacklist", default="/tmp/gpu_hang_blacklist.txt", help="Blacklist file path")
    parser.add_argument("--check-interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--hang-threshold", type=int, default=300, help="Seconds before declaring a hang")
    parser.add_argument("--kill-workflow", action="store_true", help="Also kill profiler workflow on hang")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU Hang Monitor started")
    print(f"  workspace: {workspace}")
    print(f"  blacklist: {args.blacklist}")
    print(f"  check interval: {args.check_interval}s")
    print(f"  hang threshold: {args.hang_threshold}s")
    sys.stdout.flush()

    last_progress = 0
    while True:
        time.sleep(args.check_interval)
        now = time.time()

        bench_procs = find_benchmark_pids()

        # Progress report every 5 minutes
        if now - last_progress > 300:
            logs_dir = workspace / "logs"
            if logs_dir.exists():
                built = len(list(logs_dir.glob("candidate_build_preflight_selected_kernel_batch_*.log"))) - len(list(logs_dir.glob("*_configure.log")))
                screening = len(list(logs_dir.glob("screening_*.log")))
                csv = workspace / "reports" / "gemm_profile_results.csv"
                csv_rows = len(csv.read_text().splitlines()) - 1 if csv.exists() else 0
                print(f"[{time.strftime('%H:%M:%S')}] progress: {built} built, {screening} screening, {csv_rows} CSV rows, {len(bench_procs)} benchmarks active")
                sys.stdout.flush()
                last_progress = now

        for pid, etime, cmdline in bench_procs:
            secs = runtime_seconds(etime)
            if secs < args.hang_threshold:
                continue

            state, wchan = process_state(pid)

            # Detect hang: D-state + guc_wait, or just exceeded threshold significantly
            is_hung = (state == "D" and "guc" in wchan.lower()) or secs > 600

            if not is_hung:
                continue

            print(f"\n[{time.strftime('%H:%M:%S')}] ⚠️  GPU HANG DETECTED")
            print(f"  PID={pid} runtime={secs}s state={state} wchan={wchan}")

            # Extract batch and config
            import re
            batch_match = re.search(r"selected_kernel_batch_\d+", cmdline)
            batch = batch_match.group(0) if batch_match else "unknown"
            config_match = re.search(r"--config_file=(\S+)", cmdline)
            config_file = config_match.group(1) if config_match else None

            print(f"  batch={batch} config={config_file}")

            # Find guilty kernel
            kernel = None
            if config_file:
                kernel = extract_kernel_from_config(config_file)

            if kernel:
                total = blacklist_kernel(args.blacklist, kernel)
                print(f"  blacklisted: {kernel} (total {total} in blacklist)")
            else:
                print(f"  could not identify kernel from config")

            # Kill all benchmarks
            print(f"  killing all benchmark processes...")
            for bpid, _, _ in bench_procs:
                kill_process_tree(bpid)

            # Optionally kill workflow
            if args.kill_workflow:
                try:
                    out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
                    for line in out.splitlines():
                        if "intel_gemm_profiler.workflow" in line and "grep" not in line:
                            wpid = int(line.strip().split()[0])
                            kill_process_tree(wpid)
                            print(f"  killed workflow PID={wpid}")
                except Exception:
                    pass

            print(f"  monitor exiting — restart profiler with --resume to continue")
            sys.exit(1)

        # All clear — brief heartbeat
        if not bench_procs:
            print(f"[{time.strftime('%H:%M:%S')}] no benchmarks running")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
