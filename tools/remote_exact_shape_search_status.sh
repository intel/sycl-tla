#!/bin/bash
# Progress/status helper for remote_exact_shape_search.sh runs.
#
# Example:
#   bash tools/remote_exact_shape_search_status.sh
#   bash tools/remote_exact_shape_search_status.sh /root/.../screen_runs/shape_search_20260605_123456

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/cutlass_profile_device7_b70_2500mhz}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/screen_runs}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-$RUNS_DIR/exact_shape_search_latest.txt}"

RUN_DIR="${1:-${RUN_DIR:-}}"
if [ -z "$RUN_DIR" ] && [ -f "$LATEST_RUN_FILE" ]; then
  RUN_DIR=$(cat "$LATEST_RUN_FILE")
fi

[ -n "$RUN_DIR" ] || { echo "ERROR: run dir not provided and no latest run marker found" >&2; exit 1; }
[ -d "$RUN_DIR" ] || { echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1; }

LOG_FILE="$RUN_DIR/launcher.log"
META_FILE="$RUN_DIR/run_meta.txt"
REQUEST_FILE="$RUN_DIR/requested_shapes.json"
STATUS_DIR="$RUN_DIR/status"
LAUNCHER_PID_FILE="$RUN_DIR/launcher.pid"
WORKER_PIDS_FILE="$STATUS_DIR/worker_pids.txt"

echo "run_dir=$RUN_DIR"
[ -f "$META_FILE" ] && sed -n '1,40p' "$META_FILE"

echo "---processes---"
python3 - "$RUN_DIR" "$LAUNCHER_PID_FILE" "$WORKER_PIDS_FILE" <<'PY'
import subprocess
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
launcher_pid_file = Path(sys.argv[2])
worker_pids_file = Path(sys.argv[3])

lines = subprocess.check_output(["ps", "-eo", "pid=,ppid=,stat=,etime=,cmd="], text=True).splitlines()
procs = {}
for line in lines:
    parts = line.strip().split(None, 4)
    if len(parts) < 5:
        continue
    pid = int(parts[0])
    procs[pid] = line

selected = []
if launcher_pid_file.exists():
    try:
        pid = int(launcher_pid_file.read_text().strip())
        if pid in procs:
            selected.append(procs[pid])
    except Exception:
        pass
if worker_pids_file.exists():
    for entry in worker_pids_file.read_text().splitlines():
        try:
            pid = int(entry.strip())
        except Exception:
            continue
        if pid in procs:
            selected.append(procs[pid])

for line in lines:
    if str(run_dir) in line and line not in selected:
        selected.append(line)

seen = set()
for line in selected:
    if line in seen:
        continue
    seen.add(line)
    print(line)
PY

echo "---shape-summary---"
if [ -d "$STATUS_DIR" ]; then
  [ -f "$STATUS_DIR/current_shape" ] && echo "current_shape=$(cat "$STATUS_DIR/current_shape")"
  [ -f "$STATUS_DIR/completed_shapes.txt" ] && echo "completed_shapes=$(paste -sd';' "$STATUS_DIR/completed_shapes.txt")"
fi
python3 - "$RUN_DIR" "$REQUEST_FILE" <<'PY'
import csv
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

run_dir = Path(sys.argv[1])
request_file = Path(sys.argv[2])

shape_tags = []
if request_file.exists():
    doc = json.loads(request_file.read_text())
    for shape in doc.get("shapes", []):
        shape_tags.append(f"{shape['m']}_{shape['n']}_{shape['k']}")
else:
    results_dir = run_dir / "results"
    if results_dir.exists():
        shape_tags = sorted(path.name for path in results_dir.iterdir() if path.is_dir())

for tag in shape_tags:
    result_dir = run_dir / "results" / tag
    status_dir = run_dir / "status"
    csv_files = sorted(glob.glob(str(result_dir / "*.csv")))
    print(f"shape={tag}")
    status_file = status_dir / f"{tag}.status"
    if status_file.exists():
        print(f"  marker_status={status_file.read_text().strip()}")
    print(f"  done_marker={(status_dir / f'{tag}.done').exists()}")
    print(f"  failed_marker={(status_dir / f'{tag}.failed').exists()}")
    print(f"  csv_count={len(csv_files)}")
    statuses = Counter()
    rows = 0
    for path in csv_files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                statuses[row.get("status", "")] += 1
    print(f"  rows={rows}")
    for status, count in sorted(statuses.items()):
        print(f"  status[{status}]={count}")
    for fail_path in sorted(run_dir.glob(f"failed_{tag}_gpu*.txt")):
        if not fail_path.exists():
            continue
        with fail_path.open() as handle:
            failures = [line.strip() for line in handle if line.strip()]
        gpu = fail_path.stem.rsplit("_gpu", 1)[-1]
        print(f"  failed_gpu{gpu}={len(failures)}")
PY

echo "---log-tail---"
tail -n 60 "$LOG_FILE" 2>/dev/null || true

echo "---worker-tail---"
if [ -d "$RUN_DIR/logs" ]; then
  find "$RUN_DIR/logs" -maxdepth 1 -name 'worker_*.log' | sort | while read -r path; do
    echo "FILE:$path"
    tail -n 10 "$path"
    echo "---"
  done
fi
