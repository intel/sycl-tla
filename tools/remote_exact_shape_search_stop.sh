#!/bin/bash
# Stop remote exact-shape search runs by latest marker or explicit run dir.
#
# Example:
#   bash tools/remote_exact_shape_search_stop.sh
#   RUN_DIR=/root/.../screen_runs/shape_search_xxx bash tools/remote_exact_shape_search_stop.sh
#   bash tools/remote_exact_shape_search_stop.sh /root/.../screen_runs/shape_search_xxx

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/cutlass_profile_device7_b70_2500mhz}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/screen_runs}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-$RUNS_DIR/exact_shape_search_latest.txt}"

RUN_DIR="${1:-${RUN_DIR:-}}"
if [ -z "$RUN_DIR" ] && [ -f "$LATEST_RUN_FILE" ]; then
  RUN_DIR=$(cat "$LATEST_RUN_FILE")
fi
LAUNCHER_PID_FILE=""
if [ -n "$RUN_DIR" ]; then
  LAUNCHER_PID_FILE="$RUN_DIR/launcher.pid"
fi
WORKER_PIDS_FILE=""
if [ -n "$RUN_DIR" ]; then
  WORKER_PIDS_FILE="$RUN_DIR/status/worker_pids.txt"
fi

SELF_PID="$$"

mapfile -t pids < <(RUN_DIR="$RUN_DIR" SELF_PID="$SELF_PID" LAUNCHER_PID_FILE="$LAUNCHER_PID_FILE" WORKER_PIDS_FILE="$WORKER_PIDS_FILE" python3 <<'PY'
import os
import subprocess
from collections import defaultdict

run_dir = os.environ.get("RUN_DIR", "").strip()
self_pid = int(os.environ["SELF_PID"])
launcher_pid_file = os.environ.get("LAUNCHER_PID_FILE", "").strip()
worker_pids_file = os.environ.get("WORKER_PIDS_FILE", "").strip()

lines = subprocess.check_output(["ps", "-eo", "pid=,ppid=,stat=,cmd="], text=True).splitlines()
procs = {}
children = defaultdict(list)
for line in lines:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid, ppid, stat, cmd = parts
    pid = int(pid)
    ppid = int(ppid)
    procs[pid] = (ppid, stat, cmd)
    children[ppid].append(pid)

protected = {self_pid}
cursor = self_pid
while cursor in procs:
    parent = procs[cursor][0]
    if parent <= 1 or parent in protected:
        break
    protected.add(parent)
    cursor = parent

roots = []
if launcher_pid_file and os.path.exists(launcher_pid_file):
    try:
        pid = int(open(launcher_pid_file).read().strip())
        if pid in procs and pid not in protected:
            roots.append(pid)
    except Exception:
        pass
if worker_pids_file and os.path.exists(worker_pids_file):
    for line in open(worker_pids_file):
        try:
            pid = int(line.strip())
        except Exception:
            continue
        if pid in procs and pid not in protected:
            roots.append(pid)

for pid, (_, stat, cmd) in procs.items():
    if pid in protected or stat.startswith("Z"):
        continue
    if run_dir:
        if run_dir in cmd:
            roots.append(pid)
    elif "remote_exact_shape_search.sh" in cmd or "launch_shape_" in cmd:
        roots.append(pid)

seen = set()
order = []
def visit(pid):
    for child in children.get(pid, []):
        visit(child)
    if pid in seen or pid in protected:
        return
    seen.add(pid)
    stat = procs.get(pid, ("", "Z", ""))[1]
    if not stat.startswith("Z"):
        order.append(pid)

for root in roots:
    visit(root)

for pid in order:
    print(pid)
PY
)

if [ "${#pids[@]}" -eq 0 ]; then
  echo "No exact-shape search processes found."
  exit 0
fi

echo "Stopping exact-shape search PIDs: ${pids[*]}"
for pid in "${pids[@]}"; do
  kill "$pid" 2>/dev/null || true
done
sleep 3
for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
done

echo "Stopped."
