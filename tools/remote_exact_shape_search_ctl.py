#!/usr/bin/env python3
"""Local controller for remote exact-shape B70 search workflows."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import paramiko
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit("paramiko is required for remote_exact_shape_search_ctl.py") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_ROOT = "/root/cutlass_profile_device7_b70_2500mhz"
DEFAULT_REMOTE_REPO = f"{DEFAULT_REMOTE_ROOT}/sycl-tla"
DEFAULT_RUNS_DIR = f"{DEFAULT_REMOTE_ROOT}/screen_runs"
SYNC_FILES = [
    "tools/remote_exact_shape_search.sh",
    "tools/remote_exact_shape_search_status.sh",
    "tools/remote_exact_shape_search_stop.sh",
    "media/docs/cpp/intel_b70_exact_shape_search_runbook.md",
]


@dataclass
class RemoteConfig:
    host: str
    user: str
    password: str
    remote_repo: str
    runs_dir: str
    accept_new_host_key: bool


class RemoteSession:
    def __init__(self, config: RemoteConfig):
        self.config = config
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        if config.accept_new_host_key:
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        self.client.connect(
            config.host,
            username=config.user,
            password=config.password or None,
            allow_agent=True,
            look_for_keys=True,
            timeout=20,
        )

    def close(self) -> None:
        self.client.close()

    def run(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err

    def upload(self, local_path: Path, remote_path: str, executable: bool = False) -> None:
        sftp = self.client.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
            if executable:
                sftp.chmod(remote_path, 0o755)
        finally:
            sftp.close()


def build_remote_config(args: argparse.Namespace) -> RemoteConfig:
    return RemoteConfig(
        host=args.host,
        user=args.user,
        password=args.password,
        remote_repo=args.remote_repo,
        runs_dir=args.runs_dir,
        accept_new_host_key=args.accept_new_host_key,
    )


def sync_files(session: RemoteSession) -> None:
    for rel in SYNC_FILES:
        local_path = REPO_ROOT / rel
        remote_path = posixpath.join(session.config.remote_repo, rel)
        session.upload(local_path, remote_path, executable=remote_path.endswith(".sh"))


def command_sync(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        sync_files(session)
        print(json.dumps({"status": "synced", "files": SYNC_FILES}, indent=2))
    finally:
        session.close()


def command_launch(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        skip_remote_repo_sync = args.skip_remote_repo_sync
        if not args.no_sync_files:
            sync_files(session)
            skip_remote_repo_sync = True

        run_id = args.run_id or f"shape_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = posixpath.join(session.config.runs_dir, run_id)
        log_path = posixpath.join(run_dir, "launcher.log")
        env_parts = [
            ("RUN_ID", run_id),
            ("RUN_DIR", run_dir),
            ("GPU_IDS", args.gpu_ids),
            ("SHAPES", args.shapes),
            ("STOP_EXISTING", "1" if args.stop_existing else "0"),
            ("SKIP_SYNC", "1" if skip_remote_repo_sync else "0"),
        ]
        payload = json.dumps(
            {
                "remote_repo": session.config.remote_repo,
                "run_dir": run_dir,
                "log_path": log_path,
                "env": dict(env_parts),
            }
        )
        launch_cmd = f"""python3 - <<'PY'
import json
import os
import subprocess

cfg = json.loads({shlex.quote(payload)})
os.makedirs(cfg["run_dir"], exist_ok=True)
env = os.environ.copy()
env.update(cfg["env"])
with open(cfg["log_path"], "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        ["bash", "tools/remote_exact_shape_search.sh"],
        cwd=cfg["remote_repo"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(proc.pid)
PY"""
        code, out, err = session.run(launch_cmd, timeout=60)
        if code != 0 or not out.strip():
            raise SystemExit(json.dumps({"status": "launch_failed", "stdout": out, "stderr": err}, indent=2))
        pid = out.strip().splitlines()[-1].strip()
        print(json.dumps({"status": "launched", "run_id": run_id, "run_dir": run_dir, "pid": pid, "log_file": log_path}, indent=2))
    finally:
        session.close()


def command_status(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        run_dir = args.run_dir or ""
        cmd = f"cd {shlex.quote(session.config.remote_repo)} && RUNS_DIR={shlex.quote(session.config.runs_dir)} "
        if run_dir:
            cmd += f"RUN_DIR={shlex.quote(run_dir)} "
        cmd += "bash tools/remote_exact_shape_search_status.sh"
        code, out, err = session.run(cmd, timeout=args.timeout)
        sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        raise SystemExit(code)
    finally:
        session.close()


def command_stop(args: argparse.Namespace) -> None:
    session = RemoteSession(build_remote_config(args))
    try:
        if not args.no_sync_files:
            sync_files(session)
        run_dir = args.run_dir or ""
        cmd = f"cd {shlex.quote(session.config.remote_repo)} && RUNS_DIR={shlex.quote(session.config.runs_dir)} "
        if run_dir:
            cmd += f"RUN_DIR={shlex.quote(run_dir)} "
        cmd += "bash tools/remote_exact_shape_search_stop.sh"
        code, out, err = session.run(cmd, timeout=120)
        sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        raise SystemExit(code)
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local controller for remote exact-shape B70 search workflows.")
    parser.add_argument("--host", default="10.239.11.149")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=os.environ.get("EXACT_SHAPE_REMOTE_PASSWORD", ""))
    parser.add_argument(
        "--accept-new-host-key",
        action="store_true",
        help="Trust and add a new host key for the remote host if it is not already present in known_hosts.",
    )
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Upload remote exact-shape scripts/docs to the remote repo.")
    sync_parser.set_defaults(func=command_sync)

    launch_parser = subparsers.add_parser("launch", help="Launch a remote exact-shape search run.")
    launch_parser.add_argument("--run-id", default="")
    launch_parser.add_argument("--gpu-ids", default="0,1,2,3,4")
    launch_parser.add_argument("--shapes", default="2048x384x3584;8192x384x3584")
    launch_parser.add_argument("--skip-remote-repo-sync", action="store_true", help="Pass SKIP_SYNC=1 to the remote launcher.")
    launch_parser.add_argument("--stop-existing", action="store_true", default=True, help="Stop existing exact-shape runs before launch.")
    launch_parser.add_argument("--no-stop-existing", action="store_false", dest="stop_existing")
    launch_parser.add_argument("--no-sync-files", action="store_true", help="Do not upload local scripts/docs before launch.")
    launch_parser.set_defaults(func=command_launch)

    status_parser = subparsers.add_parser("status", help="Query remote exact-shape search status.")
    status_parser.add_argument("--run-dir", default="")
    status_parser.add_argument("--timeout", type=int, default=180)
    status_parser.set_defaults(func=command_status)

    stop_parser = subparsers.add_parser("stop", help="Stop a remote exact-shape search run.")
    stop_parser.add_argument("--run-dir", default="")
    stop_parser.add_argument("--no-sync-files", action="store_true", help="Do not upload local scripts/docs before stopping.")
    stop_parser.set_defaults(func=command_stop)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
