#!/usr/bin/env python3
"""
Boot the Hominem serve process (keeps model warm in background).

This starts a uvicorn process for `apps.serve.main:app` and writes a PID file so
it can be stopped with `apps/cli/stop_model.py`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    # apps/cli/boot_model.py -> repo root
    return Path(__file__).resolve().parents[2]


def _pid_file_path(root: Path, pid_file: str | None) -> Path:
    if pid_file:
        p = Path(pid_file)
        return p if p.is_absolute() else (root / p)
    return root / "storage" / "hominem_serve.pid"


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Boot Hominem model server (warm background process).")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--pid-file", default=os.getenv("HOMINEM_SERVE_PID_FILE"))
    parser.add_argument("--env-file", default=os.getenv("HOMINEM_SERVE_ENV_FILE"))
    parser.add_argument("--log-level", default=os.getenv("HOMINEM_SERVE_LOG_LEVEL", "info"))
    args = parser.parse_args()

    root = _repo_root()
    pid_path = _pid_file_path(root, args.pid_file)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = pid_path.parent / "hominem_serve.log"

    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text().strip())
            if existing_pid and _is_running(existing_pid):
                print(f"Server already running (pid={existing_pid}).")
                return 0
        except Exception:
            pass
        # stale pid file
        try:
            pid_path.unlink()
        except Exception:
            pass

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "apps.serve.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-level",
        args.log_level,
    ]
    if args.env_file:
        cmd.extend(["--env-file", args.env_file])

    log_fh = open(log_path, "a", encoding="utf-8")

    # Start detached-ish: new process group so stop can terminate the group.
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=log_fh,
        stderr=log_fh,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        env=os.environ.copy(),
    )

    # Give it a moment to start
    time.sleep(0.4)

    pid_path.write_text(str(proc.pid))
    print(f"Started server pid={proc.pid} on http://{args.host}:{args.port}")
    print(f"PID file: {pid_path}")
    print(f"Logs: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

