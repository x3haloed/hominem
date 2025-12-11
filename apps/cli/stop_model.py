#!/usr/bin/env python3
"""
Stop the background Hominem serve process started by `apps/cli/boot_model.py`.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


def _repo_root() -> Path:
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
    parser = argparse.ArgumentParser(description="Stop Hominem model server background process.")
    parser.add_argument("--pid-file", default=os.getenv("HOMINEM_SERVE_PID_FILE"))
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    root = _repo_root()
    pid_path = _pid_file_path(root, args.pid_file)
    if not pid_path.exists():
        print(f"No PID file found at {pid_path}")
        return 0

    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        print(f"Invalid PID file: {pid_path}")
        try:
            pid_path.unlink()
        except Exception:
            pass
        return 1

    if not pid or not _is_running(pid):
        print("Server not running (stale PID file).")
        try:
            pid_path.unlink()
        except Exception:
            pass
        return 0

    # Try to terminate the process group (preferred) then fallback to the pid.
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        if not _is_running(pid):
            break
        time.sleep(0.1)

    if _is_running(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            os.kill(pid, signal.SIGKILL)

    try:
        pid_path.unlink()
    except Exception:
        pass

    print(f"Stopped server pid={pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

