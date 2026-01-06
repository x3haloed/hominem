"""Shared training CLI scaffolding."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingRunConfig:
    run_id: str
    output_dir: Path
    config: Dict[str, Any]
    emit_events: bool
    event_log_path: Optional[Path]


def default_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{int(time.time())}"


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    return json.loads(raw)


def add_common_args(parser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config.")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run ID.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory.")
    parser.add_argument("--emit-events", action="store_true", help="Emit training events to a log.")
    parser.add_argument(
        "--event-log-path",
        type=Path,
        default=None,
        help="JSONL log path for emitted events.",
    )


def build_run_config(args, *, run_prefix: str) -> TrainingRunConfig:
    run_id = args.run_id or os.environ.get("HOMINEM_RUN_ID") or default_run_id(run_prefix)
    config = load_config(args.config)
    return TrainingRunConfig(
        run_id=run_id,
        output_dir=args.output_dir,
        config=config,
        emit_events=bool(args.emit_events),
        event_log_path=args.event_log_path,
    )
