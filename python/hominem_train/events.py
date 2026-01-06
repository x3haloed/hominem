"""Event emission helpers for training workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


EVENT_SCHEMA_VERSION = 1


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventWriter:
    def __init__(self, *, enabled: bool, log_path: Optional[Path]) -> None:
        self.enabled = enabled
        self.log_path = log_path
        if self.enabled and self.log_path is None:
            raise ValueError("event log path is required when events are enabled")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled or self.log_path is None:
            return
        record = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_type": event_type,
            "ts": _now_utc_iso(),
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
