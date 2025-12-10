from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingJSONLogger:
    """
    Lightweight append-only JSONL logger for training steps and eval snapshots.

    Each run writes under `output_dir`:
      - steps.jsonl : per-step metrics
      - eval.jsonl  : periodic evaluation metrics
      - meta.json   : run-level metadata
    """

    def __init__(
        self,
        *,
        run_id: str,
        component: str,
        output_dir: Path | str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id = run_id
        self.component = component
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self._steps_path = self.output_dir / "steps.jsonl"
        self._eval_path = self.output_dir / "eval.jsonl"
        self._meta_path = self.output_dir / "meta.json"

        meta_payload = {
            "run_id": self.run_id,
            "component": self.component,
            "created_at_utc": _now_utc_iso(),
        }
        if meta:
            meta_payload.update(meta)
        self._write_json(self._meta_path, meta_payload)

    def log_step(self, payload: Dict[str, Any]) -> None:
        record = {
            "t": "step",
            "component": self.component,
            "run_id": self.run_id,
            "time_utc": _now_utc_iso(),
        }
        record.update(payload)
        self._append_jsonl(self._steps_path, record)

    def log_eval(self, payload: Dict[str, Any]) -> None:
        record = {
            "t": "eval",
            "component": self.component,
            "run_id": self.run_id,
            "time_utc": _now_utc_iso(),
        }
        record.update(payload)
        self._append_jsonl(self._eval_path, record)

    @staticmethod
    def _write_json(path: Path, obj: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
