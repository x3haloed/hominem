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
    Lightweight logger for training steps and eval snapshots.
    
    Supports both database (default) and JSONL (legacy) logging modes.
    Each run writes to database tables or under `output_dir`:
      - Database: training_runs, training_steps, training_evals tables
      - JSONL (legacy): steps.jsonl, eval.jsonl, meta.json files
    """

    def __init__(
        self,
        *,
        run_id: str,
        component: str,
        output_dir: Path | str,
        meta: Optional[Dict[str, Any]] = None,
        use_database: bool = True,
        db_path: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.component = component
        self.output_dir = Path(output_dir)
        self.use_database = use_database
        
        if use_database:
            from core.data.db import TrainingDatabase
            self.db = TrainingDatabase(db_path=db_path)
            # Create training run record
            self.db.create_training_run(
                run_id=run_id,
                component=component,
                metadata=meta,
            )
            self._step_counter = 0
            self._eval_counter = 0
        else:
            self.db = None
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
        if self.use_database and self.db:
            self._step_counter += 1
            self.db.log_training_step(
                run_id=self.run_id,
                step_number=self._step_counter,
                metrics=payload,
            )
        else:
            record = {
                "t": "step",
                "component": self.component,
                "run_id": self.run_id,
                "time_utc": _now_utc_iso(),
            }
            record.update(payload)
            self._append_jsonl(self._steps_path, record)

    def log_eval(self, payload: Dict[str, Any]) -> None:
        if self.use_database and self.db:
            self._eval_counter += 1
            self.db.log_training_eval(
                run_id=self.run_id,
                eval_number=self._eval_counter,
                metrics=payload,
            )
        else:
            record = {
                "t": "eval",
                "component": self.component,
                "run_id": self.run_id,
                "time_utc": _now_utc_iso(),
            }
            record.update(payload)
            self._append_jsonl(self._eval_path, record)

    def close(self) -> None:
        """Close database connection if using database logging."""
        if self.db:
            self.db.close()

    @staticmethod
    def _write_json(path: Path, obj: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
