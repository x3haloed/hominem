from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS sleep_event_batches (
    batch_id TEXT PRIMARY KEY,
    source TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sleep_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    event_json TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES sleep_event_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL,
    base_model_id TEXT,
    lora_config_json TEXT,
    output_dir TEXT,
    dataset_path TEXT,
    adapter_path TEXT,
    manifest_path TEXT,
    logs_path TEXT,
    error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES sleep_event_batches(batch_id)
);
"""


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    batch_id: str
    status: str
    base_model_id: Optional[str]
    lora_config: Dict[str, Any]
    output_dir: Optional[str]
    dataset_path: Optional[str]
    adapter_path: Optional[str]
    manifest_path: Optional[str]
    logs_path: Optional[str]
    error: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class TrainingFactoryDB:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def create_batch(self, *, batch_id: str, source: Optional[str], event_count: int) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO sleep_event_batches(batch_id, source, event_count)
                VALUES (?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET event_count=excluded.event_count
                """,
                (batch_id, source, int(event_count)),
            )
            self.conn.commit()

    def insert_events(self, *, batch_id: str, events: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT INTO sleep_events(batch_id, event_json) VALUES (?, ?)",
                [(batch_id, json.dumps(ev)) for ev in events],
            )
            self.conn.commit()

    def list_events(self, batch_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT event_json FROM sleep_events WHERE batch_id=? ORDER BY id ASC",
                (batch_id,),
            )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for (raw,) in rows:
            try:
                out.append(json.loads(raw))
            except Exception:
                out.append({"raw": raw})
        return out

    def replace_events(self, *, batch_id: str, events: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM sleep_events WHERE batch_id=?", (batch_id,))
            self.conn.executemany(
                "INSERT INTO sleep_events(batch_id, event_json) VALUES (?, ?)",
                [(batch_id, json.dumps(ev)) for ev in events],
            )
            self.conn.commit()

    def create_job(
        self,
        *,
        job_id: str,
        batch_id: str,
        status: str,
        base_model_id: Optional[str],
        lora_config: Optional[Dict[str, Any]],
        output_dir: Optional[str],
    ) -> None:
        lora_json = json.dumps(lora_config or {})
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO jobs(job_id, batch_id, status, base_model_id, lora_config_json, output_dir)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, batch_id, status, base_model_id, lora_json, output_dir),
            )
            self.conn.commit()

    def update_job(
        self,
        *,
        job_id: str,
        status: Optional[str] = None,
        dataset_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        logs_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        fields: List[str] = []
        params: List[Any] = []
        if status is not None:
            fields.append("status=?")
            params.append(status)
        if dataset_path is not None:
            fields.append("dataset_path=?")
            params.append(dataset_path)
        if adapter_path is not None:
            fields.append("adapter_path=?")
            params.append(adapter_path)
        if manifest_path is not None:
            fields.append("manifest_path=?")
            params.append(manifest_path)
        if logs_path is not None:
            fields.append("logs_path=?")
            params.append(logs_path)
        if error is not None:
            fields.append("error=?")
            params.append(error)

        if not fields:
            return
        fields.append("updated_at=CURRENT_TIMESTAMP")
        sql = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id=?"
        params.append(job_id)
        with self._lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
            row = cur.fetchone()
        if not row:
            return None
        columns = [col[0] for col in cur.description]  # type: ignore[union-attr]
        data = dict(zip(columns, row))
        lora_config = {}
        raw = data.get("lora_config_json")
        if raw:
            try:
                lora_config = json.loads(raw)
            except Exception:
                lora_config = {}
        return JobRecord(
            job_id=data.get("job_id"),
            batch_id=data.get("batch_id"),
            status=data.get("status"),
            base_model_id=data.get("base_model_id"),
            lora_config=lora_config,
            output_dir=data.get("output_dir"),
            dataset_path=data.get("dataset_path"),
            adapter_path=data.get("adapter_path"),
            manifest_path=data.get("manifest_path"),
            logs_path=data.get("logs_path"),
            error=data.get("error"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
