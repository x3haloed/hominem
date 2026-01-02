#!/usr/bin/env python3
"""
Run a one-off training factory job locally without the HTTP server.

This pulls sleep_events from the canonical DB, inserts them into the
training_factory DB, runs labeling + MLX training, and writes outputs
under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.training_factory.database import TrainingFactoryDB
from apps.training_factory.main import _run_job


def _default_canonical_db() -> str:
    return os.getenv(
        "DATABASE_PATH",
        os.path.join(str(Path.home()), "Documents", "hominem", "conversations.db"),
    )


def _load_sleep_events(
    *,
    db_path: str,
    limit: int,
    include_used: bool,
    conversation_id: Optional[str],
    order: str,
) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM sleep_events WHERE 1=1"
        params: List[Any] = []
        if not include_used:
            query += " AND used=0"
        if conversation_id:
            query += " AND conversation_id=?"
            params.append(conversation_id)
        order_norm = str(order or "asc").strip().lower()
        if order_norm not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")
        query += f" ORDER BY created_at {order_norm.upper()} LIMIT ?"
        params.append(int(limit))
        rows = [dict(r) for r in con.execute(query, params).fetchall()]
    finally:
        con.close()

    events: List[Dict[str, Any]] = []
    for row in rows:
        history = row.get("history_json")
        if isinstance(history, str):
            try:
                history = json.loads(history)
            except Exception:
                history = []
        if history is None:
            history = []

        metrics = row.get("metrics_json")
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        if metrics is None:
            metrics = {}

        user_message = row.get("user_message")
        if not user_message and isinstance(history, list):
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_message = msg.get("content")
                    break

        assistant = row.get("assistant") or row.get("response") or ""

        post = metrics.get("post") if isinstance(metrics, dict) else None
        if isinstance(post, dict):
            reward_intensity = post.get("reward_intensity")
            delta_phi_used = post.get("delta_phi_used")
            r_t = post.get("r_t")
        else:
            reward_intensity = metrics.get("reward_intensity") if isinstance(metrics, dict) else None
            delta_phi_used = (
                (metrics.get("delta_phi") or {}).get("used")
                if isinstance(metrics, dict)
                else None
            )
            r_t = metrics.get("r_t") if isinstance(metrics, dict) else None

        events.append(
            {
                "conversation_id": row.get("conversation_id"),
                "user_message": user_message or "",
                "assistant": assistant or "",
                "history": history,
                "metrics": metrics,
                "r_t": r_t,
                "reward_intensity": reward_intensity,
                "delta_phi_used": delta_phi_used,
            }
        )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local training factory job.")
    parser.add_argument("--db-path", default=_default_canonical_db())
    parser.add_argument("--training-factory-db", default=os.getenv("TRAINING_FACTORY_DB", "storage/training_factory.db"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--include-used", action="store_true")
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--source", default="local_job")
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument("--lora-config-json", default=None, help="Path to JSON config for MLX args.")
    args = parser.parse_args()

    events = _load_sleep_events(
        db_path=str(args.db_path),
        limit=int(args.limit),
        include_used=bool(args.include_used),
        conversation_id=str(args.conversation_id) if args.conversation_id else None,
        order=str(args.order),
    )
    if not events:
        raise SystemExit("No sleep_events found with the requested filters.")

    tfdb = TrainingFactoryDB(str(args.training_factory_db))
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    tfdb.create_batch(batch_id=batch_id, source=args.source, event_count=len(events))
    tfdb.insert_events(batch_id=batch_id, events=events)

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("artifacts/training_factory") / f"local_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_config = None
    if args.lora_config_json:
        lora_config = json.loads(Path(args.lora_config_json).read_text(encoding="utf-8"))

    tfdb.create_job(
        job_id=job_id,
        batch_id=batch_id,
        status="queued",
        base_model_id=args.base_model_id,
        lora_config=lora_config,
        output_dir=str(output_dir),
    )

    _run_job(job_id, batch_id, output_dir, args.base_model_id, lora_config)

    print("✅ Local training job complete.")
    print(f"batch_id={batch_id}")
    print(f"job_id={job_id}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
