from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from apps.training_factory.database import TrainingFactoryDB


APP_NAME = "Training Factory"

DB_PATH = os.getenv("TRAINING_FACTORY_DB", "storage/training_factory.db")
OUTPUT_ROOT = Path(os.getenv("TRAINING_FACTORY_OUTPUT_ROOT", "artifacts/training_factory"))
HOST = os.getenv("TRAINING_FACTORY_HOST", "0.0.0.0")
PORT = int(os.getenv("TRAINING_FACTORY_PORT", "8010"))

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

db = TrainingFactoryDB(DB_PATH)
app = FastAPI(title=APP_NAME)


class SleepEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    conversation_id: Optional[str] = None
    user_message: str
    assistant: str
    think: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    metrics: Optional[Dict[str, Any]] = None
    r_t: Optional[float] = None
    reward_intensity: Optional[float] = None
    delta_phi_used: Optional[float] = None
    created_at: Optional[str] = None


class SleepEventsIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_id: Optional[str] = None
    source: Optional[str] = None
    events: List[SleepEvent] = Field(default_factory=list)


class SleepEventsOut(BaseModel):
    batch_id: str
    accepted: int
    rejected: int


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_id: str
    base_model_id: Optional[str] = None
    lora_config: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    batch_id: str
    status: str
    base_model_id: Optional[str] = None
    lora_config: Dict[str, Any]
    output_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    adapter_path: Optional[str] = None
    manifest_path: Optional[str] = None
    logs_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _prompt_from_event(ev: Dict[str, Any]) -> str:
    history = ev.get("history") or []
    lines: List[str] = []
    for msg in history:
        role = str(msg.get("role") or "user").strip().lower()
        content = str(msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    user_message = str(ev.get("user_message") or "").strip()
    if user_message:
        lines.append(f"user: {user_message}")
    return "\n".join(lines).strip() or user_message


def _write_dataset(events: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            record = {
                "prompt": _prompt_from_event(ev),
                "completion": str(ev.get("assistant") or "").strip(),
                "metadata": {
                    "conversation_id": ev.get("conversation_id"),
                    "r_t": ev.get("r_t"),
                    "reward_intensity": ev.get("reward_intensity"),
                    "delta_phi_used": ev.get("delta_phi_used"),
                    "metrics": ev.get("metrics"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def _write_manifest(
    *,
    path: Path,
    job_id: str,
    batch_id: str,
    base_model_id: Optional[str],
    lora_config: Optional[Dict[str, Any]],
    dataset_path: Path,
    adapter_path: Path,
    logs_path: Path,
    event_count: int,
) -> None:
    payload = {
        "job_id": job_id,
        "batch_id": batch_id,
        "base_model_id": base_model_id,
        "lora_config": lora_config or {},
        "dataset_path": str(dataset_path),
        "adapter_path": str(adapter_path),
        "logs_path": str(logs_path),
        "event_count": int(event_count),
        "training_status": "not_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_job(job_id: str, batch_id: str, output_dir: Path, base_model_id: Optional[str], lora_config: Optional[Dict[str, Any]]) -> None:
    try:
        db.update_job(job_id=job_id, status="running")
        events = db.list_events(batch_id)
        if not events:
            raise RuntimeError(f"No events found for batch {batch_id}")

        dataset_path = output_dir / "sleep_events.jsonl"
        adapter_path = output_dir / "lora_adapter"
        manifest_path = output_dir / "manifest.json"
        logs_path = output_dir / "training.log"

        _write_dataset(events, dataset_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        logs_path.write_text("Training stub: no MLX run executed.\n", encoding="utf-8")
        _write_manifest(
            path=manifest_path,
            job_id=job_id,
            batch_id=batch_id,
            base_model_id=base_model_id,
            lora_config=lora_config,
            dataset_path=dataset_path,
            adapter_path=adapter_path,
            logs_path=logs_path,
            event_count=len(events),
        )

        db.update_job(
            job_id=job_id,
            status="succeeded",
            dataset_path=str(dataset_path),
            adapter_path=str(adapter_path),
            manifest_path=str(manifest_path),
            logs_path=str(logs_path),
        )
    except Exception as exc:
        db.update_job(job_id=job_id, status="failed", error=str(exc))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/sleep-events", response_model=SleepEventsOut)
def ingest_sleep_events(payload: SleepEventsIn) -> SleepEventsOut:
    if not payload.events:
        raise HTTPException(status_code=400, detail="No events provided.")

    batch_id = payload.batch_id or f"batch_{uuid.uuid4().hex}"
    accepted: List[Dict[str, Any]] = []
    rejected = 0

    for event in payload.events:
        data = event.model_dump()
        if not data.get("user_message") or not data.get("assistant"):
            rejected += 1
            continue
        accepted.append(data)

    if not accepted:
        raise HTTPException(status_code=400, detail="No valid events found.")

    db.create_batch(batch_id=batch_id, source=payload.source, event_count=len(accepted))
    db.insert_events(batch_id=batch_id, events=accepted)

    return SleepEventsOut(batch_id=batch_id, accepted=len(accepted), rejected=rejected)


@app.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest) -> TrainResponse:
    job_id = f"job_{uuid.uuid4().hex}"
    output_dir = Path(payload.output_dir) if payload.output_dir else OUTPUT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    db.create_job(
        job_id=job_id,
        batch_id=payload.batch_id,
        status="queued",
        base_model_id=payload.base_model_id,
        lora_config=payload.lora_config,
        output_dir=str(output_dir),
    )

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, payload.batch_id, output_dir, payload.base_model_id, payload.lora_config),
        daemon=True,
    )
    thread.start()

    return TrainResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse(
        job_id=job.job_id,
        batch_id=job.batch_id,
        status=job.status,
        base_model_id=job.base_model_id,
        lora_config=job.lora_config,
        output_dir=job.output_dir,
        dataset_path=job.dataset_path,
        adapter_path=job.adapter_path,
        manifest_path=job.manifest_path,
        logs_path=job.logs_path,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
