from __future__ import annotations

import json
import os
import threading
import uuid
import subprocess
import sys
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from apps.training_factory.database import TrainingFactoryDB


APP_NAME = "Training Factory"

DB_PATH = os.getenv("TRAINING_FACTORY_DB", "storage/training_factory.db")
OUTPUT_ROOT = Path(os.getenv("TRAINING_FACTORY_OUTPUT_ROOT", "artifacts/training_factory"))
HOST = os.getenv("TRAINING_FACTORY_HOST", "0.0.0.0")
PORT = int(os.getenv("TRAINING_FACTORY_PORT", "8010"))

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFOLD = BASE_DIR / "artifacts" / "manifold_bert_optimized" / "checkpoint-3612"
DEFAULT_REGIME = BASE_DIR / "artifacts" / "regime_bert_base" / "checkpoint-1505"
DEFAULT_REWARD = BASE_DIR / "artifacts" / "reward_manifold_bert_base" / "checkpoint-876"

MANIFOLD_CKPT = os.getenv("MANIFOLD_CKPT") or str(DEFAULT_MANIFOLD)
REGIME_CKPT = os.getenv("REGIME_CKPT") or str(DEFAULT_REGIME)
REWARD_MANIFOLD_CKPT = os.getenv("REWARD_MANIFOLD_CKPT") or str(DEFAULT_REWARD)

LABEL_BATCH_SIZE = int(os.getenv("TRAINING_FACTORY_LABEL_BATCH_SIZE", "8"))
LABEL_MAX_LENGTH = int(os.getenv("TRAINING_FACTORY_LABEL_MAX_LENGTH", "512"))
Q_RESP_K = float(os.getenv("Q_RESP_K", "0.6"))
Q_RESP_TAU = float(os.getenv("Q_RESP_TAU", "0.0"))
RELABEL_EXISTING = os.getenv("TRAINING_FACTORY_RELABEL", "false").lower() == "true"
SLEEP_UPDATE_CONFIG = os.getenv(
    "TRAINING_FACTORY_SLEEP_CONFIG",
    str(BASE_DIR / "config" / "training" / "sleep_sft_update.yaml"),
)

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


MANIFOLD_KEYS = [
    "valence",
    "arousal",
    "dominance",
    "predictive_discrepancy",
    "temporal_directionality",
    "social_broadcast",
]

REGIME_KEYS = [
    "regime_support",
    "regime_conflict",
    "regime_problem_solving",
    "regime_truth_seeking",
    "regime_crisis",
    "regime_play",
    "regime_boundary",
]

ADEQUACY_KEYS = [
    "social_coherence",
    "agency_support",
    "narrative_alignment",
    "curiosity",
    "harm_avoidance",
]


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


def _event_messages(ev: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    history = ev.get("history") or []
    if isinstance(history, list):
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user")
            content = str(msg.get("content") or "")
            if content:
                messages.append({"role": role, "content": content})
    user_message = str(ev.get("user_message") or "").strip()
    if user_message:
        messages.append({"role": "user", "content": user_message})
    assistant = str(ev.get("assistant") or "").strip()
    if assistant:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def _messages_to_text(messages: List[Dict[str, str]], *, max_turns: int) -> str:
    if max_turns > 0 and len(messages) > max_turns:
        messages = messages[-max_turns:]
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _predict_sequences(
    *,
    checkpoint: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
) -> List[List[float]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, trust_remote_code=True
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    outputs: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            for row in logits.detach().cpu().tolist():
                outputs.append([float(v) for v in row])

    del model
    del tokenizer
    gc.collect()
    if hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return outputs


def _apply_manifold_labels(events: List[Dict[str, Any]], *, relabel: bool) -> None:
    targets = []
    for ev in events:
        metrics = ev.get("metrics")
        if not relabel and isinstance(metrics, dict) and "manifold" in metrics:
            continue
        targets.append(ev)
    if not targets:
        return
    texts = [_messages_to_text(_event_messages(ev), max_turns=3) for ev in targets]
    preds = _predict_sequences(
        checkpoint=MANIFOLD_CKPT,
        texts=texts,
        batch_size=LABEL_BATCH_SIZE,
        max_length=LABEL_MAX_LENGTH,
    )
    for ev, row in zip(targets, preds):
        scores: Dict[str, float] = {}
        for key, raw in zip(MANIFOLD_KEYS, row):
            if key in ("arousal", "social_broadcast"):
                scores[key] = float(_clamp(raw, 0.0, 1.0))
            else:
                scores[key] = float(_clamp(raw, -1.0, 1.0))
        metrics = ev.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        metrics["manifold"] = scores
        ev["metrics"] = metrics


def _apply_regime_labels(events: List[Dict[str, Any]], *, relabel: bool) -> None:
    targets = []
    for ev in events:
        metrics = ev.get("metrics")
        if not relabel and isinstance(metrics, dict) and "regime_probs" in metrics:
            continue
        targets.append(ev)
    if not targets:
        return
    texts = [_messages_to_text(_event_messages(ev), max_turns=3) for ev in targets]
    preds = _predict_sequences(
        checkpoint=REGIME_CKPT,
        texts=texts,
        batch_size=LABEL_BATCH_SIZE,
        max_length=LABEL_MAX_LENGTH,
    )
    for ev, row in zip(targets, preds):
        raw = [float(v) for v in row[: len(REGIME_KEYS)]]
        sum_raw = sum(raw)
        prob_like = all(0.0 <= v <= 1.0 for v in raw) and 0.98 <= sum_raw <= 1.02
        if prob_like:
            probs = raw
        else:
            m = max(raw) if raw else 0.0
            exps = [math.exp(v - m) for v in raw]
            z = sum(exps)
            probs = [e / z for e in exps] if z > 0 else [1.0 / len(REGIME_KEYS)] * len(REGIME_KEYS)
        regime_probs = {name: p for name, p in zip(REGIME_KEYS, probs)}
        regime_argmax = max(regime_probs.items(), key=lambda kv: kv[1])[0] if regime_probs else None
        metrics = ev.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        metrics["regime_probs"] = regime_probs
        metrics["regime"] = regime_argmax
        ev["metrics"] = metrics


def _apply_reward_labels(events: List[Dict[str, Any]], *, relabel: bool) -> None:
    targets = []
    for ev in events:
        metrics = ev.get("metrics")
        if not relabel and isinstance(metrics, dict) and "reward_adequacy" in metrics:
            continue
        targets.append(ev)
    if not targets:
        return
    texts = [_messages_to_text(_event_messages(ev), max_turns=0) for ev in targets]
    preds = _predict_sequences(
        checkpoint=REWARD_MANIFOLD_CKPT,
        texts=texts,
        batch_size=LABEL_BATCH_SIZE,
        max_length=LABEL_MAX_LENGTH,
    )
    weights = {k: 1.0 / len(ADEQUACY_KEYS) for k in ADEQUACY_KEYS}
    for ev, row in zip(targets, preds):
        scores: Dict[str, float] = {}
        for key, raw in zip(ADEQUACY_KEYS, row):
            scores[key] = float(_clamp(raw, -1.0, 1.0))
        z = sum(weights[k] * scores[k] for k in ADEQUACY_KEYS) - Q_RESP_TAU
        q_resp = float(_clamp(_sigmoid(Q_RESP_K * z), 0.0, 1.0))
        metrics = ev.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        metrics["reward_adequacy"] = scores
        metrics["q_resp"] = q_resp
        metrics["q_resp_params"] = {"k": Q_RESP_K, "tau": Q_RESP_TAU, "weights": weights}
        ev["metrics"] = metrics


def _label_events_serial(events: List[Dict[str, Any]]) -> None:
    _apply_manifold_labels(events, relabel=RELABEL_EXISTING)
    _apply_regime_labels(events, relabel=RELABEL_EXISTING)
    _apply_reward_labels(events, relabel=RELABEL_EXISTING)


def _write_sleep_events_db(events: List[Dict[str, Any]], path: Path) -> None:
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS sleep_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                user_message TEXT,
                assistant TEXT,
                history_json TEXT,
                metrics_json TEXT,
                reward_intensity REAL,
                delta_phi_used REAL,
                used INTEGER DEFAULT 0,
                used_at DATETIME,
                used_in_run TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        rows = []
        for ev in events:
            rows.append(
                (
                    ev.get("conversation_id"),
                    ev.get("user_message"),
                    ev.get("assistant"),
                    json.dumps(ev.get("history") or []),
                    json.dumps(ev.get("metrics") or {}),
                    ev.get("reward_intensity"),
                    ev.get("delta_phi_used"),
                )
            )
        con.executemany(
            """
            INSERT INTO sleep_events(
                conversation_id,
                user_message,
                assistant,
                history_json,
                metrics_json,
                reward_intensity,
                delta_phi_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
    finally:
        con.close()


def _run_mlx_training(
    *,
    db_path: Path,
    output_dir: Path,
    base_model_id: Optional[str],
    lora_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "core.lora_trainer.sleep_sft_update_mlx",
        "--config",
        str(SLEEP_UPDATE_CONFIG),
        "--db-path",
        str(db_path),
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(output_dir / "logs"),
    ]
    if base_model_id:
        cmd.extend(["--base-model-id", str(base_model_id)])
    if lora_config:
        for key, flag in (
            ("iters", "--iters"),
            ("epochs", "--epochs"),
            ("batch_size", "--batch-size"),
            ("min_r_t", "--min-r-t"),
            ("min_reward_intensity", "--min-reward-intensity"),
            ("alpha", "--alpha"),
            ("base_memory_weight", "--base-memory-weight"),
            ("self_fraction_power", "--self-fraction-power"),
            ("reward_clip", "--reward-clip"),
            ("num_samples", "--num-samples"),
            ("priority_w_delta_phi", "--priority-w-delta-phi"),
            ("priority_w_intensity", "--priority-w-intensity"),
            ("priority_w_social", "--priority-w-social"),
            ("priority_w_self", "--priority-w-self"),
            ("high_priority_fraction", "--high-priority-fraction"),
        ):
            if key in lora_config and lora_config[key] is not None:
                cmd.extend([flag, str(lora_config[key])])
        if lora_config.get("require_positive_r_t"):
            cmd.append("--require-positive-r-t")
        if lora_config.get("include_used"):
            cmd.append("--include-used")
        if lora_config.get("order"):
            cmd.extend(["--order", str(lora_config["order"])])
        if lora_config.get("limit"):
            cmd.extend(["--limit", str(lora_config["limit"])])
        if lora_config.get("mlx_args"):
            cmd.append("--mlx-args")
            cmd.extend([str(v) for v in lora_config["mlx_args"]])

    env = os.environ.copy()
    local_mlx = str(BASE_DIR / "third_party" / "mlx_vlm")
    env["PYTHONPATH"] = f"{local_mlx}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"sleep_sft_update_mlx failed with return code {result.returncode}")

    current_path = output_dir / "current.json"
    if current_path.exists():
        try:
            return json.loads(current_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


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

        _label_events_serial(events)
        db.replace_events(batch_id=batch_id, events=events)

        valid_events: List[Dict[str, Any]] = []
        invalid_events: List[Dict[str, Any]] = []
        for idx, ev in enumerate(events):
            history = ev.get("history") or []
            has_history = isinstance(history, list) and any(
                isinstance(m, dict) and str(m.get("content") or "").strip() for m in history
            )
            has_user = bool(str(ev.get("user_message") or "").strip())
            has_assistant = bool(str(ev.get("assistant") or "").strip())
            if has_assistant and (has_user or has_history):
                valid_events.append(ev)
            else:
                invalid_events.append(
                    {
                        "index": idx,
                        "conversation_id": ev.get("conversation_id"),
                        "has_history": has_history,
                        "has_user_message": has_user,
                        "has_assistant": has_assistant,
                    }
                )
        if invalid_events:
            invalid_path = output_dir / "invalid_events.jsonl"
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            with invalid_path.open("w", encoding="utf-8") as f:
                for row in invalid_events:
                    f.write(json.dumps(row, ensure_ascii=False))
                    f.write("\n")
            print(f"⚠️  Skipped {len(invalid_events)} invalid events; details in {invalid_path}")

        db_path = output_dir / "sleep_events.db"
        _write_sleep_events_db(valid_events, db_path)
        manifest = _run_mlx_training(
            db_path=db_path,
            output_dir=output_dir,
            base_model_id=base_model_id,
            lora_config=lora_config,
        )

        db.update_job(
            job_id=job_id,
            status="succeeded",
            dataset_path=manifest.get("train_jsonl"),
            adapter_path=manifest.get("adapter_path"),
            manifest_path=manifest.get("output_dir") and str(Path(manifest.get("output_dir")) / "manifest.json"),
            logs_path=str(output_dir / "logs"),
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
