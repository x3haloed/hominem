"""Shared utilities for training scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hominem_train.common import messages_to_text, normalize_messages, prompt_from_event
from hominem_train.data_io import read_jsonl_records
from hominem_train.events import EventWriter


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: Optional[str]
    dataset_path: Optional[Path]
    query_json: Optional[str]
    dataset_index_path: Optional[Path]


def resolve_dataset_path(spec: DatasetSpec) -> Optional[Path]:
    if spec.dataset_path:
        return spec.dataset_path
    if spec.dataset_id:
        candidate = Path(spec.dataset_id)
        if candidate.exists():
            return candidate
        if not spec.dataset_index_path:
            raise SystemExit("dataset-id provided but no dataset index path supplied.")
        resolved = _lookup_dataset_path(spec.dataset_index_path, spec.dataset_id)
        if not resolved:
            raise SystemExit(f"dataset-id '{spec.dataset_id}' not found in {spec.dataset_index_path}.")
        return resolved
    return None


def load_dataset(spec: DatasetSpec) -> List[Dict[str, Any]]:
    dataset_path = resolve_dataset_path(spec)
    if dataset_path:
        if dataset_path.suffix == ".parquet":
            raise SystemExit("Parquet loading not wired yet. Use JSONL for now.")
        return read_jsonl_records(dataset_path)
    if spec.query_json:
        raise SystemExit("Query-based dataset building is not implemented yet.")
    raise SystemExit("No dataset specified.")


def _lookup_dataset_path(index_path: Path, dataset_id: str) -> Optional[Path]:
    if not index_path.exists():
        raise SystemExit(f"dataset index path not found: {index_path}")
    if index_path.suffix == ".json":
        data = json.loads(index_path.read_text(encoding="utf-8"))
        items = data.get("datasets")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("dataset_id") == dataset_id and item.get("dataset_path"):
                    return Path(str(item["dataset_path"]))
        return None
    if index_path.suffix == ".jsonl":
        for record in read_jsonl_records(index_path):
            if record.get("dataset_id") == dataset_id and record.get("dataset_path"):
                return Path(str(record["dataset_path"]))
        return None
    raise SystemExit("dataset index must be .json or .jsonl")


def record_to_text(rec: Dict[str, Any], *, max_history_turns: int) -> str:
    history = normalize_messages(rec.get("history") or [], drop_system=True)
    if max_history_turns > 0 and len(history) > max_history_turns:
        history = history[-max_history_turns:]
    target = rec.get("target")
    if isinstance(target, dict):
        role = str(target.get("role") or "assistant")
        if role not in {"system", "developer"}:
            history.append(
                {
                    "role": role,
                    "content": str(target.get("content") or ""),
                }
            )
        return messages_to_text(history)

    if "user_message" in rec or "assistant" in rec:
        return prompt_from_event(rec, drop_system=True)

    return messages_to_text(history)


def build_text_label_dataset(
    records: Iterable[Dict[str, Any]],
    *,
    label_keys: Sequence[str],
    label_bounds: Dict[str, Tuple[float, float]],
    max_history_turns: int,
    record_limit: int,
    min_records: int,
    empty_message_error: str,
) -> tuple[List[str], List[List[float]]]:
    texts: List[str] = []
    labels: List[List[float]] = []
    for rec in records:
        label_vec = _extract_label_vector(rec, label_keys=label_keys, label_bounds=label_bounds)
        if label_vec is None:
            continue
        texts.append(record_to_text(rec, max_history_turns=max_history_turns))
        labels.append(label_vec)
        if record_limit > 0 and len(texts) >= record_limit:
            break
    if min_records > 0 and len(texts) < min_records:
        raise SystemExit(f"Need at least {min_records} usable records but only found {len(texts)}.")
    if not texts:
        raise SystemExit(empty_message_error)
    return texts, labels


def _extract_label_vector(
    rec: Dict[str, Any],
    *,
    label_keys: Sequence[str],
    label_bounds: Dict[str, Tuple[float, float]],
) -> Optional[List[float]]:
    labels = rec.get("labels")
    if not isinstance(labels, dict):
        return None
    out: List[float] = []
    for key in label_keys:
        if key not in labels:
            return None
        try:
            val = float(labels[key])
        except (TypeError, ValueError):
            return None
        lo, hi = label_bounds[key]
        if val < lo or val > hi:
            return None
        out.append(val)
    return out


def write_manifest(
    output_dir: Path,
    *,
    model_type: str,
    dataset_spec: DatasetSpec,
    row_count: int,
    config: Dict[str, Any],
    training: Dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_type": model_type,
        "created_at_utc": _now_utc_iso(),
        "config": config,
        "training": training,
        "dataset": {
            "dataset_id": dataset_spec.dataset_id,
            "dataset_path": str(dataset_spec.dataset_path) if dataset_spec.dataset_path else None,
            "query_json": dataset_spec.query_json,
            "row_count": row_count,
        },
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


class MetricsCallback:
    def __init__(self, *, events: EventWriter, run_id: str, model_type: str) -> None:
        self.events = events
        self.run_id = run_id
        self.model_type = model_type

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "run_id": self.run_id,
            "model_type": self.model_type,
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "metrics": {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))},
        }
        self.events.emit("TrainingBatchMetrics", payload)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        payload = {
            "run_id": self.run_id,
            "model_type": self.model_type,
            "step": int(state.global_step),
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        self.events.emit("TrainingEvalMetrics", payload)
