"""CLI entry point for manifold model training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hominem_train.common import messages_to_text, normalize_messages, prompt_from_event
from hominem_train.data_io import read_jsonl_records
from hominem_train.events import EventWriter
from hominem_train.train_base import add_common_args, build_run_config, TrainingRunConfig


DEFAULT_MANIFOLD_MODEL_ID = "bert-base-uncased"

MANIFOLD_KEYS = [
    "valence",
    "arousal",
    "dominance",
    "predictive_discrepancy",
    "temporal_directionality",
    "social_broadcast",
]
LABEL_BOUNDS = {
    "valence": (-1.0, 1.0),
    "arousal": (0.0, 1.0),
    "dominance": (-1.0, 1.0),
    "predictive_discrepancy": (-1.0, 1.0),
    "temporal_directionality": (-1.0, 1.0),
    "social_broadcast": (0.0, 1.0),
}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: Optional[str]
    dataset_path: Optional[Path]
    query_json: Optional[str]
    dataset_index_path: Optional[Path]


def _load_dataset(spec: DatasetSpec) -> List[Dict[str, Any]]:
    dataset_path = _resolve_dataset_path(spec)
    if dataset_path:
        if dataset_path.suffix == ".parquet":
            raise SystemExit("Parquet loading not wired yet. Use JSONL for now.")
        return read_jsonl_records(dataset_path)
    if spec.query_json:
        raise SystemExit("Query-based dataset building is not implemented yet.")
    raise SystemExit("No dataset specified.")


def _resolve_dataset_path(spec: DatasetSpec) -> Optional[Path]:
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


def _record_to_text(rec: Dict[str, Any], *, max_history_turns: int) -> str:
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


def _extract_label_vector(rec: Dict[str, Any]) -> Optional[List[float]]:
    labels = rec.get("labels")
    if not isinstance(labels, dict):
        return None
    out: List[float] = []
    for key in MANIFOLD_KEYS:
        if key not in labels:
            return None
        try:
            val = float(labels[key])
        except (TypeError, ValueError):
            return None
        lo, hi = LABEL_BOUNDS[key]
        if val < lo or val > hi:
            return None
        out.append(val)
    return out


def _build_dataset(
    records: List[Dict[str, Any]],
    *,
    max_history_turns: int,
    record_limit: int,
    min_records: int,
) -> tuple[List[str], List[List[float]]]:
    texts: List[str] = []
    labels: List[List[float]] = []
    for rec in records:
        label_vec = _extract_label_vector(rec)
        if label_vec is None:
            continue
        texts.append(_record_to_text(rec, max_history_turns=max_history_turns))
        labels.append(label_vec)
        if record_limit > 0 and len(texts) >= record_limit:
            break
    if min_records > 0 and len(texts) < min_records:
        raise SystemExit(f"Need at least {min_records} usable records but only found {len(texts)}.")
    if not texts:
        raise SystemExit("No usable records with manifold labels.")
    return texts, labels


def _write_manifest(
    output_dir: Path,
    *,
    dataset_spec: DatasetSpec,
    row_count: int,
    config: Dict[str, Any],
    training: Dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_type": "manifold",
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


def _train_manifold(
    *,
    dataset: List[Dict[str, Any]],
    run_config: TrainingRunConfig,
    events: EventWriter,
    dataset_spec: DatasetSpec,
    args: argparse.Namespace,
) -> None:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            TrainerCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc}.") from exc

    run_id = run_config.run_id
    output_dir = run_config.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    events.emit(
        "TrainingRunStarted",
        {
            "run_id": run_id,
            "model_type": "manifold",
            "dataset_id": dataset_spec.dataset_id,
            "dataset_path": str(dataset_spec.dataset_path) if dataset_spec.dataset_path else None,
            "row_count": len(dataset),
        },
    )

    texts, labels = _build_dataset(
        dataset,
        max_history_turns=args.max_history_turns,
        record_limit=args.record_limit,
        min_records=args.min_records,
    )
    hf_dataset = Dataset.from_dict({"text": texts, "labels": labels})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(MANIFOLD_KEYS),
        problem_type="regression",
        trust_remote_code=True,
    )

    device = torch.device(
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )

    tokenized = hf_dataset.map(tokenize, batched=True)
    if args.validation_split > 0:
        split = tokenized.train_test_split(
            test_size=args.validation_split,
            seed=args.validation_seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = tokenized
        eval_dataset = None

    for subset in (train_dataset, eval_dataset):
        if subset is not None:
            subset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def compute_metrics(eval_pred):
        preds, lbls = eval_pred
        preds = preds.squeeze(-1) if preds.ndim == 3 else preds
        mse = float(np.mean((preds - lbls) ** 2))
        return {"mse": mse}

    eval_strategy = "steps" if eval_dataset else "no"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        eval_strategy=eval_strategy,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    class _MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            payload = {
                "run_id": run_id,
                "model_type": "manifold",
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "metrics": {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))},
            }
            events.emit("TrainingBatchMetrics", payload)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            payload = {
                "run_id": run_id,
                "model_type": "manifold",
                "step": int(state.global_step),
                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            events.emit("TrainingEvalMetrics", payload)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[_MetricsCallback()],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    events.emit(
        "TrainingRunCompleted",
        {
            "run_id": run_id,
            "model_type": "manifold",
            "status": "success",
        },
    )
    manifest_path = _write_manifest(
        output_dir,
        dataset_spec=dataset_spec,
        row_count=len(texts),
        config=run_config.config,
        training={
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "max_length": args.max_length,
            "validation_split": args.validation_split,
            "validation_seed": args.validation_seed,
            "record_limit": args.record_limit,
            "min_records": args.min_records,
            "max_history_turns": args.max_history_turns,
        },
    )
    events.emit(
        "ModelArtifactProduced",
        {
            "run_id": run_id,
            "model_type": "manifold",
            "manifest_path": str(manifest_path),
        },
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train manifold classifier.")
    parser.add_argument("--dataset-id", type=str, default=None, help="Dataset ID from the index/log.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to JSONL dataset.")
    parser.add_argument("--query-json", type=str, default=None, help="Inline dataset query JSON.")
    parser.add_argument(
        "--dataset-index-path",
        type=Path,
        default=None,
        help="JSON/JSONL dataset index for resolving dataset IDs.",
    )
    parser.add_argument(
        "--record-limit",
        type=int,
        default=0,
        help="Stop after this many usable records (0 = no limit)",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=0,
        help="Require at least this many usable records before training starts",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="Fraction of usable records held out for validation (0 = disabled)",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=42,
        help="Random seed used when splitting off the validation set",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MANIFOLD_MODEL_ID,
        help="Base model ID for regression head",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-turns", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    add_common_args(parser)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.record_limit < 0 or args.min_records < 0:
        raise SystemExit("record-limit and min-records must be non-negative")
    if not 0 <= args.validation_split < 1:
        raise SystemExit("validation-split must be between 0 (inclusive) and 1 (exclusive)")

    spec = DatasetSpec(
        dataset_id=args.dataset_id,
        dataset_path=args.dataset_path,
        query_json=args.query_json,
        dataset_index_path=args.dataset_index_path,
    )
    resolved_path = _resolve_dataset_path(spec)
    if resolved_path and spec.dataset_path is None:
        spec = DatasetSpec(
            dataset_id=spec.dataset_id,
            dataset_path=resolved_path,
            query_json=spec.query_json,
            dataset_index_path=spec.dataset_index_path,
        )
    dataset = _load_dataset(spec)
    run_config = build_run_config(args, run_prefix="manifold")
    events = EventWriter(enabled=run_config.emit_events, log_path=run_config.event_log_path)
    _train_manifold(
        dataset=dataset,
        run_config=run_config,
        events=events,
        dataset_spec=spec,
        args=args,
    )


if __name__ == "__main__":
    main()
