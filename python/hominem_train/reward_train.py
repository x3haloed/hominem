"""CLI entry point for reward model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hominem_train.common import messages_to_text, normalize_messages
from hominem_train.events import EventWriter
from hominem_train.train_base import add_common_args, build_run_config, TrainingRunConfig
from hominem_train.trainer_utils import (
    DatasetSpec,
    MetricsCallback,
    load_dataset,
    resolve_dataset_path,
    write_manifest,
)


DEFAULT_REWARD_MODEL_ID = "bert-base-uncased"

ADEQUACY_KEYS = [
    "social_coherence",
    "agency_support",
    "narrative_alignment",
    "curiosity",
    "harm_avoidance",
]
LABEL_BOUNDS = {key: (0.0, 1.0) for key in ADEQUACY_KEYS}

PRESETS = {
    "distilbert": {
        "model_id": "distilbert-base-uncased",
        "batch_size": 16,
        "grad_accum": 1,
        "lr": 5e-5,
    },
    "bert_base": {
        "model_id": "bert-base-uncased",
        "batch_size": 8,
        "grad_accum": 1,
        "lr": 5e-5,
    },
    "roberta_base": {
        "model_id": "roberta-base",
        "batch_size": 8,
        "grad_accum": 1,
        "lr": 5e-5,
    },
}


def _messages_to_text(messages: List[Dict[str, str]], *, max_turns: int) -> str:
    normalized = normalize_messages(messages, drop_system=True)
    if max_turns > 0 and len(normalized) > max_turns:
        normalized = normalized[-max_turns:]
    return messages_to_text(normalized)


def _extract_reward_labels(scores: Dict[str, Any]) -> Optional[List[float]]:
    out: List[float] = []
    for key in ADEQUACY_KEYS:
        if key not in scores:
            return None
        try:
            val = float(scores[key])
        except (TypeError, ValueError):
            return None
        lo, hi = LABEL_BOUNDS[key]
        if val < lo or val > hi:
            return None
        out.append(val)
    return out


def _load_reward_records(
    records: Iterable[Dict[str, Any]],
    *,
    max_turns: int,
    min_records: int,
) -> Tuple[List[str], List[List[float]], Dict[str, int]]:
    texts: List[str] = []
    labels: List[List[float]] = []
    stats = {
        "total": 0,
        "usable": 0,
        "missing_messages": 0,
        "missing_scores": 0,
        "missing_keys": 0,
        "out_of_bounds": 0,
    }
    for row in records:
        stats["total"] += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            stats["missing_messages"] += 1
            continue
        scores = row.get("scores")
        if not isinstance(scores, dict):
            stats["missing_scores"] += 1
            continue
        label_vec = _extract_reward_labels(scores)
        if label_vec is None:
            if all(k in scores for k in ADEQUACY_KEYS):
                stats["out_of_bounds"] += 1
            else:
                stats["missing_keys"] += 1
            continue
        texts.append(_messages_to_text(messages, max_turns=max_turns))
        labels.append(label_vec)
        stats["usable"] += 1

    if stats["usable"] == 0:
        raise SystemExit("No usable records found with adequacy labels.")
    if min_records > 0 and stats["usable"] < min_records:
        raise SystemExit(
            f"Need at least {min_records} usable records but only found {stats['usable']}."
        )
    return texts, labels, stats


def _compute_metrics(preds, labels) -> Dict[str, Dict[str, Any]]:
    import numpy as np

    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    metrics: Dict[str, Dict[str, Any]] = {"overall": {}, "per_axis": {}, "distribution": {}}

    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    rmse = float(np.sqrt(mse))
    metrics["overall"] = {"mse": float(mse), "mae": float(mae), "rmse": rmse}

    per_axis = {}
    for idx, key in enumerate(ADEQUACY_KEYS):
        p = preds[:, idx]
        y = labels[:, idx]
        mse_k = np.mean((p - y) ** 2)
        mae_k = np.mean(np.abs(p - y))
        rmse_k = float(np.sqrt(mse_k))
        mean_err = float(np.mean(p - y))
        std_err = float(np.std(p - y))
        corr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else 0.0
        per_axis[key] = {
            "mse": float(mse_k),
            "mae": float(mae_k),
            "rmse": rmse_k,
            "correlation": corr,
            "mean_error": mean_err,
            "std_error": std_err,
        }
    metrics["per_axis"] = per_axis
    corrs = [v["correlation"] for v in per_axis.values()]
    metrics["overall"]["mean_correlation"] = float(sum(corrs) / len(corrs)) if corrs else 0.0

    metrics["distribution"] = {
        "predictions": {
            "mean": preds.mean(axis=0).tolist(),
            "std": preds.std(axis=0).tolist(),
            "min": preds.min(axis=0).tolist(),
            "max": preds.max(axis=0).tolist(),
        },
        "ground_truth": {
            "mean": labels.mean(axis=0).tolist(),
            "std": labels.std(axis=0).tolist(),
            "min": labels.min(axis=0).tolist(),
            "max": labels.max(axis=0).tolist(),
        },
    }
    return metrics


def _train_reward(
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
            "model_type": "reward",
            "dataset_id": dataset_spec.dataset_id,
            "dataset_path": str(dataset_spec.dataset_path) if dataset_spec.dataset_path else None,
            "row_count": len(dataset),
        },
    )

    texts, labels, stats = _load_reward_records(
        dataset,
        max_turns=args.max_turns,
        min_records=args.min_records,
    )
    print(
        "📦 Loaded records: total={total} usable={usable} missing_messages={missing_messages} "
        "missing_scores={missing_scores} missing_keys={missing_keys} out_of_bounds={out_of_bounds}".format(
            **stats
        )
    )
    hf_dataset = Dataset.from_dict({"text": texts, "labels": labels})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(ADEQUACY_KEYS),
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
        if preds.ndim == 3:
            preds = preds.squeeze(-1)
        mse = float(np.mean((preds - lbls) ** 2))
        mae = float(np.mean(np.abs(preds - lbls)))
        corrs = []
        for idx in range(lbls.shape[1]):
            p = preds[:, idx]
            y = lbls[:, idx]
            if np.std(p) > 0 and np.std(y) > 0:
                corrs.append(float(np.corrcoef(p, y)[0, 1]))
        mean_corr = float(np.mean(corrs)) if corrs else 0.0
        return {"mse": mse, "mae": mae, "mean_correlation": mean_corr}

    eval_strategy = "steps" if eval_dataset else "no"
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0 and args.warmup_ratio > 0 and eval_strategy != "no":
        steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
        total_steps = int(steps_per_epoch * args.num_epochs)
        warmup_steps = max(1, int(total_steps * float(args.warmup_ratio)))

    class HuberTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.SmoothL1Loss()
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

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
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="mean_correlation",
        greater_is_better=True,
    )

    trainer = HuberTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback(events=events, run_id=run_id, model_type="reward")],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if eval_dataset and args.eval_json:
        eval_output = trainer.predict(eval_dataset)
        eval_metrics = _compute_metrics(eval_output.predictions, eval_output.label_ids)
        out_path = Path(args.eval_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(eval_metrics, indent=2))
        print(f"📊 Wrote eval summary to {out_path}")

    events.emit(
        "TrainingRunCompleted",
        {
            "run_id": run_id,
            "model_type": "reward",
            "status": "success",
        },
    )
    manifest_path = write_manifest(
        output_dir,
        model_type="reward",
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
            "min_records": args.min_records,
            "max_turns": args.max_turns,
            "warmup_steps": warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "preset": args.preset,
        },
    )
    events.emit(
        "ModelArtifactProduced",
        {
            "run_id": run_id,
            "model_type": "reward",
            "manifest_path": str(manifest_path),
        },
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train reward regressor.")
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
        "--model-id",
        default=DEFAULT_REWARD_MODEL_ID,
        help="Base model ID for regression head",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Use a preset model/batch/lr configuration",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=0, help="0 = keep all messages")
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--min-records", type=int, default=0)
    parser.add_argument("--eval-json", default="", help="Optional path to save eval summary JSON")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=1)
    add_common_args(parser)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.preset:
        preset = PRESETS[args.preset]
        args.model_id = preset["model_id"]
        args.batch_size = preset["batch_size"]
        args.gradient_accumulation_steps = preset["grad_accum"]
        args.lr = preset["lr"]

    if not 0 <= args.validation_split < 1:
        raise SystemExit("validation-split must be between 0 (inclusive) and 1 (exclusive)")

    spec = DatasetSpec(
        dataset_id=args.dataset_id,
        dataset_path=args.dataset_path,
        query_json=args.query_json,
        dataset_index_path=args.dataset_index_path,
    )
    resolved_path = resolve_dataset_path(spec)
    if resolved_path and spec.dataset_path is None:
        spec = DatasetSpec(
            dataset_id=spec.dataset_id,
            dataset_path=resolved_path,
            query_json=spec.query_json,
            dataset_index_path=spec.dataset_index_path,
        )
    dataset = load_dataset(spec)
    run_config = build_run_config(args, run_prefix="reward")
    events = EventWriter(enabled=run_config.emit_events, log_path=run_config.event_log_path)
    _train_reward(
        dataset=dataset,
        run_config=run_config,
        events=events,
        dataset_spec=spec,
        args=args,
    )


if __name__ == "__main__":
    main()
