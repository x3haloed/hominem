"""CLI entry point for regime classifier training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hominem_train.events import EventWriter
from hominem_train.train_base import add_common_args, build_run_config, TrainingRunConfig
from hominem_train.trainer_utils import (
    DatasetSpec,
    MetricsCallback,
    build_text_label_dataset,
    load_dataset,
    resolve_dataset_path,
    write_manifest,
)


DEFAULT_REGIME_MODEL_ID = "bert-base-uncased"

REGIME_KEYS = [
    "regime_support",
    "regime_conflict",
    "regime_problem_solving",
    "regime_truth_seeking",
    "regime_crisis",
    "regime_play",
    "regime_boundary",
]
LABEL_BOUNDS = {key: (0.0, 1.0) for key in REGIME_KEYS}

def _train_regime(
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
            "model_type": "regime",
            "dataset_id": dataset_spec.dataset_id,
            "dataset_path": str(dataset_spec.dataset_path) if dataset_spec.dataset_path else None,
            "row_count": len(dataset),
        },
    )

    texts, labels = build_text_label_dataset(
        dataset,
        label_keys=REGIME_KEYS,
        label_bounds=LABEL_BOUNDS,
        max_history_turns=args.max_history_turns,
        record_limit=args.record_limit,
        min_records=args.min_records,
        empty_message_error="No usable records with regime labels.",
    )
    hf_dataset = Dataset.from_dict({"text": texts, "labels": labels})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(REGIME_KEYS),
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback(events=events, run_id=run_id, model_type="regime")],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    events.emit(
        "TrainingRunCompleted",
        {
            "run_id": run_id,
            "model_type": "regime",
            "status": "success",
        },
    )
    manifest_path = write_manifest(
        output_dir,
        model_type="regime",
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
            "model_type": "regime",
            "manifest_path": str(manifest_path),
        },
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train regime classifier.")
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
        default=DEFAULT_REGIME_MODEL_ID,
        help="Base model ID for regime classification head",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument("--num-epochs", type=int, default=5)
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
    resolved_path = resolve_dataset_path(spec)
    if resolved_path and spec.dataset_path is None:
        spec = DatasetSpec(
            dataset_id=spec.dataset_id,
            dataset_path=resolved_path,
            query_json=spec.query_json,
            dataset_index_path=spec.dataset_index_path,
        )
    dataset = load_dataset(spec)
    run_config = build_run_config(args, run_prefix="regime")
    events = EventWriter(enabled=run_config.emit_events, log_path=run_config.event_log_path)
    _train_regime(
        dataset=dataset,
        run_config=run_config,
        events=events,
        dataset_spec=spec,
        args=args,
    )


if __name__ == "__main__":
    main()
