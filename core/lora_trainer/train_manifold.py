#!/usr/bin/env python3
"""
Train a 6-axis emotion manifold head from labeled unified-theory shards.

Assumptions:
- Input JSONL records follow the shard schema with `history`, `target`, and `labels`.
- History is already trimmed to the last few turns (as produced by the generator).
- Labels contain the 6 manifold axes: valence, arousal, dominance,
  predictive_discrepancy, temporal_directionality, social_broadcast.

This is a minimal Trainer-based entry point to get a regression head checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from core.data.shard_loader import MissingDatasetError, ShardLoader, print_shard_summary


MANIFOLD_KEYS = [
    "valence",
    "arousal",
    "dominance",
    "predictive_discrepancy",
    "temporal_directionality",
    "social_broadcast",
]


def record_to_text(rec: Dict) -> str:
    """Concatenate history and target into a plain-text training sample."""
    # Clamp history to last 3 turns as per spec (section 3.1: "history up to 3 turns")
    history = rec.get("history", [])
    if len(history) > 3:
        history = history[-3:]  # Keep only the last 3 turns

    turns = history + [rec.get("target", {})]
    parts = []
    for t in turns:
        role = t.get("role", "user")
        content = t.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def build_dataset(records: List[Dict]) -> Dataset:
    texts = []
    labels = []
    for rec in records:
        lbl = rec.get("labels", {})
        if not all(k in lbl for k in MANIFOLD_KEYS):
            continue
        texts.append(record_to_text(rec))
        labels.append([float(lbl[k]) for k in MANIFOLD_KEYS])
    if not texts:
        raise ValueError("No usable records with manifold labels.")
    return Dataset.from_dict({"text": texts, "labels": labels})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion manifold regressor head.")
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=["data/processed_datasets_unified"],
        help="Root folders containing labeled shard datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ultrachat_trajectories", "ultrachat_synthetic_trajectories"],
        help="Names of dataset directories to load from the data roots",
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
        "--output-dir",
        default="artifacts/manifold_head",
        help="Where to save the checkpoint",
    )
    parser.add_argument(
        "--model-id",
        default="distilbert-base-uncased",
        help="Base model ID for regression head",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if args.record_limit < 0 or args.min_records < 0:
        raise SystemExit("record-limit and min-records must be non-negative")
    if not 0 <= args.validation_split < 1:
        raise SystemExit("validation-split must be between 0 (inclusive) and 1 (exclusive)")

    loader = ShardLoader(root_paths=[Path(p) for p in args.data_roots], dataset_filters=args.datasets)
    try:
        max_records = args.record_limit if args.record_limit > 0 else None
        records, summary = loader.load_records(MANIFOLD_KEYS, max_records=max_records)
    except MissingDatasetError as exc:
        raise SystemExit(f"Dataset loading failed: {exc}")

    if summary.usable_records == 0:
        raise SystemExit("No usable records found after enforcing required manifold labels.")
    if args.min_records > 0 and summary.usable_records < args.min_records:
        raise SystemExit(
            f"Need at least {args.min_records} usable records but only found {summary.usable_records}."
        )

    print_shard_summary(summary)

    dataset = build_dataset(records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    # Handle models without pad token (like GPT models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(MANIFOLD_KEYS),
        problem_type="regression",
        trust_remote_code=True,
    )
    model.to(torch.device("mps"))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True)
    if args.validation_split > 0:
        split = tokenized.train_test_split(
            test_size=args.validation_split, seed=args.validation_seed
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = tokenized
        eval_dataset = None

    for subset in (train_dataset, eval_dataset):
        if subset is not None:
            subset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(
        f"⚙️  Training samples: {len(train_dataset)}"
        + (f", validation samples: {len(eval_dataset)}" if eval_dataset else "")
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.squeeze(-1) if preds.ndim == 3 else preds
        mse = np.mean((preds - labels) ** 2)
        return {"mse": float(mse)}

    eval_strategy = "steps" if eval_dataset else "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        eval_strategy=eval_strategy,
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Manifold head trained and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
