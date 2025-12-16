#!/usr/bin/env python3
"""
Train a 7-way regime classifier head from labeled unified-theory shards.

Assumptions:
- Input JSONL records follow the shard schema with `history`, `target`, and `labels`.
- Labels contain soft probabilities for 7 regimes; we train a classifier to predict
  the argmax regime (single-label) for simplicity. For full soft-label training,
  replace with a custom loss.
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
from sklearn.metrics import accuracy_score, f1_score

from core.data.shard_loader import MissingDatasetError, ShardLoader, print_shard_summary


REGIME_KEYS = [
    "regime_support",
    "regime_conflict",
    "regime_problem_solving",
    "regime_truth_seeking",
    "regime_crisis",
    "regime_play",
    "regime_boundary",
]


def record_to_text(rec: Dict) -> str:
    turns = rec.get("history", []) + [rec.get("target", {})]
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
        if not all(k in lbl for k in REGIME_KEYS):
            continue
        probs = np.array([float(lbl[k]) for k in REGIME_KEYS])
        if probs.sum() <= 0:
            continue
        label_idx = int(probs.argmax())
        texts.append(record_to_text(rec))
        labels.append(label_idx)
    if not texts:
        raise ValueError("No usable records with regime labels.")
    return Dataset.from_dict({"text": texts, "labels": labels})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regime classifier head.")
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
        default="artifacts/regime_classifier",
        help="Where to save the checkpoint",
    )
    parser.add_argument(
        "--model-id",
        default="distilbert-base-uncased",
        help="Base model ID for classifier head",
    )
    parser.add_argument("--batch-size", type=int, default=8)
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
        records, summary = loader.load_records(REGIME_KEYS, max_records=max_records)
    except MissingDatasetError as exc:
        raise SystemExit(f"Dataset loading failed: {exc}")

    if summary.usable_records == 0:
        raise SystemExit("No usable records found after enforcing required regime labels.")
    if args.min_records > 0 and summary.usable_records < args.min_records:
        raise SystemExit(
            f"Need at least {args.min_records} usable records but only found {summary.usable_records}."
        )

    print_shard_summary(summary)

    dataset = build_dataset(records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(REGIME_KEYS),
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
        preds = preds.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    evaluation_strategy = "steps" if eval_dataset else "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        evaluation_strategy=evaluation_strategy,
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
    print(f"✅ Regime classifier trained and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
