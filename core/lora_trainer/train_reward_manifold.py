#!/usr/bin/env python3
"""
Train a reward-manifold regression head from labeled reward samples.

Inputs:
- JSONL files with fields: messages (list of {role, content}) and scores (dict).
- Scores include adequacy axes: social_coherence, agency_support, narrative_alignment,
  curiosity, harm_avoidance.

Outputs:
- HuggingFace-style checkpoint in the output dir.
- Optional eval JSON summary (mirrors prior manifold evaluation artifacts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


ADEQUACY_KEYS = [
    "social_coherence",
    "agency_support",
    "narrative_alignment",
    "curiosity",
    "harm_avoidance",
]


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


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _messages_to_text(messages: List[Dict[str, str]], *, max_turns: int) -> str:
    if max_turns > 0 and len(messages) > max_turns:
        messages = messages[-max_turns:]
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _load_records(paths: List[Path], *, max_turns: int) -> Tuple[List[str], List[List[float]], Dict[str, int]]:
    texts: List[str] = []
    labels: List[List[float]] = []
    stats = {
        "total": 0,
        "usable": 0,
        "missing_messages": 0,
        "missing_scores": 0,
        "missing_keys": 0,
    }
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["total"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    stats["missing_messages"] += 1
                    continue
                scores = row.get("scores")
                if not isinstance(scores, dict):
                    stats["missing_scores"] += 1
                    continue
                if not all(k in scores for k in ADEQUACY_KEYS):
                    stats["missing_keys"] += 1
                    continue
                texts.append(_messages_to_text(messages, max_turns=max_turns))
                labels.append([float(scores[k]) for k in ADEQUACY_KEYS])
                stats["usable"] += 1
    return texts, labels, stats


def _compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Dict]:
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    metrics: Dict[str, Dict] = {"overall": {}, "per_axis": {}, "distribution": {}}

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
    metrics["overall"]["mean_correlation"] = float(np.mean(corrs)) if corrs else 0.0

    metrics["distribution"] = {
        "predictions": {
            "mean": np.mean(preds, axis=0).tolist(),
            "std": np.std(preds, axis=0).tolist(),
            "min": np.min(preds, axis=0).tolist(),
            "max": np.max(preds, axis=0).tolist(),
        },
        "ground_truth": {
            "mean": np.mean(labels, axis=0).tolist(),
            "std": np.std(labels, axis=0).tolist(),
            "min": np.min(labels, axis=0).tolist(),
            "max": np.max(labels, axis=0).tolist(),
        },
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reward-manifold adequacy regressor.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/labeled/reward_samples_lowered_k.jsonl"],
        help="One or more labeled JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/reward_manifold",
        help="Where to save the checkpoint",
    )
    parser.add_argument(
        "--model-id",
        default="bert-base-uncased",
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
    args = parser.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        args.model_id = preset["model_id"]
        args.batch_size = preset["batch_size"]
        args.gradient_accumulation_steps = preset["grad_accum"]
        args.lr = preset["lr"]

    if not 0 <= args.validation_split < 1:
        raise SystemExit("validation-split must be between 0 (inclusive) and 1 (exclusive)")

    input_paths = [Path(p) for p in args.inputs]
    texts, labels, stats = _load_records(input_paths, max_turns=args.max_turns)
    if stats["usable"] == 0:
        raise SystemExit("No usable records found with adequacy labels.")
    if args.min_records > 0 and stats["usable"] < args.min_records:
        raise SystemExit(
            f"Need at least {args.min_records} usable records but only found {stats['usable']}."
        )

    print(
        "📦 Loaded records: total={total} usable={usable} missing_messages={missing_messages} "
        "missing_scores={missing_scores} missing_keys={missing_keys}".format(**stats)
    )

    dataset = Dataset.from_dict({"text": texts, "labels": labels})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(ADEQUACY_KEYS),
        problem_type="regression",
        trust_remote_code=True,
    )
    model.to(_device())

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
    # Compute warmup steps if ratio requested.
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0 and args.warmup_ratio > 0 and eval_strategy != "no":
        steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
        total_steps = int(steps_per_epoch * args.num_epochs)
        warmup_steps = max(1, int(total_steps * float(args.warmup_ratio)))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        eval_strategy=eval_strategy,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="mean_correlation",
        greater_is_better=True,
    )

    class HuberTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.SmoothL1Loss()
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = HuberTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if eval_dataset and args.eval_json:
        eval_output = trainer.predict(eval_dataset)
        eval_metrics = _compute_metrics(eval_output.predictions, eval_output.label_ids)
        out_path = Path(args.eval_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(eval_metrics, indent=2))
        print(f"📊 Wrote eval summary to {out_path}")

    print(f"✅ Reward-manifold head trained and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
