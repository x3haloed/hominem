from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import yaml

from core.data.schema import REWARD_DIMENSIONS, REWARD_MODEL_TARGETS
from core.reward_model.dataset import (
    RewardTorchDataset,
    load_reward_samples,
    train_val_split,
)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_dataloaders(
    data_path: Path,
    tokenizer,
    *,
    max_length: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    label_mean: torch.Tensor | None = None,
    label_std: torch.Tensor | None = None,
) -> Dict[str, DataLoader]:
    samples = load_reward_samples(data_path)
    train_samples, val_samples = train_val_split(
        samples, val_ratio=val_ratio, seed=seed
    )

    train_dataset = RewardTorchDataset(
        train_samples,
        tokenizer=tokenizer,
        max_length=max_length,
        label_mean=label_mean,
        label_std=label_std,
    )
    val_dataset = RewardTorchDataset(
        val_samples,
        tokenizer=tokenizer,
        max_length=max_length,
        label_mean=label_mean,
        label_std=label_std,
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    }


def train(
    *,
    data_path: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    cfg = load_config(config_path)

    model_id = cfg.get("model_id", "distilbert-base-uncased")
    max_length = int(cfg.get("max_length", 512))
    batch_size = int(cfg.get("batch_size", 4))
    num_epochs = int(cfg.get("num_epochs", 1))
    learning_rate = float(cfg.get("learning_rate", 5e-5))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    seed = int(cfg.get("seed", 42))
    standardize_labels = bool(cfg.get("standardize_labels", False))
    per_dim_weights_cfg = cfg.get("per_dim_weights") or None
    warmup_steps = int(cfg.get("warmup_steps", 0))

    torch.manual_seed(seed)

    label_mean_t: torch.Tensor | None = None
    label_std_t: torch.Tensor | None = None

    # Precompute label mean/std on the full dataset (train+val) and pass to splits.
    if standardize_labels:
        raw_samples = load_reward_samples(data_path)
        if not raw_samples:
            raise ValueError("No reward samples found for standardization.")
        label_matrix = np.array(
            [
                [getattr(s.reward, name) for name in REWARD_MODEL_TARGETS]
                for s in raw_samples
            ],
            dtype=np.float32,
        )
        means = label_matrix.mean(axis=0)
        stds = label_matrix.std(axis=0)
        stds = np.maximum(stds, 1e-6)  # avoid div-by-zero
        label_mean_t = torch.tensor(means, dtype=torch.float32)
        label_std_t = torch.tensor(stds, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(tuple(REWARD_MODEL_TARGETS)),
        problem_type="regression",
    )

    # Prefer Apple Silicon GPU (MPS) on macOS, then CUDA, then CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    dataloaders = create_dataloaders(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        label_mean=label_mean_t,
        label_std=label_std_t,
    )

    total_steps = len(dataloaders["train"]) * num_epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    per_dim_weights = None
    if per_dim_weights_cfg is not None:
        per_dim_weights = torch.tensor(
            [float(x) for x in per_dim_weights_cfg],
            dtype=torch.float32,
            device=device,
        )
        if per_dim_weights.numel() != len(tuple(REWARD_MODEL_TARGETS)):
            raise ValueError(
                f"per_dim_weights length {per_dim_weights.numel()} "
                f"does not match number of targets {len(tuple(REWARD_MODEL_TARGETS))}."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        model.train()
        train_loss_sum = 0.0
        for step, batch in enumerate(dataloaders["train"], start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits
            diff = logits - labels
            if per_dim_weights is not None:
                # Per-dimension weighting, then mean over dims and batch.
                loss = (diff.pow(2) * per_dim_weights).mean()
            else:
                loss = diff.pow(2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()

            if step % 10 == 0 or step == len(dataloaders["train"]):
                avg_loss = train_loss_sum / step
                print(f"  Step {step}/{len(dataloaders['train'])} - Train loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in dataloaders["val"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                outputs = model(**batch)
                logits = outputs.logits
                diff = logits - labels
                if per_dim_weights is not None:
                    loss = (diff.pow(2) * per_dim_weights).mean()
                else:
                    loss = diff.pow(2).mean()

                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(1, val_batches)
        print(f"Validation loss: {val_loss:.4f}")

    # Save model and tokenizer
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Metadata
    metadata: Dict[str, Any] = {
        "model_id": model_id,
        "dimensions": list(REWARD_DIMENSIONS),
        "targets": list(REWARD_MODEL_TARGETS),
        "training_data": str(data_path),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "val_ratio": val_ratio,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if standardize_labels and label_mean_t is not None and label_std_t is not None:
        metadata["label_mean"] = metadata.get("label_mean") or label_mean_t.tolist()
        metadata["label_std"] = metadata.get("label_std") or label_std_t.tolist()
    if per_dim_weights is not None:
        metadata["per_dim_weights"] = per_dim_weights.cpu().tolist()

    with (output_dir / "METADATA.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved reward model to {model_dir}")
    print(f"Saved metadata to {output_dir / 'METADATA.json'}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a reward model on labeled JSONL data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/labeled/reward_samples.jsonl"),
        help="Path to labeled reward JSONL data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training/reward_model.yaml"),
        help="Training configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reward_model/default"),
        help="Directory to save model and metadata.",
    )

    args = parser.parse_args(argv)
    train(data_path=args.data, config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()


