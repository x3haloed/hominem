from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import yaml

from core.data.schema import REWARD_DIMENSIONS
from core.reward_model.dataset import RewardTorchDataset, load_reward_samples, train_val_split


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
) -> Dict[str, DataLoader]:
    samples = load_reward_samples(data_path)
    train_samples, val_samples = train_val_split(
        samples, val_ratio=val_ratio, seed=seed
    )

    train_dataset = RewardTorchDataset(
        train_samples, tokenizer=tokenizer, max_length=max_length
    )
    val_dataset = RewardTorchDataset(
        val_samples, tokenizer=tokenizer, max_length=max_length
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

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(tuple(REWARD_DIMENSIONS)),
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
    )

    total_steps = len(dataloaders["train"]) * num_epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.MSELoss()

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

            loss = loss_fn(logits, labels)
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
                loss = loss_fn(logits, labels)

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
        "training_data": str(data_path),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "val_ratio": val_ratio,
        "seed": seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

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


