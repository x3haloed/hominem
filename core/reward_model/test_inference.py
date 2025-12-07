from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.data.schema import REWARD_DIMENSIONS, RewardVector


def load_samples(path: Path, limit: int) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        return []
    return random.sample(records, min(limit, len(records)))


def run_inference(
    *,
    model_dir: Path,
    data_path: Path,
    samples: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    records = load_samples(data_path, samples)
    if not records:
        print("No records found; nothing to test.")
        return

    for record in records:
        prompt = record.get("prompt", "")
        response = record.get("response", "")
        reward_data = record.get("reward") or {}
        teacher_reward = RewardVector.from_mapping(reward_data)

        text = f"User: {prompt}\nAssistant: {response}"
        encoded = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits.squeeze(0).cpu().tolist()

        print("-" * 60)
        print(f"ID: {record.get('id')}")
        for idx, dim in enumerate(REWARD_DIMENSIONS):
            teacher_val = getattr(teacher_reward, dim)
            pred_val = float(logits[idx])
            print(f"{dim:20s} teacher={teacher_val:+.3f}  pred={pred_val:+.3f}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test reward model predictions against teacher labels."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/reward_model/default/model"),
        help="Directory containing the trained reward model and tokenizer.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/labeled/reward_samples.jsonl"),
        help="Labeled reward data JSONL to sample from.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of random samples to test.",
    )

    args = parser.parse_args(argv)
    run_inference(model_dir=args.model_dir, data_path=args.data, samples=args.samples)


if __name__ == "__main__":  # pragma: no cover
    main()


