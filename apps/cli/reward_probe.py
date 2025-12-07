from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.data.schema import REWARD_DIMENSIONS, REWARD_MODEL_TARGETS


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_reward_model(model_dir: Path) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def score_candidate(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    prompt: str,
    response: str,
    max_length: int,
) -> Dict[str, float]:
    text = f"User: {prompt}\nAssistant: {response}"
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits.squeeze(0).cpu().tolist()

    scores: Dict[str, float] = {}
    for idx, name in enumerate(REWARD_MODEL_TARGETS):
        if idx >= len(logits):
            break
        scores[name] = float(logits[idx])
    return scores


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the reward model on a prompt and one or more candidate responses."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/reward_model/default/model"),
        help="Directory containing the trained reward model and tokenizer.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt text.",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        action="append",
        required=True,
        help="Candidate response text. Specify multiple times to compare several responses.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for reward model inputs.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    tokenizer, model, device = load_reward_model(args.model_dir)

    for idx, cand in enumerate(args.candidate, start=1):
        scores = score_candidate(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt=args.prompt,
            response=cand,
            max_length=args.max_length,
        )

        print("-" * 60)
        print(f"Candidate {idx}")
        print("RESPONSE:")
        print(cand)
        print("\nREWARD VECTOR:")
        for dim in REWARD_DIMENSIONS:
            val = scores.get(dim, 0.0)
            print(f"{dim:20s} {val:+.3f}")

        intensity = scores.get("reward_intensity")
        safety = scores.get("safety_score")
        print("\nSCALARS:")
        if intensity is not None:
            print(f"{'reward_intensity':20s} {intensity:+.3f}")
        else:
            print("reward_intensity: N/A")
        if safety is not None:
            print(f"{'safety_score':20s} {safety:+.3f}")
        else:
            print("safety_score: N/A")
        print()


if __name__ == "__main__":  # pragma: no cover
    main()



