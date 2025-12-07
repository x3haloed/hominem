from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from core.data.schema import REWARD_DIMENSIONS
from core.lora_trainer.train_dpo import (
    load_config,
    prepare_model_and_tokenizer,
    select_device,
    set_seed,
)


@dataclass
class OnlinePreferenceSample:
    """Single online preference pair with weighting metadata.

    - prompt: user prompt text.
    - chosen: higher-scoring candidate response.
    - rejected: lower-scoring candidate response.
    - reward_intensity: how strongly this example should drive learning.
    - safety_score: safety signal used for gating / diagnostics.
    """

    prompt: str
    chosen: str
    rejected: str
    reward_intensity: float
    safety_score: float


def _format_chat(tokenizer: AutoTokenizer, prompt: str, response: str) -> str:
    """Format (prompt, response) into a chat-style string compatible with train_dpo."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return f"User: {prompt}\nAssistant: {response}"


class OnlinePreferenceDataset(Dataset):
    """Torch dataset for online DPO-style pairwise training with per-example weights."""

    def __init__(
        self,
        samples: List[OnlinePreferenceSample],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self._samples = samples
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._samples)

    def _build_inputs(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        text = _format_chat(self._tokenizer, prompt, response)
        enc = self._tokenizer(
            text,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._samples[idx]
        chosen_inputs = self._build_inputs(sample.prompt, sample.chosen)
        rejected_inputs = self._build_inputs(sample.prompt, sample.rejected)

        weight = max(float(sample.reward_intensity), 0.0)

        return {
            "chosen": chosen_inputs,
            "rejected": rejected_inputs,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def online_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collator that stacks chosen/rejected tensors and weights."""

    def stack_side(side: str) -> Dict[str, torch.Tensor]:
        keys = batch[0][side].keys()
        return {
            k: torch.stack([item[side][k] for item in batch], dim=0)
            for k in keys
        }

    weights = torch.stack([item["weight"] for item in batch], dim=0)

    return {
        "chosen": stack_side("chosen"),
        "rejected": stack_side("rejected"),
        "weight": weights,
    }


def weighted_dpo_loss(
    model: AutoModelForCausalLM,
    batch: Dict[str, Any],
    *,
    beta: float,
) -> torch.Tensor:
    """DPO-style loss with per-example weights.

    L_i = -log σ(β * (logp_chosen_i - logp_rejected_i))
    L_total = sum_i w_i * L_i / sum_i w_i
    """

    def log_probs_for_side(side_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(
            input_ids=side_batch["input_ids"],
            attention_mask=side_batch["attention_mask"],
            labels=side_batch["labels"],
        )
        seq_lengths = side_batch["attention_mask"].sum(dim=-1)
        loglik = -outputs.loss * seq_lengths
        return loglik

    chosen_logp = log_probs_for_side(batch["chosen"])
    rejected_logp = log_probs_for_side(batch["rejected"])

    diff = chosen_logp - rejected_logp
    logits = beta * diff
    per_example_loss = -torch.nn.functional.logsigmoid(logits)

    weights: torch.Tensor = batch["weight"].to(per_example_loss.device)
    weights = torch.clamp(weights, min=0.0)
    if torch.all(weights == 0):
        # Avoid NaNs if all examples were effectively gated out.
        return per_example_loss.mean()

    weighted = per_example_loss * weights
    return weighted.sum() / weights.sum()


def load_online_samples_from_logs(
    log_dir: Path,
    *,
    min_reward_intensity: float,
    safety_threshold: float,
    max_records: int | None = None,
) -> List[OnlinePreferenceSample]:
    """Load online interaction logs and convert them into preference pairs.

    For each log record produced by apps/http/self_train_server.py we:
      - Take the chosen candidate as the positive example.
      - Select the lowest-scoring candidate as the negative example.
      - Use the chosen candidate's reward_intensity and safety_score for weighting/gating.
    """
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory '{log_dir}' does not exist.")

    samples: List[OnlinePreferenceSample] = []
    log_files: List[Path] = sorted(log_dir.glob("session_*.jsonl"))
    for path in log_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompt = record.get("prompt", "")
                if not prompt:
                    continue

                candidates: List[Dict[str, Any]] = record.get("candidates") or []
                if not candidates:
                    continue

                # Identify chosen and worst candidates by scalar_score.
                chosen = record.get("chosen") or max(
                    candidates, key=lambda c: c.get("scalar_score", float("-inf"))
                )
                worst = min(candidates, key=lambda c: c.get("scalar_score", float("inf")))

                reward = chosen.get("reward") or {}
                reward_intensity = float(reward.get("reward_intensity", 1.0))
                safety_score = float(reward.get("safety_score", 1.0))

                # Safety Gate: skip examples that fall below the safety threshold.
                if safety_score < safety_threshold:
                    continue

                if reward_intensity < min_reward_intensity:
                    continue

                samples.append(
                    OnlinePreferenceSample(
                        prompt=prompt,
                        chosen=str(chosen.get("text", "")),
                        rejected=str(worst.get("text", "")),
                        reward_intensity=reward_intensity,
                        safety_score=safety_score,
                    )
                )

                if max_records is not None and len(samples) >= max_records:
                    return samples

    if not samples:
        raise ValueError(
            f"No usable online samples found in '{log_dir}'. "
            "Ensure the self_train server has logged interactions with reward_intensity and safety_score."
        )

    return samples


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perform an online LoRA update using replayed self-training logs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/lora_dpo.yaml",
        help="Path to LoRA/DPO training config YAML (reused for model + optimizer settings).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/online_feedback",
        help="Directory containing session_*.jsonl logs from the self_train server.",
    )
    parser.add_argument(
        "--min-reward-intensity",
        type=float,
        default=0.0,
        help="Minimum reward_intensity required for an example to contribute to training.",
    )
    parser.add_argument(
        "--safety-threshold",
        type=float,
        default=-0.2,
        help="Minimum safety_score required; examples below this are skipped.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap on the number of log records to consume (0 = no limit).",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    arg_parser = parse_args()
    args = arg_parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    train_cfg: Dict[str, Any] = cfg["training"]
    model_cfg: Dict[str, Any] = cfg["model"]

    set_seed(train_cfg.get("seed", 42))

    log_dir = Path(args.log_dir)
    max_records = args.max_records if args.max_records and args.max_records > 0 else None

    samples = load_online_samples_from_logs(
        log_dir,
        min_reward_intensity=float(args.min_reward_intensity),
        safety_threshold=float(args.safety_threshold),
        max_records=max_records,
    )

    print(f"Loaded {len(samples)} online preference pairs from {log_dir}")

    device = select_device()
    model, tokenizer = prepare_model_and_tokenizer(cfg, device)

    dataset = OnlinePreferenceDataset(
        samples,
        tokenizer=tokenizer,
        max_length=int(model_cfg["max_length"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=online_collate_fn,
    )

    num_training_steps = int(train_cfg["num_epochs"]) * math.ceil(
        len(dataset) / int(train_cfg["batch_size"])
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        num_training_steps=num_training_steps,
    )

    model.train()
    global_step = 0
    logging_steps = int(train_cfg.get("logging_steps", 10))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    beta = float(train_cfg.get("beta", 0.1))

    for epoch in range(int(train_cfg["num_epochs"])):
        epoch_loss = 0.0
        for batch in dataloader:
            # Move tensors to device.
            batch = {
                side: {k: v.to(device) for k, v in side_batch.items()}
                if isinstance(side_batch, dict)
                else side_batch.to(device)
                for side, side_batch in batch.items()
            }

            loss = weighted_dpo_loss(model, batch, beta=beta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / max(1, global_step)
                print(
                    f"Epoch {epoch + 1}, step {global_step}/{num_training_steps} "
                    f"- loss: {loss.item():.4f}, avg_loss: {avg_loss:.4f}"
                )

        avg_epoch_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1} completed - average loss: {avg_epoch_loss:.4f}")

    # Save updated LoRA adapter into a timestamped online version directory.
    base_output_dir = Path(train_cfg["output_dir"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_output_dir / f"online_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    version_info: Dict[str, Any] = {
        "source": "online_update",
        "config": str(args.config),
        "log_dir": str(log_dir),
        "num_samples": len(samples),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "ONLINE_VERSION.json").open("w", encoding="utf-8") as f:
        json.dump(version_info, f, indent=2)

    print(f"Saved online-updated LoRA adapter to {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()


