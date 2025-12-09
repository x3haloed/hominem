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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from core.data.replay_buffer import ReplayBufferStore, ReplayPair
from core.lora_trainer.train_dpo import (
    load_config,
    prepare_model_and_tokenizer,
    select_device,
    set_seed,
)

# Type alias for clarity - OnlinePreferenceSample is just a ReplayPair
OnlinePreferenceSample = ReplayPair


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
            "safety_mode": getattr(sample, 'safety_mode', None),
        }


def online_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collator that stacks chosen/rejected tensors and weights, accounting for safety modes."""

    def stack_side(side: str) -> Dict[str, torch.Tensor]:
        keys = batch[0][side].keys()
        return {
            k: torch.stack([item[side][k] for item in batch], dim=0)
            for k in keys
        }

    weights = []
    for item in batch:
        base_weight = float(item["weight"])
        safety_mode = item.get("safety_mode")

        # Apply safety-based weight adjustments
        if safety_mode and hasattr(safety_mode, 'value'):
            if safety_mode.value == "downweight":
                base_weight *= 0.3  # Reduce influence of moderately unsafe examples
            elif safety_mode.value == "inverse":
                base_weight *= 0.8  # Slightly reduce but still use for regularization

        weights.append(max(0.0, min(5.0, base_weight)))  # Clamp to prevent extreme values

    return {
        "chosen": stack_side("chosen"),
        "rejected": stack_side("rejected"),
        "weight": torch.tensor(weights, dtype=torch.float32),
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
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of replay pairs to sample for this update (0 = use all).",
    )
    parser.add_argument(
        "--w-sft",
        type=float,
        default=1.0,
        help="Weight for the SFT loss component.",
    )
    parser.add_argument(
        "--w-reward",
        type=float,
        default=1.0,
        help="Weight for the reward (DPO-style) loss component.",
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

    buffer = ReplayBufferStore.from_self_train_logs(
        log_dir,
        safety_threshold=float(args.safety_threshold),
        min_reward_intensity=float(args.min_reward_intensity),
        max_records=max_records,
    )

    if args.num_samples and args.num_samples > 0:
        replay_pairs: List[ReplayPair] = buffer.sample_pairs(num_samples=args.num_samples)
    else:
        # Use all available pairs.
        replay_pairs = buffer.sample_pairs(num_samples=len(buffer._pairs))  # type: ignore[attr-defined]

    print(f"Loaded {len(replay_pairs)} online preference pairs from {log_dir}")

    device = select_device()
    model, tokenizer = prepare_model_and_tokenizer(cfg, device)

    dataset = OnlinePreferenceDataset(
        replay_pairs,
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
    w_sft = float(args.w_sft)
    w_reward = float(args.w_reward)

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

            # Reward (DPO-style) loss with per-example RewardIntensity weights.
            reward_loss = weighted_dpo_loss(model, batch, beta=beta)

            # SFT loss on chosen responses only.
            chosen_batch = batch["chosen"]
            outputs = model(
                input_ids=chosen_batch["input_ids"],
                attention_mask=chosen_batch["attention_mask"],
                labels=chosen_batch["labels"],
            )
            sft_loss = outputs.loss

            loss = w_sft * sft_loss + w_reward * reward_loss
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
        "num_samples": len(replay_pairs),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(output_dir),
    }

    # current.json always points at the latest adapter (offline or online).
    pointer_path = base_output_dir / "current.json"
    with pointer_path.open("w", encoding="utf-8") as f:
        json.dump(version_info, f, indent=2)

    # Append to history.json for audit / rollback.
    history_path = base_output_dir / "history.json"
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as f:
                history = json.load(f) or []
        except Exception:
            history = []
    history.append(version_info)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (output_dir / "ONLINE_VERSION.json").open("w", encoding="utf-8") as f:
        json.dump(version_info, f, indent=2)

    print(f"Saved online-updated LoRA adapter to {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()


