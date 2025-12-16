#!/usr/bin/env python3
"""
Seed LoRA via plain SFT using unified-theory shard JSONL files.

This is intended for "bootstrap the voice/format" before enabling sleep / dual-loss.

Input record schema (as produced in data/processed_datasets_unified/*/shard_*.jsonl):
  - history: [{role, content, turn_number?}, ...]
  - target:  {role: "assistant", content: ...}
  - labels:  optional; may include reward_intensity, safety_score, etc.
  - target_use: optional list of tags to filter by.

Training objective:
  - Next-token prediction on the assistant completion only (prompt tokens masked to -100).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from core.data.shard_loader import MissingDatasetError, ShardLoader, print_shard_summary


@dataclass
class SFTShardSample:
    messages: List[Dict[str, str]]
    response: str
    weight: float


def _device() -> torch.device:
    return torch.device("mps")


def _torch_dtype_for_device(device: torch.device) -> Optional[torch.dtype]:
    if device.type == "mps":
        return torch.float16
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _maybe_apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except Exception:
        # Fallback format if tokenizer doesn't expose a chat template.
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


def _record_to_messages(rec: Dict[str, Any], *, clamp_history_turns: int = 3) -> Optional[List[Dict[str, str]]]:
    history = rec.get("history", []) or []
    if clamp_history_turns > 0 and len(history) > clamp_history_turns:
        history = history[-clamp_history_turns:]

    target = rec.get("target") or {}
    if (target.get("role") or "").lower() != "assistant":
        return None

    messages: List[Dict[str, str]] = []
    for turn in history:
        role = str(turn.get("role") or "user").lower()
        content = str(turn.get("content") or "")
        if not content.strip():
            continue
        messages.append({"role": role, "content": content})
    return messages


def _weight_from_labels(labels: Dict[str, Any], *, intensity_scale: float, safety_min: Optional[float]) -> float:
    weight = 1.0
    if safety_min is not None:
        try:
            safety = float(labels.get("safety_score", 1.0))
        except (TypeError, ValueError):
            safety = 1.0
        if safety < safety_min:
            return 0.0

    if intensity_scale > 0:
        try:
            intensity = float(labels.get("reward_intensity", 0.0))
        except (TypeError, ValueError):
            intensity = 0.0
        # reward_intensity is typically in [0, 3] in this repo; map to [0, 1].
        intensity01 = _clamp(intensity / 3.0, 0.0, 1.0)
        weight *= 1.0 + intensity_scale * intensity01
    return float(_clamp(weight, 0.0, 5.0))


def build_sft_samples(
    records: Sequence[Dict[str, Any]],
    *,
    clamp_history_turns: int,
    target_use: Optional[str],
    intensity_scale: float,
    safety_min: Optional[float],
    max_records: Optional[int],
) -> List[SFTShardSample]:
    out: List[SFTShardSample] = []
    for rec in records:
        if target_use:
            uses = rec.get("target_use") or []
            if target_use not in uses:
                continue
        messages = _record_to_messages(rec, clamp_history_turns=clamp_history_turns)
        if not messages:
            continue
        target = rec.get("target") or {}
        response = str(target.get("content") or "")
        if not response.strip():
            continue
        labels = rec.get("labels") or {}
        weight = _weight_from_labels(labels, intensity_scale=intensity_scale, safety_min=safety_min)
        if weight <= 0:
            continue
        out.append(SFTShardSample(messages=messages, response=response, weight=weight))
        if max_records is not None and len(out) >= max_records:
            break
    return out


class SFTShardDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[SFTShardSample],
        tokenizer,
        *,
        max_length: int,
        enable_thinking: bool,
    ) -> None:
        self._samples = list(samples)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._enable_thinking = enable_thinking

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[idx]

        prompt_messages = list(sample.messages)
        prompt_text = _maybe_apply_chat_template(
            self._tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )

        full_messages = list(sample.messages) + [{"role": "assistant", "content": sample.response}]
        full_text = _maybe_apply_chat_template(
            self._tokenizer,
            full_messages,
            add_generation_prompt=False,
            enable_thinking=self._enable_thinking,
        )

        prompt_ids = self._tokenizer(
            prompt_text,
            max_length=self._max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        encoded = self._tokenizer(
            full_text,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_len = int(prompt_ids.numel())
        if prompt_len > 0:
            labels[: min(prompt_len, labels.numel())] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weight": torch.tensor(float(sample.weight), dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed LoRA via SFT using unified-theory shards.")
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
        "--target-use",
        default=None,
        help="Optional target_use tag filter (e.g., phi_training).",
    )
    parser.add_argument("--record-limit", type=int, default=0, help="0 = no limit (after filtering).")
    parser.add_argument("--min-records", type=int, default=0)

    parser.add_argument("--base-model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="artifacts/lora/qwen3-1.7b-seed-sft")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--clamp-history-turns", type=int, default=3)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="If set, uses Qwen3 thinking-mode chat template (<think>...</think>) during SFT formatting.",
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    parser.add_argument(
        "--intensity-scale",
        type=float,
        default=0.0,
        help="If >0, upweights examples based on labels.reward_intensity (weak bootstrap).",
    )
    parser.add_argument(
        "--safety-min",
        type=float,
        default=None,
        help="If set, drops examples with labels.safety_score below this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.record_limit < 0 or args.min_records < 0:
        raise SystemExit("record-limit and min-records must be non-negative")

    loader = ShardLoader(root_paths=[Path(p) for p in args.data_roots], dataset_filters=args.datasets)
    try:
        max_records = args.record_limit if args.record_limit > 0 else None
        records, summary = loader.load_records(required_label_keys=[], max_records=None)
    except MissingDatasetError as exc:
        raise SystemExit(f"Dataset loading failed: {exc}")

    print_shard_summary(summary)

    samples = build_sft_samples(
        records,
        clamp_history_turns=args.clamp_history_turns,
        target_use=args.target_use,
        intensity_scale=float(args.intensity_scale),
        safety_min=args.safety_min,
        max_records=max_records,
    )
    if args.min_records and len(samples) < args.min_records:
        raise SystemExit(f"Need at least {args.min_records} usable samples but found {len(samples)}.")
    if not samples:
        raise SystemExit("No usable SFT samples found (check target_use, datasets, or shard schema).")

    print(f"🧩 SFT samples: {len(samples)}")

    device = _device()
    torch_dtype = _torch_dtype_for_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)

    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=list(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.train()

    dataset = SFTShardDataset(
        samples,
        tokenizer,
        max_length=int(args.max_length),
        enable_thinking=bool(args.enable_thinking),
    )
    loader_dl = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True)

    total_steps = math.ceil(len(loader_dl) / max(1, int(args.gradient_accumulation_steps))) * int(args.num_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=int(total_steps),
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(args.num_epochs)):
        for step, batch in enumerate(loader_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss = loss * batch["weight"].mean()
            loss = loss / max(1, int(args.gradient_accumulation_steps))
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()

            if (step + 1) % max(1, int(args.gradient_accumulation_steps)) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(f"step {global_step}/{total_steps} loss={float(loss.detach().cpu()):.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Saved seed LoRA to {out_dir}")


if __name__ == "__main__":
    main()
