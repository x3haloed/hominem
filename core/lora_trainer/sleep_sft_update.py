#!/usr/bin/env python3
"""
Sleep-cycle LoRA consolidation via weighted SFT on "memory" events.

This is the first integration pass for sleep:
- Pulls sleep events from the canonical conversations SQLite (sleep_events table).
- Continues training from an existing LoRA adapter (no re-init).
- Trains an SFT objective on the assistant completion only (prompt tokens masked).

Manual cadence: run this script (or POST /sleep with SLEEP_UPDATE_ENABLED=true).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import uuid
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import yaml

from core.training_logger import TrainingJSONLogger


def _default_db_path() -> str:
    return os.getenv(
        "DATABASE_PATH",
        os.path.join(str(Path.home()), "Documents", "hominem", "conversations.db"),
    )


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _torch_dtype_for_device(device: torch.device) -> Optional[torch.dtype]:
    if device.type == "mps":
        return torch.float16
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _linear_warmup(*, step: int, warmup_steps: int, start: float, end: float) -> float:
    if warmup_steps <= 0:
        return float(end)
    t = float(_clamp(step, 0, warmup_steps)) / float(warmup_steps)
    return float(start + (end - start) * t)

def _split_train_val_indices(n: int, *, validation_split: float, seed: int) -> tuple[list[int], list[int]]:
    if validation_split <= 0:
        return list(range(n)), []
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1 (exclusive)")
    rng = random.Random(int(seed))
    idx = list(range(n))
    rng.shuffle(idx)
    val_size = int(n * float(validation_split))
    val_size = max(1, min(val_size, n - 1))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return train_idx, val_idx


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
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


@dataclass
class SleepSFTSample:
    messages: List[Dict[str, str]]
    response: str
    memory_weight: float
    gravity_reward: float
    mean_self_fraction: float
    reward_intensity: float
    delta_phi_used: float
    event_id: int


def _coerce_messages(history: Any, *, clamp_history_turns: int) -> Optional[List[Dict[str, str]]]:
    if not isinstance(history, list):
        return None
    msgs: List[Dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        msgs.append({"role": role, "content": content})
    if clamp_history_turns > 0 and len(msgs) > clamp_history_turns:
        msgs = msgs[-clamp_history_turns:]
    return msgs


def _reward_intensity_from_s(s: Dict[str, Any]) -> float:
    try:
        arousal = float(s.get("arousal", 0.0))
    except Exception:
        arousal = 0.0
    try:
        valence = float(s.get("valence", 0.0))
    except Exception:
        valence = 0.0
    try:
        discrepancy = float(s.get("predictive_discrepancy", 0.0))
    except Exception:
        discrepancy = 0.0

    arousal = _clamp(arousal, 0.0, 1.0)
    valence = _clamp(valence, -1.0, 1.0)
    discrepancy = _clamp(discrepancy, -1.0, 1.0)

    base = arousal * math.sqrt((abs(valence) ** 1.0) * abs(discrepancy))
    if valence < 0:
        base *= 1.8
    return float(_clamp(base, 0.0, 3.0))


def _mean_self_fraction_from_metrics(post: Dict[str, Any]) -> float:
    sf = post.get("self_fractions")
    if isinstance(sf, dict) and sf:
        values = []
        for v in sf.values():
            try:
                values.append(float(v))
            except Exception:
                pass
        if values:
            return float(_clamp(sum(values) / len(values), 0.0, 1.0))
    try:
        return float(_clamp(float(post.get("mean_self", 0.0)), 0.0, 1.0))
    except Exception:
        return 0.0


def _derive_post_metrics_from_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    metrics = ev.get("metrics_json")
    if isinstance(metrics, dict):
        post = metrics.get("post")
        if isinstance(post, dict):
            return post
    return {}


def _derive_values(ev: Dict[str, Any], *, alpha: float) -> Tuple[float, float, float]:
    post = _derive_post_metrics_from_event(ev)

    reward_intensity = ev.get("reward_intensity")
    if reward_intensity is None:
        reward_intensity = post.get("reward_intensity")
    if reward_intensity is None:
        s = post.get("s")
        if isinstance(s, dict):
            reward_intensity = _reward_intensity_from_s(s)
    try:
        reward_intensity_f = float(reward_intensity) if reward_intensity is not None else 0.0
    except Exception:
        reward_intensity_f = 0.0

    delta_phi_used = ev.get("delta_phi_used")
    if delta_phi_used is None:
        delta_phi_used = post.get("delta_phi_used", post.get("delta_phi_ema", post.get("delta_phi_raw")))
    try:
        delta_phi_used_f = float(delta_phi_used) if delta_phi_used is not None else 0.0
    except Exception:
        delta_phi_used_f = 0.0

    r_t = ev.get("r_t")
    if r_t is None:
        r_t = post.get("r_t")
    if r_t is None:
        r_t = delta_phi_used_f + float(alpha) * reward_intensity_f
    try:
        r_t_f = float(r_t) if r_t is not None else 0.0
    except Exception:
        r_t_f = 0.0

    return float(r_t_f), float(reward_intensity_f), float(delta_phi_used_f)


def _priority_score(
    *,
    delta_phi_used: float,
    reward_intensity: float,
    social_broadcast: float,
    mean_self_fraction: float,
    w_delta_phi: float,
    w_intensity: float,
    w_social: float,
    w_self: float,
) -> float:
    return float(
        w_delta_phi * abs(delta_phi_used)
        + w_intensity * max(0.0, reward_intensity)
        + w_social * max(0.0, social_broadcast)
        + w_self * max(0.0, mean_self_fraction)
    )

class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self._indices = list(indices)

    def __iter__(self) -> Iterable[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def build_balanced_epoch_indices(
    *,
    samples: Sequence[SleepSFTSample],
    priorities: Sequence[float],
    seed: int,
    high_bucket_fraction: float,
    mix_high: float,
    mix_low: float,
    low_intensity_threshold: float,
    low_delta_phi_threshold: float,
) -> List[int]:
    if len(samples) != len(priorities):
        raise ValueError("samples/priorities length mismatch")

    n = len(samples)
    if n == 0:
        return []

    rng = random.Random(int(seed))

    # Define high-priority bucket by top quantile.
    k_high = max(1, int(n * float(_clamp(high_bucket_fraction, 0.0, 1.0))))
    ranked = sorted(range(n), key=lambda i: float(priorities[i]), reverse=True)
    high_idx = ranked[:k_high]

    # Define explicit low-intensity bucket.
    low_idx: List[int] = []
    for i, s in enumerate(samples):
        if float(s.reward_intensity) <= float(low_intensity_threshold) and abs(float(s.delta_phi_used)) <= float(low_delta_phi_threshold):
            low_idx.append(i)

    # Remaining bucket.
    high_set = set(high_idx)
    low_set = set(low_idx)
    rest_idx = [i for i in range(n) if i not in high_set and i not in low_set]

    # Clamp mixing fractions and ensure they sum <= 1.
    mix_high = float(_clamp(mix_high, 0.0, 1.0))
    mix_low = float(_clamp(mix_low, 0.0, 1.0))
    if mix_high + mix_low > 1.0:
        scale = 1.0 / (mix_high + mix_low)
        mix_high *= scale
        mix_low *= scale
    mix_rest = 1.0 - mix_high - mix_low

    def choose_from(pool: Sequence[int]) -> int:
        # Sample with replacement; avoids exhaustion.
        return pool[rng.randrange(0, len(pool))]

    # Build epoch index order: per-example bucket choice, then sample with replacement.
    epoch: List[int] = []
    for _ in range(n):
        r = rng.random()
        if r < mix_high and high_idx:
            epoch.append(choose_from(high_idx))
        elif r < mix_high + mix_low and low_idx:
            epoch.append(choose_from(low_idx))
        elif rest_idx:
            epoch.append(choose_from(rest_idx))
        elif high_idx:
            epoch.append(choose_from(high_idx))
        elif low_idx:
            epoch.append(choose_from(low_idx))
        else:
            epoch.append(rng.randrange(0, n))

    return epoch


def load_sleep_events(
    *,
    db_path: str,
    only_unused: bool,
    limit: int,
    conversation_id: Optional[str],
) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM sleep_events WHERE 1=1"
        params: List[Any] = []
        if only_unused:
            query += " AND used=0"
        if conversation_id:
            query += " AND conversation_id=?"
            params.append(conversation_id)
        query += " ORDER BY created_at ASC"
        query += " LIMIT ?"
        params.append(int(limit))
        cur = con.execute(query, params)
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    for row in rows:
        if row.get("history_json"):
            try:
                row["history_json"] = json.loads(row["history_json"])
            except Exception:
                pass
        if row.get("metrics_json"):
            try:
                row["metrics_json"] = json.loads(row["metrics_json"])
            except Exception:
                pass
    return rows


def mark_events_used(*, db_path: str, event_ids: List[int], run_id: str) -> None:
    if not event_ids:
        return
    con = sqlite3.connect(db_path)
    try:
        placeholders = ",".join(["?"] * len(event_ids))
        con.execute(
            f"""
            UPDATE sleep_events
            SET used=1, used_at=CURRENT_TIMESTAMP, used_in_run=?
            WHERE id IN ({placeholders})
            """,
            (run_id, *event_ids),
        )
        con.commit()
    finally:
        con.close()


def build_sleep_sft_samples(
    events: Sequence[Dict[str, Any]],
    *,
    clamp_history_turns: int,
    require_positive_r_t: bool,
    min_r_t: float,
    min_reward_intensity: float,
    alpha: float,
    base_memory_weight: float,
    self_fraction_power: float,
    reward_clip: float,
    max_events: Optional[int],
) -> List[SleepSFTSample]:
    out: List[SleepSFTSample] = []
    for ev in events:
        history = ev.get("history_json")
        messages = _coerce_messages(history, clamp_history_turns=clamp_history_turns)
        if not messages:
            continue
        response = str(ev.get("assistant") or "").strip()
        if not response:
            continue
        event_id = int(ev.get("id"))

        post = _derive_post_metrics_from_event(ev)
        mean_self_fraction = _mean_self_fraction_from_metrics(post)
        r_t, reward_intensity, delta_phi_used = _derive_values(ev=ev, alpha=float(alpha))
        if min_reward_intensity and reward_intensity < float(min_reward_intensity):
            continue
        if min_r_t:
            if require_positive_r_t:
                if r_t < float(min_r_t):
                    continue
            else:
                if abs(r_t) < float(min_r_t):
                    continue
        if require_positive_r_t and r_t <= 0:
            continue

        gravity_reward = float(_clamp(r_t, -float(reward_clip), float(reward_clip)))
        # Weight gravity more when the event is strongly self-owned.
        gravity_reward *= float(_clamp(mean_self_fraction, 0.0, 1.0)) ** float(self_fraction_power)

        memory_weight = float(_clamp(float(base_memory_weight), 0.0, 5.0))

        out.append(
            SleepSFTSample(
                messages=messages,
                response=response,
                memory_weight=memory_weight,
                gravity_reward=gravity_reward,
                mean_self_fraction=mean_self_fraction,
                reward_intensity=float(reward_intensity),
                delta_phi_used=float(delta_phi_used),
                event_id=event_id,
            )
        )
        if max_events is not None and len(out) >= max_events:
            break
    return out


class SleepSFTDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[SleepSFTSample],
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
        prompt_text = _maybe_apply_chat_template(
            self._tokenizer,
            list(sample.messages),
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        full_text = _maybe_apply_chat_template(
            self._tokenizer,
            list(sample.messages) + [{"role": "assistant", "content": sample.response}],
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
            # We always mask the prompt portion, but tokenizers may use left padding
            # (we set padding_side="left"), so the prompt doesn't necessarily start at index 0.
            nonpad = (attention_mask == 1).nonzero(as_tuple=False)
            nonpad_start = int(nonpad[0].item()) if nonpad.numel() else 0
            start = nonpad_start
            end = min(start + prompt_len, labels.numel())
            if start < end:
                labels[start:end] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "memory_weight": torch.tensor(float(sample.memory_weight), dtype=torch.float32),
            "gravity_reward": torch.tensor(float(sample.gravity_reward), dtype=torch.float32),
            "mean_self_fraction": torch.tensor(float(sample.mean_self_fraction), dtype=torch.float32),
            "reward_intensity": torch.tensor(float(sample.reward_intensity), dtype=torch.float32),
            "delta_phi_used": torch.tensor(float(sample.delta_phi_used), dtype=torch.float32),
            "event_id": torch.tensor(int(sample.event_id), dtype=torch.int64),
        }


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sleep-cycle LoRA update via weighted SFT on sleep_events.")
    parser.add_argument("--config", default="config/training/sleep_sft_update.yaml")
    parser.add_argument("--db-path", default=None, help="Path to conversations.db (canonical).")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional log root dir. If set, writes JSONL logs to log-dir/<run-id>/ when database logging is disabled.",
    )
    parser.add_argument(
        "--no-database-logging",
        action="store_true",
        help="Log training metrics to JSONL instead of SQLite training tables.",
    )
    parser.add_argument("--conversation-id", default=None, help="Optional conversation_id filter.")
    parser.add_argument("--only-unused", action="store_true", help="Only use sleep_events.used=0 (default).")
    parser.add_argument("--include-used", action="store_true", help="If set, allows using already-used events.")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--min-r-t", type=float, default=0.3)
    parser.add_argument("--min-reward-intensity", type=float, default=0.0)
    parser.add_argument(
        "--require-positive-r-t",
        action="store_true",
        help="If set, drop non-positive r_t examples (recommended for SFT-only updates).",
    )

    parser.add_argument("--init-adapter", default="artifacts/lora/qwen3-1.7b-seed-sft-v3")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--clamp-history-turns", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true", help="Use thinking-mode chat template if supported.")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)

    parser.add_argument("--num-samples", type=int, default=0, help="0 = use all usable sleep events.")

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--w-memory", type=float, default=None)
    parser.add_argument("--w-gravity", type=float, default=None)
    parser.add_argument("--base-memory-weight", type=float, default=1.0)
    parser.add_argument("--self-fraction-power", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=1.0)

    parser.add_argument("--priority-w-delta-phi", type=float, default=1.0)
    parser.add_argument("--priority-w-intensity", type=float, default=1.2)
    parser.add_argument("--priority-w-social", type=float, default=0.4)
    parser.add_argument("--priority-w-self", type=float, default=0.2)
    parser.add_argument("--high-priority-fraction", type=float, default=0.7)
    parser.add_argument("--random-fraction", type=float, default=0.3)

    parser.add_argument("--no-mark-used", action="store_true", help="Do not mark events used after training.")

    parser.add_argument("--seed", type=int, default=42)

    # Balanced replay sampling (explicit low-intensity mixing).
    parser.add_argument("--balanced-batches", action="store_true", help="Enable explicit high/low/random batch mixing.")
    parser.add_argument("--high-bucket-fraction", type=float, default=0.3, help="Fraction of samples considered high-priority bucket.")
    parser.add_argument("--mix-high", type=float, default=0.6, help="Fraction of draws from high-priority bucket per epoch.")
    parser.add_argument("--mix-low", type=float, default=0.2, help="Fraction of draws from low-intensity bucket per epoch.")
    parser.add_argument("--low-intensity-threshold", type=float, default=0.12, help="reward_intensity <= threshold => low-intensity bucket.")
    parser.add_argument("--low-delta-phi-threshold", type=float, default=0.08, help="|delta_phi_used| <= threshold => low-intensity bucket.")

    # Gravity warm-up schedule.
    parser.add_argument("--gravity-warmup-steps", type=int, default=0, help="Warm up w_gravity over this many optimizer steps.")
    parser.add_argument("--gravity-warmup-start", type=float, default=0.0, help="Starting w_gravity during warmup.")

    # Validation/eval during training.
    parser.add_argument("--validation-split", type=float, default=0.0, help="Hold out a fraction of sleep samples for validation.")
    parser.add_argument("--eval-steps", type=int, default=0, help="If >0, run validation every N optimizer steps.")
    parser.add_argument("--eval-max-batches", type=int, default=0, help="If >0, cap validation batches per eval run.")
    parser.add_argument("--save-best", action="store_true", help="Save best adapter (by val_loss_total) to <output_root>/best/.")
    return parser.parse_args(argv)


def _resolve_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(str(args.config))
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    return {"model": model_cfg, "training": train_cfg, "data": data_cfg}


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _resolve_cfg(args)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    db_path = args.db_path or data_cfg.get("db_path") or _default_db_path()
    init_adapter = args.init_adapter or model_cfg.get("init_adapter") or "artifacts/lora/qwen3-1.7b-seed-sft-v3"

    output_dir = args.output_dir or train_cfg.get("output_dir") or init_adapter
    out_root = Path(output_dir)
    run_id = train_cfg.get("run_id") or f"sleep_sft_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    base_model_id = args.base_model_id or model_cfg.get("base_model_id") or "Qwen/Qwen3-1.7B"
    max_length = int(args.max_length or model_cfg.get("max_length") or 1024)
    clamp_history_turns = int(args.clamp_history_turns or data_cfg.get("clamp_history_turns") or 6)
    enable_thinking = bool(args.enable_thinking or model_cfg.get("enable_thinking") or False)

    batch_size = int(args.batch_size or train_cfg.get("batch_size") or 1)
    grad_accum = int(args.gradient_accumulation_steps or train_cfg.get("gradient_accumulation_steps") or 4)
    num_epochs = int(args.num_epochs or train_cfg.get("num_epochs") or 1)
    lr = float(args.lr or train_cfg.get("learning_rate") or 2e-5)
    warmup_steps = int(args.warmup_steps or train_cfg.get("warmup_steps") or 50)
    max_grad_norm = float(args.max_grad_norm or train_cfg.get("max_grad_norm") or 0.3)
    logging_steps = int(args.logging_steps or train_cfg.get("logging_steps") or 10)

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    alpha = float(args.alpha if args.alpha is not None else train_cfg.get("alpha", 0.5))
    w_memory = float(args.w_memory if args.w_memory is not None else train_cfg.get("w_memory", 1.0))
    w_gravity_target = float(args.w_gravity if args.w_gravity is not None else train_cfg.get("w_gravity", 0.8))
    base_memory_weight = float(args.base_memory_weight if args.base_memory_weight is not None else train_cfg.get("base_memory_weight", 1.0))
    self_fraction_power = float(args.self_fraction_power if args.self_fraction_power is not None else train_cfg.get("self_fraction_power", 1.0))
    reward_clip = float(args.reward_clip if args.reward_clip is not None else train_cfg.get("reward_clip", 1.0))

    priority_w_delta_phi = float(args.priority_w_delta_phi)
    priority_w_intensity = float(args.priority_w_intensity)
    priority_w_social = float(args.priority_w_social)
    priority_w_self = float(args.priority_w_self)
    high_priority_fraction = float(args.high_priority_fraction)
    random_fraction = float(args.random_fraction)

    balanced_batches = bool(args.balanced_batches)
    high_bucket_fraction = float(args.high_bucket_fraction)
    mix_high = float(args.mix_high)
    mix_low = float(args.mix_low)
    low_intensity_threshold = float(args.low_intensity_threshold)
    low_delta_phi_threshold = float(args.low_delta_phi_threshold)

    gravity_warmup_steps = int(args.gravity_warmup_steps if args.gravity_warmup_steps is not None else train_cfg.get("gravity_warmup_steps", 0))
    gravity_warmup_start = float(args.gravity_warmup_start if args.gravity_warmup_start is not None else train_cfg.get("gravity_warmup_start", 0.0))

    validation_split = float(args.validation_split if args.validation_split is not None else train_cfg.get("validation_split", 0.0))
    eval_steps = int(args.eval_steps if args.eval_steps is not None else train_cfg.get("eval_steps", 0))
    eval_max_batches = int(args.eval_max_batches if args.eval_max_batches is not None else train_cfg.get("eval_max_batches", 0))
    save_best = bool(args.save_best)

    only_unused = not bool(args.include_used)
    if args.only_unused:
        only_unused = True
    require_positive_r_t = bool(args.require_positive_r_t or data_cfg.get("require_positive_r_t") or False)

    events = load_sleep_events(
        db_path=str(db_path),
        only_unused=only_unused,
        limit=int(args.limit),
        conversation_id=str(args.conversation_id) if args.conversation_id else None,
    )
    # Build samples + compute priorities for replay selection.
    raw_samples = build_sleep_sft_samples(
        events,
        clamp_history_turns=clamp_history_turns,
        require_positive_r_t=require_positive_r_t,
        min_r_t=float(args.min_r_t or 0.0),
        min_reward_intensity=float(args.min_reward_intensity or 0.0),
        alpha=alpha,
        base_memory_weight=base_memory_weight,
        self_fraction_power=self_fraction_power,
        reward_clip=reward_clip,
        max_events=None,
    )
    if not raw_samples:
        print("No usable sleep_events for training (filters removed everything).")
        return

    # Train/val split before prioritization to keep eval "unseen" by construction.
    train_idx, val_idx = _split_train_val_indices(len(raw_samples), validation_split=validation_split, seed=seed)
    train_pool = [raw_samples[i] for i in train_idx]
    val_pool = [raw_samples[i] for i in val_idx]

    # Prioritized mixing (spec-inspired): top-K by priority + random mix from remainder (train pool only).
    events_by_id: Dict[int, Dict[str, Any]] = {}
    for ev in events:
        try:
            events_by_id[int(ev.get("id"))] = ev
        except Exception:
            continue

    scored: List[Tuple[float, SleepSFTSample]] = []
    for s in train_pool:
        # Extract social_broadcast from metrics if available (best effort).
        ev = events_by_id.get(int(s.event_id))
        social_broadcast = 0.0
        if isinstance(ev, dict):
            post = _derive_post_metrics_from_event(ev)
            ss = post.get("s")
            if isinstance(ss, dict):
                try:
                    social_broadcast = float(ss.get("social_broadcast", 0.0))
                except Exception:
                    social_broadcast = 0.0
        priority = _priority_score(
            delta_phi_used=s.delta_phi_used,
            reward_intensity=s.reward_intensity,
            social_broadcast=social_broadcast,
            mean_self_fraction=s.mean_self_fraction,
            w_delta_phi=priority_w_delta_phi,
            w_intensity=priority_w_intensity,
            w_social=priority_w_social,
            w_self=priority_w_self,
        )
        scored.append((priority, s))

    scored.sort(key=lambda t: t[0], reverse=True)
    requested = int(args.num_samples or 0)
    num_samples = min(len(scored), requested) if requested > 0 else len(scored)
    k_high = max(1, int(num_samples * high_priority_fraction))
    k_high = min(k_high, len(scored))
    k_random = max(0, num_samples - k_high)
    high = [s for _, s in scored[:k_high]]
    remaining = [s for _, s in scored[k_high:]]
    random_part: List[SleepSFTSample] = []
    if remaining and k_random > 0:
        import random as _random
        random_part = _random.sample(remaining, min(k_random, len(remaining)))
    samples = (high + random_part)[:num_samples]
    val_samples = list(val_pool)

    device = _device()
    torch_dtype = _torch_dtype_for_device(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)

    if init_adapter:
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
    else:
        raise ValueError("--init-adapter is required for sleep_sft_update (to preserve existing weights).")

    model.train()

    dataset = SleepSFTDataset(samples, tokenizer, max_length=max_length, enable_thinking=enable_thinking)
    # Default: shuffle=True (simple). If balanced_batches enabled, we create a per-epoch sampler.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not balanced_batches)
    val_dataset = SleepSFTDataset(val_samples, tokenizer, max_length=max_length, enable_thinking=enable_thinking)
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size), shuffle=False) if val_samples else None

    total_steps = int(num_epochs) * math.ceil(len(dataset) / max(1, batch_size * grad_accum))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )

    logging_root = Path(train_cfg.get("logging_dir", "logs/train"))
    if args.log_dir:
        run_log_dir = Path(args.log_dir) / run_id
    else:
        run_log_dir = logging_root / run_id

    use_database_logging = not bool(args.no_database_logging)
    if use_database_logging:
        # The canonical conversations DB may not include training_* tables.
        # Detect this and fall back to JSONL logging to avoid hard failure.
        try:
            con = sqlite3.connect(str(db_path))
            try:
                exists = con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_runs'"
                ).fetchone()
                if not exists:
                    use_database_logging = False
            finally:
                con.close()
        except Exception:
            use_database_logging = False

    logger = TrainingJSONLogger(
        run_id=run_id,
        component="sleep_sft_update",
        output_dir=run_log_dir,
        meta={
            "db_path": str(db_path),
            "conversation_id": args.conversation_id,
            "init_adapter": str(init_adapter),
            "output_root": str(out_root),
            "num_events_loaded": len(events),
            "num_samples": len(samples),
            "num_samples_val": len(val_samples),
            "require_positive_r_t": require_positive_r_t,
            "min_r_t": float(args.min_r_t),
            "min_reward_intensity": float(args.min_reward_intensity),
            "max_length": max_length,
            "clamp_history_turns": clamp_history_turns,
            "enable_thinking": enable_thinking,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "num_epochs": num_epochs,
            "lr": lr,
            "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "alpha": alpha,
            "w_memory": w_memory,
            "w_gravity_target": w_gravity_target,
            "gravity_warmup_steps": gravity_warmup_steps,
            "gravity_warmup_start": gravity_warmup_start,
            "base_memory_weight": base_memory_weight,
            "self_fraction_power": self_fraction_power,
            "reward_clip": reward_clip,
            "priority_weights": {
                "delta_phi": priority_w_delta_phi,
                "intensity": priority_w_intensity,
                "social": priority_w_social,
                "self": priority_w_self,
            },
            "priority_stats": {"high_priority_fraction": high_priority_fraction, "random_fraction": random_fraction},
            "balanced_batches": balanced_batches,
            "balanced_high_bucket_fraction": high_bucket_fraction,
            "balanced_mix": {"high": mix_high, "low": mix_low, "rest": max(0.0, 1.0 - mix_high - mix_low)},
            "balanced_low_thresholds": {"reward_intensity": low_intensity_threshold, "abs_delta_phi_used": low_delta_phi_threshold},
            "validation_split": validation_split,
            "eval_steps": eval_steps,
            "eval_max_batches": eval_max_batches,
            "save_best": save_best,
            "base_model_id": str(base_model_id),
            "device": str(device),
            "torch_dtype": str(torch_dtype) if torch_dtype is not None else None,
        },
        use_database=use_database_logging,
        db_path=str(db_path),
    )

    def per_example_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.shape[-1]
        flat_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        tok_loss = flat_loss.view(shift_labels.shape)
        mask = shift_labels.ne(-100)
        denom = mask.sum(dim=1).clamp(min=1)
        seq_loss = (tok_loss * mask).sum(dim=1) / denom
        return seq_loss

    @torch.no_grad()
    def evaluate(*, step: int) -> Dict[str, float]:
        if val_loader is None:
            return {}
        model.eval()
        max_batches = int(eval_max_batches)
        mem_losses: List[float] = []
        grav_losses: List[float] = []
        tot_losses: List[float] = []
        reward_means: List[float] = []
        self_means: List[float] = []

        w_gravity_eval = _linear_warmup(
            step=step,
            warmup_steps=gravity_warmup_steps,
            start=gravity_warmup_start,
            end=w_gravity_target,
        )

        for i, batch in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            seq_losses = per_example_ce_loss(outputs.logits, batch["labels"])
            memory_w = torch.clamp(batch["memory_weight"], min=0.0)
            gravity_r = torch.clamp(batch["gravity_reward"], min=-reward_clip, max=reward_clip)

            mem_num = (seq_losses * memory_w).sum()
            mem_den = memory_w.sum().clamp(min=1e-6)
            memory_loss = mem_num / mem_den
            gravity_loss = (seq_losses * gravity_r).mean()
            total_loss = w_memory * memory_loss + w_gravity_eval * gravity_loss

            mem_losses.append(float(memory_loss.detach().cpu()))
            grav_losses.append(float(gravity_loss.detach().cpu()))
            tot_losses.append(float(total_loss.detach().cpu()))
            reward_means.append(float(gravity_r.mean().detach().cpu()))
            self_means.append(float(batch["mean_self_fraction"].mean().detach().cpu()))

        model.train()

        if not tot_losses:
            return {
                "val_loss_total": float("nan"),
                "val_loss_memory": float("nan"),
                "val_loss_gravity": float("nan"),
                "val_reward_mean": float("nan"),
                "val_mean_self_fraction": float("nan"),
                "val_w_gravity": float(w_gravity_eval),
            }

        return {
            "val_loss_total": float(sum(tot_losses) / len(tot_losses)),
            "val_loss_memory": float(sum(mem_losses) / len(mem_losses)),
            "val_loss_gravity": float(sum(grav_losses) / len(grav_losses)),
            "val_reward_mean": float(sum(reward_means) / len(reward_means)),
            "val_mean_self_fraction": float(sum(self_means) / len(self_means)),
            "val_w_gravity": float(w_gravity_eval),
        }

    global_step = 0
    accum_micro = 0
    accum_memory_loss = 0.0
    accum_gravity_loss = 0.0
    accum_total_loss = 0.0
    accum_reward_mean = 0.0
    accum_self_mean = 0.0
    optimizer.zero_grad(set_to_none=True)
    best_val_loss: Optional[float] = None

    try:
        for epoch in range(num_epochs):
            if balanced_batches:
                # Rebuild a sampler each epoch so we re-mix buckets.
                # Priorities were computed on the pre-selected `samples` list; recompute in that order.
                epoch_priorities: List[float] = []
                # We only have priorities for `raw_samples` in scored; reconstruct for current dataset ordering.
                priority_map = {s.event_id: float(p) for (p, s) in scored}
                for s in samples:
                    epoch_priorities.append(priority_map.get(int(s.event_id), 0.0))
                indices = build_balanced_epoch_indices(
                    samples=samples,
                    priorities=epoch_priorities,
                    seed=seed + epoch,
                    high_bucket_fraction=high_bucket_fraction,
                    mix_high=mix_high,
                    mix_low=mix_low,
                    low_intensity_threshold=low_intensity_threshold,
                    low_delta_phi_threshold=low_delta_phi_threshold,
                )
                epoch_loader = DataLoader(dataset, batch_size=batch_size, sampler=FixedOrderSampler(indices), shuffle=False)
            else:
                epoch_loader = loader

            for micro_step, batch in enumerate(epoch_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                seq_losses = per_example_ce_loss(outputs.logits, batch["labels"])
                memory_w = torch.clamp(batch["memory_weight"], min=0.0)
                gravity_r = torch.clamp(batch["gravity_reward"], min=-reward_clip, max=reward_clip)
                # Memory loss: standard SFT weighted by memory_weight.
                mem_num = (seq_losses * memory_w).sum()
                mem_den = memory_w.sum().clamp(min=1e-6)
                memory_loss = mem_num / mem_den
                # Gravity loss: reward-weighted CE (negative rewards push unlearning).
                gravity_loss = (seq_losses * gravity_r).mean()
                w_gravity = _linear_warmup(
                    step=global_step,
                    warmup_steps=gravity_warmup_steps,
                    start=gravity_warmup_start,
                    end=w_gravity_target,
                )
                total_loss = w_memory * memory_loss + w_gravity * gravity_loss

                loss_for_backward = total_loss / max(1, grad_accum)
                if not torch.isfinite(loss_for_backward):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss_for_backward.backward()

                accum_memory_loss += float(memory_loss.detach().cpu())
                accum_gravity_loss += float(gravity_loss.detach().cpu())
                accum_total_loss += float(total_loss.detach().cpu())
                accum_reward_mean += float(gravity_r.mean().detach().cpu())
                accum_self_mean += float(batch["mean_self_fraction"].mean().detach().cpu())
                accum_micro += 1

                if (micro_step + 1) % grad_accum != 0:
                    continue

                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                step_mem = accum_memory_loss / max(1, accum_micro)
                step_grav = accum_gravity_loss / max(1, accum_micro)
                step_total = accum_total_loss / max(1, accum_micro)
                step_reward_mean = accum_reward_mean / max(1, accum_micro)
                step_self_mean = accum_self_mean / max(1, accum_micro)
                accum_memory_loss = 0.0
                accum_gravity_loss = 0.0
                accum_total_loss = 0.0
                accum_reward_mean = 0.0
                accum_self_mean = 0.0
                accum_micro = 0

                if global_step % max(1, logging_steps) == 0:
                    logger.log_step(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "num_training_steps": max(1, total_steps),
                            "loss_total": step_total,
                            "loss_memory": step_mem,
                            "loss_gravity": step_grav,
                            "reward_mean": step_reward_mean,
                            "mean_self_fraction": step_self_mean,
                            "w_gravity": w_gravity,
                            "grad_norm": grad_norm,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "dataset_size": len(dataset),
                        }
                    )
                    print(
                        f"step {global_step}/{max(1, total_steps)} "
                        f"loss={step_total:.4f} mem={step_mem:.4f} grav={step_grav:.4f} "
                        f"r̄={step_reward_mean:+.3f} self̄={step_self_mean:.2f}"
                    )

                if val_loader and eval_steps and global_step % max(1, eval_steps) == 0:
                    metrics = evaluate(step=global_step)
                    logger.log_eval(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "num_training_steps": max(1, total_steps),
                            **metrics,
                        }
                    )
                    if metrics:
                        print(
                            f"eval step {global_step}/{max(1, total_steps)} "
                            f"val_loss={metrics.get('val_loss_total', float('nan')):.4f}"
                        )
                    if save_best and metrics and math.isfinite(metrics.get("val_loss_total", float("nan"))):
                        v = float(metrics["val_loss_total"])
                        if best_val_loss is None or v < best_val_loss:
                            best_val_loss = v
                            best_dir = out_root / "best"
                            best_dir.mkdir(parents=True, exist_ok=True)
                            model.save_pretrained(best_dir)
                            tokenizer.save_pretrained(best_dir)
                            print(f"🏆 Saved best checkpoint to {best_dir} (val_loss_total={best_val_loss:.4f})")

            # Epoch-end eval.
            if val_loader:
                metrics = evaluate(step=global_step)
                logger.log_eval(
                    {
                        "event": "epoch_end",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "num_training_steps": max(1, total_steps),
                        **metrics,
                    }
                )
                if metrics:
                    print(
                        f"epoch {epoch + 1} end "
                        f"val_loss={metrics.get('val_loss_total', float('nan')):.4f}"
                    )

        # Save to a timestamped subdir under output root, but keep root stable for manual server switch.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = out_root / f"sleep_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        version_info: Dict[str, Any] = {
            "source": "sleep_sft_update",
            "run_id": run_id,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_model_id": str(base_model_id),
            "init_adapter": str(init_adapter),
            "db_path": str(db_path),
            "num_samples": len(samples),
            "path": str(out_dir),
        }
        (out_root / "current.json").write_text(json.dumps(version_info, indent=2), encoding="utf-8")

        history_path = out_root / "history.json"
        history: List[Dict[str, Any]] = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8")) or []
            except Exception:
                history = []
        history.append(version_info)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        logger.log_eval({"event": "training_complete", "output_dir": str(out_dir), **version_info})
        print(f"✅ Saved sleep LoRA update to {out_dir}")

        if not args.no_mark_used:
            used_ids = [s.event_id for s in samples] + [s.event_id for s in val_samples]
            mark_events_used(db_path=str(db_path), event_ids=used_ids, run_id=run_id)
            print(f"🧾 Marked {len(used_ids)} sleep_events as used (run_id={run_id})")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
