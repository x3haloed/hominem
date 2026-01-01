#!/usr/bin/env python3
"""
Sleep-cycle LoRA consolidation via weighted SFT on "memory" events (MLX).

This MLX-VLM version:
- Pulls sleep events from the canonical conversations SQLite (sleep_events table).
- Selects a prioritized subset of events (ΔΦ/intensity/self/soc mix).
- Writes MLX-ready JSONL data.
- Invokes mlx_vlm.lora to train LoRA adapters.

Manual cadence: run this script (or POST /sleep with SLEEP_UPDATE_ENABLED=true).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from core.training_logger import TrainingJSONLogger


def _default_db_path() -> str:
    return os.getenv(
        "DATABASE_PATH",
        os.path.join(str(Path.home()), "Documents", "hominem", "conversations.db"),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _maybe_apply_chat_template(
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
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
    # RewardIntensity is a gain scalar; keep it bounded to avoid domination/reward hacking.
    return float(_clamp(base, 0.0, 1.0))


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
    # Clamp even if the DB stored an out-of-range intensity from older runs.
    reward_intensity_f = float(_clamp(reward_intensity_f, 0.0, 1.0))

    delta_phi_used = ev.get("delta_phi_used")
    if delta_phi_used is None:
        delta_phi_used = post.get("delta_phi_used", post.get("delta_phi_ema", post.get("delta_phi_raw")))
    try:
        delta_phi_used_f = float(delta_phi_used) if delta_phi_used is not None else 0.0
    except Exception:
        delta_phi_used_f = 0.0

    # Unified Theory update: multiplicative gain so RewardIntensity cannot flip the sign of ΔΦ_used.
    r_t_f = float(delta_phi_used_f * (1.0 + float(alpha) * reward_intensity_f))

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


def load_sleep_events(
    *,
    db_path: str,
    only_unused: bool,
    limit: int,
    conversation_id: Optional[str],
    order: str = "asc",
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
        order_norm = str(order or "asc").strip().lower()
        if order_norm not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")
        query += f" ORDER BY created_at {order_norm.upper()}"
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


def _write_jsonl_dataset(samples: Sequence[SleepSFTSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            messages = list(sample.messages) + [{"role": "assistant", "content": sample.response}]
            example_weight = float(sample.memory_weight) * max(0.0, float(sample.gravity_reward))
            record = {
                "messages": messages,
                "images": [],
                "example_weight": example_weight,
                "metadata": {
                    "memory_weight": sample.memory_weight,
                    "gravity_reward": sample.gravity_reward,
                    "mean_self_fraction": sample.mean_self_fraction,
                    "reward_intensity": sample.reward_intensity,
                    "delta_phi_used": sample.delta_phi_used,
                    "event_id": sample.event_id,
                    "images_placeholder": True,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sleep-cycle LoRA update via MLX-VLM on sleep_events.")
    parser.add_argument("--config", default="config/training/sleep_sft_update.yaml")
    parser.add_argument("--db-path", default=None, help="Path to conversations.db (canonical).")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional log root dir. If set, writes JSONL logs to log-dir/<run-id>.",
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
    parser.add_argument(
        "--order",
        choices=["asc", "desc"],
        default="asc",
        help="Order sleep_events by created_at (asc=oldest-first, desc=newest-first).",
    )
    parser.add_argument("--min-r-t", type=float, default=0.3)
    parser.add_argument("--min-reward-intensity", type=float, default=0.0)
    parser.add_argument(
        "--require-positive-r-t",
        action="store_true",
        help="If set, drop non-positive r_t examples (recommended for SFT-only updates).",
    )

    parser.add_argument("--output-dir", default=None, help="Output root for MLX runs.")
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument("--iters", type=int, default=None, help="Number of MLX optimizer steps.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--mlx-args",
        nargs="*",
        default=None,
        help="Extra args passed to mlx_vlm.lora (space-separated).",
    )

    parser.add_argument("--num-samples", type=int, default=0, help="0 = use all usable sleep events.")

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--base-memory-weight", type=float, default=1.0)
    parser.add_argument("--self-fraction-power", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=1.0)

    parser.add_argument("--priority-w-delta-phi", type=float, default=1.0)
    parser.add_argument("--priority-w-intensity", type=float, default=1.2)
    parser.add_argument("--priority-w-social", type=float, default=0.4)
    parser.add_argument("--priority-w-self", type=float, default=0.2)
    parser.add_argument("--high-priority-fraction", type=float, default=0.7)

    parser.add_argument("--no-mark-used", action="store_true", help="Do not mark events used after training.")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args(argv)


def _resolve_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(str(args.config))
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    return {"model": model_cfg, "training": train_cfg, "data": data_cfg}


def _build_mlx_command(
    *,
    base_model_id: str,
    dataset_path: Path,
    iters: int,
    epochs: int,
    batch_size: int,
    adapter_path: Path,
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm.lora",
        "--model-path",
        base_model_id,
        "--dataset",
        str(dataset_path),
        "--steps",
        str(iters),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--output-path",
        str(adapter_path),
    ]
    if extra_args:
        cmd.extend(list(extra_args))
    return cmd


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _resolve_cfg(args)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    db_path = args.db_path or data_cfg.get("db_path") or _default_db_path()

    base_model_id = args.base_model_id or model_cfg.get("base_model_id") or "Qwen/Qwen3-1.7B"
    iters = int(args.iters or train_cfg.get("iters") or 200)
    epochs = int(args.epochs or train_cfg.get("epochs") or 1)
    batch_size = int(args.batch_size or train_cfg.get("batch_size") or 4)
    extra_args = args.mlx_args if args.mlx_args is not None else train_cfg.get("mlx_args") or []
    if isinstance(extra_args, str):
        extra_args = extra_args.split()

    output_root = Path(args.output_dir or train_cfg.get("output_dir") or "artifacts/lora_mlx")
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = train_cfg.get("run_id") or f"sleep_mlx_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    alpha = float(args.alpha if args.alpha is not None else train_cfg.get("alpha", 0.5))
    base_memory_weight = float(args.base_memory_weight if args.base_memory_weight is not None else train_cfg.get("base_memory_weight", 1.0))
    self_fraction_power = float(args.self_fraction_power if args.self_fraction_power is not None else train_cfg.get("self_fraction_power", 1.0))
    reward_clip = float(args.reward_clip if args.reward_clip is not None else train_cfg.get("reward_clip", 1.0))

    priority_w_delta_phi = float(args.priority_w_delta_phi)
    priority_w_intensity = float(args.priority_w_intensity)
    priority_w_social = float(args.priority_w_social)
    priority_w_self = float(args.priority_w_self)
    high_priority_fraction = float(args.high_priority_fraction)

    only_unused = not bool(args.include_used)
    if args.only_unused:
        only_unused = True
    require_positive_r_t = bool(args.require_positive_r_t or data_cfg.get("require_positive_r_t") or False)

    events = load_sleep_events(
        db_path=str(db_path),
        only_unused=only_unused,
        limit=int(args.limit),
        conversation_id=str(args.conversation_id) if args.conversation_id else None,
        order=str(args.order),
    )
    raw_samples = build_sleep_sft_samples(
        events,
        clamp_history_turns=int(data_cfg.get("clamp_history_turns", 6)),
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

    rng = random.Random(int(seed))

    events_by_id: Dict[int, Dict[str, Any]] = {}
    for ev in events:
        try:
            events_by_id[int(ev.get("id"))] = ev
        except Exception:
            continue

    scored: List[Tuple[float, SleepSFTSample]] = []
    for s in raw_samples:
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
        random_part = rng.sample(remaining, min(k_random, len(remaining)))
    samples = (high + random_part)[:num_samples]

    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = data_dir / "train.jsonl"
    _write_jsonl_dataset(samples, train_jsonl)

    logging_root = Path(train_cfg.get("logging_dir", "logs/train"))
    run_log_dir = Path(args.log_dir) / run_id if args.log_dir else logging_root / run_id

    use_database_logging = not bool(args.no_database_logging)
    if use_database_logging:
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
        component="sleep_sft_update_mlx",
        output_dir=run_log_dir,
        meta={
            "db_path": str(db_path),
            "conversation_id": args.conversation_id,
            "output_root": str(output_root),
            "num_events_loaded": len(events),
            "num_samples": len(samples),
            "require_positive_r_t": require_positive_r_t,
            "min_r_t": float(args.min_r_t),
            "min_reward_intensity": float(args.min_reward_intensity),
            "batch_size": batch_size,
            "iters": iters,
            "priority_weights": {
                "delta_phi": priority_w_delta_phi,
                "intensity": priority_w_intensity,
                "social": priority_w_social,
                "self": priority_w_self,
            },
            "priority_stats": {
                "high_priority_fraction": high_priority_fraction,
            },
            "base_model_id": str(base_model_id),
            "train_jsonl": str(train_jsonl),
            "extra_args": list(extra_args),
        },
        use_database=use_database_logging,
        db_path=str(db_path),
    )

    lora_path = run_dir / "lora"
    lora_path.mkdir(parents=True, exist_ok=True)
    adapter_path = lora_path / "adapters.safetensors"

    cmd = _build_mlx_command(
        base_model_id=str(base_model_id),
        dataset_path=train_jsonl,
        iters=iters,
        epochs=epochs,
        batch_size=batch_size,
        adapter_path=adapter_path,
        extra_args=extra_args,
    )

    manifest = {
        "source": "sleep_sft_update_mlx",
        "run_id": run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": str(base_model_id),
        "db_path": str(db_path),
        "num_samples": len(samples),
        "output_dir": str(run_dir),
        "data_dir": str(data_dir),
        "train_jsonl": str(train_jsonl),
        "lora_path": str(lora_path),
        "adapter_path": str(adapter_path),
        "mlx_command": cmd,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.log_eval({"event": "mlx_train_start", "command": cmd, "run_dir": str(run_dir)})

    try:
        result = subprocess.run(cmd, check=False)
    except Exception as exc:
        logger.log_eval({"event": "mlx_train_failed", "error": str(exc)})
        raise

    if result.returncode != 0:
        logger.log_eval({"event": "mlx_train_failed", "returncode": result.returncode})
        raise SystemExit(result.returncode)

    logger.log_eval({"event": "mlx_train_complete", "lora_path": str(lora_path)})

    current_path = output_root / "current.json"
    current_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    history_path = output_root / "history.json"
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8")) or []
        except Exception:
            history = []
    history.append(manifest)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if not args.no_mark_used:
        used_ids = [s.event_id for s in samples]
        mark_events_used(db_path=str(db_path), event_ids=used_ids, run_id=run_id)
        print(f"🧾 Marked {len(used_ids)} sleep_events as used (run_id={run_id})")

    logger.close()
    print(f"✅ MLX sleep LoRA update complete: {run_dir}")


if __name__ == "__main__":
    main()
