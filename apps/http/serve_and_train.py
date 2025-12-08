from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from fastapi import FastAPI

from apps.http import self_train_server as sts
from core.lora_trainer.train_dpo import load_config as load_lora_config
from core.lora_trainer import online_update as online_update_module


@dataclass
class OnlineConfig:
    lora_config_path: Path
    reward_model_dir: Path
    log_dir: Path
    min_reward_intensity: float
    safety_threshold: float
    num_samples: int
    w_sft: float
    w_reward: float
    interactions_per_update: int
    check_interval_seconds: int


def _load_online_config(path: Path) -> OnlineConfig:
    if not path.exists():
        # Reasonable defaults tied to existing training config.
        return OnlineConfig(
            lora_config_path=Path("config/training/lora_dpo.yaml"),
            reward_model_dir=Path("artifacts/reward_model/default/model"),
            log_dir=Path("data/online_feedback"),
            min_reward_intensity=0.2,
            safety_threshold=-0.1,
            num_samples=32,
            w_sft=1.0,
            w_reward=1.0,
            interactions_per_update=32,
            check_interval_seconds=60,
        )

    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    return OnlineConfig(
        lora_config_path=Path(data.get("lora_config_path", "config/training/lora_dpo.yaml")),
        reward_model_dir=Path(data.get("reward_model_dir", "artifacts/reward_model/default/model")),
        log_dir=Path(data.get("log_dir", "data/online_feedback")),
        min_reward_intensity=float(data.get("min_reward_intensity", 0.2)),
        safety_threshold=float(data.get("safety_threshold", -0.1)),
        num_samples=int(data.get("num_samples", 32)),
        w_sft=float(data.get("w_sft", 1.0)),
        w_reward=float(data.get("w_reward", 1.0)),
        interactions_per_update=int(data.get("interactions_per_update", 32)),
        check_interval_seconds=int(data.get("check_interval_seconds", 60)),
    )


def _resolve_current_lora_dir(output_dir: Path) -> Path:
    """Resolve the active LoRA directory using current.json if present."""
    pointer_path = output_dir / "current.json"
    if pointer_path.exists():
        try:
            with pointer_path.open("r", encoding="utf-8") as f:
                obj = json.load(f) or {}
            path_str = obj.get("path")
            if isinstance(path_str, str):
                resolved = Path(path_str)
                if resolved.exists():
                    return resolved
        except Exception:
            pass
    return output_dir


app = FastAPI(title="Hominem Serve + Train API")

_ctx: Optional[sts.SelfTrainContext] = None
_online_cfg: Optional[OnlineConfig] = None
_lora_output_dir: Optional[Path] = None
_interactions_since_update: int = 0


def _reload_lora_in_context() -> None:
    """Reload the latest LoRA adapter into the existing SelfTrainContext."""
    global _ctx, _lora_output_dir
    if _ctx is None or _lora_output_dir is None:
        return

    # Determine current LoRA directory from history/pointer.
    lora_dir = _resolve_current_lora_dir(_lora_output_dir)

    # Rebuild base model + LoRA adapter and move to the existing device.
    device = _ctx.device
    # Reuse the same base model id via the tokenizer config.
    base_model_id = _ctx.gen_tokenizer.name_or_path

    gen_tokenizer = sts.AutoTokenizer.from_pretrained(base_model_id)  # type: ignore[attr-defined]
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"

    base_model = sts.AutoModelForCausalLM.from_pretrained(base_model_id)  # type: ignore[attr-defined]
    base_model.to(device)
    base_model.eval()

    lora_model = sts.PeftModel.from_pretrained(base_model, lora_dir)  # type: ignore[attr-defined]
    lora_model.to(device)
    lora_model.eval()

    _ctx.gen_tokenizer = gen_tokenizer
    _ctx.gen_model = lora_model


async def _online_update_loop() -> None:
    """Periodically run online LoRA updates based on logged interactions."""
    global _interactions_since_update, _online_cfg

    assert _online_cfg is not None

    while True:
        await asyncio.sleep(_online_cfg.check_interval_seconds)
        if _interactions_since_update < _online_cfg.interactions_per_update:
            continue

        # Run the online update using the existing CLI entrypoint with config-based args.
        args = [
            "--config",
            str(_online_cfg.lora_config_path),
            "--log-dir",
            str(_online_cfg.log_dir),
            "--min-reward-intensity",
            str(_online_cfg.min_reward_intensity),
            "--safety-threshold",
            str(_online_cfg.safety_threshold),
            "--num-samples",
            str(_online_cfg.num_samples),
            "--w-sft",
            str(_online_cfg.w_sft),
            "--w-reward",
            str(_online_cfg.w_reward),
        ]
        print(f"[online_update] Starting online LoRA update with args: {args}")
        online_update_module.main(args)
        print("[online_update] Completed online LoRA update.")

        # Hot-swap to the latest LoRA adapter.
        _reload_lora_in_context()

        _interactions_since_update = 0


@app.on_event("startup")
async def _startup() -> None:
    """Load models and start the background online-update loop."""
    global _ctx, _online_cfg, _lora_output_dir

    _online_cfg = _load_online_config(Path("config/training/online_update.yaml"))

    # Resolve base_model_id and LoRA output_dir from the shared LoRA config.
    lora_cfg = load_lora_config(_online_cfg.lora_config_path)
    model_cfg = lora_cfg["model"]
    train_cfg = lora_cfg["training"]
    base_model_id = model_cfg["base_model_id"]
    _lora_output_dir = Path(train_cfg["output_dir"])

    lora_dir = _resolve_current_lora_dir(_lora_output_dir)

    # Initialize the shared SelfTrainContext used for generation + reward scoring.
    _ctx = sts.SelfTrainContext(
        base_model_id=base_model_id,
        lora_dir=lora_dir,
        reward_model_dir=_online_cfg.reward_model_dir,
        log_dir=_online_cfg.log_dir,
    )

    # Kick off the background online-update scheduler.
    asyncio.create_task(_online_update_loop())


@app.post("/self_train", response_model=sts.GenerateResponse)
def self_train(req: sts.GenerateRequest) -> sts.GenerateResponse:
    """Generate candidates, score with reward model, log, and schedule online updates."""
    global _ctx, _interactions_since_update

    assert _ctx is not None, "Context not initialized"
    ctx = _ctx

    candidates: list[sts.CandidateResult] = []
    for _ in range(req.num_candidates):
        text = sts.generate_one(
            ctx=ctx,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        reward_vec = sts.score_with_reward_model(
            ctx=ctx,
            prompt=req.prompt,
            response=text,
        )
        score = sts.scalar_score(reward_vec)
        candidates.append(
            sts.CandidateResult(
                text=text,
                reward=reward_vec,
                scalar_score=score,
            )
        )

    chosen = max(candidates, key=lambda c: c.scalar_score)

    log_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": req.prompt,
        "num_candidates": req.num_candidates,
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "device": str(ctx.device),
        "chosen": chosen.dict(),
        "candidates": [c.dict() for c in candidates],
    }
    with ctx.log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    _interactions_since_update += 1

    return sts.GenerateResponse(
        prompt=req.prompt,
        chosen=chosen,
        candidates=candidates,
        logfile=str(ctx.log_path),
    )



