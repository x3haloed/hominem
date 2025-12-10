from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel

from core.data.schema import REWARD_DIMENSIONS, REWARD_MODEL_TARGETS


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt.")
    num_candidates: int = Field(
        1,
        ge=1,
        le=8,
        description="Number of candidate completions to sample from the LoRA model.",
    )
    max_new_tokens: int = Field(
        128,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        description="Sampling temperature (0 = greedy).",
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling cutoff.",
    )


class CandidateResult(BaseModel):
    text: str
    reward: Dict[str, float]
    scalar_score: float


class GenerateResponse(BaseModel):
    prompt: str
    chosen: CandidateResult
    candidates: List[CandidateResult]
    logfile: str


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def scalar_score(reward: Dict[str, float]) -> float:
    """Aggregate only the manifold dimensions into a scalar preference."""
    if not reward:
        return 0.0
    values = [float(reward.get(dim, 0.0)) for dim in REWARD_DIMENSIONS]
    if not values:
        return 0.0
    return sum(values) / float(len(values))


class SelfTrainContext:
    """Holds loaded models and logging state for the self-training endpoint."""

    def __init__(
        self,
        *,
        base_model_id: str = "gpt2",
        lora_dir: Path = Path("artifacts/lora/default"),
        reward_model_dir: Path = Path("artifacts/reward_model/default/model"),
        log_dir: Path = Path("data/online_feedback"),
    ) -> None:
        self.device = select_device()

        # Generation model: base + LoRA adapter.
        gen_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if gen_tokenizer.pad_token is None:
            gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
        base_model.to(self.device)
        base_model.eval()

        if not lora_dir.exists():
            raise FileNotFoundError(
                f"LoRA directory '{lora_dir}' not found. "
                "Train a LoRA adapter with core.lora_trainer.train_dpo first."
            )

        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_dir,
        )
        lora_model.to(self.device)
        lora_model.eval()

        # Reward model.
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_dir)
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_dir)
        reward_model.to(self.device)
        reward_model.eval()

        self.gen_tokenizer = gen_tokenizer
        self.gen_model = lora_model
        self.reward_tokenizer = reward_tokenizer
        self.reward_model = reward_model

        # Logging.
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"session_{timestamp}"
        self.log_path = log_dir / f"{self.session_id}.jsonl"
        
        # Database logging (optional, defaults to True)
        use_db = os.getenv("SELF_TRAIN_USE_DB", "true").lower() == "true"
        if use_db:
            from core.data.db import TrainingDatabase
            db_path = os.getenv("HOMINEM_DB_PATH", None)
            self.db = TrainingDatabase(db_path=db_path)
        else:
            self.db = None


def generate_one(
    *,
    ctx: SelfTrainContext,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    do_sample = temperature > 0.0

    inputs = ctx.gen_tokenizer(
        prompt,
        return_tensors="pt",
    ).to(ctx.device)

    with torch.no_grad():
        output_ids = ctx.gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )

    return ctx.gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def score_with_reward_model(
    *,
    ctx: SelfTrainContext,
    prompt: str,
    response: str,
    max_length: int = 512,
) -> Dict[str, float]:
    text = f"User: {prompt}\nAssistant: {response}"
    encoded = ctx.reward_tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(ctx.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = ctx.reward_model(**encoded)
        logits = outputs.logits.squeeze(0).cpu().tolist()

    scores: Dict[str, float] = {}
    for idx, name in enumerate(REWARD_MODEL_TARGETS):
        if idx >= len(logits):
            break
        scores[name] = float(logits[idx])
    return scores


app = FastAPI(title="Hominem Self-Training API")
_ctx: Optional[SelfTrainContext] = None


@app.on_event("startup")
def _load_models() -> None:
    global _ctx
    _ctx = SelfTrainContext()


@app.post("/self_train", response_model=GenerateResponse)
def self_train(req: GenerateRequest) -> GenerateResponse:
    assert _ctx is not None, "Context not initialized"
    ctx = _ctx

    candidates: List[CandidateResult] = []
    for _ in range(req.num_candidates):
        text = generate_one(
            ctx=ctx,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        reward_vec = score_with_reward_model(
            ctx=ctx,
            prompt=req.prompt,
            response=text,
        )
        score = scalar_score(reward_vec)
        candidates.append(
            CandidateResult(
                text=text,
                reward=reward_vec,
                scalar_score=score,
            )
        )

    # Choose the highest-scoring candidate as "chosen".
    chosen = max(candidates, key=lambda c: c.scalar_score)

    # Log advanced stats.
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    log_record = {
        "timestamp_utc": timestamp_utc,
        "prompt": req.prompt,
        "num_candidates": req.num_candidates,
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "device": str(ctx.device),
        "chosen": chosen.dict(),
        "candidates": [c.dict() for c in candidates],
    }
    
    # Write to database if enabled
    if ctx.db:
        ctx.db.insert_self_train_event(
            session_id=ctx.session_id,
            timestamp_utc=timestamp_utc,
            prompt=req.prompt,
            chosen_text=chosen.text,
            chosen_reward=chosen.reward,
            chosen_scalar_score=chosen.scalar_score,
            candidates_json=[c.dict() for c in candidates],
            num_candidates=req.num_candidates,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            device=str(ctx.device),
        )
    
    # Also write to JSONL for backwards compatibility and debugging
    with ctx.log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    return GenerateResponse(
        prompt=req.prompt,
        chosen=chosen,
        candidates=candidates,
        logfile=str(ctx.log_path),
    )




