#!/usr/bin/env python3
"""
Unified Theory chat server (fresh implementation)

Features:
- Loads base LM (Qwen3-1.7B) + optional LoRA adapter
- Loads frozen manifold/regime heads
- Runs agent loop with self-tagging, anchors, Φ/ΔΦ, RewardIntensity
- Provides simple FastAPI with /chat and /sleep endpoints
"""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from apps.serve.agent_runtime import AgentRuntime, ConversationState, TurnMetrics
from apps.serve.database import ConversationDB


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFOLD = BASE_DIR / "artifacts" / "manifold_bert_optimized" / "checkpoint-3612"
DEFAULT_REGIME = BASE_DIR / "artifacts" / "regime_bert_base" / "checkpoint-1505"
DEFAULT_LORA = BASE_DIR / "artifacts" / "lora" / "qwen3-1.7b-seed-sft-v3"

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    str(Path.home() / "Documents" / "hominem" / "conversations.db"),
)
BASE_MODEL_ID = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen3-1.7B") or "Qwen/Qwen3-1.7B"
LORA_PATH = os.getenv("AUTO_LOAD_LORA")
if LORA_PATH:
    # allow bare name under artifacts/lora
    lp = Path(LORA_PATH)
    if not lp.is_absolute():
        lp = BASE_DIR / "artifacts" / "lora" / LORA_PATH
    LORA_PATH = str(lp)
elif DEFAULT_LORA.exists():
    LORA_PATH = str(DEFAULT_LORA)
else:
    LORA_PATH = None

MANIFOLD_CKPT = os.getenv("MANIFOLD_CKPT") or str(DEFAULT_MANIFOLD)
REGIME_CKPT = os.getenv("REGIME_CKPT") or str(DEFAULT_REGIME)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SLEEP_UPDATE_ENABLED = os.getenv("SLEEP_UPDATE_ENABLED", "false").lower() == "true"
SLEEP_UPDATE_CONFIG = os.getenv("SLEEP_UPDATE_CONFIG", str(BASE_DIR / "config" / "training" / "lora_dpo.yaml"))
SLEEP_LOG_DIR = Path(os.getenv("SLEEP_LOG_DIR", BASE_DIR / "data" / "online_feedback"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def state_from_dict(d: Dict[str, Any]) -> ConversationState:
    if not d:
        return ConversationState()
    return ConversationState(
        history=d.get("history", []),
        phi_prev=d.get("phi_prev", 0.0),
        ema_delta_phi=d.get("ema_delta_phi", 0.0),
        mean_self_prev=d.get("mean_self_prev", 0.0),
        manifold_history=d.get("manifold_history", []),
        sleep_queue=d.get("sleep_queue", []),
        intervention_state=d.get("intervention_state", {}),
    )


def state_to_dict(state: ConversationState) -> Dict[str, Any]:
    return {
        "history": state.history,
        "phi_prev": state.phi_prev,
        "ema_delta_phi": state.ema_delta_phi,
        "mean_self_prev": state.mean_self_prev,
        "manifold_history": state.manifold_history,
        "sleep_queue": state.sleep_queue,
        "intervention_state": state.intervention_state,
    }


def metrics_to_dict(m: TurnMetrics) -> Dict[str, Any]:
    return {
        "s": m.s,
        "s_self": m.s_self,
        "s_world": m.s_world,
        "self_fractions": m.self_fractions,
        "regime_probs": m.regime_probs,
        "regime_argmax": m.regime_argmax,
        "lambdas": m.lambdas,
        "anchors": m.anchors,
        "phi": {"value": m.phi_value, "components": m.phi_components},
        "delta_phi": {"raw": m.delta_phi_raw, "ema": m.delta_phi_ema, "used": m.delta_phi_used},
        "reward_intensity": m.reward_intensity,
        "r_t": m.r_t,
        "think_gate": m.think_gate,
    }


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    conversation_id: str = "canonical"
    user_message: str
    enable_thinking: bool = False


class ChatResponse(BaseModel):
    assistant: str
    metrics: Dict[str, Any]
    sleep_queue_len: int


class SleepRequest(BaseModel):
    conversation_id: str = "canonical"
    max_items: int = 10


class SleepResponse(BaseModel):
    processed: int
    remaining: int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Unified Theory Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

db = ConversationDB(DATABASE_PATH)
runtime = AgentRuntime(
    base_model_id=BASE_MODEL_ID,
    lora_path=LORA_PATH,
    manifold_checkpoint=MANIFOLD_CKPT,
    regime_checkpoint=REGIME_CKPT,
    device="mps",
    )


def _write_sleep_logs(conversation_id: str, entries: list[dict[str, Any]]) -> Path:
    SLEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = SLEEP_LOG_DIR / f"session_{conversation_id}_{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            metrics = entry.get("metrics", {}) or {}
            reward_intensity = float(metrics.get("reward_intensity", metrics.get("r_t", 0.0) or 0.0))
            delta_phi_used = float(
                (metrics.get("delta_phi") or {}).get("used", metrics.get("delta_phi_used", 0.0) or 0.0)
            )
            scalar_score = float(metrics.get("r_t", delta_phi_used))
            # We don't currently compute safety_score; leave as neutral 0.0
            safety_score = 0.0
            think_text = entry.get("think")
            record = {
                "prompt": entry.get("user_message", ""),
                "think": think_text,
                "candidates": [
                    {
                        "text": entry.get("assistant", ""),
                        "reward": {
                            "reward_intensity": reward_intensity,
                            "safety_score": safety_score,
                            "delta_phi_used": delta_phi_used,
                            "scalar_score": scalar_score,
                        },
                        "scalar_score": scalar_score,
                        "metrics": metrics,
                    }
                ],
                "timestamp_utc": ts,
            }
            f.write(json.dumps(record))
            f.write("\n")
    return path


def _run_sleep_update(log_dir: Path, config_path: str) -> None:
    cmd = [
        "python",
        "-m",
        "core.lora_trainer.online_update",
        "--log-dir",
        str(log_dir),
        "--config",
        config_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"⚠️ Sleep update failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    st_dict = db.get_state(req.conversation_id)
    state = state_from_dict(st_dict)

    assistant, think_content, metrics, new_state = runtime.run_turn(
        state, req.user_message, enable_thinking=req.enable_thinking
    )

    db.append_message(req.conversation_id, "user", req.user_message)
    db.append_message(req.conversation_id, "assistant", assistant, think=think_content)
    db.save_state(req.conversation_id, state_to_dict(new_state))

    return ChatResponse(
        assistant=assistant,
        metrics=metrics_to_dict(metrics),
        sleep_queue_len=len(new_state.sleep_queue),
    )


@app.post("/sleep", response_model=SleepResponse)
def sleep(req: SleepRequest):
    st_dict = db.get_state(req.conversation_id)
    state = state_from_dict(st_dict)
    to_process = state.sleep_queue[: req.max_items]
    if to_process:
        log_path = _write_sleep_logs(req.conversation_id, to_process)
        if SLEEP_UPDATE_ENABLED:
            _run_sleep_update(SLEEP_LOG_DIR, SLEEP_UPDATE_CONFIG)
    remaining = state.sleep_queue[req.max_items :]
    state.sleep_queue = remaining
    db.save_state(req.conversation_id, state_to_dict(state))
    return SleepResponse(processed=len(to_process), remaining=len(remaining))


@app.get("/messages")
def messages(conversation_id: str = "canonical", limit: int = 50):
    msgs = db.list_messages(conversation_id, limit=limit)
    return {"messages": [{"role": r, "content": c} for r, c in msgs]}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
