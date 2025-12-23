#!/usr/bin/env python3
"""
Unified Theory chat server (fresh implementation)

Features:
- Loads base LM (Qwen3-1.7B) + optional LoRA adapter
- Loads frozen manifold/regime heads
- Runs agent loop with self-tagging, anchors, Φ/ΔΦ, RewardIntensity
- Provides OpenAI-compatible /v1/chat/completions endpoint
"""

from __future__ import annotations

import os
import json
import re
import time
import uuid
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
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
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "10000"))
SLEEP_UPDATE_ENABLED = os.getenv("SLEEP_UPDATE_ENABLED", "false").lower() == "true"
SLEEP_UPDATE_CONFIG = os.getenv(
    "SLEEP_UPDATE_CONFIG",
    str(BASE_DIR / "config" / "training" / "sleep_sft_update.yaml"),
)
SLEEP_LOG_DIR = Path(os.getenv("SLEEP_LOG_DIR", BASE_DIR / "data" / "online_feedback"))
SLEEP_TRIGGER_TOKENS = int(os.getenv("SLEEP_TRIGGER_TOKENS", "8000"))
SLEEP_TARGET_TOKENS = int(os.getenv("SLEEP_TARGET_TOKENS", "1600"))
SLEEP_CONTINUITY_HEADER = os.getenv("SLEEP_CONTINUITY_HEADER", "true").lower() == "true"
SLEEP_CONTINUITY_MAX_CHARS = int(os.getenv("SLEEP_CONTINUITY_MAX_CHARS", "900"))


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
        last_post=d.get("last_post", {}),
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
        "last_post": state.last_post,
    }


def metrics_to_dict(m: TurnMetrics) -> Dict[str, Any]:
    def snapshot_to_dict(snap: Any) -> Dict[str, Any]:
        return {
            "s": snap.s,
            "s_self": snap.s_self,
            "s_world": snap.s_world,
            "self_fractions": snap.self_fractions,
            "mean_self": snap.mean_self,
            "regime_probs": snap.regime_probs,
            "regime_argmax": snap.regime_argmax,
            "lambdas": snap.lambdas,
            "anchors": snap.anchors,
            "phi": {"value": snap.phi_value, "components": snap.phi_components},
            "delta_phi": {"raw": snap.delta_phi_raw, "ema": snap.delta_phi_ema, "used": snap.delta_phi_used},
            "reward_intensity": snap.reward_intensity,
            "r_t": snap.r_t,
        }

    return {
        "pre": snapshot_to_dict(m.pre),
        "post": snapshot_to_dict(m.post),
        "think_gate": m.think_gate,
    }


def _prompt_token_count(history: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    # Count tokens for the actual model prompt after applying the model's chat template.
    if not history:
        return 0
    try:
        prompt = runtime._format_prompt(  # type: ignore[attr-defined]
            history,
            think_block=None,
            enable_thinking=False,
            tools=tools,
        )
        encoded = runtime.lm_tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
        return int(encoded["input_ids"].shape[-1])
    except Exception:
        # Conservative fallback (rare): approximate count using the model tokenizer
        # on a manual <|im_start|> chat serialization.
        parts: list[str] = []
        for msg in history:
            role = (msg.get("role") or "user").lower()
            content = msg.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)
        encoded = runtime.lm_tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
        return int(encoded["input_ids"].shape[-1])

def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _normalize_messages(messages: list[dict[str, Any]]) -> Tuple[list[dict[str, Any]], str]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").lower()
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "assistant"
        content = _extract_message_text(msg.get("content"))
        normalized_msg: dict[str, Any] = {"role": role, "content": content}
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            normalized_msg["tool_calls"] = tool_calls
        tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId")
        if role == "tool" and tool_call_id:
            normalized_msg["tool_call_id"] = tool_call_id
        normalized.append(normalized_msg)
    for idx in range(len(normalized) - 1, -1, -1):
        if normalized[idx]["role"] == "user":
            user_message = normalized[idx]["content"]
            return normalized[:idx], user_message
    raise HTTPException(status_code=400, detail="No user message found in messages.")


def _merge_canonical_history(
    canonical: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not incoming:
        return list(canonical), []
    if not canonical:
        return list(incoming), list(incoming)
    max_k = min(len(canonical), len(incoming))

    def message_key(msg: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")
        tool_calls_key = None
        if tool_calls is not None:
            try:
                tool_calls_key = json.dumps(tool_calls, sort_keys=True)
            except Exception:
                tool_calls_key = str(tool_calls)
        return (role, content, tool_calls_key, tool_call_id)

    canonical_keys = [message_key(m) for m in canonical]
    incoming_keys = [message_key(m) for m in incoming]
    for k in range(max_k, 0, -1):
        if canonical_keys[-k:] == incoming_keys[:k]:
            appended = list(incoming[k:])
            return list(canonical) + appended, appended
    appended = list(incoming)
    return list(canonical) + appended, appended


def _extract_tool_calls_from_content(content: str) -> Tuple[str, list[dict[str, Any]]]:
    if not content:
        return "", []
    tool_calls: list[dict[str, Any]] = []
    pattern = re.compile(r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", re.DOTALL)
    matches = list(pattern.finditer(content))
    if not matches:
        return content, []
    cleaned = content
    for match in matches:
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if name:
            if not isinstance(arguments, str):
                try:
                    arguments = json.dumps(arguments, ensure_ascii=False)
                except Exception:
                    arguments = str(arguments)
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments or "{}"},
                }
            )
    cleaned = pattern.sub("", cleaned).strip()
    return cleaned, tool_calls


def _strip_assistant_prefix(text: str | None) -> str:
    if not text:
        return ""
    prefix = "<|im_start|>assistant\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def _count_tokens_text(text: str) -> int:
    if not text:
        return 0
    encoded = runtime.lm_tokenizer(text, return_tensors="pt")  # type: ignore[union-attr]
    return int(encoded["input_ids"].shape[-1])


def _excerpt(text: str, limit: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "…"


def _avg_dict(dicts: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not dicts:
        return {k: 0.0 for k in keys}
    for k in keys:
        vals: list[float] = []
        for d in dicts:
            try:
                vals.append(float(d.get(k, 0.0)))
            except Exception:
                pass
        out[k] = float(sum(vals) / max(1, len(vals))) if vals else 0.0
    return out


def _mean_self_fraction_from_post(post: dict[str, Any]) -> float:
    sf = post.get("self_fractions")
    if isinstance(sf, dict) and sf:
        vals: list[float] = []
        for v in sf.values():
            try:
                vals.append(float(v))
            except Exception:
                pass
        if vals:
            return float(sum(vals) / max(1, len(vals)))
    try:
        return float(post.get("mean_self", 0.0) or 0.0)
    except Exception:
        return 0.0


def _sleep_event_score(entry: dict[str, Any]) -> float:
    metrics = entry.get("metrics") or {}
    post = (metrics.get("post") or {}) if isinstance(metrics, dict) else {}
    try:
        reward_intensity = float(post.get("reward_intensity", 0.0) or 0.0)
    except Exception:
        reward_intensity = 0.0
    reward_intensity = float(max(0.0, min(1.0, reward_intensity)))
    try:
        dphi = float(post.get("delta_phi_used", post.get("delta_phi_ema", post.get("delta_phi_raw", 0.0))) or 0.0)
    except Exception:
        dphi = 0.0
    mean_self = _mean_self_fraction_from_post(post)
    return float(reward_intensity * max(0.0, min(1.0, mean_self)) * abs(dphi))


def _build_continuity_header(state: ConversationState, drained: list[dict[str, Any]]) -> str:
    # Keep this compact; it's in the active prompt after trimming.
    last_post = state.last_post or {}
    manifold_hist = state.manifold_history or []
    manifold_keys = ["valence", "arousal", "dominance", "predictive_discrepancy", "temporal_directionality", "social_broadcast"]
    baseline = _avg_dict(manifold_hist[-5:], manifold_keys)
    baseline_str = ", ".join(f"{k}={baseline[k]:+.2f}" for k in baseline)

    # Very lightweight "pending" heuristic (we don't have an explicit commitments store yet).
    pending = "—"
    try:
        if float(last_post.get("s_self", {}).get("temporal_directionality", 0.0)) > 0.5:
            pending = "follow through on recent forward-looking commitments"
    except Exception:
        pass

    # Top etched events from what we're consolidating in this sleep.
    etched: list[str] = []
    if drained:
        scored = sorted([(float(_sleep_event_score(e)), e) for e in drained], key=lambda t: t[0], reverse=True)
        for score, e in scored[:3]:
            if score <= 0:
                continue
            u = _excerpt(str(e.get("user_message") or ""), 90)
            a = _excerpt(str(e.get("assistant") or ""), 120)
            etched.append(f"- {u} → {a}")

    # Include tiny numeric anchors for the runtime's persisted scalars.
    try:
        phi = float(last_post.get("phi_value", state.phi_prev) or 0.0)
    except Exception:
        phi = 0.0
    try:
        dphi_used = float(last_post.get("delta_phi_used", state.ema_delta_phi) or 0.0)
    except Exception:
        dphi_used = 0.0

    parts = [
        "Continuity (post-sleep anchor):",
        f"- baseline self-manifold: {baseline_str}",
        f"- pending: {pending}",
        f"- Φ={phi:+.2f}, ΔΦ_used={dphi_used:+.2f}",
    ]
    if etched:
        parts.append("- etched:")
        parts.extend(etched)
    text = "\n".join(parts).strip()
    if len(text) > int(SLEEP_CONTINUITY_MAX_CHARS):
        text = text[: int(SLEEP_CONTINUITY_MAX_CHARS)].rstrip() + "…"
    return text


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict[str, Any]]
    stream: bool | None = False
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    metadata: Dict[str, Any] | None = None
    reasoning_effort: str | None = None
    stream_options: Dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None


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
    force: bool = False


class SleepResponse(BaseModel):
    processed: int
    remaining: int
    token_count_before: int | None = None
    token_count_after: int | None = None
    dropped_history_messages: int | None = None
    continuity_header_added: bool | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenAI Chat Completions")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            # Support legacy (flat metrics dict) and new ({"pre":..., "post":...}) schemas.
            post_metrics = metrics.get("post") if isinstance(metrics, dict) else None
            if isinstance(post_metrics, dict):
                reward_intensity = float(post_metrics.get("reward_intensity", 0.0) or 0.0)
                delta_phi_used = float(post_metrics.get("delta_phi_used", 0.0) or 0.0)
                scalar_score = float(post_metrics.get("r_t", 0.0) or 0.0)
            else:
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
        sys.executable,
        "-m",
        "core.lora_trainer.sleep_sft_update",
        "--db-path",
        DATABASE_PATH,
        "--init-adapter",
        LORA_PATH or str(DEFAULT_LORA),
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

    projected_history = list(state.history) + [{"role": "user", "content": req.user_message}]
    projected_tokens = _prompt_token_count(projected_history)
    if projected_tokens >= int(MAX_CONTEXT_TOKENS):
        raise HTTPException(
            status_code=413,
            detail=f"Context too large ({projected_tokens} tokens). Call /sleep to trim history (trigger at {SLEEP_TRIGGER_TOKENS}).",
        )

    assistant, think_content, metrics, new_state, _think_block, _pass1_think = runtime.run_turn(
        state,
        req.user_message,
        enable_thinking=req.enable_thinking,
    )

    db.append_message(req.conversation_id, "user", req.user_message)
    db.append_message(req.conversation_id, "assistant", assistant, think=think_content)
    merged_state = dict(st_dict or {})
    merged_state.update(state_to_dict(new_state))
    db.save_state(req.conversation_id, merged_state)

    return ChatResponse(
        assistant=assistant,
        metrics=metrics_to_dict(metrics),
        sleep_queue_len=len(new_state.sleep_queue),
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    history, user_message = _normalize_messages(req.messages)
    conversation_id = (req.metadata or {}).get("conversation_id") or "canonical"
    enable_thinking = bool((req.metadata or {}).get("enable_thinking"))
    if req.reasoning_effort and req.reasoning_effort not in {"none", "minimal"}:
        enable_thinking = True

    tools = req.tools
    if req.tool_choice == "none":
        tools = None
    elif isinstance(req.tool_choice, dict):
        tool_name = ((req.tool_choice.get("function") or {}).get("name"))
        if tool_name and tools:
            tools = [t for t in tools if (t.get("function") or {}).get("name") == tool_name]
            if not tools:
                raise HTTPException(status_code=400, detail=f"tool_choice requested unknown tool '{tool_name}'.")

    st_dict = db.get_state(conversation_id)
    state = state_from_dict(st_dict)
    state.history, appended_history = _merge_canonical_history(state.history, history)
    for msg in appended_history:
        db.append_message(conversation_id, msg.get("role", "user"), msg.get("content", ""))

    projected_history = list(state.history) + [{"role": "user", "content": user_message}]
    projected_tokens = _prompt_token_count(projected_history, tools=tools)
    if projected_tokens >= int(MAX_CONTEXT_TOKENS):
        raise HTTPException(
            status_code=413,
            detail=f"Context too large ({projected_tokens} tokens). Reduce messages or truncate history.",
        )

    max_new_tokens = max(1, int(req.max_completion_tokens or req.max_tokens or 256))
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def build_response(
        assistant_text: str | None,
        output_think_block: str,
        model_name: str,
        tool_calls: list[dict[str, Any]] | None = None,
        finish_reason: str = "stop",
    ) -> Dict[str, Any]:
        output_content = f"{output_think_block}{assistant_text or ''}"
        prompt_tokens = _prompt_token_count(projected_history, tools=tools)
        completion_tokens = _count_tokens_text(output_content)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_content if output_content else None,
                        "tool_calls": tool_calls,
                        "refusal": None,
                        "annotations": [],
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
            "service_tier": "default",
        }

    if req.stream:
        def event_stream() -> Iterable[str]:
            think_block = None
            merged_think_block = None
            pass1_think = None
            pre_history, pre, think_gate, think_block, injected_think, hist = runtime._prepare_turn(
                state,
                user_message,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            temperature, top_p = runtime._resolve_sampling(think_block, enable_thinking, req.temperature, req.top_p)

            merged_think_block = think_block
            if think_block and enable_thinking:
                stop_ids = runtime.lm_tokenizer.encode("</think>", add_special_tokens=False)  # type: ignore[union-attr]
                prompt_first = runtime._format_prompt(
                    pre_history,
                    think_block=None,
                    enable_thinking=True,
                    add_generation_prompt=True,
                )
                first_generated = runtime._run_once(
                    prompt_first,
                    stop_ids=stop_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                _, pass1_think = runtime._extract_think(first_generated)
                merged_think_block = runtime._inject_think_block(think_block, pass1_think)

            output_think_block = _strip_assistant_prefix(merged_think_block)

            def send_chunk(
                delta_content: str | None = None,
                delta_role: str | None = None,
                finish_reason: str | None = None,
                delta_tool_calls: list[dict[str, Any]] | None = None,
            ) -> str:
                delta: Dict[str, Any] = {}
                if delta_role:
                    delta["role"] = delta_role
                if delta_content is not None:
                    delta["content"] = delta_content
                if delta_tool_calls is not None:
                    delta["tool_calls"] = delta_tool_calls
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if tools:
                assistant, think_content, _metrics, new_state, think_block, pass1_think = runtime.run_turn(
                    state,
                    user_message,
                    enable_thinking=enable_thinking,
                    max_new_tokens=max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    tools=tools,
                )
                output_think_block = think_block
                if pass1_think:
                    output_think_block = runtime._inject_think_block(think_block, pass1_think) if think_block else None
                output_think_block = _strip_assistant_prefix(output_think_block)
                cleaned, tool_calls = _extract_tool_calls_from_content(assistant)
                output_content = f"{output_think_block}{cleaned}"
                content_delta = output_content if output_content else None
                yield send_chunk(delta_role="assistant")
                if content_delta:
                    yield send_chunk(delta_content=content_delta)
                if tool_calls:
                    yield send_chunk(delta_tool_calls=tool_calls)
                db.append_message(conversation_id, "user", user_message)
                db.append_message(conversation_id, "assistant", assistant, think=think_content)
                merged_state = dict(st_dict or {})
                merged_state.update(state_to_dict(new_state))
                db.save_state(conversation_id, merged_state)
                yield send_chunk(
                    delta_content=None,
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
                yield "data: [DONE]\n\n"
                return

            yield send_chunk(delta_role="assistant")
            if output_think_block:
                yield send_chunk(delta_content=output_think_block)

            if think_block and enable_thinking:
                prompt_second = runtime._format_prompt(
                    pre_history,
                    think_block=merged_think_block,
                    enable_thinking=True,
                    add_generation_prompt=False,
                    tools=tools,
                )
                stream_iter = runtime._stream_once(
                    prompt_second,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                prompt = runtime._format_prompt(
                    pre_history,
                    think_block=think_block,
                    enable_thinking=enable_thinking,
                    add_generation_prompt=None,
                    tools=tools,
                )
                stream_iter = runtime._stream_once(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            generated_parts: list[str] = []
            for chunk in stream_iter:
                generated_parts.append(chunk)
                yield send_chunk(delta_content=chunk)

            generated = "".join(generated_parts)
            assistant_content, generated_think = runtime._extract_think(generated)

            _metrics, new_state, stored_think = runtime._finalize_turn(
                state=state,
                user_message=user_message,
                pre_history=pre_history,
                pre=pre,
                think_gate=think_gate,
                hist=hist,
                assistant_content=assistant_content,
                think_block=think_block,
                pass1_think=pass1_think,
                injected_think=injected_think,
            )

            db.append_message(conversation_id, "user", user_message)
            db.append_message(conversation_id, "assistant", assistant_content, think=stored_think)
            merged_state = dict(st_dict or {})
            merged_state.update(state_to_dict(new_state))
            db.save_state(conversation_id, merged_state)

            yield send_chunk(delta_content=None, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    assistant, think_content, _metrics, new_state, think_block, pass1_think = runtime.run_turn(
        state,
        user_message,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tools=tools,
    )

    output_think_block = think_block
    if pass1_think:
        output_think_block = runtime._inject_think_block(think_block, pass1_think) if think_block else None
    output_think_block = _strip_assistant_prefix(output_think_block)

    cleaned, tool_calls = _extract_tool_calls_from_content(assistant)

    db.append_message(conversation_id, "user", user_message)
    db.append_message(conversation_id, "assistant", assistant, think=think_content)
    merged_state = dict(st_dict or {})
    merged_state.update(state_to_dict(new_state))
    db.save_state(conversation_id, merged_state)

    finish_reason = "tool_calls" if tool_calls else "stop"
    return build_response(cleaned, output_think_block, req.model, tool_calls=tool_calls or None, finish_reason=finish_reason)


@app.post("/chat/completions")
def chat_completions_alias(req: ChatCompletionRequest):
    return chat_completions(req)


@app.post("/sleep", response_model=SleepResponse)
def sleep(req: SleepRequest):
    st_dict = db.get_state(req.conversation_id)
    state = state_from_dict(st_dict)
    token_before = _prompt_token_count(state.history)

    # Only perform a "real sleep" (flush + trim) once the context is large enough,
    # unless the caller forces it.
    should_sleep = bool(req.force) or token_before >= int(SLEEP_TRIGGER_TOKENS)

    to_process = state.sleep_queue if should_sleep else state.sleep_queue[: req.max_items]
    if to_process:
        _write_sleep_logs(req.conversation_id, to_process)
        for entry in to_process:
            metrics = entry.get("metrics") or {}
            post = (metrics.get("post") or {}) if isinstance(metrics, dict) else {}
            r_t = post.get("r_t") if isinstance(post, dict) else None
            reward_intensity = post.get("reward_intensity") if isinstance(post, dict) else None
            delta_phi_used = post.get("delta_phi_used") if isinstance(post, dict) else None
            try:
                r_t_val = float(r_t) if r_t is not None else None
            except Exception:
                r_t_val = None
            try:
                reward_intensity_val = float(reward_intensity) if reward_intensity is not None else None
            except Exception:
                reward_intensity_val = None
            try:
                delta_phi_used_val = float(delta_phi_used) if delta_phi_used is not None else None
            except Exception:
                delta_phi_used_val = None

            db.insert_sleep_event(
                conversation_id=req.conversation_id,
                user_message=str(entry.get("user_message") or ""),
                assistant=str(entry.get("assistant") or ""),
                think=entry.get("think"),
                history=entry.get("history"),
                metrics=metrics if isinstance(metrics, dict) else None,
                r_t=r_t_val,
                reward_intensity=reward_intensity_val,
                delta_phi_used=delta_phi_used_val,
            )
        if SLEEP_UPDATE_ENABLED:
            _run_sleep_update(SLEEP_LOG_DIR, SLEEP_UPDATE_CONFIG)

    if should_sleep:
        # Flush the queue during sleep.
        drained = list(to_process)
        state.sleep_queue = []

        header_added = False
        if SLEEP_CONTINUITY_HEADER:
            header_text = _build_continuity_header(state, drained=drained)
            header_msg = {"role": "system", "content": header_text}
            # Idempotent: replace existing continuity header if present.
            if state.history and (state.history[0].get("role") or "").lower() == "system":
                if (state.history[0].get("content") or "").strip().startswith("Continuity (post-sleep anchor):"):
                    state.history[0] = header_msg
                    header_added = True
            if not header_added:
                state.history.insert(0, header_msg)
                header_added = True

        # Drop oldest history until the full prompt token count is under target.
        dropped = 0
        token_after = token_before
        token_after = _prompt_token_count(state.history)
        # Preserve an optional leading system continuity header if present.
        start_idx = 1 if state.history and (state.history[0].get("role") or "").lower() == "system" else 0
        while len(state.history) > start_idx and token_after > int(SLEEP_TARGET_TOKENS):
            state.history.pop(start_idx)
            dropped += 1
            # Ensure we don't leave an assistant-leading fragment.
            while len(state.history) > start_idx and (state.history[start_idx].get("role") or "").lower() == "assistant":
                state.history.pop(start_idx)
                dropped += 1
            token_after = _prompt_token_count(state.history)
    else:
        remaining = state.sleep_queue[req.max_items :]
        state.sleep_queue = remaining
        dropped = 0
        token_after = token_before
        header_added = False

    merged_state = dict(st_dict or {})
    merged_state.update(state_to_dict(state))
    db.save_state(req.conversation_id, merged_state)
    return SleepResponse(
        processed=len(to_process),
        remaining=len(state.sleep_queue),
        token_count_before=token_before,
        token_count_after=token_after,
        dropped_history_messages=dropped,
        continuity_header_added=bool(header_added),
    )


@app.get("/messages")
def messages(conversation_id: str = "canonical", limit: int = 50):
    msgs = db.list_messages(conversation_id, limit=limit)
    return {"messages": [{"role": r, "content": c} for r, c in msgs]}

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
