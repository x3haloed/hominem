from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from hominem_agent.agent import build_agent, default_tools


app = FastAPI(title="Hominem Agent")

_CORS_ORIGINS = [o.strip() for o in os.getenv("HOMINEM_UI_CORS_ORIGINS", "").split(",") if o.strip()]
if not _CORS_ORIGINS:
    # Default dev origin for `apps/hominem-ui` (Vite).
    _CORS_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class Session:
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)


_SESSIONS: Dict[str, Session] = {}
_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent(tools=default_tools())
    return _AGENT


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    session_id: Optional[str] = None
    message: str
    # OpenAI-style list-of-parts content is allowed for multimodal.
    content: Optional[List[Dict[str, Any]]] = None


class ChatOut(BaseModel):
    session_id: str
    assistant: str
    messages: List[Dict[str, Any]]

def _merge_maybe_delta_str(existing: Any, incoming: Any) -> Any:
    """
    Merge a possibly-streamed string field.

    Some clients emit cumulative strings; others emit deltas. We handle both:
    - If incoming starts with existing, treat it as cumulative and take incoming.
    - If existing starts with incoming, keep existing.
    - Otherwise, append incoming (treat as delta).
    """
    if not isinstance(incoming, str):
        return incoming
    if not isinstance(existing, str) or not existing:
        return incoming
    if incoming.startswith(existing):
        return incoming
    if existing.startswith(incoming):
        return existing
    return existing + incoming


def _merge_tool_calls(existing: Any, incoming: Any) -> Any:
    if not incoming:
        return existing
    if not existing:
        return incoming
    if isinstance(existing, list) and isinstance(incoming, list):
        seen = set()
        out: List[Any] = []
        for tc in existing + incoming:
            if not isinstance(tc, dict):
                out.append(tc)
                continue
            key = tc.get("id") or json.dumps(tc, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(tc)
        return out
    return incoming


def _accumulate_agent_chunks(chunks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Accumulate qwen-agent streamed output into a single list of new messages to append.

    `qwen-agent` often yields many chunks; each chunk may contain partial updates
    ("deltas") for fields like `content` and `reasoning_content`. Those deltas
    should be merged into a single assistant message for the turn, not treated as
    separate messages.
    """
    out: List[Dict[str, Any]] = []

    for chunk in chunks:
        if not chunk:
            continue
        for msg in chunk:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role") or "").strip().lower()

            if role == "assistant" and out and (out[-1].get("role") or "").strip().lower() == "assistant":
                # Merge streamed assistant fields into the last assistant message.
                out[-1]["content"] = _merge_maybe_delta_str(out[-1].get("content", ""), msg.get("content", ""))
                if "reasoning_content" in msg:
                    out[-1]["reasoning_content"] = _merge_maybe_delta_str(
                        out[-1].get("reasoning_content", ""),
                        msg.get("reasoning_content", ""),
                    )
                if "tool_calls" in msg:
                    out[-1]["tool_calls"] = _merge_tool_calls(out[-1].get("tool_calls"), msg.get("tool_calls"))
                continue

            # If we see an assistant message that only has reasoning_content, we still create it,
            # expecting a later delta to fill `content` into the same message.
            out.append(dict(msg))

    # Drop any assistant messages that never received any payload (shouldn't happen, but avoids UI artifacts).
    cleaned: List[Dict[str, Any]] = []
    for msg in out:
        role = (msg.get("role") or "").strip().lower()
        if role == "assistant":
            if not (msg.get("content") or msg.get("reasoning_content") or msg.get("tool_calls")):
                continue
        cleaned.append(msg)
    return cleaned

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        """<!doctype html>
<html>
  <head><meta charset="utf-8" /><title>Hominem Agent</title></head>
  <body style="font-family: system-ui, -apple-system, sans-serif; padding: 24px;">
    <h1 style="margin: 0 0 8px 0;">Hominem Agent</h1>
    <p style="margin: 0 0 16px 0;">API server for orchestration + tool calling.</p>
    <ul>
      <li><code>POST /api/chat</code></li>
      <li><code>GET /health</code></li>
    </ul>
    <p style="margin-top: 16px;">
      UI is now a separate SPA: <code>apps/hominem-ui</code>.
    </p>
  </body>
</html>"""
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn) -> ChatOut:
    agent = _get_agent()
    req_id = f"req_{uuid.uuid4().hex}"

    session_id = (payload.session_id or "").strip() or f"sess_{uuid.uuid4().hex}"
    session = _SESSIONS.get(session_id)
    if session is None:
        session = Session(session_id=session_id)
        _SESSIONS[session_id] = session

    if payload.content is not None:
        user_content: Any = payload.content
    else:
        user_content = payload.message
    session.messages.append({"role": "user", "content": user_content})

    try:
        responses_iter = agent.run(messages=session.messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc

    chunks: List[List[Dict[str, Any]]] = []
    try:
        for chunk in responses_iter:
            if chunk:
                chunks.append(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent streaming failed: {exc}") from exc

    if not chunks:
        raise HTTPException(status_code=500, detail="Agent produced no response.")

    last_norm = _accumulate_agent_chunks(chunks)

    # Qwen-Agent returns a list of message dicts; take the last assistant content if present.
    assistant_text = ""
    for msg_obj in reversed(last_norm):
        if msg_obj.get("role") == "assistant":
            assistant_text = str(msg_obj.get("content") or "")
            break

    session.messages.extend(last_norm)
    return ChatOut(session_id=session_id, assistant=assistant_text, messages=session.messages)

@app.post("/api/chat/stream")
def chat_stream(payload: ChatIn):
    """
    Stream chat updates as newline-delimited JSON (NDJSON).

    Events:
    - {"type":"start","session_id":...}
    - {"type":"assistant","assistant": {...}}  (snapshot-style updates)
    - {"type":"done","session_id":...,"assistant":...,"messages":[...]}
    - {"type":"error","detail":...}
    """
    agent = _get_agent()
    req_id = f"req_{uuid.uuid4().hex}"

    session_id = (payload.session_id or "").strip() or f"sess_{uuid.uuid4().hex}"
    session = _SESSIONS.get(session_id)
    if session is None:
        session = Session(session_id=session_id)
        _SESSIONS[session_id] = session

    if payload.content is not None:
        user_content: Any = payload.content
    else:
        user_content = payload.message
    session.messages.append({"role": "user", "content": user_content})

    def gen():
        yield json.dumps({"type": "start", "session_id": session_id}, ensure_ascii=False) + "\n"

        chunks: List[List[Dict[str, Any]]] = []
        assistant_state: Dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": ""}
        last_sent_key = ""

        try:
            responses_iter = agent.run(messages=session.messages)
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": f"Agent run failed: {exc}"}, ensure_ascii=False) + "\n"
            return

        try:
            for chunk in responses_iter:
                if not chunk:
                    continue
                chunks.append(chunk)

                # Merge assistant updates within this chunk into a single snapshot state.
                for msg in chunk:
                    if not isinstance(msg, dict):
                        continue
                    if (msg.get("role") or "").strip().lower() != "assistant":
                        continue
                    assistant_state["content"] = _merge_maybe_delta_str(
                        assistant_state.get("content", ""),
                        msg.get("content", ""),
                    )
                    if "reasoning_content" in msg:
                        assistant_state["reasoning_content"] = _merge_maybe_delta_str(
                            assistant_state.get("reasoning_content", ""),
                            msg.get("reasoning_content", ""),
                        )
                    if "tool_calls" in msg:
                        assistant_state["tool_calls"] = _merge_tool_calls(
                            assistant_state.get("tool_calls"),
                            msg.get("tool_calls"),
                        )

                key = json.dumps(
                    {
                        "content": assistant_state.get("content") or "",
                        "reasoning_content": assistant_state.get("reasoning_content") or "",
                        "tool_calls": assistant_state.get("tool_calls"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if key != last_sent_key:
                    last_sent_key = key
                    yield json.dumps({"type": "assistant", "assistant": assistant_state}, ensure_ascii=False) + "\n"

        except Exception as exc:
            yield json.dumps({"type": "error", "detail": f"Agent streaming failed: {exc}"}, ensure_ascii=False) + "\n"
            return

        last_norm = _accumulate_agent_chunks(chunks)
        if not last_norm:
            yield json.dumps({"type": "error", "detail": "Agent produced no response."}, ensure_ascii=False) + "\n"
            return

        assistant_text = ""
        for msg_obj in reversed(last_norm):
            if msg_obj.get("role") == "assistant":
                assistant_text = str(msg_obj.get("content") or "")
                break

        session.messages.extend(last_norm)
        yield json.dumps(
            {
                "type": "done",
                "session_id": session_id,
                "assistant": assistant_text,
                "messages": session.messages,
            },
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
