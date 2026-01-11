from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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

    last: List[Dict[str, Any]] = []
    try:
        for chunk in responses_iter:
            if chunk:
                last = chunk
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent streaming failed: {exc}") from exc

    if not last:
        raise HTTPException(status_code=500, detail="Agent produced no response.")

    # Qwen-Agent returns a list of message dicts; take the last assistant content if present.
    assistant_text = ""
    for msg_obj in reversed(last):
        if msg_obj.get("role") == "assistant":
            assistant_text = str(msg_obj.get("content") or "")
            break

    session.messages.extend(last)
    return ChatOut(session_id=session_id, assistant=assistant_text, messages=session.messages)
