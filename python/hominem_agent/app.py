from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from hominem_agent.agent import build_agent, default_tools


app = FastAPI(title="Hominem Agent")


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


INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Hominem Agent</title>
    <style>
      body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #0b0f16; color: #e6edf3; }
      header { padding: 12px 16px; border-bottom: 1px solid #222b3a; background: #0b0f16; position: sticky; top: 0; }
      main { max-width: 900px; margin: 0 auto; padding: 16px; }
      #log { white-space: pre-wrap; background: #0f1623; border: 1px solid #222b3a; border-radius: 8px; padding: 12px; min-height: 240px; }
      form { display: flex; gap: 8px; margin-top: 12px; }
      input[type=text] { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #222b3a; background: #0f1623; color: #e6edf3; }
      button { padding: 10px 12px; border-radius: 8px; border: 1px solid #2b3a52; background: #1a2638; color: #e6edf3; cursor: pointer; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .meta { color: #9db0c6; font-size: 12px; margin-top: 8px; }
      code { background: #0b1220; padding: 2px 4px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <header>
      <div><strong>Hominem Agent</strong> <span class="meta">UI → Qwen-Agent → infer → Qwen-Agent → UI</span></div>
    </header>
    <main>
      <div class="meta">Agent server: <code>/api/chat</code> • Infer base URL from <code>HOMINEM_INFER_BASE_URL</code></div>
      <div id="log"></div>
      <form id="form">
        <input id="msg" type="text" placeholder="Say something…" autocomplete="off" />
        <button id="send" type="submit">Send</button>
      </form>
    </main>
    <script>
      const log = document.getElementById("log");
      const form = document.getElementById("form");
      const msg = document.getElementById("msg");
      const send = document.getElementById("send");
      let sessionId = localStorage.getItem("hominem_session_id") || "";

      function append(text) {
        log.textContent += text + "\\n";
        log.scrollTop = log.scrollHeight;
      }

      async function chat(userText) {
        send.disabled = true;
        append("> " + userText);
        const body = { session_id: sessionId || null, message: userText };
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) {
          append("! error: " + (data.detail || JSON.stringify(data)));
          send.disabled = false;
          return;
        }
        sessionId = data.session_id;
        localStorage.setItem("hominem_session_id", sessionId);
        append(data.assistant);
        send.disabled = false;
      }

      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = (msg.value || "").trim();
        if (!text) return;
        msg.value = "";
        chat(text).catch((err) => {
          append("! exception: " + err);
          send.disabled = false;
        });
      });
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


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

