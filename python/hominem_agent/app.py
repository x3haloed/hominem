from __future__ import annotations

import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from hominem_agent.agent import build_agent, default_tools


app = FastAPI(title="Hominem Agent")

_CORS_ORIGINS = [o.strip() for o in os.getenv("HOMINEM_UI_CORS_ORIGINS", "").split(",") if o.strip()]
if not _CORS_ORIGINS:
    # Common Open WebUI dev/prod origins.
    _CORS_ORIGINS = [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent(tools=default_tools())
    return _AGENT


def _get_model_id() -> str:
    model = os.getenv("HOMINEM_AGENT_MODEL", os.getenv("INFER_MODEL_ID", "")).strip()
    if not model:
        model = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    return model


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    stream: bool | None = False
    tools: List[Dict[str, Any]] | None = None
    tool_choice: Any | None = None
    stream_options: Dict[str, Any] | None = None


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").strip().lower()
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = msg.get("content")
        if content is None:
            content = ""
        normalized_msg: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and msg.get("tool_calls") is not None:
            normalized_msg["tool_calls"] = msg.get("tool_calls")
        if role == "tool":
            tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId")
            if tool_call_id:
                normalized_msg["tool_call_id"] = tool_call_id
        normalized.append(normalized_msg)
    return normalized


def _merge_maybe_delta_str(existing: Any, incoming: Any) -> Any:
    if not isinstance(incoming, str):
        return incoming
    if not isinstance(existing, str) or not existing:
        return incoming
    if incoming.startswith(existing):
        return incoming
    if existing.startswith(incoming):
        return existing
    return existing + incoming


def _extract_delta(previous: str, incoming: str) -> tuple[str, str]:
    if not incoming:
        return previous, ""
    if not previous:
        return incoming, incoming
    if incoming.startswith(previous):
        return incoming, incoming[len(previous):]
    if previous.startswith(incoming):
        return previous, ""
    return previous + incoming, incoming


def _process_qwen_tool_calls(chunks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert Qwen-Agent tool call format to OpenAI format."""
    import re
    from hominem_agent.tools.openai_tools import execute_tool

    processed = []
    tool_call_id = f"call_{uuid.uuid4().hex}"

    for chunk in chunks:
        for msg in chunk:
            # Ensure msg is a dict
            if not isinstance(msg, dict):
                continue

            if msg.get("role") == "assistant" and "content" in msg:
                content = msg["content"]
                # Look for Qwen tool call format: {"name": "...", "arguments": {...}}</tool_call>
                tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                if tool_match:
                    try:
                        tool_data = json.loads(tool_match.group(1))
                        tool_name = tool_data.get("name")
                        tool_args = tool_data.get("arguments", {})

                        # Execute the tool
                        tool_result = execute_tool(tool_name, **tool_args)

                        # Create OpenAI format tool call
                        tool_call = {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args)
                            }
                        }

                        # Replace content with clean version and add tool_calls
                        clean_content = content.replace(tool_match.group(0), "").strip()
                        msg["content"] = clean_content if clean_content else None
                        msg["tool_calls"] = [tool_call]

                        # Add tool result message
                        processed.append(msg)
                        processed.append({
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": tool_call_id
                        })
                        continue
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        # If parsing fails, keep original content
                        pass

            processed.append(msg)

    return processed


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
    out: List[Dict[str, Any]] = []

    for chunk in chunks:
        if not chunk:
            continue
        for msg in chunk:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role") or "").strip().lower()

            if role == "assistant" and out and (out[-1].get("role") or "").strip().lower() == "assistant":
                out[-1]["content"] = _merge_maybe_delta_str(out[-1].get("content", ""), msg.get("content", ""))
                if "reasoning_content" in msg:
                    out[-1]["reasoning_content"] = _merge_maybe_delta_str(
                        out[-1].get("reasoning_content", ""),
                        msg.get("reasoning_content", ""),
                    )
                if "tool_calls" in msg:
                    out[-1]["tool_calls"] = _merge_tool_calls(out[-1].get("tool_calls"), msg.get("tool_calls"))
                continue

            out.append(dict(msg))

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
      <li><code>GET /v1/models</code></li>
      <li><code>POST /v1/chat/completions</code></li>
      <li><code>GET /health</code></li>
    </ul>
  </body>
</html>"""
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    model_id = _get_model_id()
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "hominem",
            }
        ],
    }


def _run_agent(
    *,
    agent: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    tool_choice: Any | None,
) -> Iterable[List[Dict[str, Any]]]:
    kwargs: Dict[str, Any] = {"messages": messages}
    sig = None
    try:
        sig = inspect.signature(agent.run)
    except Exception:
        sig = None
    if sig is not None:
        if "tools" in sig.parameters:
            kwargs["tools"] = tools
        if "tool_choice" in sig.parameters:
            kwargs["tool_choice"] = tool_choice
    return agent.run(**kwargs)


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionRequest) -> Any:
    agent = _get_agent()
    req_id = f"req_{uuid.uuid4().hex}"
    model_name = payload.model or _get_model_id()
    normalized_messages = _normalize_messages(payload.messages)

    tools = payload.tools
    if payload.tool_choice == "none":
        tools = None

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        responses_iter = _run_agent(
            agent=agent,
            messages=normalized_messages,
            tools=tools,
            tool_choice=payload.tool_choice,
        )

        # Collect all responses to process tool calls
        all_chunks = []
        for chunk in responses_iter:
            if chunk:
                all_chunks.extend(chunk)

        # Process tool calls from Qwen-Agent format to OpenAI format
        if all_chunks:
            processed_chunks = _process_qwen_tool_calls([all_chunks])
            responses_iter = iter([processed_chunks])
        else:
            # If no chunks, create empty response
            responses_iter = iter([[]])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc

    def build_response(
        *,
        assistant_text: str | None,
        tool_calls: List[Dict[str, Any]] | None,
        finish_reason: str,
    ) -> Dict[str, Any]:
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
                        "content": assistant_text if assistant_text else None,
                        "tool_calls": tool_calls,
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    if payload.stream:
        def event_stream() -> Iterable[str]:
            def send_chunk(
                *,
                delta_content: str | None = None,
                delta_role: str | None = None,
                delta_tool_calls: List[Dict[str, Any]] | None = None,
                finish_reason: str | None = None,
                usage: Dict[str, Any] | None = None,
            ) -> str:
                delta: Dict[str, Any] = {}
                if delta_role:
                    delta["role"] = delta_role
                if delta_content is not None:
                    delta["content"] = delta_content
                if delta_tool_calls is not None:
                    delta["tool_calls"] = delta_tool_calls
                if usage is not None:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                        "usage": usage,
                    }
                else:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                    }
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # Process all chunks first for tool calls
            all_chunks = []
            try:
                for chunk in responses_iter:
                    if chunk:
                        all_chunks.extend(chunk)
            except Exception as exc:
                error_payload = {
                    "error": {
                        "message": f"Agent streaming failed: {exc}",
                        "type": "server_error",
                    }
                }
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Process tool calls
            processed_chunks = _process_qwen_tool_calls([all_chunks]) if all_chunks else []

            # Stream the processed chunks
            assistant_content = ""
            tool_calls: List[Dict[str, Any]] | None = None
            yielded_assistant = False

            for msg in processed_chunks:
                if msg.get("role") == "assistant":
                    if not yielded_assistant:
                        yield send_chunk(delta_role="assistant")
                        yielded_assistant = True

                    content = msg.get("content") or ""
                    if content:
                        assistant_content, delta = _extract_delta(assistant_content, str(content))
                        if delta:
                            yield send_chunk(delta_content=delta)

                    if "tool_calls" in msg:
                        merged = _merge_tool_calls(tool_calls, msg.get("tool_calls"))
                        if merged != tool_calls:
                            tool_calls = merged
                            if tool_calls:
                                yield send_chunk(delta_tool_calls=tool_calls)

                elif msg.get("role") == "tool":
                    # Tool results are sent as separate messages, but for streaming
                    # we might need to handle this differently. For now, skip tool results in streaming.
                    pass

            finish_reason = "tool_calls" if tool_calls else "stop"
            if payload.stream_options and payload.stream_options.get("include_usage"):
                yield send_chunk(
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                )
            yield send_chunk(finish_reason=finish_reason)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # For non-streaming, process all chunks and tool calls
    chunks: List[List[Dict[str, Any]]] = []
    try:
        for chunk in responses_iter:
            if chunk:
                chunks.append(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent streaming failed: {exc}") from exc

    if not chunks:
        raise HTTPException(status_code=500, detail="Agent produced no response.")

    # Process tool calls from Qwen format to OpenAI format
    processed_chunks = _process_qwen_tool_calls(chunks)

    assistant_text = None
    tool_calls: List[Dict[str, Any]] | None = None
    for msg_obj in reversed(processed_chunks):
        if msg_obj.get("role") == "assistant":
            assistant_text = str(msg_obj.get("content") or "") or None
            tool_calls = msg_obj.get("tool_calls")
            break

    finish_reason = "tool_calls" if tool_calls else "stop"
    return build_response(
        assistant_text=assistant_text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )
