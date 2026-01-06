"""OpenAI-compatible inference API (slim)."""

from __future__ import annotations

import gc
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from hominem_infer.events import EventWriter


DEFAULT_MODEL_ID = os.getenv("INFER_MODEL_ID", "mlx-community/Qwen2-VL-2B-Instruct-4bit")
EVENT_LOG_PATH = os.getenv("INFER_EVENT_LOG")
EVENTS_ENABLED = os.getenv("INFER_EVENT_LOG", "").strip() != ""
BACKEND = os.getenv("INFER_BACKEND", "mlx_vlm")

_MODEL_CACHE: Dict[str, Any] = {}


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = DEFAULT_MODEL_ID
    messages: List[Dict[str, Any]]
    stream: bool | None = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: Any | None = None
    metadata: Dict[str, Any] | None = None
    stream_options: Dict[str, Any] | None = None
    adapter_path: Optional[str] = None
    resize_shape: Optional[List[int]] = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: Any
    model: str = DEFAULT_MODEL_ID
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool | None = False
    adapter_path: Optional[str] = None
    resize_shape: Optional[List[int]] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if (msg.get("role") or "").lower() == "user":
            return _extract_text(msg.get("content"))
    return ""


def _stub_completion(messages: List[Dict[str, Any]]) -> str:
    user_text = _last_user_text(messages)
    if user_text:
        return f"(stub) {user_text}"
    return "(stub) Hello."


def _stream_chunks(text: str, *, chunk_size: int = 40) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def _usage_stub() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _make_response(model: str, content: str, *, finish_reason: str = "stop") -> Dict[str, Any]:
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_stub(),
    }


def _stream_response(model: str, content: str) -> Iterator[str]:
    resp_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())
    first = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"
    for chunk in _stream_chunks(content):
        payload = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    done = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


app = FastAPI(title="Hominem Infer")
event_writer = EventWriter(
    enabled=EVENTS_ENABLED,
    log_path=(Path(EVENT_LOG_PATH) if EVENT_LOG_PATH else None),
)


def _mlx_load_model(model_id: str, adapter_path: Optional[str]):
    cache_key = f"{model_id}|{adapter_path or ''}"
    if _MODEL_CACHE.get("cache_key") == cache_key:
        return _MODEL_CACHE["model"], _MODEL_CACHE["processor"], _MODEL_CACHE["config"]

    try:
        from mlx_vlm.utils import load
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"mlx_vlm not installed: {exc}") from exc

    model, processor = load(model_id, adapter_path, trust_remote_code=True)
    config = model.config
    _MODEL_CACHE.clear()
    _MODEL_CACHE.update(
        {
            "cache_key": cache_key,
            "model": model,
            "processor": processor,
            "config": config,
        }
    )
    return model, processor, config


def _extract_media(messages: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    images: List[str] = []
    audio: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "input_image":
                images.append(str(item.get("image_url") or ""))
            elif item_type == "image_url":
                image_url = item.get("image_url") or {}
                images.append(str(image_url.get("url") or ""))
            elif item_type == "input_audio":
                input_audio = item.get("input_audio") or {}
                audio.append(str(input_audio.get("data") or ""))
    images = [img for img in images if img]
    audio = [aud for aud in audio if aud]
    return images, audio


def _responses_input_to_messages(payload: ResponsesRequest) -> List[Dict[str, Any]]:
    raw = payload.input
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if isinstance(raw, list):
        messages: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict) and "role" in item:
                messages.append(item)
        if messages:
            return messages
    raise HTTPException(status_code=400, detail="Unsupported responses input format.")


@app.get("/v1/models")
def list_models() -> ModelList:
    return ModelList(data=[{"id": DEFAULT_MODEL_ID, "object": "model"}])


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionRequest):
    if BACKEND == "stub":
        content = _stub_completion(payload.messages)
        event_writer.emit(
            "TurnEvent",
            {
                "model": payload.model or DEFAULT_MODEL_ID,
                "messages": payload.messages,
                "assistant": {"role": "assistant", "content": content},
                "stream": bool(payload.stream),
            },
        )
        if payload.stream:
            return StreamingResponse(
                _stream_response(payload.model or DEFAULT_MODEL_ID, content),
                media_type="text/event-stream",
            )
        return JSONResponse(_make_response(payload.model or DEFAULT_MODEL_ID, content))

    if BACKEND != "mlx_vlm":
        raise HTTPException(status_code=501, detail=f"Unknown backend: {BACKEND}")

    model_id = payload.model or DEFAULT_MODEL_ID
    model, processor, config = _mlx_load_model(model_id, payload.adapter_path)
    images, audio = _extract_media(payload.messages)

    try:
        from mlx_vlm.generate import generate, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"mlx_vlm not installed: {exc}") from exc

    kwargs: Dict[str, Any] = {}
    if payload.resize_shape:
        if len(payload.resize_shape) not in (1, 2):
            raise HTTPException(status_code=400, detail="resize_shape must have 1 or 2 integers")
        if len(payload.resize_shape) == 1:
            kwargs["resize_shape"] = (payload.resize_shape[0], payload.resize_shape[0])
        else:
            kwargs["resize_shape"] = (payload.resize_shape[0], payload.resize_shape[1])

    prompt = apply_chat_template(
        processor,
        config,
        payload.messages,
        num_images=len(images),
        num_audios=len(audio),
    )

    max_tokens = payload.max_tokens or payload.max_completion_tokens
    temperature = payload.temperature if payload.temperature is not None else 0.2
    top_p = payload.top_p if payload.top_p is not None else 1.0

    if payload.stream:
        def stream_generator():
            try:
                for chunk in stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    image=images or None,
                    audio=audio or None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs,
                ):
                    if not chunk or not hasattr(chunk, "text"):
                        continue
                    payload_chunk = {
                        "id": f"chatcmpl_{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload_chunk)}\n\n"
                done = {
                    "id": f"chatcmpl_{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                gc.collect()
        event_writer.emit(
            "TurnEvent",
            {
                "model": model_id,
                "messages": payload.messages,
                "assistant": {"role": "assistant", "content": "<stream>"},
                "stream": True,
            },
        )
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    gen_result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=images or None,
        audio=audio or None,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        verbose=False,
        **kwargs,
    )
    gc.collect()
    content = gen_result.text
    event_writer.emit(
        "TurnEvent",
        {
            "model": model_id,
            "messages": payload.messages,
            "assistant": {"role": "assistant", "content": content},
            "stream": bool(payload.stream),
        },
    )
    return JSONResponse(_make_response(model_id, content))


@app.post("/v1/responses")
def responses(payload: ResponsesRequest):
    if BACKEND == "stub":
        messages = _responses_input_to_messages(payload)
        content = _stub_completion(messages)
        created_at = int(time.time())
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"
        response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "instructions": None,
            "max_output_tokens": payload.max_output_tokens,
            "model": payload.model or DEFAULT_MODEL_ID,
            "output": [
                {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            ],
            "output_text": content,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "usage": _usage_stub(),
        }
        return JSONResponse(response)

    if BACKEND != "mlx_vlm":
        raise HTTPException(status_code=501, detail=f"Unknown backend: {BACKEND}")

    model_id = payload.model or DEFAULT_MODEL_ID
    model, processor, config = _mlx_load_model(model_id, payload.adapter_path)
    messages = _responses_input_to_messages(payload)
    images, audio = _extract_media(messages)

    try:
        from mlx_vlm.generate import generate, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"mlx_vlm not installed: {exc}") from exc

    kwargs: Dict[str, Any] = {}
    if payload.resize_shape:
        if len(payload.resize_shape) not in (1, 2):
            raise HTTPException(status_code=400, detail="resize_shape must have 1 or 2 integers")
        if len(payload.resize_shape) == 1:
            kwargs["resize_shape"] = (payload.resize_shape[0], payload.resize_shape[0])
        else:
            kwargs["resize_shape"] = (payload.resize_shape[0], payload.resize_shape[1])

    prompt = apply_chat_template(
        processor,
        config,
        messages,
        num_images=len(images),
        num_audios=len(audio),
    )
    max_tokens = payload.max_output_tokens
    temperature = payload.temperature if payload.temperature is not None else 0.2
    top_p = payload.top_p if payload.top_p is not None else 1.0

    if payload.stream:
        def stream_generator():
            response_id = f"resp_{uuid.uuid4().hex}"
            message_id = f"msg_{uuid.uuid4().hex}"
            created_at = int(time.time())
            base_response = {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "instructions": None,
                "max_output_tokens": payload.max_output_tokens,
                "model": model_id,
                "output": [],
                "output_text": "",
                "temperature": temperature,
                "top_p": top_p,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
            yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': base_response})}\n\n"
            yield f"event: response.in_progress\ndata: {json.dumps({'type': 'response.in_progress', 'response': base_response})}\n\n"
            message_item = {
                "id": message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': message_item})}\n\n"
            content_part = {"type": "output_text", "text": "", "annotations": []}
            yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': content_part})}\n\n"

            usage_stats = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            full_text = ""
            try:
                for chunk in stream_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    image=images or None,
                    audio=audio or None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs,
                ):
                    if not chunk or not hasattr(chunk, "text"):
                        continue
                    delta = chunk.text
                    full_text += delta
                    usage_stats = {
                        "input_tokens": getattr(chunk, "prompt_tokens", 0),
                        "output_tokens": getattr(chunk, "generation_tokens", 0),
                        "total_tokens": getattr(chunk, "prompt_tokens", 0)
                        + getattr(chunk, "generation_tokens", 0),
                    }
                    yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'delta': delta})}\n\n"
                yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'text': full_text})}\n\n"
                final_content_part = {"type": "output_text", "text": full_text, "annotations": []}
                yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': final_content_part})}\n\n"
                final_message_item = {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [final_content_part],
                }
                yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': final_message_item})}\n\n"
                completed = dict(base_response)
                completed.update(
                    {
                        "status": "completed",
                        "output": [final_message_item],
                        "output_text": full_text,
                        "usage": usage_stats,
                    }
                )
                yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': completed})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                gc.collect()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    gen_result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=images or None,
        audio=audio or None,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        verbose=False,
        **kwargs,
    )
    gc.collect()
    content = gen_result.text
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "instructions": None,
        "max_output_tokens": payload.max_output_tokens,
        "model": model_id,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        ],
        "output_text": content,
        "temperature": temperature,
        "top_p": top_p,
        "usage": {
            "input_tokens": getattr(gen_result, "prompt_tokens", 0),
            "output_tokens": getattr(gen_result, "generation_tokens", 0),
            "total_tokens": getattr(gen_result, "total_tokens", 0),
        },
    }
    return JSONResponse(response)
