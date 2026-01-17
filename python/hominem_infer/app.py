"""OpenAI-compatible inference API (slim)."""

from __future__ import annotations

import gc
import json
import os
import time
import uuid
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from hominem_infer.events import EventWriter
from hominem_infer.media import extract_media, materialize_video_path, normalize_video_elements
from hominem_infer.parsing import ThinkStreamParser, extract_reasoning_from_text, extract_tool_calls_from_text


DEFAULT_MODEL_ID = os.getenv("INFER_MODEL_ID", "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx")
EVENT_LOG_PATH = os.getenv("INFER_EVENT_LOG")
EVENTS_ENABLED = os.getenv("INFER_EVENT_LOG", "").strip() != ""
BACKEND = os.getenv("INFER_BACKEND", "mlx_vlm")

_MODEL_CACHE: Dict[str, Any] = {}
_MLX_LOCK = threading.Lock()

_DEBUG_PROMPT_DUMP_ENABLED = os.getenv("INFER_DEBUG_PROMPT_DUMP", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_DEBUG_PROMPT_DUMP_DIR = os.getenv("INFER_DEBUG_PROMPT_DUMP_DIR", "").strip()


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
    tools: List[Dict[str, Any]] | None = None
    tool_choice: Any | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]


def _dump_templatized_prompt(*, prompt: str, model_id: str, endpoint: str) -> None:
    if not _DEBUG_PROMPT_DUMP_ENABLED:
        return
    try:
        target_dir = Path(_DEBUG_PROMPT_DUMP_DIR) if _DEBUG_PROMPT_DUMP_DIR else Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        dump_path = target_dir / f"infer_prompt_{endpoint}_{int(time.time())}_{uuid.uuid4().hex}.txt"
        dump_path.write_text(prompt, encoding="utf-8")
    except Exception:
        # Debug-only; never fail inference because of prompt dumping.
        return


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


def _make_response(
    model: str,
    content: str,
    *,
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    reasoning_content: Optional[str] = None,
) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
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


#
# NOTE: parsing helpers live in `hominem_infer.parsing`.
#


def _tool_calls_to_responses_required_action(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "submit_tool_outputs",
        "submit_tool_outputs": {
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": tc["function"],
                }
                for tc in tool_calls
            ]
        },
    }


def _tool_calls_to_responses_output_items(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        items.append(
            {
                "id": tc.get("id"),
                "type": "function_call",
                "status": "completed",
                "name": fn.get("name"),
                "arguments": fn.get("arguments"),
            }
        )
    return items


app = FastAPI(title="Hominem Infer")
event_writer = EventWriter(
    enabled=EVENTS_ENABLED,
    log_path=(Path(EVENT_LOG_PATH) if EVENT_LOG_PATH else None),
)


def _mlx_load_model(model_id: str, adapter_path: Optional[str]):
    cache_key = f"{model_id}|{adapter_path or ''}"
    with _MLX_LOCK:
        if _MODEL_CACHE.get("cache_key") == cache_key:
            return _MODEL_CACHE["model"], _MODEL_CACHE["processor"], _MODEL_CACHE["config"]

        try:
            from mlx_vlm.utils import load
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=f"mlx_vlm not installed: {exc}") from exc

        try:
            model, processor = load(model_id, adapter_path, trust_remote_code=True)
        except AttributeError as exc:
            if "'list' object has no attribute 'keys'" in str(exc):
                # Try loading without trust_remote_code for models with tokenizer config issues
                model, processor = load(model_id, adapter_path, trust_remote_code=False)
            else:
                raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc
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
    normalized_messages, video_specs = normalize_video_elements(payload.messages)
    images, audio = extract_media(normalized_messages)

    try:
        from mlx_vlm.generate import generate, stream_generate
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

    # Video support: route through mlx_vlm.video_generate so we produce pixel_values_videos + grid_thw.
    if video_specs:
        try:
            import mlx.core as mx
            from mlx_vlm.video_generate import process_vision_info
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=f"mlx_vlm video dependencies missing: {exc}") from exc

        # Materialize file-based videos so OpenCV can read them.
        # (Frame-list videos are handled inside mlx_vlm.video_generate.fetch_video.)
        for spec in video_specs:
            if not spec.is_file_like:
                continue
            try:
                spec.ele["video"] = materialize_video_path(str(spec.ele.get("video") or ""))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid video input: {exc}") from exc

        chat_template_kwargs: Dict[str, Any] = {}
        if payload.tools:
            chat_template_kwargs["tools"] = payload.tools
        if payload.tool_choice is not None:
            chat_template_kwargs["tool_choice"] = payload.tool_choice

        try:
            prompt = processor.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except TypeError as exc:
            if payload.tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model chat_template does not support tools: {exc}",
                ) from exc
            raise HTTPException(
                status_code=500,
                detail=f"Processor chat_template failure: {exc}",
            ) from exc

        image_inputs, video_inputs, _video_kwargs = process_vision_info(normalized_messages, True)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        with _MLX_LOCK:
            input_ids = mx.array(inputs["input_ids"])
            pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
            if pixel_values is None:
                raise HTTPException(status_code=400, detail="Video input produced no pixel values.")
            pixel_values = mx.array(pixel_values)
            mask = mx.array(inputs["attention_mask"])
            kwargs["input_ids"] = input_ids
            kwargs["pixel_values"] = pixel_values
            kwargs["mask"] = mask
            if inputs.get("video_grid_thw") is not None:
                kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
            if inputs.get("image_grid_thw") is not None:
                kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])
    else:
        chat_template_kwargs = {}
        if payload.tools:
            chat_template_kwargs["tools"] = payload.tools
        if payload.tool_choice is not None:
            chat_template_kwargs["tool_choice"] = payload.tool_choice
        try:
            prompt = processor.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except TypeError as exc:
            if payload.tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model chat_template does not support tools: {exc}",
                ) from exc
            raise HTTPException(
                status_code=500,
                detail=f"Processor chat_template failure: {exc}",
            ) from exc
    _dump_templatized_prompt(prompt=prompt, model_id=model_id, endpoint="chat_completions")

    max_tokens = payload.max_tokens or payload.max_completion_tokens
    temperature = payload.temperature if payload.temperature is not None else 0.2
    top_p = payload.top_p if payload.top_p is not None else 1.0

    if payload.stream:
        def stream_generator():
            resp_id = f"chatcmpl_{uuid.uuid4().hex}"
            created = int(time.time())
            buffer = ""
            saw_tool_call = False
            think_parser = ThinkStreamParser()
            try:
                first = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first)}\n\n"

                with _MLX_LOCK:
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
                        delta = str(getattr(chunk, "text", "") or "")
                        if not delta:
                            continue
                        buffer += delta
                        if saw_tool_call:
                            continue

                        tool_idx = delta.find("<tool_call>")
                        if tool_idx != -1:
                            prefix = delta[:tool_idx]
                            if prefix:
                                c_delta, r_delta = think_parser.feed(prefix)
                                if r_delta:
                                    payload_chunk = {
                                        "id": resp_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_id,
                                        "choices": [{"index": 0, "delta": {"reasoning_content": r_delta}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(payload_chunk)}\n\n"
                                if c_delta:
                                    payload_chunk = {
                                        "id": resp_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_id,
                                        "choices": [{"index": 0, "delta": {"content": c_delta}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(payload_chunk)}\n\n"
                            saw_tool_call = True
                            continue

                        c_delta, r_delta = think_parser.feed(delta)
                        if r_delta:
                            payload_chunk = {
                                "id": resp_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"reasoning_content": r_delta}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(payload_chunk)}\n\n"
                        if c_delta:
                            payload_chunk = {
                                "id": resp_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": c_delta}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(payload_chunk)}\n\n"

                c_final, r_final = think_parser.finish()
                if r_final:
                    payload_chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"reasoning_content": r_final}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload_chunk)}\n\n"
                if c_final:
                    payload_chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": c_final}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload_chunk)}\n\n"

                _final_text_no_tools, tool_calls = extract_tool_calls_from_text(buffer)
                if tool_calls:
                    delta_tc = {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": tc["type"],
                                "function": tc["function"],
                            }
                            for i, tc in enumerate(tool_calls)
                        ]
                    }
                    payload_chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta_tc, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload_chunk)}\n\n"
                    finish_reason = "tool_calls"
                else:
                    finish_reason = "stop"

                done = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                gc.collect()
        event_writer.emit(
            "TurnEvent",
            {
                "model": model_id,
                "messages": normalized_messages,
                "assistant": {"role": "assistant", "content": "<stream>"},
                "stream": True,
            },
        )
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    with _MLX_LOCK:
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
    content_raw = gen_result.text
    content_no_tools, tool_calls = extract_tool_calls_from_text(content_raw)
    content, reasoning_content = extract_reasoning_from_text(content_no_tools)
    finish_reason = "tool_calls" if tool_calls else "stop"
    event_writer.emit(
        "TurnEvent",
        {
            "model": model_id,
            "messages": normalized_messages,
            "assistant": {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning_content or None,
                "tool_calls": tool_calls or None,
            },
            "stream": bool(payload.stream),
        },
    )
    return JSONResponse(
        _make_response(
            model_id,
            content,
            finish_reason=finish_reason,
            tool_calls=tool_calls or None,
            reasoning_content=reasoning_content or None,
        )
    )


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
    normalized_messages, video_specs = normalize_video_elements(messages)
    images, audio = extract_media(normalized_messages)

    try:
        from mlx_vlm.generate import generate, stream_generate
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

    chat_template_kwargs = {}
    if payload.tools:
        chat_template_kwargs["tools"] = payload.tools
    if payload.tool_choice is not None:
        chat_template_kwargs["tool_choice"] = payload.tool_choice

    if video_specs:
        try:
            import mlx.core as mx
            from mlx_vlm.video_generate import process_vision_info
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=f"mlx_vlm video dependencies missing: {exc}") from exc

        for spec in video_specs:
            if not spec.is_file_like:
                continue
            try:
                spec.ele["video"] = materialize_video_path(str(spec.ele.get("video") or ""))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid video input: {exc}") from exc

        try:
            prompt = processor.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except TypeError as exc:
            if payload.tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model chat_template does not support tools: {exc}",
                ) from exc
            raise HTTPException(status_code=500, detail=f"Processor chat_template failure: {exc}") from exc

        image_inputs, video_inputs, _video_kwargs = process_vision_info(normalized_messages, True)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        with _MLX_LOCK:
            kwargs["input_ids"] = mx.array(inputs["input_ids"])
            pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
            if pixel_values is None:
                raise HTTPException(status_code=400, detail="Video input produced no pixel values.")
            kwargs["pixel_values"] = mx.array(pixel_values)
            kwargs["mask"] = mx.array(inputs["attention_mask"])
            if inputs.get("video_grid_thw") is not None:
                kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
            if inputs.get("image_grid_thw") is not None:
                kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])
        # Media already packed into pixel_values; avoid redundant loader paths.
        images = []
        audio = []
    else:
        try:
            prompt = processor.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except TypeError as exc:
            if payload.tools:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model chat_template does not support tools: {exc}",
                ) from exc
            raise HTTPException(status_code=500, detail=f"Processor chat_template failure: {exc}") from exc
    _dump_templatized_prompt(prompt=prompt, model_id=model_id, endpoint="responses")
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
                with _MLX_LOCK:
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
                cleaned, tool_calls = extract_tool_calls_from_text(full_text)
                yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'text': cleaned})}\n\n"
                final_content_part = {"type": "output_text", "text": cleaned, "annotations": []}
                yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': final_content_part})}\n\n"
                final_message_item = {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [final_content_part],
                }
                yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': final_message_item})}\n\n"

                tool_items: List[Dict[str, Any]] = []
                required_action: Optional[Dict[str, Any]] = None
                status = "completed"
                if tool_calls:
                    status = "requires_action"
                    required_action = _tool_calls_to_responses_required_action(tool_calls)
                    tool_items = _tool_calls_to_responses_output_items(tool_calls)
                    for idx, item in enumerate(tool_items, start=1):
                        yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': idx, 'item': item})}\n\n"
                        yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': idx, 'item': item})}\n\n"

                completed = dict(base_response)
                completed.update(
                    {
                        "status": status,
                        "required_action": required_action,
                        "output": [final_message_item, *tool_items],
                        "output_text": cleaned,
                        "usage": usage_stats,
                    }
                )
                yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': completed})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                gc.collect()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    with _MLX_LOCK:
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
    content, tool_calls = extract_tool_calls_from_text(gen_result.text)
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    tool_items = _tool_calls_to_responses_output_items(tool_calls) if tool_calls else []
    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": ("requires_action" if tool_calls else "completed"),
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
        ]
        + tool_items,
        "output_text": content,
        "temperature": temperature,
        "top_p": top_p,
        "required_action": (_tool_calls_to_responses_required_action(tool_calls) if tool_calls else None),
        "usage": {
            "input_tokens": getattr(gen_result, "prompt_tokens", 0),
            "output_tokens": getattr(gen_result, "generation_tokens", 0),
            "total_tokens": getattr(gen_result, "total_tokens", 0),
        },
    }
    return JSONResponse(response)
