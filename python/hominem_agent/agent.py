from __future__ import annotations

import json
import os
from types import MethodType
from typing import Any, Dict, List, Optional


def _safe_preview(obj: Any, *, max_len: int = 4000) -> Any:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) <= max_len:
        return obj
    return s[:max_len] + "…<truncated>"


def _instrument_qwen_agent_llm(llm: Any) -> None:
    """
    Add high-signal trace events around the Qwen-Agent -> OpenAI client hop.

    This is the only place we can reliably see what qwen-agent *thinks* it is
    sending to hominem_infer before the HTTP request happens.
    """
    try:
        from hominem_observability.trace import trace_event
    except Exception:
        return

    if getattr(llm, "_hominem_instrumented", False):
        return
    setattr(llm, "_hominem_instrumented", True)

    if hasattr(llm, "raw_chat"):
        orig_raw_chat = llm.raw_chat

        def raw_chat_wrapped(self, messages, functions=None, stream=True, generate_cfg=None):
            try:
                msg_dump = []
                for m in messages or []:
                    if hasattr(m, "model_dump"):
                        msg_dump.append(m.model_dump())
                    else:
                        msg_dump.append(m)
            except Exception:
                msg_dump = "<unavailable>"

            trace_event(
                "agent.llm.raw_chat.request",
                {
                    "llm_type": type(self).__name__,
                    "messages_len": (len(messages) if isinstance(messages, list) else None),
                    "messages": _safe_preview(msg_dump),
                    "functions_len": (len(functions) if isinstance(functions, list) else None),
                    "generate_cfg_keys": (sorted(list((generate_cfg or {}).keys())) if isinstance(generate_cfg, dict) else None),
                },
                source="hominem_agent",
            )
            return orig_raw_chat(messages=messages, functions=functions, stream=stream, generate_cfg=generate_cfg)

        llm.raw_chat = MethodType(raw_chat_wrapped, llm)

    if hasattr(llm, "_chat_complete_create"):
        orig = llm._chat_complete_create

        def chat_create_wrapped(*args, **kwargs):
            trace_event(
                "agent.llm.oai.request",
                {
                    "keys": sorted(list(kwargs.keys())),
                    "messages_len": (len(kwargs.get("messages")) if isinstance(kwargs.get("messages"), list) else None),
                    "messages": _safe_preview(kwargs.get("messages")),
                    "tools_len": (len(kwargs.get("tools")) if isinstance(kwargs.get("tools"), list) else None),
                    "tool_choice": kwargs.get("tool_choice"),
                    "stream": kwargs.get("stream"),
                },
                source="hominem_agent",
            )
            return orig(*args, **kwargs)

        llm._chat_complete_create = chat_create_wrapped


def build_agent(*, tools: Optional[List[Any]] = None):
    """
    Build a Qwen-Agent Assistant wired to talk to `hominem_infer`.

    This module intentionally owns orchestration + tool execution. `hominem_infer`
    remains inference-only.
    """
    try:
        from qwen_agent.agents import Assistant
    except ImportError as exc:
        raise RuntimeError(
            "qwen-agent is not installed. Install with `pip install -e .[agent]`."
        ) from exc

    infer_base_url = os.getenv("HOMINEM_INFER_BASE_URL", "http://127.0.0.1:8000/v1").strip()
    model = os.getenv("HOMINEM_AGENT_MODEL", os.getenv("INFER_MODEL_ID", "")).strip()
    if not model:
        model = "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx"

    llm_cfg: Dict[str, Any] = {
        "model": model,
        "model_server": infer_base_url,
        "api_key": os.getenv("HOMINEM_INFER_API_KEY", "EMPTY").strip() or "EMPTY",
        # Use the OpenAI-compatible (multimodal) client wrapper.
        "model_type": os.getenv("HOMINEM_AGENT_MODEL_TYPE", "qwenvl_oai").strip() or "qwenvl_oai",
        "generate_cfg": {
            # Critical: send native OpenAI `tools=` and expect native `tool_calls` back.
            "use_raw_api": True,
        },
    }

    function_list = list(tools or [])
    assistant = Assistant(function_list=function_list, llm=llm_cfg)
    try:
        _instrument_qwen_agent_llm(assistant.llm)
    except Exception:
        # Never break agent creation due to debugging hooks.
        pass
    return assistant


def default_tools() -> List[Any]:
    """Return Qwen-Agent compatible tool wrappers."""
    try:
        from hominem_agent.tools.wrapper_tools import get_wrapped_tools
    except Exception as exc:
        raise RuntimeError(f"Failed to import default tools: {exc}") from exc

    return get_wrapped_tools()
