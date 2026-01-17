from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


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
    return Assistant(function_list=function_list, llm=llm_cfg)


def default_tools() -> List[Any]:
    """Return Qwen-Agent compatible tool wrappers."""
    try:
        from hominem_agent.tools.wrapper_tools import get_wrapped_tools
    except Exception as exc:
        raise RuntimeError(f"Failed to import default tools: {exc}") from exc

    return get_wrapped_tools()
