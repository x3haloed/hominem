"""
Tabstack-backed fetch tools.

These tools fetch and normalize web content using the Tabstack SDK.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _get_api_key() -> str:
    api_key = os.getenv("TABSTACK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TABSTACK_API_KEY is not set.")
    return api_key


def _build_client():
    try:
        from tabstack import Tabstack
    except ImportError as exc:
        raise RuntimeError(
            "tabstack SDK is not installed. Install with `./.venv/bin/python -m pip install tabstack`."
        ) from exc
    return Tabstack(api_key=_get_api_key())


def fetch_markdown(
    *,
    url: str,
    timeout_s: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    if not url or not str(url).strip():
        raise ValueError("url is required")

    with _build_client() as client:
        if timeout_s is not None or max_retries is not None:
            opts: Dict[str, Any] = {}
            if timeout_s is not None:
                opts["timeout"] = timeout_s
            if max_retries is not None:
                opts["max_retries"] = max_retries
            client = client.with_options(**opts)
        result = client.extract.markdown(url=str(url).strip())

    content = getattr(result, "content", None)
    return {
        "url": str(url).strip(),
        "content": content if content is not None else None,
        "metadata": getattr(result, "metadata", None),
        "title": getattr(result, "title", None),
        "description": getattr(result, "description", None),
    }


def fetch_json(
    *,
    url: str,
    json_schema: Dict[str, Any],
    timeout_s: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    if not url or not str(url).strip():
        raise ValueError("url is required")
    if not isinstance(json_schema, dict):
        raise ValueError("json_schema must be an object")

    with _build_client() as client:
        if timeout_s is not None or max_retries is not None:
            opts: Dict[str, Any] = {}
            if timeout_s is not None:
                opts["timeout"] = timeout_s
            if max_retries is not None:
                opts["max_retries"] = max_retries
            client = client.with_options(**opts)
        result = client.extract.json(url=str(url).strip(), json_schema=json_schema)

    if isinstance(result, dict):
        return {"url": str(url).strip(), "data": result}
    return {"url": str(url).strip(), "data": result}
