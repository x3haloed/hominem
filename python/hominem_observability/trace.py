from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


TRACE_SCHEMA_VERSION = 1

_trace_id_var: ContextVar[Optional[str]] = ContextVar("hominem_trace_id", default=None)
_span_id_var: ContextVar[Optional[str]] = ContextVar("hominem_span_id", default=None)

_write_lock = threading.Lock()


def _truthy(x: str) -> bool:
    return x.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_trace_path() -> Path:
    # Prefer workspace so logs stay out of package directories.
    return Path.cwd() / "workspace" / "agent_trace.jsonl"


def trace_enabled() -> bool:
    if _truthy(os.getenv("HOMINEM_TRACE", "")):
        return True
    return bool(os.getenv("HOMINEM_TRACE_LOG", "").strip())


def trace_path() -> Optional[Path]:
    if not trace_enabled():
        return None
    raw = os.getenv("HOMINEM_TRACE_LOG", "").strip()
    return Path(raw) if raw else _default_trace_path()


def get_trace_id() -> Optional[str]:
    return _trace_id_var.get()


def set_trace_id(trace_id: Optional[str]) -> None:
    _trace_id_var.set(trace_id)


def get_span_id() -> Optional[str]:
    return _span_id_var.get()


def set_span_id(span_id: Optional[str]) -> None:
    _span_id_var.set(span_id)


def new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def new_span_id() -> str:
    return f"span_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str


@contextmanager
def span(name: str, *, trace_id: Optional[str] = None) -> Iterator[Span]:
    if not trace_enabled():
        yield Span(trace_id=trace_id or (get_trace_id() or ""), span_id="", parent_span_id=None, name=name)
        return
    tid = trace_id or get_trace_id() or new_trace_id()
    parent = get_span_id()
    sid = new_span_id()
    prev_tid = get_trace_id()
    prev_sid = get_span_id()
    set_trace_id(tid)
    set_span_id(sid)
    try:
        trace_event(
            "span.start",
            {"name": name, "parent_span_id": parent},
            trace_id=tid,
            span_id=sid,
            parent_span_id=parent,
        )
        yield Span(trace_id=tid, span_id=sid, parent_span_id=parent, name=name)
    finally:
        trace_event(
            "span.end",
            {"name": name},
            trace_id=tid,
            span_id=sid,
            parent_span_id=parent,
        )
        set_trace_id(prev_tid)
        set_span_id(prev_sid)


def trace_event(
    event_type: str,
    payload: Dict[str, Any],
    *,
    source: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> None:
    path = trace_path()
    if path is None:
        return

    tid = trace_id or get_trace_id() or new_trace_id()
    sid = span_id or get_span_id()

    record = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "ts_unix": time.time(),
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "source": source,
        "trace_id": tid,
        "span_id": sid,
        "parent_span_id": parent_span_id,
        "payload": payload,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Observability must never break core behavior.
        return

