from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def extract_tool_calls_from_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    if not text:
        return "", []

    tool_calls: List[Dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid <tool_call> JSON: {exc}") from exc
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=400, detail="Invalid <tool_call>: missing 'name' string.")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            try:
                arguments_json = json.dumps(arguments, ensure_ascii=False)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid <tool_call> arguments JSON: {exc}") from exc
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": name, "arguments": arguments_json},
            }
        )

    cleaned = _TOOL_CALL_RE.sub("", text).strip()
    return cleaned, tool_calls


_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def extract_reasoning_from_text(text: str) -> Tuple[str, str]:
    """
    Extract reasoning from model output.

    Supported patterns:
    - `<think> ... </think>` blocks (possibly multiple).
    - If `<think>` is missing but a `</think>` appears, treat everything before the first `</think>`
      as reasoning_content and everything after as content.
    """
    if not text:
        return "", ""

    if "<think" in text.lower():
        parts: List[str] = []
        for m in _THINK_BLOCK_RE.finditer(text):
            inner = m.group(1)
            if inner:
                parts.append(inner.strip())
        reasoning = "\n\n".join([p for p in parts if p]).strip()
        cleaned = _THINK_BLOCK_RE.sub("", text).strip()
        return cleaned, reasoning

    close_idx = text.lower().find(_THINK_CLOSE)
    if close_idx != -1:
        reasoning = text[:close_idx].strip()
        cleaned = text[close_idx + len(_THINK_CLOSE) :].strip()
        return cleaned, reasoning

    return text.strip(), ""


@dataclass
class ThinkStreamParser:
    """
    Streaming parser that separates `reasoning_content` from `content`.

    Behavior matches the requested policy:
    - If we ever see a closing `</think>` without having seen `<think>`, treat everything before it as reasoning.
      To do that, we buffer output until we can decide.
    - If `<think>...</think>` blocks are used, we stream deltas normally.
    """

    decided: bool = False
    mode: str = "undecided"  # undecided|implicit_close|explicit
    in_think: bool = False
    buffer: str = ""
    carry: str = ""
    strip_next_content_prefix: bool = False

    def feed(self, delta: str) -> Tuple[str, str]:
        if not delta:
            return "", ""

        s = self.carry + delta
        self.carry = ""

        # Always keep a small carry to handle split tags.
        keep = max(len(_THINK_OPEN), len(_THINK_CLOSE)) - 1
        if len(s) <= keep:
            self.carry = s
            return "", ""

        emit_now = s[:-keep]
        self.carry = s[-keep:]

        if not self.decided:
            self.buffer += emit_now
            lower = self.buffer.lower()
            open_idx = lower.find(_THINK_OPEN)
            close_idx = lower.find(_THINK_CLOSE)
            if open_idx == -1 and close_idx == -1:
                return "", ""
            # Decide based on first occurrence of either tag.
            if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
                self.decided = True
                self.mode = "explicit"
                # Content before <think> is content.
                content_prefix = self.buffer[:open_idx]
                rest = self.buffer[open_idx + len(_THINK_OPEN) :]
                self.buffer = ""
                self.in_think = True
                # Process rest in explicit mode immediately.
                c2, r2 = self._feed_explicit(rest)
                return content_prefix, r2
            # Closing tag first: treat everything before as reasoning, then switch to content mode.
            self.decided = True
            self.mode = "implicit_close"
            reasoning_prefix = self.buffer[:close_idx]
            rest = self.buffer[close_idx + len(_THINK_CLOSE) :]
            self.buffer = ""
            self.in_think = False
            self.strip_next_content_prefix = True
            # After the close, we treat as content (but still allow explicit think blocks later).
            c2, r2 = self._feed_explicit(rest)
            return c2, reasoning_prefix + (("\n" + r2) if r2 else "")

        return self._feed_explicit(emit_now)

    def finish(self) -> Tuple[str, str]:
        # No more input. Decide if needed.
        tail = self.buffer + self.carry
        self.buffer = ""
        self.carry = ""
        if not tail:
            return "", ""
        if not self.decided:
            # No evidence of thinking tags; treat as content.
            return tail, ""
        return self._feed_explicit(tail)

    def _feed_explicit(self, s: str) -> Tuple[str, str]:
        out_c: List[str] = []
        out_r: List[str] = []

        if not self.in_think and self.strip_next_content_prefix:
            stripped = s.lstrip()
            if not stripped:
                # Entire chunk is whitespace after </think>; drop it and keep stripping.
                return "", ""
            s = stripped
            self.strip_next_content_prefix = False

        while s:
            lower = s.lower()
            if not self.in_think:
                idx = lower.find(_THINK_OPEN)
                if idx >= 0:
                    if idx > 0:
                        out_c.append(s[:idx])
                    s = s[idx + len(_THINK_OPEN) :]
                    self.in_think = True
                    continue
                out_c.append(s)
                break
            else:
                idx = lower.find(_THINK_CLOSE)
                if idx >= 0:
                    if idx > 0:
                        out_r.append(s[:idx])
                    s = s[idx + len(_THINK_CLOSE) :]
                    self.in_think = False
                    self.strip_next_content_prefix = True
                    continue
                out_r.append(s)
                break

        return "".join(out_c), "".join(out_r)
