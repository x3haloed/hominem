"""Shared prompt serialization helpers for training data."""

from __future__ import annotations

from typing import Dict, Iterable, List


def normalize_messages(
    history: Iterable[Dict[str, str]],
    *,
    drop_system: bool = False,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        if drop_system and role in {"system", "developer"}:
            continue
        content = str(msg.get("content") or "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def messages_to_text(messages: Iterable[Dict[str, str]], *, max_turns: int = 0) -> str:
    items = list(messages)
    if max_turns > 0 and len(items) > max_turns:
        items = items[-max_turns:]
    parts: List[str] = []
    for msg in items:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def prompt_from_event(event: Dict[str, str], *, drop_system: bool = False) -> str:
    history = normalize_messages(event.get("history") or [], drop_system=drop_system)
    user_message = str(event.get("user_message") or "").strip()
    if user_message:
        history.append({"role": "user", "content": user_message})
    return messages_to_text(history)
