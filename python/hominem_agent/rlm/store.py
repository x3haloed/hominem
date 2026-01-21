from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_LOCK = threading.Lock()


def _workspace_root() -> Path:
    # Repo root is cwd in typical runs.
    return Path.cwd()


def _default_rlm_dir() -> Path:
    return _workspace_root() / "workspace" / "rlm"


def _canonical_path() -> Path:
    raw = os.getenv("HOMINEM_RLM_CANONICAL_PATH", "").strip()
    if raw:
        return Path(raw)
    return _default_rlm_dir() / "canonical.jsonl"


def _state_path() -> Path:
    raw = os.getenv("HOMINEM_RLM_STATE_PATH", "").strip()
    if raw:
        return Path(raw)
    return _default_rlm_dir() / "state.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _now_unix() -> float:
    return time.time()


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _stringify_record(obj: Dict[str, Any]) -> str:
    # Try common fields first.
    for key in ("text", "content", "message", "body", "value"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            parts: List[str] = []
            for it in v:
                if isinstance(it, str) and it.strip():
                    parts.append(it.strip())
                elif isinstance(it, dict):
                    t = it.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            if parts:
                return "\n".join(parts).strip()
    return json.dumps(obj, ensure_ascii=False, default=str)


def _truncate(s: str, *, max_chars: int = 240) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


@dataclass
class RlmCommitment:
    cid: str
    text: str
    created_at: float
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[float] = None


class RlmStore:
    def __init__(self, *, canonical_path: Path, state_path: Path) -> None:
        self.canonical_path = canonical_path
        self.state_path = state_path

    @classmethod
    def from_env(cls) -> "RlmStore":
        return cls(canonical_path=_canonical_path(), state_path=_state_path())

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"mem": {}, "commitments": []}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"mem": {}, "commitments": []}
        if not isinstance(data, dict):
            return {"mem": {}, "commitments": []}
        data.setdefault("mem", {})
        data.setdefault("commitments", [])
        if not isinstance(data["mem"], dict):
            data["mem"] = {}
        if not isinstance(data["commitments"], list):
            data["commitments"] = []
        return data

    def _save_state(self, state: Dict[str, Any]) -> None:
        _ensure_parent(self.state_path)
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def read_canonical(self, *, limit: int = 200, cursor: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Read canonical JSONL records.

        cursor: optional line index (0-based) to start from. When omitted, reads the last `limit`.
        Returns (records, next_cursor).
        """
        if not self.canonical_path.exists():
            return [], None
        try:
            lines = self.canonical_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return [], None

        start = 0
        if cursor is not None:
            start = max(0, int(cursor))
        else:
            start = max(0, len(lines) - max(1, int(limit)))

        end = min(len(lines), start + max(1, int(limit)))
        out: List[Dict[str, Any]] = []
        for ln in lines[start:end]:
            obj = _safe_json_loads(ln)
            if obj is None:
                continue
            out.append(obj)
        next_cursor = end if end < len(lines) else None
        return out, next_cursor

    def search_canonical(self, *, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        ql = q.lower()
        recs, _ = self.read_canonical(limit=5000, cursor=0)
        hits: List[Dict[str, Any]] = []
        for idx, obj in enumerate(recs):
            text = _stringify_record(obj)
            if ql in text.lower():
                hits.append({"type": "canonical", "idx": idx, "snippet": _truncate(text)})
                if len(hits) >= max(1, int(limit)):
                    break
        return hits

    def mem_get(self, key: str) -> Optional[str]:
        with _LOCK:
            state = self._load_state()
            v = state["mem"].get(key)
            return str(v) if v is not None else None

    def mem_set(self, key: str, value: str) -> None:
        with _LOCK:
            state = self._load_state()
            state["mem"][key] = value
            self._save_state(state)

    def commit_add(self, text: str) -> RlmCommitment:
        now = _now_unix()
        cid = f"c_{uuid.uuid4().hex}"
        com = RlmCommitment(cid=cid, text=text.strip(), created_at=now)
        with _LOCK:
            state = self._load_state()
            state["commitments"].append(com.__dict__)
            self._save_state(state)
        return com

    def commit_list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        with _LOCK:
            state = self._load_state()
            items = [x for x in state.get("commitments", []) if isinstance(x, dict)]
        open_items = [x for x in items if not x.get("resolved")]
        return open_items[: max(1, int(limit))]

    def commit_resolve(self, cid: str, resolution: str) -> bool:
        with _LOCK:
            state = self._load_state()
            items = state.get("commitments", [])
            if not isinstance(items, list):
                items = []
            changed = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("cid") != cid:
                    continue
                if item.get("resolved"):
                    return True
                item["resolved"] = True
                item["resolution"] = resolution
                item["resolved_at"] = _now_unix()
                changed = True
                break
            if changed:
                state["commitments"] = items
                self._save_state(state)
            return changed

    def view(self) -> str:
        with _LOCK:
            state = self._load_state()
        mem = state.get("mem", {}) if isinstance(state.get("mem"), dict) else {}
        commits = [c for c in (state.get("commitments") or []) if isinstance(c, dict) and not c.get("resolved")]
        recs, next_cursor = self.read_canonical(limit=30, cursor=None)
        lines: List[str] = []
        lines.append("# RLM View")
        lines.append("")
        lines.append(f"- canonical_records_shown: {len(recs)}")
        lines.append(f"- canonical_next_cursor: {next_cursor}")
        lines.append("")
        lines.append("## Working Memory")
        if mem:
            for k in sorted(mem.keys()):
                lines.append(f"- {k}: {_truncate(str(mem.get(k)), max_chars=300)}")
        else:
            lines.append("- (empty)")
        lines.append("")
        lines.append("## Commitments (open)")
        if commits:
            for c in commits[:25]:
                lines.append(f"- {c.get('cid')}: {_truncate(str(c.get('text') or ''), max_chars=300)}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Canonical (recent)")
        if recs:
            for obj in recs:
                role = obj.get("role") if isinstance(obj.get("role"), str) else obj.get("type")
                snippet = _truncate(_stringify_record(obj), max_chars=400)
                lines.append(f"- {role}: {snippet}")
        else:
            lines.append("- (no canonical records)")
        return "\n".join(lines).strip() + "\n"


_STORE: Optional[RlmStore] = None


def get_store() -> RlmStore:
    global _STORE
    if _STORE is None:
        _STORE = RlmStore.from_env()
    return _STORE
