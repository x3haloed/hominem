from __future__ import annotations

from typing import Any, Dict, Optional

from hominem_agent.rlm.store import get_store

try:
    from hominem_observability.trace import trace_event
except Exception:  # pragma: no cover
    def trace_event(*_args, **_kwargs):
        return


def rlm_repl(
    *,
    action: str,
    query: Optional[str] = None,
    limit: int = 20,
    text: Optional[str] = None,
    cid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single-tool surface for RLM state inspection/mutation.

    Actions:
    - view
    - search (query required)
    - get_messages (optional query="cursor:<int>")
    - mem_get (query=key)
    - mem_set (query=key, text=value)
    - commit_add (text required)
    - commit_list
    - commit_resolve (cid required, text=resolution)
    """
    act = (action or "").strip().lower()
    store = get_store()

    trace_event(
        "rlm.repl.call",
        {"action": act, "query": query, "limit": limit, "cid": cid, "has_text": bool(text)},
        source="hominem_agent",
    )

    if act == "view":
        return {"action": act, "text": store.view()}
    if act == "search":
        if not query or not str(query).strip():
            raise ValueError("query is required for action=search")
        return {"action": act, "query": query, "results": store.search_canonical(query=str(query), limit=limit)}
    if act == "get_messages":
        cursor = None
        q = (query or "").strip()
        if q.startswith("cursor:"):
            try:
                cursor = int(q.split(":", 1)[1].strip())
            except Exception:
                cursor = None
        recs, next_cursor = store.read_canonical(limit=limit, cursor=cursor)
        return {"action": act, "cursor": cursor, "next_cursor": next_cursor, "messages": recs}
    if act == "mem_get":
        if not query or not str(query).strip():
            raise ValueError("query (key) is required for action=mem_get")
        return {"action": act, "key": str(query), "value": store.mem_get(str(query))}
    if act == "mem_set":
        if not query or not str(query).strip():
            raise ValueError("query (key) is required for action=mem_set")
        if text is None:
            raise ValueError("text (value) is required for action=mem_set")
        store.mem_set(str(query), str(text))
        return {"action": act, "key": str(query), "ok": True}
    if act == "commit_add":
        if not text or not str(text).strip():
            raise ValueError("text is required for action=commit_add")
        com = store.commit_add(str(text))
        return {"action": act, "cid": com.cid, "text": com.text}
    if act == "commit_list":
        return {"action": act, "commitments": store.commit_list(limit=limit)}
    if act == "commit_resolve":
        if not cid or not str(cid).strip():
            raise ValueError("cid is required for action=commit_resolve")
        if text is None:
            raise ValueError("text (resolution) is required for action=commit_resolve")
        ok = store.commit_resolve(str(cid), str(text))
        return {"action": act, "cid": str(cid), "ok": bool(ok)}

    raise ValueError(f"Unknown rlm_repl action: {act}")

