"""
YaCy search/status tools.

These tools are intentionally narrow: they only talk to a YaCy instance
and return structured search results/health information.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_BASE_URL = "http://127.0.0.1:8090"
_DEFAULT_TIMEOUT_S = 10
_MAX_LIMIT = 20
_MAX_OFFSET = 100


def _get_base_url() -> str:
    base_url = os.getenv("HOMINEM_YACY_BASE_URL", _DEFAULT_BASE_URL).strip()
    if not base_url:
        base_url = _DEFAULT_BASE_URL
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


def _allow_nonlocal() -> bool:
    return os.getenv("HOMINEM_YACY_ALLOW_NONLOCAL", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _assert_localhost(base_url: str) -> None:
    if _allow_nonlocal():
        return
    parsed = urllib.parse.urlparse(base_url)
    host = (parsed.hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError(
        "Non-local YaCy base URL is blocked. "
        "Set HOMINEM_YACY_ALLOW_NONLOCAL=true to allow."
    )


def _timeout_s() -> float:
    raw = os.getenv("HOMINEM_YACY_TIMEOUT_S", "").strip()
    if not raw:
        return float(_DEFAULT_TIMEOUT_S)
    try:
        return float(raw)
    except ValueError:
        return float(_DEFAULT_TIMEOUT_S)


def _http_get_json(url: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}" if query else url
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": "hominem-agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=_timeout_s()) as resp:
        status = int(resp.status)
        raw = resp.read()
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("Unexpected JSON response format.")
    return data, status


def _http_get_text(url: str) -> Tuple[str, int]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "hominem-agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=_timeout_s()) as resp:
        status = int(resp.status)
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    return text, status


def _normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    title = None
    for key in ("title", "title_s", "title_t"):
        if key in item:
            title = item.get(key)
            break
    if not title and "title" in item:
        title = item.get("title")

    url = None
    for key in ("url", "link", "link_s", "sku", "id"):
        if key in item:
            url = item.get(key)
            break

    snippet = None
    for key in ("snippet", "description", "description_txt", "text_t", "content"):
        if key in item:
            snippet = item.get(key)
            break

    score = item.get("score")
    return {
        "title": str(title) if title is not None else None,
        "url": str(url) if url is not None else None,
        "snippet": str(snippet) if snippet is not None else None,
        "score": score,
        "source": "yacy",
    }


def _parse_solr_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    resp = payload.get("response")
    if not isinstance(resp, dict):
        return []
    docs = resp.get("docs")
    if not isinstance(docs, list):
        return []
    results: List[Dict[str, Any]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        results.append(_normalize_result(doc))
    return results


def _parse_yacysearch_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = None
    for key in ("channels", "items", "results", "links"):
        if key in payload:
            items = payload.get(key)
            break
    if isinstance(items, dict):
        items = items.get("items") or items.get("results")
    if not isinstance(items, list):
        return []
    results: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(_normalize_result(item))
    return results


def search(
    *,
    query: str,
    limit: int = 5,
    offset: int = 0,
    site: Optional[str] = None,
    since: Optional[str] = None,
    format: str = "auto",
    dedupe: bool = True,
    include_raw: bool = False,
) -> Dict[str, Any]:
    if not query or not str(query).strip():
        raise ValueError("query is required")

    base_url = _get_base_url()
    _assert_localhost(base_url)

    limit = int(limit)
    offset = int(offset)
    if limit < 1:
        limit = 1
    if limit > _MAX_LIMIT:
        limit = _MAX_LIMIT
    if offset < 0:
        offset = 0
    if offset > _MAX_OFFSET:
        offset = _MAX_OFFSET

    q = str(query).strip()
    if site:
        q = f"{q} site:{site}"
    if since:
        # Best-effort; if YaCy ignores, harmless.
        q = f"{q} since:{since}"

    endpoints = []
    fmt = (format or "auto").strip().lower()
    if fmt in {"solr", "auto"}:
        endpoints.append(("solr", f"{base_url}/solr/select"))
    if fmt in {"yacysearch_json", "auto"}:
        endpoints.append(("yacysearch_json", f"{base_url}/yacysearch.json"))

    last_error = None
    for name, url in endpoints:
        start = time.time()
        try:
            if name == "solr":
                payload, status = _http_get_json(
                    url,
                    {
                        "q": q,
                        "rows": limit,
                        "start": offset,
                        "wt": "json",
                    },
                )
                results = _parse_solr_results(payload)
            else:
                payload, status = _http_get_json(
                    url,
                    {
                        "query": q,
                        "count": limit,
                        "start": offset,
                        "format": "json",
                    },
                )
                results = _parse_yacysearch_results(payload)

            if dedupe:
                seen = set()
                deduped: List[Dict[str, Any]] = []
                for item in results:
                    key = (item.get("url") or "").strip()
                    if not key:
                        deduped.append(item)
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(item)
                results = deduped

            elapsed_ms = int((time.time() - start) * 1000)
            out: Dict[str, Any] = {
                "query": q,
                "limit": limit,
                "offset": offset,
                "results": results,
                "engine": {
                    "base_url": base_url,
                    "endpoint": url,
                    "http_status": status,
                    "elapsed_ms": elapsed_ms,
                },
            }
            if include_raw:
                out["raw"] = payload
            return out
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"YaCy search failed: {last_error}")


def search_status() -> Dict[str, Any]:
    base_url = _get_base_url()
    _assert_localhost(base_url)
    start = time.time()
    ok = False
    http_status = None
    error = None
    try:
        _, http_status = _http_get_text(base_url)
        ok = True if http_status and 200 <= http_status < 500 else False
    except Exception as exc:
        error = str(exc)

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "ok": ok,
        "base_url": base_url,
        "http_status": http_status,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }
