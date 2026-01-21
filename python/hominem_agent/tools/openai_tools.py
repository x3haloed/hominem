"""
OpenAI-style tool definitions and implementations.

These replace the Qwen-Agent BaseTool classes with pure OpenAI-compatible
function schemas and execution functions.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List

# OpenAI tool schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rlm_repl",
            "description": "RLM state access tool: view/search/messages/working_memory/commitments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action name (view, search, get_messages, mem_get, mem_set, commit_add, commit_list, commit_resolve).",
                    },
                    "query": {"type": "string", "description": "Search query or key/cursor depending on action."},
                    "limit": {"type": "integer", "description": "Limit for search/get_messages/commit_list."},
                    "text": {"type": "string", "description": "Text value for mem_set / commit_add / commit_resolve."},
                    "cid": {"type": "string", "description": "Commitment id for commit_resolve."},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_file",
            "description": "Describe a local file (outline for .py, otherwise head preview). Accepts absolute paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to a file."},
                    "max_lines": {"type": "integer", "description": "Max preview lines for non-.py files."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_section",
            "description": "Extract a section from a local file. For .py supports selector `function:<name>` or `class:<name>`. Otherwise supports `lines:<start>-<end>`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to a file."},
                    "selector": {"type": "string", "description": "Selector string."},
                },
                "required": ["path", "selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_section",
            "description": "Replace a line range in a local file. Selector must be `lines:<start>-<end>`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to a file."},
                    "selector": {"type": "string", "description": "lines:<start>-<end>"},
                    "value": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "selector", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "limit": {"type": "integer", "description": "Max results (1-20)."},
                    "offset": {"type": "integer", "description": "Start offset (0-100)."},
                    "site": {"type": "string", "description": "Optional site filter (e.g. example.com)."},
                    "since": {"type": "string", "description": "Optional since filter (best-effort)."},
                    "format": {
                        "type": "string",
                        "description": "Preferred endpoint: auto, solr, yacysearch_json.",
                    },
                    "dedupe": {"type": "boolean", "description": "Deduplicate results by URL."},
                    "include_raw": {"type": "boolean", "description": "Include raw response payload."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_status",
            "description": "Check reachability of the local search index.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_markdown",
            "description": "Fetch a URL and return cleaned Markdown content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch."},
                    "timeout_s": {"type": "number", "description": "Override request timeout in seconds."},
                    "max_retries": {"type": "integer", "description": "Override retry count for this request."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_json",
            "description": "Fetch a URL and extract structured data using a JSON schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch."},
                    "json_schema": {"type": "object", "description": "JSON schema to extract."},
                    "timeout_s": {"type": "number", "description": "Override request timeout in seconds."},
                    "max_retries": {"type": "integer", "description": "Override retry count for this request."},
                },
                "required": ["url", "json_schema"],
            },
        },
    },
]


def describe_file(path: str, max_lines: int = 120) -> Dict[str, Any]:
    """Describe a local file (outline for .py, otherwise head preview)."""
    if not path:
        raise ValueError("path is required")
    p = Path(path)
    if not p.is_absolute():
        raise ValueError("path must be absolute")
    if not p.exists():
        raise ValueError(f"path does not exist: {p}")
    if p.is_dir():
        entries = sorted([x.name for x in p.iterdir()])
        return {"kind": "dir", "path": str(p), "entries": entries[:200], "total": len(entries)}

    suffix = p.suffix.lower()
    raw = p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".py":
        try:
            mod = ast.parse(raw)
        except SyntaxError as exc:
            return {"kind": "python", "path": str(p), "error": f"syntax_error: {exc}", "preview": raw[:4000]}
        outline: List[Dict[str, Any]] = []
        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                outline.append({"type": "function", "name": node.name, "line": getattr(node, "lineno", None)})
            elif isinstance(node, ast.ClassDef):
                outline.append({"type": "class", "name": node.name, "line": getattr(node, "lineno", None)})
        return {"kind": "python", "path": str(p), "outline": outline, "size_bytes": p.stat().st_size}

    lines = raw.splitlines()
    head = "\n".join(lines[:max_lines])
    return {"kind": "text", "path": str(p), "size_bytes": p.stat().st_size, "preview": head}


def extract_section(path: str, selector: str) -> Dict[str, Any]:
    """Extract a section from a local file."""
    if not path or not selector:
        raise ValueError("path and selector are required")
    p = Path(path)
    if not p.is_absolute():
        raise ValueError("path must be absolute")
    if not p.exists() or p.is_dir():
        raise ValueError(f"file not found: {p}")
    raw = p.read_text(encoding="utf-8", errors="replace")

    if p.suffix.lower() == ".py":
        mod = ast.parse(raw)
        want_type, _, want_name = selector.partition(":")
        want_type = want_type.strip().lower()
        want_name = want_name.strip()
        for node in mod.body:
            if want_type == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == want_name:
                return {"path": str(p), "selector": selector, "text": ast.get_source_segment(raw, node) or ""}
            if want_type == "class" and isinstance(node, ast.ClassDef) and node.name == want_name:
                return {"path": str(p), "selector": selector, "text": ast.get_source_segment(raw, node) or ""}
        return {"path": str(p), "selector": selector, "text": "", "error": "not_found"}

    if selector.lower().startswith("lines:"):
        spec = selector.split(":", 1)[1].strip()
        start_s, _, end_s = spec.partition("-")
        start = int(start_s.strip() or "1")
        end = int(end_s.strip() or str(start))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = raw.splitlines()
        chunk = "\n".join(lines[start - 1:end])
        return {"path": str(p), "selector": selector, "text": chunk}

    raise ValueError("unsupported selector for this file type")


def replace_section(path: str, selector: str, value: str) -> Dict[str, Any]:
    """Replace a line range in a local file."""
    if not path or not selector:
        raise ValueError("path and selector are required")
    p = Path(path)
    if not p.is_absolute():
        raise ValueError("path must be absolute")
    if not p.exists() or p.is_dir():
        raise ValueError(f"file not found: {p}")
    if not selector.lower().startswith("lines:"):
        raise ValueError("unsupported selector; only lines:<start>-<end> is supported")
    spec = selector.split(":", 1)[1].strip()
    start_s, _, end_s = spec.partition("-")
    start = int(start_s.strip() or "1")
    end = int(end_s.strip() or str(start))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    raw = p.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    new_lines = lines[: start - 1] + value.splitlines() + lines[end:]
    p.write_text("\n".join(new_lines) + ("\n" if raw.endswith("\n") else ""), encoding="utf-8")
    return {"ok": True, "path": str(p), "selector": selector}


# Tool execution dispatcher
TOOL_FUNCTIONS = {
    "rlm_repl": None,
    "describe_file": describe_file,
    "extract_section": extract_section,
    "replace_section": replace_section,
    "search": None,
    "search_status": None,
    "fetch_markdown": None,
    "fetch_json": None,
}


_ABS_PATH_SUBSTR_RE = re.compile(
    r"(?P<p>"
    r"(?:/Users/[^ \n\t\"'<>]+)"
    r"|(?:/home/[^ \n\t\"'<>]+)"
    r"|(?:/private/[^ \n\t\"'<>]+)"
    r"|(?:/var/[^ \n\t\"'<>]+)"
    r"|(?:/tmp/[^ \n\t\"'<>]+)"
    r"|(?:[A-Za-z]:\\\\[^ \n\t\"'<>]+)"
    r")"
)


def _redact_path_token(p: str) -> str:
    try:
        name = Path(p).name
    except Exception:
        name = "redacted"
    return f"<path:{name or 'redacted'}>"


def _redact_paths_in_string(s: str) -> str:
    if not s:
        return s
    return _ABS_PATH_SUBSTR_RE.sub(lambda m: _redact_path_token(m.group("p")), s)


def _sanitize_tool_result(obj: Any) -> Any:
    """
    Remove local on-disk file paths from tool outputs before they are exposed to the model.
    Tool *inputs* may still require absolute paths; this only affects returned content.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return _redact_paths_in_string(obj)
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_sanitize_tool_result(x) for x in obj]
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in {"path", "filepath", "file_path", "filename", "canonical_path", "state_path"}:
                if isinstance(v, str):
                    out[k] = _redact_path_token(v) if (v.startswith(("/", "\\")) or (":\\" in v)) else v
                    continue
            out[k] = _sanitize_tool_result(v)
        return out
    return obj


def execute_tool(tool_name: str, **kwargs) -> Any:
    """Execute a tool by name with the given arguments."""
    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name}")
    func = TOOL_FUNCTIONS[tool_name]
    if func is None and tool_name == "rlm_repl":
        from hominem_agent.rlm.tool import rlm_repl

        TOOL_FUNCTIONS["rlm_repl"] = rlm_repl
        func = TOOL_FUNCTIONS[tool_name]
    if func is None and tool_name in {"search", "search_status"}:
        from hominem_agent.tools.yacy import search, search_status

        TOOL_FUNCTIONS["search"] = search
        TOOL_FUNCTIONS["search_status"] = search_status
        func = TOOL_FUNCTIONS[tool_name]
    if func is None and tool_name in {"fetch_markdown", "fetch_json"}:
        from hominem_agent.tools.tabstack_tools import fetch_json, fetch_markdown

        TOOL_FUNCTIONS["fetch_markdown"] = fetch_markdown
        TOOL_FUNCTIONS["fetch_json"] = fetch_json
        func = TOOL_FUNCTIONS[tool_name]
    return _sanitize_tool_result(func(**kwargs))
