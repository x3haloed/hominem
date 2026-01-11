from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Union


from qwen_agent.tools.base import BaseTool


class DescribeFile(BaseTool):
    """
    A small, dependency-free file outline tool.

    This is intentionally minimal: it's enough to make the UI→agent→infer loop testable.
    """

    name = "describe_file"
    description = "Describe a local file (outline for .py, otherwise head preview). Accepts absolute paths."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to a file."},
            "max_lines": {"type": "integer", "description": "Max preview lines for non-.py files."},
        },
        "required": ["path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            raise ValueError("params must be an object")
        path = str(params.get("path") or "")
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

        max_lines = int(params.get("max_lines") or 120)
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
