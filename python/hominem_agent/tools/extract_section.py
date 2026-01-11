from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Union


from qwen_agent.tools.base import BaseTool


class ExtractSection(BaseTool):
    name = "extract_section"
    description = (
        "Extract a section from a local file. For .py supports selector "
        "`function:<name>` or `class:<name>`. Otherwise supports `lines:<start>-<end>`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to a file."},
            "selector": {"type": "string", "description": "Selector string."},
        },
        "required": ["path", "selector"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            raise ValueError("params must be an object")
        path = str(params.get("path") or "")
        selector = str(params.get("selector") or "")
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
