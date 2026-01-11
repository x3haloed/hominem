from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union


from qwen_agent.tools.base import BaseTool


class ReplaceSection(BaseTool):
    name = "replace_section"
    description = "Replace a line range in a local file. Selector must be `lines:<start>-<end>`."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to a file."},
            "selector": {"type": "string", "description": "lines:<start>-<end>"},
            "value": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "selector", "value"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        if isinstance(params, str):
            raise ValueError("params must be an object")
        path = str(params.get("path") or "")
        selector = str(params.get("selector") or "")
        value = str(params.get("value") or "")
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
