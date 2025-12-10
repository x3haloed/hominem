"""
Simple admin flag storage for operational toggles (e.g., introspection inclusion).
Persists a small JSON file under config/admin_flags.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_FLAGS = {
    "include_introspection": False,
}


def _flags_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "admin_flags.json"


def load_flags() -> Dict[str, Any]:
    path = _flags_path()
    if not path.exists():
        return DEFAULT_FLAGS.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return DEFAULT_FLAGS.copy()
            merged = DEFAULT_FLAGS.copy()
            merged.update(data)
            return merged
    except Exception:
        return DEFAULT_FLAGS.copy()


def save_flags(flags: Dict[str, Any]) -> None:
    path = _flags_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)


def set_flag(key: str, value: Any) -> None:
    flags = load_flags()
    flags[key] = value
    save_flags(flags)


def get_flag(key: str, default: Optional[Any] = None) -> Any:
    return load_flags().get(key, default)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Admin flags utility.")
    parser.add_argument("--set", dest="set_flag", nargs=2, metavar=("KEY", "VALUE"), help="Set flag to value (true/false for bool).")
    parser.add_argument("--get", dest="get_flag", metavar="KEY", help="Get flag value.")
    args = parser.parse_args()

    if args.set_flag:
        key, raw = args.set_flag
        val: Any
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            val = lowered == "true"
        else:
            val = raw
        set_flag(key, val)
        print(f"{key}={val}")
    elif args.get_flag:
        val = get_flag(args.get_flag, None)
        print(f"{args.get_flag}={val}")
    else:
        print(json.dumps(load_flags(), indent=2))


if __name__ == "__main__":
    main()
