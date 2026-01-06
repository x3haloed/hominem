"""Lightweight JSONL helpers for training datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple


def iter_jsonl_raw(path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Yield (raw_line, parsed_obj) pairs from a JSONL file.

    Invalid JSON lines yield an empty dict so callers can preserve them.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                yield raw, {}
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                yield raw, {}
                continue
            if not isinstance(obj, dict):
                yield raw, {}
                continue
            yield raw, obj


def iter_jsonl_records(path: Path, *, skip_invalid: bool = True) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    for _, obj in iter_jsonl_raw(path):
        if not obj and skip_invalid:
            continue
        yield obj


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL records from a file or directory of shards."""
    records: List[Dict[str, Any]] = []
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for fp in files:
        for obj in iter_jsonl_records(fp, skip_invalid=True):
            records.append(obj)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], *, ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii))
            f.write("\n")


def append_jsonl(path: Path, record: Dict[str, Any], *, ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=ensure_ascii))
        f.write("\n")


def validate_required_keys(
    records: Iterable[Dict[str, Any]],
    required: Sequence[str],
) -> List[str]:
    missing: List[str] = []
    for rec in records:
        missing.extend([k for k in required if k not in rec])
    return sorted(set(missing))
