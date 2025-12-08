#!/usr/bin/env python3
"""
Strip prompt-engineering preambles that precede the real assistant reply.

Some trajectory rows contain long explanations such as:

    User prompt: ...
    Original assistant response: ...
    Rewritten assistant response:
    <actual answer>

This script removes everything that appears before the last occurrence of one
of the sentinel strings provided below (e.g. ``"response:\\n\\n"``). The text
immediately following that sentinel is kept and trimmed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


SENTINELS = ("response:\n\n", "response:\n", "response: ")


@dataclass
class StripResult:
    text: str
    changed: bool


def strip_response_preamble(text: str) -> StripResult:
    lowered = text.lower()
    best_match: Tuple[int, int] | None = None  # (index, sentinel_len)

    for sentinel in SENTINELS:
        idx = lowered.rfind(sentinel)
        if idx == -1:
            continue
        if (
            best_match is None
            or idx > best_match[0]
            or (idx == best_match[0] and len(sentinel) > best_match[1])
        ):
            best_match = (idx, len(sentinel))

    if best_match is None:
        return StripResult(text=text, changed=False)

    idx, sentinel_len = best_match
    cleaned = text[idx + sentinel_len :].lstrip("\n ")
    if not cleaned:
        return StripResult(text=text, changed=False)

    return StripResult(text=cleaned, changed=True)


def load_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        yield from handle


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)


def process_file(input_path: Path, output_path: Path | None, dry_run: bool) -> None:
    total = 0
    mutated = 0
    updated_lines = []

    for raw_line in load_lines(input_path):
        total += 1
        record = json.loads(raw_line)
        response = record.get("response")

        if isinstance(response, str):
            result = strip_response_preamble(response)
            if result.changed:
                record["response"] = result.text
                mutated += 1
        updated_lines.append(json.dumps(record, ensure_ascii=False) + "\n")

    if dry_run:
        print(f"[dry-run] {mutated} / {total} rows would be updated.")
        return

    destination = output_path or input_path
    if destination == input_path:
        tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        write_lines(tmp_path, updated_lines)
        tmp_path.replace(input_path)
    else:
        write_lines(destination, updated_lines)

    print(f"Wrote cleaned trajectories to {destination} ({mutated} rows updated).")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip response preambles from a trajectories JSONL file."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to trajectories JSONL file (e.g., data/raw/trajectories.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to in-place overwrite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any files; just report how many rows would change.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    process_file(args.input_path, args.output, args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])

