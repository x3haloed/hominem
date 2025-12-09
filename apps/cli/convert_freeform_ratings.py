"""Convert free-form reward ratings into structured JSON scores.

This script scans a JSONL dataset (default: data/labeled/reward_samples.jsonl)
for entries produced with --no-json (which record a `freeform_rating` field),
then re-sends them to the teacher model with structured-output enabled so the
dataset regains numeric reward vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from core.data.schema import RewardVector
from core.data.teacher_client import TeacherClient


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert free-form reward annotations into structured JSON ratings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/labeled/reward_samples.jsonl"),
        help="JSONL file to scan for free-form ratings.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Maximum number of entries to convert. Defaults to all available.",
    )
    return parser.parse_args(argv)


def convert_freeform_ratings(input_path: Path, max_count: int | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    client = TeacherClient.from_default_config()
    updated = 0
    for idx, raw in enumerate(lines):
        record = json.loads(raw)
        freeform = record.get("freeform_rating")
        if not freeform:
            continue
        if max_count is not None and updated >= max_count:
            break

        try:
            rating = client.normalize_freeform_rating(notes=freeform)
        except ValueError as exc:
            print(
                f"[convert_freeform_ratings] Failed to normalize rating for record "
                f"{record.get('id')}: {exc}"
            )
            continue

        try:
            reward_vector = RewardVector.from_mapping(rating)
        except ValueError as exc:
            print(
                f"[convert_freeform_ratings] Invalid structured rating for record "
                f"{record.get('id')}: {exc}"
            )
            continue

        record["reward"] = reward_vector.to_dict()
        record["rationale"] = rating.get("rationale", freeform.strip())
        record.pop("freeform_rating", None)

        lines[idx] = json.dumps(record, ensure_ascii=False) + "\n"
        updated += 1

    if updated == 0:
        return

    with input_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    convert_freeform_ratings(args.input, args.max_count)


if __name__ == "__main__":  # pragma: no cover
    main()

