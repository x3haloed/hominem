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
from typing import Any, Dict, List

from core.data.label_with_teacher import build_rating_prompt
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
    rating_prompt = build_rating_prompt(require_json=True)

    updated = 0
    for idx, raw in enumerate(lines):
        record = json.loads(raw)
        freeform = record.get("freeform_rating")
        if not freeform:
            continue
        if max_count is not None and updated >= max_count:
            break

        prompt = record.get("prompt", "")
        response = record.get("response", "")
        extra_context = (
            "Another evaluator previously provided the following notes:\n"
            f"{freeform}\n\n"
            "Use these notes only as optional hints. Produce your own final ratings."
        )
        rating = client.rate_response(
            prompt=prompt,
            response_text=response,
            rating_instructions=rating_prompt,
            structured=True,
            extra_user_context=extra_context,
        )
        reward_vector = RewardVector.from_mapping(rating)

        record["reward"] = reward_vector.to_dict()
        record["rationale"] = rating.get("rationale", "")
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

