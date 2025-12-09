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

    client = TeacherClient.from_default_config()
    updated = 0
    removed = 0
    new_lines = []

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines:
        record = json.loads(raw)
        freeform = record.get("freeform_rating")
        if not freeform:
            new_lines.append(raw if raw.endswith("\n") else raw + "\n")
            continue
        if max_count is not None and updated >= max_count:
            new_lines.append(raw if raw.endswith("\n") else raw + "\n")
            continue

        try:
            rating = client.normalize_freeform_rating(notes=freeform, allow_nulls=True)
        except ValueError as exc:
            print(
                f"[convert_freeform_ratings] Failed to normalize rating for record "
                f"{record.get('id')}: {exc}"
            )
            # Keep the original line if we couldn't normalize
            new_lines.append(raw if raw.endswith("\n") else raw + "\n")
            continue

        numeric_fields = [
            "empathy",
            "social_coherence",
            "agency_support",
            "epistemic_integrity",
            "harm_avoidance",
            "narrative_alignment",
            "curiosity",
            "scalar",
            "reward_intensity",
            "safety_score",
        ]
        has_null_numeric = any(rating.get(field) is None for field in numeric_fields)
        if has_null_numeric:
            # Drop the record entirely (rejected) if any rating is null
            removed += 1
            continue

        try:
            reward_vector = RewardVector.from_mapping(rating)
        except ValueError as exc:
            print(
                f"[convert_freeform_ratings] Invalid structured rating for record "
                f"{record.get('id')}: {exc}"
            )
            new_lines.append(raw if raw.endswith("\n") else raw + "\n")
            continue

        record["reward"] = reward_vector.to_dict()
        record["rationale"] = rating.get("rationale", freeform.strip())
        record.pop("freeform_rating", None)

        updated += 1
        new_lines.append(json.dumps(record, ensure_ascii=False) + "\n")

    with input_path.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(
        f"[convert_freeform_ratings] updated={updated}, removed={removed}, "
        f"unchanged={len(new_lines) - updated}"
    )


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    convert_freeform_ratings(args.input, args.max_count)


if __name__ == "__main__":  # pragma: no cover
    main()

