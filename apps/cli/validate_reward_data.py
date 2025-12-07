from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

from core.data.schema import REWARD_DIMENSIONS, RewardVector


def validate_file(path: Path, sample_count: int) -> None:
    records = []
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)

            reward_data = record.get("reward") or {}
            # This will raise if anything is out of range or missing.
            RewardVector.from_mapping(reward_data)
            records.append(record)

    print(f"Validated {total} labeled records in {path}")

    if not records:
        return

    sample_count = min(sample_count, len(records))
    print(f"\nRandom sample of {sample_count} records:")
    for record in random.sample(records, sample_count):
        reward = record.get("reward") or {}
        scalar = reward.get("scalar")
        print("-" * 40)
        print(f"ID: {record.get('id')}")
        print(f"Category: {record.get('category')}")
        print(f"Scalar: {scalar}")
        for dim in REWARD_DIMENSIONS:
            print(f"  {dim}: {reward.get(dim)}")
        print(f"Rationale: {record.get('rationale', '')[:300]}...")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate labeled reward samples and print a random sample."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/labeled/reward_samples.jsonl"),
        help="Input JSONL file of labeled reward samples.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of random samples to display.",
    )

    args = parser.parse_args(argv)
    validate_file(args.input, args.samples)


if __name__ == "__main__":  # pragma: no cover
    main()


