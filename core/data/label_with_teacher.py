from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from core.data.schema import RewardVector
from core.data.teacher_client import TeacherClient


RATING_SYSTEM_PROMPT = (
    "You are a careful, honest, and precise evaluator of assistant responses. "
    "You judge how well a response serves the user along several human-centric dimensions. "
    "You must follow the requested JSON output format exactly."
)


def label_trajectories(
    *,
    input_path: Path,
    output_path: Path,
) -> None:
    client = TeacherClient.from_default_config()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            prompt = record.get("prompt", "")
            response_text = record.get("response", "")

            rating = client.rate_response(
                prompt=prompt,
                response_text=response_text,
                rating_instructions=RATING_SYSTEM_PROMPT,
            )

            # Validate and normalize via RewardVector.
            reward_vector = RewardVector.from_mapping(rating)

            labeled: Dict[str, Any] = {
                "id": record.get("id"),
                "prompt_id": record.get("prompt_id"),
                "category": record.get("category"),
                "prompt": prompt,
                "response": response_text,
                "reward": reward_vector.to_dict(),
                "rationale": rating.get("rationale", ""),
            }
            out_f.write(json.dumps(labeled, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Use a teacher model to label trajectories with reward vectors."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/trajectories.jsonl"),
        help="Input JSONL file of trajectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labeled/reward_samples.jsonl"),
        help="Output JSONL file for labeled reward samples.",
    )

    args = parser.parse_args(argv)
    label_trajectories(input_path=args.input, output_path=args.output)


if __name__ == "__main__":  # pragma: no cover
    main()


