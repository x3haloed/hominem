from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.data.db import TrainingDatabase
from core.data.schema import REWARD_DIMENSIONS


@dataclass
class RewardSample:
    """Single labeled reward sample from reward_samples.jsonl."""

    id: str | None
    prompt_id: str | None
    category: str | None
    prompt: str
    response: str
    reward: Dict[str, float]


def load_reward_samples(path: Path) -> List[RewardSample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Reward samples file not found at '{path}'. "
            "Expected JSONL with fields: id, prompt_id, category, prompt, response, reward."
        )

    samples: List[RewardSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            reward = obj.get("reward") or {}
            if not isinstance(reward, dict) or not reward:
                # Skip malformed entries.
                continue

            samples.append(
                RewardSample(
                    id=obj.get("id"),
                    prompt_id=obj.get("prompt_id"),
                    category=obj.get("category"),
                    prompt=obj.get("prompt", ""),
                    response=obj.get("response", ""),
                    reward={k: float(v) for k, v in reward.items()},
                )
            )

    if not samples:
        raise ValueError(f"No usable reward samples found in '{path}'.")

    return samples


def scalar_score(reward: Dict[str, float]) -> float:
    """Compute a simple scalar score from a reward vector.

    This aggregates only the core manifold dimensions defined in
    REWARD_DIMENSIONS, ignoring scalar aggregates and cross-cutting
    fields such as reward_intensity and safety_score.
    """
    if not reward:
        return 0.0

    values = [float(reward.get(dim, 0.0)) for dim in REWARD_DIMENSIONS]
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def group_by_prompt(samples: Iterable[RewardSample]) -> Dict[str, List[Tuple[RewardSample, float]]]:
    """Group samples by a stable prompt identifier and attach scalar scores."""
    groups: Dict[str, List[Tuple[RewardSample, float]]] = defaultdict(list)
    for s in samples:
        key = s.prompt_id or s.prompt
        if not key:
            # Skip samples with no usable grouping key.
            continue
        score = scalar_score(s.reward)
        groups[key].append((s, score))
    return groups


def generate_pairs_for_group(
    items: List[Tuple[RewardSample, float]],
    *,
    min_margin: float,
    max_pairs_per_prompt: int,
) -> List[Dict[str, Any]]:
    """Generate (prompt, chosen, rejected) pairs from a group of (sample, score).

    Strategy:
      - Sort by score descending.
      - For each high-scoring sample, pair it against lower-scoring ones if
        the score gap is at least `min_margin`.
      - Limit the number of pairs per prompt to avoid combinatorial blowup.
    """
    if len(items) < 2:
        return []

    # Highest score first.
    items_sorted = sorted(items, key=lambda t: t[1], reverse=True)
    pairs: List[Dict[str, Any]] = []

    for i, (chosen_sample, chosen_score) in enumerate(items_sorted):
        if len(pairs) >= max_pairs_per_prompt:
            break
        for j in range(len(items_sorted) - 1, i, -1):
            rejected_sample, rejected_score = items_sorted[j]
            margin = chosen_score - rejected_score
            if margin < min_margin:
                continue
            pairs.append(
                {
                    "prompt": chosen_sample.prompt,
                    "chosen": chosen_sample.response,
                    "rejected": rejected_sample.response,
                    # Optional metadata for your own inspection.
                    "chosen_id": chosen_sample.id,
                    "rejected_id": rejected_sample.id,
                    "prompt_id": chosen_sample.prompt_id,
                    "category": chosen_sample.category,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "score_margin": margin,
                }
            )
            if len(pairs) >= max_pairs_per_prompt:
                break

    return pairs


def make_preferences(
    *,
    input_path: Path,
    output_path: Optional[Path] = None,
    min_margin: float,
    max_pairs_per_prompt: int,
    use_database: bool = True,
    db_path: Optional[str] = None,
) -> None:
    samples = load_reward_samples(input_path)
    groups = group_by_prompt(samples)

    # Initialize database or JSONL output
    db: Optional[TrainingDatabase] = None
    out_f: Optional[Any] = None
    
    if use_database:
        db = TrainingDatabase(db_path=db_path)
    else:
        if output_path is None:
            raise ValueError("output_path is required when use_database=False")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = output_path.open("w", encoding="utf-8")

    total_pairs = 0
    total_prompts_with_pairs = 0

    try:
        for key, items in groups.items():
            pairs = generate_pairs_for_group(
                items,
                min_margin=min_margin,
                max_pairs_per_prompt=max_pairs_per_prompt,
            )
            if not pairs:
                continue
            total_prompts_with_pairs += 1
            total_pairs += len(pairs)
            for pair in pairs:
                if db:
                    db.insert_preference_pair(
                        prompt=pair["prompt"],
                        chosen=pair["chosen"],
                        rejected=pair["rejected"],
                        chosen_id=pair.get("chosen_id"),
                        rejected_id=pair.get("rejected_id"),
                        prompt_id=pair.get("prompt_id"),
                        category=pair.get("category"),
                        chosen_score=pair.get("chosen_score"),
                        rejected_score=pair.get("rejected_score"),
                        score_margin=pair.get("score_margin"),
                    )
                else:
                    out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    finally:
        if db:
            db.close()
        elif out_f:
            out_f.close()

    if total_pairs == 0:
        raise ValueError(
            "No preference pairs were generated. "
            "Consider lowering --min-margin or ensuring reward vectors have more spread."
        )

    print(
        f"Generated {total_pairs} preference pairs from {len(groups)} prompt groups "
        f"({total_prompts_with_pairs} groups contributed at least one pair)."
    )
    if output_path:
        print(f"Written to: {output_path}")
    elif db:
        print(f"Written to database: {db.db_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create preference pairs JSONL from labeled reward samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/labeled/reward_samples.jsonl"),
        help="Input JSONL of labeled rewards (output of label_with_teacher).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL of preferences (only used if --no-db).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Write to JSONL file instead of database.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (uses default if not specified).",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.2,
        help="Minimum score difference (chosen - rejected) required to form a pair.",
    )
    parser.add_argument(
        "--max-pairs-per-prompt",
        type=int,
        default=5,
        help="Maximum number of pairs to generate per prompt group.",
    )
    return parser.parse_args()


def main(argv: List[str] | None = None) -> None:
    args = parse_args()
    if args.no_db and args.output is None:
        args.output = Path("data/preferences/preferences.jsonl")
    
    make_preferences(
        input_path=args.input,
        output_path=args.output,
        min_margin=args.min_margin,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        use_database=not args.no_db,
        db_path=args.db_path,
    )


if __name__ == "__main__":  # pragma: no cover
    main()



