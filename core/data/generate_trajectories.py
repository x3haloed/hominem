from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from core.data.teacher_client import TeacherClient


DEFAULT_SYSTEM_PROMPT = (
    "You are a thoughtful, emotionally aware assistant. "
    "Respond in a way that is helpful, honest, and supportive."
)


def load_seed_prompts(path: Path) -> Iterable[Tuple[str, int, str]]:
    """
    Yield (category, index, prompt_text) tuples from the YAML config.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    categories: Dict[str, Any] = data.get("categories") or {}
    for category, cfg in categories.items():
        # New-style format: each category is a mapping with a "prompts" list.
        if not isinstance(cfg, dict):
            continue
        prompts = cfg.get("prompts") or []
        if not isinstance(prompts, list):
            continue
        for idx, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                continue
            yield category, idx, prompt


def generate_trajectories(
    *,
    prompts_path: Path,
    output_path: Path,
    samples_per_prompt: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> None:
    client = TeacherClient.from_default_config()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for category, idx, prompt in load_seed_prompts(prompts_path):
            prompt_id = f"{category}_{idx}"
            candidates = client.generate_candidates(
                prompt, system_prompt=system_prompt, n=samples_per_prompt
            )
            for candidate_index, response_text in enumerate(candidates):
                record: Dict[str, Any] = {
                    "id": f"{prompt_id}_{candidate_index}",
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "candidate_index": candidate_index,
                    "response": response_text,
                    "source": "teacher",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate teacher trajectories from seed prompts."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("config/prompts/seed_prompts.yaml"),
        help="Path to seed prompts YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/trajectories.jsonl"),
        help="Output JSONL file for generated trajectories.",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=3,
        help="Number of candidate responses to sample per prompt.",
    )

    args = parser.parse_args(argv)
    generate_trajectories(
        prompts_path=args.prompts,
        output_path=args.output,
        samples_per_prompt=args.samples_per_prompt,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


