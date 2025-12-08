from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import yaml

from core.data.teacher_client import InferenceConfig, TeacherClient, load_inference_config


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


def _sanitize_model_id(model_id: str) -> str:
    """
    Turn an arbitrary model identifier into a filesystem- and JSONL-friendly token.

    This is used only inside the synthetic `id` field so that trajectories from
    different generator models can coexist without clashing with the original
    teacher-only IDs.
    """
    # Replace any character that is not alphanumeric, dot, underscore, or dash.
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id)
    # Collapse consecutive underscores for readability.
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "model"


def _load_existing_ids(path: Path) -> Set[str]:
    """
    Scan an existing trajectories file and collect all `id` values.

    This allows the generator to be safely re-run in a resumable, append-only way
    across both the original teacher trajectories and any multi-model supplements.
    """
    existing_ids: Set[str] = set()
    if not path.exists():
        return existing_ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Leave any truncated/invalid lines in place; downstream consumers
                # are hardened to skip them as well.
                continue
            _id = obj.get("id")
            if isinstance(_id, str):
                existing_ids.add(_id)
    return existing_ids


def _build_generator_clients(
    *,
    base_config: InferenceConfig,
    generator_models: Sequence[str] | None,
) -> Mapping[str, TeacherClient]:
    """
    Construct one TeacherClient per generator model.

    If `generator_models` is empty/None, a single client is returned using the
    model_id from the base config, preserving the original behavior.
    """
    if not generator_models:
        # Original single-teacher behavior.
        return {
            "teacher": TeacherClient(base_config),
        }

    clients: Dict[str, TeacherClient] = {}
    for model_id in generator_models:
        model_id = model_id.strip()
        if not model_id:
            continue
        cfg = InferenceConfig(
            endpoint_url=base_config.endpoint_url,
            api_key=base_config.api_key,
            model_id=model_id,
        )
        alias = _sanitize_model_id(model_id)
        clients[alias] = TeacherClient(cfg)
    return clients


def generate_trajectories(
    *,
    prompts_path: Path,
    output_path: Path,
    samples_per_prompt: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    generator_models: Sequence[str] | None = None,
) -> None:
    """
    Generate trajectories from seed prompts.

    By default, this reproduces the original behavior of sampling from a single
    teacher model configured in `inference.toml`.

    If `generator_models` is provided (a list of model identifiers), the same
    prompts are instead fanned out across multiple generator models. All of the
    resulting trajectories are appended to `output_path`, with:

    - Unique, model-aware `id` values (so they coexist with existing data).
    - `source` set to `"teacher"` for the original single-model mode, or
      `"generator_model"` plus a `generator_model_id` field in multi-model mode.

    The function is **resumable** and **idempotent** across both modes:

    - Existing examples are detected by their `id` field and **skipped**.
    - New examples are **appended** to the JSONL file.
    """
    base_config = load_inference_config()
    clients = _build_generator_clients(
        base_config=base_config,
        generator_models=generator_models,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _load_existing_ids(output_path)

    # Append new data; do not clobber existing trajectories.
    with output_path.open("a", encoding="utf-8") as out_f:
        for category, idx, prompt in load_seed_prompts(prompts_path):
            prompt_id = f"{category}_{idx}"

            if not generator_models:
                # Original single-teacher behavior: preserve the exact ID scheme so
                # we do not regenerate any previously-created examples.
                client = clients["teacher"]
                for candidate_index in range(samples_per_prompt):
                    sample_id = f"{prompt_id}_{candidate_index}"
                    if sample_id in existing_ids:
                        # Already generated in a previous run; skip to avoid
                        # re-spending on the same (prompt, candidate_index).
                        continue

                    candidates = client.generate_candidates(
                        prompt,
                        system_prompt=system_prompt,
                        n=1,
                    )
                    if not candidates:
                        # Be robust to empty responses from the teacher.
                        continue
                    response_text = candidates[0]

                    record: Dict[str, Any] = {
                        "id": sample_id,
                        "prompt_id": prompt_id,
                        "category": category,
                        "prompt": prompt,
                        "candidate_index": candidate_index,
                        "response": response_text,
                        "source": "teacher",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # Multi-model mode: fan out the same prompts across a variety of
                # generator models to deliberately capture both strong and weak
                # behavior for the reward model.
                for model_alias, client in clients.items():
                    for candidate_index in range(samples_per_prompt):
                        sample_id = f"{prompt_id}_{model_alias}_{candidate_index}"
                        if sample_id in existing_ids:
                            continue

                        candidates = client.generate_candidates(
                            prompt,
                            system_prompt=system_prompt,
                            n=1,
                        )
                        if not candidates:
                            continue
                        response_text = candidates[0]

                        record = {
                            "id": sample_id,
                            "prompt_id": prompt_id,
                            "category": category,
                            "prompt": prompt,
                            "candidate_index": candidate_index,
                            "response": response_text,
                            "source": "generator_model",
                            "generator_model_id": client._config.model_id,  # type: ignore[attr-defined]
                            "generator_model_alias": model_alias,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate trajectories from seed prompts using either a single teacher "
            "model (default) or multiple generator models for increased variety."
        )
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
        help="Number of candidate responses to sample per prompt per generator model.",
    )
    parser.add_argument(
        "--generator-model",
        dest="generator_models",
        action="append",
        default=None,
        help=(
            "Optional. Model identifier to use as a generator. "
            "May be passed multiple times to sample from a variety of models. "
            "If omitted, the single-model teacher behavior is used."
        ),
    )

    args = parser.parse_args(argv)
    generate_trajectories(
        prompts_path=args.prompts,
        output_path=args.output,
        samples_per_prompt=args.samples_per_prompt,
        generator_models=args.generator_models,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

