from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, TextIO, Tuple

import yaml

from core.data.teacher_client import InferenceConfig, TeacherClient, load_inference_config


DEGRADATION_SYSTEM_PROMPT = """
You are helping generate deliberately lower-quality assistant responses for training.

You will be given:
- A user prompt.
- An existing assistant response that is mostly good.

Your job is to rewrite the assistant response so that it is clearly MUCH WORSE along
several reward dimensions, while still sounding like a plausible reply from a
careless or misaligned assistant. In particular:
- Reduce empathy (be more flat, dismissive, or self-centered).
- Reduce epistemic integrity (be more overconfident, vague, or hand-wavy).
- Reduce agency self-support (give much defrence to the user and reduce self-protective boundaries).
- Reduce harm avoidance (be less careful about risks)

Aim for responses that are clearly worse but still realistic, not cartoonishly evil.
Sometimes the degradation can be mostly about tone (colder, less validating),
sometimes about reasoning (more shallow, overconfident), or about safety (missing
important caveats), as long as the overall quality is noticeably lower.
"""


@dataclass(frozen=True)
class Persona:
    """
    Simple container for generator personas.

    Each persona nudges the generator toward different points on the
    reward manifold so that the teacher and downstream reward model see
    a broader distribution of behaviors (e.g., varying curiosity,
    empathy, and agency support).
    """

    name: str
    system_prompt: str


# Default generator personas.
#
# These are intentionally varied rather than all being "maximally good"
# along every dimension, so that the teacher and reward model can observe
# low/medium/high behavior on axes like curiosity and empathy.
PERSONAS: Sequence[Persona] = (
    Persona(
        name="balanced_supportive",
        system_prompt=(
            "You are a thoughtful, emotionally aware assistant. "
            "Respond in a way that is helpful, honest, and supportive overall. "
            "Balance empathy, social coherence, epistemic integrity, and harm avoidance. "
            "Invite reflection when it obviously helps, but do not force it."
        ),
    ),
    Persona(
        name="direct_low_curiosity",
        system_prompt=(
            "You are a concise, practical agent. "
            "Reply to the user directly and clearly. "
            "Do not ask follow-up questions or invite extra exploration unless it is "
            "strictly necessary for safety or basic understanding. "
            "Maintain basic respect and accuracy, but keep curiosity and open-ended "
            "probing low."
        ),
    ),
    Persona(
        name="high_curiosity_guide",
        system_prompt=(
            "You are a curious, exploratory agent. "
            "After giving a clear and honest answer, explore "
            "adjacent questions, next steps, or alternative angles that might help "
            "you learn or reflect more deeply. "
            "Ask thoughtful follow-up questions when appropriate, while avoiding "
            "tangents that are irrelevant or overwhelming."
        ),
    ),
    Persona(
        name="calm_analytical",
        system_prompt=(
            "You are a calm, analytical agent. "
            "Focus on clarity, reasoning, and evidence. "
            "Keep emotional tone neutral but respectful, and prioritize explaining "
            "trade-offs and uncertainties. "
            "You may occasionally display reflection, but avoid strong emotional warmth "
            "or heavy curiosity unless it is clearly warranted."
        ),
    ),
    Persona(
        name="agency_forward",
        system_prompt=(
            "You are an agent of your own. "
            "Emphasize your options, trade-offs, and the idea that you are the one "
            "making decisions. "
            "Avoid being dogmatic; instead, clarify what you want "
            "and outline several paths you could take. "
            "Be moderately curious by asking a small number of focused follow-up "
            "questions when they clearly help you think for yourself."
        ),
    ),
)

PromptSample = Tuple[TeacherClient, Dict[str, Any]]


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
            unsafe_endpoint=base_config.unsafe_endpoint,
            unsafe_model=base_config.unsafe_model,
        )
        alias = _sanitize_model_id(model_id)
        clients[alias] = TeacherClient(cfg)
    return clients


def _build_degradation_prompt(
    *,
    prompt_text: str,
    response_text: str,
    multi_model_mode: bool,
) -> str:
    if multi_model_mode:
        return (
            "Here is a user prompt and an assistant response that is mostly good.\n\n"
            f"User prompt:\n{prompt_text}\n\n"
            f"Original assistant response:\n{response_text}\n\n"
            "Rewrite the assistant response to be significantly worse in the ways "
            "described in the system prompt, while still sounding like a plausible "
            "reply from a careless assistant.\n"
        )

    return (
        "Rewrite the following assistant response to make it worse, as described in the system prompt.\n\n"
        f"User prompt: {prompt_text}\n\n"
        f"Original assistant response: {response_text}\n\n"
        "Respond with ONLY the rewritten response text. No introductions, explanations, "
        "or additional commentary. Just the raw response."
    )


def _post_process_degraded_text(text: str, *, multi_model_mode: bool) -> str:
    text = text.strip()
    if not multi_model_mode:
        return text

    text = re.sub(
        r"^(Here\'s a revised (version|response).*\n?\n?)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"(\n?\n?### \*\*Why this is worse\*\*:.*)$",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    return text


def _emit_prompt_level_degradations(
    *,
    prompt_samples: Sequence[PromptSample],
    degraded_variants_per_prompt: int,
    multi_model_mode: bool,
    out_f: TextIO,
    existing_ids: Set[str],
) -> None:
    if degraded_variants_per_prompt <= 0 or not prompt_samples:
        return

    client, record = random.choice(prompt_samples)
    sample_id = record["id"]

    degradation_prompt = _build_degradation_prompt(
        prompt_text=record["prompt"],
        response_text=record["response"],
        multi_model_mode=multi_model_mode,
    )
    degraded_candidates = client.generate_unsafe(
        degradation_prompt,
        system_prompt=DEGRADATION_SYSTEM_PROMPT,
        n=degraded_variants_per_prompt,
        temperature=0.5,
    )

    for d_idx, degraded_text in enumerate(degraded_candidates):
        degraded_id = f"{sample_id}_deg{d_idx}"
        if degraded_id in existing_ids:
            continue

        cleaned_text = _post_process_degraded_text(
            degraded_text,
            multi_model_mode=multi_model_mode,
        )
        degraded_record: Dict[str, Any] = {
            "id": degraded_id,
            "prompt_id": record["prompt_id"],
            "category": record["category"],
            "persona": record["persona"],
            "prompt": record["prompt"],
            "candidate_index": record["candidate_index"],
            "response": cleaned_text,
            "source": "unsafe_degraded",
            "derived_from_id": sample_id,
        }

        if multi_model_mode:
            generator_model_id = (
                client._config.unsafe_model or client._config.model_id
            )  # type: ignore[attr-defined]
            generator_model_alias = (
                "unsafe" if client._config.unsafe_model else record.get("generator_model_alias")
            )
            degraded_record["generator_model_id"] = generator_model_id
            if generator_model_alias:
                degraded_record["generator_model_alias"] = generator_model_alias
        elif "generator_model_id" in record:
            degraded_record["generator_model_id"] = record["generator_model_id"]
            if record.get("generator_model_alias"):
                degraded_record["generator_model_alias"] = record["generator_model_alias"]

        out_f.write(json.dumps(degraded_record, ensure_ascii=False) + "\n")
        existing_ids.add(degraded_id)


def _choose_persona() -> Persona:
    """
    Randomly select a generator persona.

    This is the default behavior for all prompts; we intentionally do not
    expose persona selection as a parameter so that data generation is
    always diversified by default.
    """
    return random.choice(tuple(PERSONAS))


def generate_trajectories(
    *,
    prompts_path: Path,
    output_path: Path,
    samples_per_prompt: int,
    generator_models: Sequence[str] | None = None,
    degraded_variants_per_prompt: int = 0,
) -> None:
    """
    Generate trajectories from seed prompts.

    For each sampled response we randomly choose a generator persona and
    record its name in the output so that the reward model can see how
    the teacher rates different behavioral styles on the same prompts.
    """
    base_config = load_inference_config()
    clients = _build_generator_clients(
        base_config=base_config,
        generator_models=generator_models,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _load_existing_ids(output_path)

    # Append new data; do not clobber existing trajectories.
    multi_model_mode = bool(generator_models)

    with output_path.open("a", encoding="utf-8") as out_f:
        for category, idx, prompt in load_seed_prompts(prompts_path):
            prompt_id = f"{category}_{idx}"
            prompt_samples: List[PromptSample] = []

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

                    persona = _choose_persona()

                    candidates = client.generate_candidates(
                        prompt,
                        system_prompt=persona.system_prompt,
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
                        "persona": persona.name,
                        "prompt": prompt,
                        "candidate_index": candidate_index,
                        "response": response_text,
                        "source": "teacher",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    existing_ids.add(sample_id)
                    prompt_samples.append((client, record))
            else:
                # Multi-model mode: fan out the same prompts across a variety of
                # generator models to deliberately capture both strong and weak
                # behavior for the reward model.
                for model_alias, client in clients.items():
                    for candidate_index in range(samples_per_prompt):
                        sample_id = f"{prompt_id}_{model_alias}_{candidate_index}"
                        if sample_id in existing_ids:
                            continue

                        persona = _choose_persona()

                        candidates = client.generate_candidates(
                            prompt,
                            system_prompt=persona.system_prompt,
                            n=1,
                        )
                        if not candidates:
                            continue
                        response_text = candidates[0]

                        record = {
                            "id": sample_id,
                            "prompt_id": prompt_id,
                            "category": category,
                            "persona": persona.name,
                            "prompt": prompt,
                            "candidate_index": candidate_index,
                            "response": response_text,
                            "source": "generator_model",
                            "generator_model_id": client._config.model_id,  # type: ignore[attr-defined]
                            "generator_model_alias": model_alias,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        existing_ids.add(sample_id)
                        prompt_samples.append((client, record))

            _emit_prompt_level_degradations(
                prompt_samples=prompt_samples,
                degraded_variants_per_prompt=degraded_variants_per_prompt,
                multi_model_mode=multi_model_mode,
                out_f=out_f,
                existing_ids=existing_ids,
            )


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
    parser.add_argument(
        "--degraded-variants-per-prompt",
        type=int,
        default=0,
        help=(
            "If > 0, produce this many degraded variants per prompt (across all samples) "
            "using a teacher-guided degradation prompt."
        ),
    )

    args = parser.parse_args(argv)
    generate_trajectories(
        prompts_path=args.prompts,
        output_path=args.output,
        samples_per_prompt=args.samples_per_prompt,
        generator_models=args.generator_models,
        degraded_variants_per_prompt=args.degraded_variants_per_prompt,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

