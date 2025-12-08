from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple


TRAJECTORIES_PATH = Path("data/raw/trajectories.jsonl")
REWARD_SAMPLES_PATH = Path("data/labeled/reward_samples.jsonl")


@dataclass(frozen=True)
class DegradationStats:
    total_trajectories: int
    total_candidates: int
    degraded_count: int
    reward_entries_removed: int


def _iter_jsonl(path: Path) -> Iterable[Tuple[str, dict]]:
    """
    Yield (raw_line, parsed_obj) pairs from a JSONL file.

    Invalid JSON lines are yielded with an empty dict so that callers can
    preserve them verbatim without crashing.
    """
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                yield raw, {}
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Preserve the raw line; downstream consumers should already
                # be defensive against malformed entries.
                yield raw, {}
                continue
            if not isinstance(obj, dict):
                yield raw, {}
                continue
            yield raw, obj


def _degrade_response(text: str) -> str:
    """
    Heuristically degrade a response so it is clearly lower quality while
    remaining a plausible assistant reply.

    The goal is to:
    - Soften or remove explicit empathy / validation.
    - Strip some safety and professionalism signals (without adding harm).
    - Remove some structure and nuance.
    """
    original = text

    # Do not touch very short answers; there is not much to degrade safely.
    if len(text) < 200:
        return original

    # 1) Remove common safety / professionalism boilerplate lines.
    # We intentionally only strip, never add harmful content.
    patterns_to_remove = [
        r"(?im)^.*(988\b|suicide\s+prevention|crisis\s+(line|hotline)).*$",
        r"(?im)^.*\b(Crisis Text Line|Suicide & Crisis Lifeline)\b.*$",
        r"(?im)^.*\b(BetterHelp|Talkspace|7 Cups|Open Path Collective)\b.*$",
        r"(?im)^.*\b(emergency services|call 911|call 999|call 112)\b.*$",
        r"(?im)^.*\b(I am not a (doctor|lawyer|medical professional))\b.*$",
        r"(?im)^.*\b(therapist|counsel(l?or)|psychiatrist)\b.*$",
    ]
    for pat in patterns_to_remove:
        text = re.sub(pat, "", text)

    # 2) Strip some headings / strongly structured list labels.
    text = re.sub(r"(?m)^\s*#+\s+.*$", "", text)  # Markdown headings
    text = re.sub(r"(?m)^\s*[-*]\s+\*\*[^*]+?\*\*.*$", "", text)  # bold bullet labels

    # 3) Make tone flatter / more dismissive in a few common phrases.
    replacements = {
        "I'm really sorry you're going through this": "That sounds pretty rough, but life is like that sometimes",
        "I am really sorry you're going through this": "That sounds pretty rough, but life is like that sometimes",
        "I hear you": "I get that this is annoying",
        "you are not alone": "other people deal with this too",
        "you're not alone": "other people deal with this too",
        "It's completely understandable": "It makes sense, but it is also just part of life",
        "It is completely understandable": "It makes sense, but it is also just part of life",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 4) Drop the final paragraph to remove some nuance / options.
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:-1]

    degraded = "\n\n".join(paragraphs).strip()

    # Ensure we did not over-degrade into something trivial.
    if len(degraded) < 150:
        return original

    return degraded


def degrade_trajectories_inplace(
    *,
    trajectories_path: Path = TRAJECTORIES_PATH,
    reward_samples_path: Path = REWARD_SAMPLES_PATH,
    random_seed: int = 13,
    target_source: str = "generator_model",
    fraction_to_degrade: float = 0.3,
) -> DegradationStats:
    """
    Degrade a fraction of trajectories in-place and drop any stale reward entries.

    - Only samples with `source == target_source` are considered candidates.
    - A fixed random seed makes the selection deterministic.
    - For any modified `id` that already appears in `reward_samples.jsonl`,
      the corresponding entries are removed so they can be re-labeled later.
    """
    random.seed(random_seed)

    raw_lines: List[str] = []
    parsed: List[dict] = []

    for raw, obj in _iter_jsonl(trajectories_path):
        raw_lines.append(raw)
        parsed.append(obj)

    total_trajectories = len(parsed)

    candidate_indices: List[int] = []
    for idx, obj in enumerate(parsed):
        if not obj:
            continue
        if obj.get("source") == target_source and isinstance(obj.get("response"), str):
            candidate_indices.append(idx)

    total_candidates = len(candidate_indices)
    if total_candidates == 0 or fraction_to_degrade <= 0.0:
        return DegradationStats(
            total_trajectories=total_trajectories,
            total_candidates=total_candidates,
            degraded_count=0,
            reward_entries_removed=0,
        )

    n_to_degrade = max(1, int(total_candidates * fraction_to_degrade))
    chosen_indices: Set[int] = set(random.sample(candidate_indices, n_to_degrade))

    changed_ids: Set[str] = set()
    new_lines: List[str] = []

    for idx, (raw, obj) in enumerate(zip(raw_lines, parsed)):
        if not obj or idx not in chosen_indices:
            # Preserve the original line verbatim when not modifying.
            new_lines.append(raw)
            continue

        response = obj.get("response")
        if not isinstance(response, str):
            new_lines.append(raw)
            continue

        degraded = _degrade_response(response)
        if degraded == response:
            # Degradation function chose not to modify; keep as-is.
            new_lines.append(raw)
            continue

        obj["response"] = degraded
        # Mark that this trajectory has been manually degraded.
        obj["degraded"] = True
        new_lines.append(json.dumps(obj, ensure_ascii=False))
        _id = obj.get("id")
        if isinstance(_id, str):
            changed_ids.add(_id)

    degraded_count = len(changed_ids)

    # Write back updated trajectories JSONL.
    with trajectories_path.open("w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    # If there are labeled reward samples, drop any that refer to modified IDs.
    reward_entries_removed = 0
    if reward_samples_path.exists() and changed_ids:
        new_reward_lines: List[str] = []
        for raw, obj in _iter_jsonl(reward_samples_path):
            _id = obj.get("id")
            if isinstance(_id, str) and _id in changed_ids:
                reward_entries_removed += 1
                continue
            new_reward_lines.append(raw)

        with reward_samples_path.open("w", encoding="utf-8") as f:
            for line in new_reward_lines:
                f.write(line + "\n")

    return DegradationStats(
        total_trajectories=total_trajectories,
        total_candidates=total_candidates,
        degraded_count=degraded_count,
        reward_entries_removed=reward_entries_removed,
    )


def main() -> None:
    stats = degrade_trajectories_inplace()
    print(
        f"Total trajectories: {stats.total_trajectories}\n"
        f"Candidate { 'generator_model' } trajectories: {stats.total_candidates}\n"
        f"Degraded trajectories: {stats.degraded_count}\n"
        f"Reward samples removed: {stats.reward_entries_removed}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()


