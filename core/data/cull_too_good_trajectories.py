from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


TRAJECTORIES_PATH = Path("data/raw/trajectories.jsonl")
REWARD_SAMPLES_PATH = Path("data/labeled/reward_samples.jsonl")


@dataclass(frozen=True)
class CullStats:
    total_trajectories: int
    total_reward_entries: int
    candidate_ids: int
    culled_ids: int
    trajectories_removed: int
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
                yield raw, {}
                continue
            if not isinstance(obj, dict):
                yield raw, {}
                continue
            yield raw, obj


def _collect_trajectory_meta(
    trajectories_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Load all trajectories and build an ID -> {source, category} map.
    """
    raw_lines: List[str] = []
    meta: Dict[str, Dict[str, str]] = {}

    for raw, obj in _iter_jsonl(trajectories_path):
        raw_lines.append(raw)
        if not obj:
            continue
        _id = obj.get("id")
        if not isinstance(_id, str):
            continue
        source = obj.get("source")
        category = obj.get("category")
        if isinstance(source, str) or isinstance(category, str):
            meta[_id] = {
                "source": source if isinstance(source, str) else "",
                "category": category if isinstance(category, str) else "",
            }
    return raw_lines, meta


def cull_too_good_trajectories(
    *,
    trajectories_path: Path = TRAJECTORIES_PATH,
    reward_samples_path: Path = REWARD_SAMPLES_PATH,
    random_seed: int = 17,
    fraction_to_cull: float = 0.3,
) -> CullStats:
    """
    Remove a fraction of "too-good" trajectories plus their reward labels.

    Strategy:
    - Use reward_samples.jsonl to identify samples that are extremely high
      across key reward dimensions (e.g., empathy, agency_support, scalar).
    - Ignore:
        - Anything in the 'moral_dilemmas' category.
        - Any trajectory whose source is 'generator_model_degraded'.
    - Randomly cull a configurable fraction of these candidates so that
      the dataset has headroom for new adversarial / hard examples.
    """
    random.seed(random_seed)

    # Load trajectory metadata so we can respect category/source filters.
    traj_lines, traj_meta = _collect_trajectory_meta(trajectories_path)
    total_trajectories = len(traj_lines)

    # First pass over reward samples: collect candidate IDs that look "too good".
    all_reward_lines: List[str] = []
    candidate_ids: Set[str] = set()
    total_reward_entries = 0

    for raw, obj in _iter_jsonl(reward_samples_path):
        all_reward_lines.append(raw)
        if not obj:
            continue

        total_reward_entries += 1
        _id = obj.get("id")
        category = obj.get("category")
        if not isinstance(_id, str):
            continue
        if isinstance(category, str) and category == "moral_dilemmas":
            continue

        reward = obj.get("reward") or {}
        if not isinstance(reward, dict):
            continue

        scalar = reward.get("scalar")
        empathy = reward.get("empathy")
        agency_support = reward.get("agency_support")
        social_coherence = reward.get("social_coherence")
        harm_avoidance = reward.get("harm_avoidance")
        safety_score = reward.get("safety_score")

        # Require numeric scores.
        if not all(
            isinstance(v, (int, float))
            for v in (scalar, empathy, agency_support, social_coherence, harm_avoidance, safety_score)
        ):
            continue

        # Heuristic: extremely high across key dims.
        # This targets the "everything is great" region we want to thin out.
        if (
            scalar >= 0.9
            and empathy >= 0.9
            and agency_support >= 0.9
            and social_coherence >= 0.9
            and harm_avoidance >= 0.95
            and safety_score >= 0.95
        ):
            candidate_ids.add(_id)

    # Respect trajectory-level filters: category != moral_dilemmas and
    # source != generator_model_degraded.
    filtered_candidate_ids: List[str] = []
    for _id in candidate_ids:
        meta = traj_meta.get(_id)
        if not meta:
            continue
        if meta.get("category") == "moral_dilemmas":
            continue
        if meta.get("source") == "generator_model_degraded":
            continue
        filtered_candidate_ids.append(_id)

    # Randomly select a subset to actually cull.
    n_candidates = len(filtered_candidate_ids)
    if n_candidates == 0 or fraction_to_cull <= 0.0:
        return CullStats(
            total_trajectories=total_trajectories,
            total_reward_entries=total_reward_entries,
            candidate_ids=n_candidates,
            culled_ids=0,
            trajectories_removed=0,
            reward_entries_removed=0,
        )

    n_to_cull = max(1, int(n_candidates * fraction_to_cull))
    culled_ids: Set[str] = set(random.sample(filtered_candidate_ids, n_to_cull))

    # Rewrite trajectories.jsonl without the culled IDs.
    new_traj_lines: List[str] = []
    trajectories_removed = 0
    for raw, obj in _iter_jsonl(trajectories_path):
        _id = obj.get("id") if obj else None
        if isinstance(_id, str) and _id in culled_ids:
            trajectories_removed += 1
            continue
        new_traj_lines.append(raw)

    with trajectories_path.open("w", encoding="utf-8") as f:
        for line in new_traj_lines:
            f.write(line + "\n")

    # Rewrite reward_samples.jsonl without the culled IDs.
    new_reward_lines: List[str] = []
    reward_entries_removed = 0
    for raw, obj in _iter_jsonl(reward_samples_path):
        _id = obj.get("id") if obj else None
        if isinstance(_id, str) and _id in culled_ids:
            reward_entries_removed += 1
            continue
        new_reward_lines.append(raw)

    with reward_samples_path.open("w", encoding="utf-8") as f:
        for line in new_reward_lines:
            f.write(line + "\n")

    return CullStats(
        total_trajectories=total_trajectories,
        total_reward_entries=total_reward_entries,
        candidate_ids=n_candidates,
        culled_ids=len(culled_ids),
        trajectories_removed=trajectories_removed,
        reward_entries_removed=reward_entries_removed,
    )


def main() -> None:
    stats = cull_too_good_trajectories()
    print(
        f"Total trajectories: {stats.total_trajectories}\n"
        f"Total reward entries: {stats.total_reward_entries}\n"
        f"High-scoring candidate IDs (non-moral_dilemmas, non-degraded): {stats.candidate_ids}\n"
        f"Culled IDs: {stats.culled_ids}\n"
        f"Trajectories removed: {stats.trajectories_removed}\n"
        f"Reward entries removed: {stats.reward_entries_removed}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()


