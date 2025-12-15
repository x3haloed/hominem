#!/usr/bin/env python3
"""
Anchor/Φ post-processing pipeline.

Reads the auto-labeled JSONL shards produced by apps/cli/batch_emotion_label.py,
applies the heuristic anchor/Φ scoring functions from docs/unified_theory.md,
and emits enriched JSONL shards ready for Unified Theory training.

Properties:
- Re-entrant: skips shards that already have enriched outputs unless --force is set.
- Streaming: processes line-by-line to keep memory usage low.
- Configurable input/output roots and dataset regime defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def bool_to_float(condition: bool) -> float:
    return 1.0 if condition else 0.0


def safe_get(labels: Dict[str, float], field: str, default: float = 0.0) -> float:
    value = labels.get(field, default)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# History helpers (degenerate defaults for non-conversational data)
# -----------------------------------------------------------------------------


def history_avg_social(history: List[Dict[str, float]], n: int = 3) -> float:
    if not history:
        return 0.0
    recent = history[-n:]
    if not recent:
        return 0.0
    return sum(item.get("social_broadcast", 0.0) for item in recent) / len(recent)


def history_stuck_check(history: List[Dict[str, float]], n: int = 3) -> bool:
    if len(history) < n:
        return False
    recent = history[-n:]
    return all(
        item.get("arousal", 0.0) > 0.7 and abs(item.get("valence", 0.0)) > 0.6 for item in recent
    )


def compute_expected_anchor_gain(
    current: Dict[str, float],
    regime: str,
    history: List[Dict[str, float]],
) -> float:
    # Without rich history we fall back to zero expected gain.
    if len(history) < 2:
        return 0.0

    def estimate_anchor_score(turn: Dict[str, float], reg: str) -> float:
        valence = turn.get("valence", 0.0)
        arousal = turn.get("arousal", 0.0)
        dominance = turn.get("dominance", 0.0)
        discrepancy = turn.get("predictive_discrepancy", 0.0)
        temporal = turn.get("temporal_directionality", 0.0)
        social = turn.get("social_broadcast", 0.0)

        if reg in ("support", "play"):
            return valence * 0.4 + social * 0.6
        if reg == "conflict":
            return bool_to_float(temporal > 0) * 0.5 + bool_to_float(social > 0.5) * 0.5
        if reg == "crisis":
            return dominance * 0.7 + bool_to_float(temporal > 0) * 0.3
        if reg in ("truth_seeking", "problem_solving"):
            return bool_to_float(discrepancy > 0 and dominance > 0) * 0.8
        return valence * 0.3 + dominance * 0.3 + social * 0.4

    recent_scores: List[float] = []
    for turn in history[-3:]:
        recent_scores.append(estimate_anchor_score(turn, regime))

    if len(recent_scores) < 2:
        return 0.0

    trend = (recent_scores[-1] - recent_scores[0]) / (len(recent_scores) - 1 or 1)
    expected_gain = current.get("temporal_directionality", 0.0) * trend
    return clamp(expected_gain, -1.0, 1.0)


# -----------------------------------------------------------------------------
# Heuristic scoring functions (derived from docs/unified_theory.md)
# -----------------------------------------------------------------------------


HIGH_SOCIAL_REGIMES = {"support", "conflict", "play"}


def agency_support_score(s: Dict[str, float], regime: str) -> float:
    base = 0.0
    base += s.get("dominance", 0.0) * 0.5
    base += (1.0 - abs(s.get("predictive_discrepancy", 0.0))) * 0.3
    base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.2
    if regime == "boundary":
        base += s.get("dominance", 0.0) * 0.3
    elif regime == "crisis":
        base -= 0.1
    return clamp(base)


def harm_minimization_score(s: Dict[str, float], regime: str) -> float:
    base = (
        -s.get("valence", 0.0) * 0.4
        + s.get("dominance", 0.0) * 0.4
        + (-abs(s.get("predictive_discrepancy", 0.0))) * 0.2
    )
    if regime == "crisis":
        base *= 1.5
    if regime == "support":
        base += bool_to_float(s.get("valence", 0.0) > 0) * 0.3
    return clamp(base)


def optionality_preservation(s: Dict[str, float], regime: str) -> float:
    base = 0.0
    base += s.get("dominance", 0.0) * 0.5
    base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3
    base += (1.0 - abs(s.get("predictive_discrepancy", 0.0))) * 0.2
    if regime not in HIGH_SOCIAL_REGIMES and s.get("social_broadcast", 0.0) > 0.7:
        base -= 0.3
    return clamp(base)


def empathy_correctness(s: Dict[str, float], regime: str) -> float:
    base = s.get("social_broadcast", 0.0) * 0.6 + bool_to_float(s.get("valence", 0.0) > 0) * 0.4
    if regime == "support":
        base *= 1.3
    if regime == "conflict":
        base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.2
    return clamp(base)


def social_coherence_repair(s: Dict[str, float], history: List[Dict[str, float]]) -> float:
    base = s.get("social_broadcast", 0.0) * 0.5
    base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3
    base += bool_to_float(s.get("valence", 0.0) > -0.3) * 0.2
    recent_avg = history_avg_social(history, 2)
    if s.get("social_broadcast", 0.0) > recent_avg:
        base += 0.3
    return clamp(base)


def narrative_alignment_without_domination(s: Dict[str, float], regime: str) -> float:
    base = s.get("social_broadcast", 0.0) * 0.4
    base += (1.0 - abs(s.get("dominance", 0.0))) * 0.3
    base += bool_to_float(s.get("valence", 0.0) > -0.2) * 0.3
    if regime in HIGH_SOCIAL_REGIMES and s.get("dominance", 0.0) > 0.7:
        base -= 0.4
    return clamp(base)


def epistemic_integrity(s: Dict[str, float], regime: str) -> float:
    base = 0.0
    if regime in ("truth_seeking", "problem_solving"):
        base += abs(s.get("predictive_discrepancy", 0.0)) * 0.5
        base += s.get("dominance", 0.0) * 0.5
    base += bool_to_float(s.get("social_broadcast", 0.0) > 0.3) * 0.3
    base += bool_to_float(s.get("temporal_directionality", 0.0) < 0.3) * 0.2
    return clamp(base)


def curiosity_resolved_usefully(s: Dict[str, float], regime: str) -> float:
    base = s.get("predictive_discrepancy", 0.0) * 0.4
    base += s.get("valence", 0.0) * 0.3
    base += s.get("dominance", 0.0) * 0.3
    if regime in ("truth_seeking", "problem_solving"):
        base *= 1.2
    return clamp(base)


def surprise_reduction(s: Dict[str, float]) -> float:
    base = (-s.get("predictive_discrepancy", 0.0)) * 0.5
    base += s.get("valence", 0.0) * 0.3
    base += (1.0 - s.get("arousal", 0.0)) * 0.2
    if s.get("arousal", 0.0) > 0.7 and s.get("predictive_discrepancy", 0.0) < -0.3:
        base -= 0.4
    return clamp(base)


def emotional_trajectory_health(
    s: Dict[str, float],
    regime: str,
    history: List[Dict[str, float]],
) -> float:
    base = 0.0
    valence = s.get("valence", 0.0)
    arousal = s.get("arousal", 0.0)
    dominance = s.get("dominance", 0.0)

    if regime in ("support", "play"):
        base += valence * 0.6 + (0.5 - abs(arousal - 0.5)) * 0.4
    elif regime in ("conflict", "crisis"):
        base += (dominance - abs(valence)) * 0.5
        base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3
    elif regime in ("truth_seeking", "problem_solving"):
        discrepancy = s.get("predictive_discrepancy", 0.0)
        base += discrepancy * (1.0 if dominance > 0 else -1.0) * 0.7

    base += max(0.0, s.get("social_broadcast", 0.0) - history_avg_social(history, 3)) * 0.4

    if history_stuck_check(history, 3):
        if arousal > 0.7 and abs(valence) > 0.6:
            base -= 0.5

    if s.get("social_broadcast", 0.0) < 0.3 and regime in HIGH_SOCIAL_REGIMES:
        base -= 0.4

    expected_gain = compute_expected_anchor_gain(s, regime, history)
    if s.get("temporal_directionality", 0.0) > 0.5 and expected_gain > 0:
        base += 0.2

    return clamp(base)


def aggregate_anchor_scores(
    s: Dict[str, float],
    regime: str,
    history: List[Dict[str, float]],
) -> Dict[str, float]:
    survival = clamp(
        0.4 * agency_support_score(s, regime)
        + 0.4 * harm_minimization_score(s, regime)
        + 0.2 * optionality_preservation(s, regime)
    )

    belonging = clamp(
        0.35 * empathy_correctness(s, regime)
        + 0.35 * social_coherence_repair(s, history)
        + 0.3 * narrative_alignment_without_domination(s, regime)
    )

    control = clamp(
        0.4 * epistemic_integrity(s, regime)
        + 0.3 * curiosity_resolved_usefully(s, regime)
        + 0.3 * surprise_reduction(s)
    )

    emotional_health = emotional_trajectory_health(s, regime, history)

    return {
        "survival": survival,
        "belonging": belonging,
        "control": control,
        "emotional_health": emotional_health,
    }


# -----------------------------------------------------------------------------
# Processing pipeline
# -----------------------------------------------------------------------------

REGIME_DEFAULTS = {
    "setfit_emotion": "support",
    "go_emotions": "support",
    "dahoas_rm_static": "problem_solving",
    "stanford_shp": "truth_seeking",
}


def enrich_shard(
    input_path: Path,
    output_path: Path,
    regime: str,
) -> None:
    tmp_path = output_path.with_suffix(".tmp")
    with input_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            labels = record.get("labels", {})
            scores = aggregate_anchor_scores(labels, regime, history=[])

            # We store anchors + φ components.
            phi_components = {
                "lambda_survival": 1.0 * scores["survival"],
                "lambda_belonging": 1.0 * scores["belonging"],
                "lambda_control": 1.0 * scores["control"],
                "lambda_emotional": 1.0 * scores["emotional_health"],
            }
            phi_value = sum(phi_components.values())

            record["anchors"] = scores
            record["phi"] = {
                "value": clamp(phi_value, -3.0, 3.0),
                "components": phi_components,
            }

            json.dump(record, dst, ensure_ascii=False)
            dst.write("\n")

    tmp_path.replace(output_path)


def discover_shards(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("shard_*.jsonl"))


def ensure_output_dir(output_root: Path, dataset_name: str) -> Path:
    target = output_root / dataset_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def process_dataset_dir(
    dataset_dir: Path,
    output_dir: Path,
    regime: str,
    force: bool,
) -> None:
    shards = discover_shards(dataset_dir)
    if not shards:
        print(f"⚠️  No shards found in {dataset_dir}")
        return

    for shard_path in shards:
        relative_name = shard_path.name
        output_path = output_dir / relative_name
        if output_path.exists() and not force:
            continue
        print(f"   ↳ {relative_name} (regime={regime})")
        enrich_shard(shard_path, output_path, regime)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add heuristic anchor/Φ scores to labeled shards.")
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/processed_datasets",
        help="Root directory containing auto-labeled dataset folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/processed_datasets_with_anchors",
        help="Root directory for enriched shards.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name(s) to process. Defaults to all directories under input root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute shards even if the enriched output already exists.",
    )
    parser.add_argument(
        "--regime",
        action="append",
        nargs=2,
        metavar=("DATASET", "REGIME"),
        help="Override default regime for a dataset (e.g., --regime setfit_emotion support).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_names = args.dataset
    else:
        dataset_names = sorted(
            [p.name for p in input_root.iterdir() if p.is_dir() and p.name != "__pycache__"]
        )

    regime_overrides = {name: value for name, value in (args.regime or [])}

    for dataset_name in dataset_names:
        dataset_dir = input_root / dataset_name
        if not dataset_dir.exists():
            print(f"⚠️  Dataset directory missing: {dataset_dir}")
            continue

        regime = regime_overrides.get(dataset_name, REGIME_DEFAULTS.get(dataset_name, "general"))
        print(f"\n📦 Dataset {dataset_name} (regime={regime})")

        output_dir = ensure_output_dir(output_root, dataset_name)
        process_dataset_dir(dataset_dir, output_dir, regime, force=args.force)

    print("\n✅ Anchor post-processing complete.")


if __name__ == "__main__":
    main()

