#!/usr/bin/env python3
"""
Validate a labeled UltraChat trajectory shard to catch collapsed distributions
before training. This is intentionally conservative: if the shard is too small,
mode-collapsed (all positive valence, no crisis/conflict/boundary, ΔΦ flat), or
lacks ownership variety, it fails fast.

Usage:
  python core/data/validate_ultrachat_shard.py \\
      --path data/processed_datasets_unified/ultrachat_trajectories/shard_00000.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EXPECTED_LABEL_KEYS: Tuple[str, ...] = (
    "valence",
    "arousal",
    "dominance",
    "predictive_discrepancy",
    "temporal_directionality",
    "social_broadcast",
    "valence_self_fraction",
    "arousal_self_fraction",
    "dominance_self_fraction",
    "predictive_discrepancy_self_fraction",
    "temporal_directionality_self_fraction",
    "social_broadcast_self_fraction",
    "anchor_survival",
    "anchor_belonging",
    "anchor_control",
    "phi_value",
    "regime_support",
    "regime_conflict",
    "regime_problem_solving",
    "regime_truth_seeking",
    "regime_crisis",
    "regime_play",
    "regime_boundary",
    "delta_phi",
    "reward_intensity",
    "safety_score",
    "agent_initiated",
    "user_triggered",
    "commitment_active",
    "confidence",
)

REGIME_KEYS: Tuple[str, ...] = (
    "regime_support",
    "regime_conflict",
    "regime_problem_solving",
    "regime_truth_seeking",
    "regime_crisis",
    "regime_play",
    "regime_boundary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate UltraChat trajectory label distributions (single file or directory of shards)"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to shard .jsonl file OR directory containing multiple shard .jsonl files",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=1000,
        help="Minimum records required to pass (spec-scale would be 300k+)",
    )
    parser.add_argument(
        "--min-regime-mass",
        type=float,
        default=0.05,
        help="Minimum average mass per regime to avoid collapse",
    )
    parser.add_argument(
        "--min-delta-phi-span",
        type=float,
        default=0.05,
        help="Minimum span (max-min) required for delta_phi",
    )
    parser.add_argument(
        "--min-valence-floor",
        type=float,
        default=0.2,
        help="Require valence to reach this low to ensure negative coverage",
    )
    parser.add_argument(
        "--min-arousal-ceiling",
        type=float,
        default=0.8,
        help="Require arousal to reach at least this high to ensure intensity coverage",
    )
    parser.add_argument(
        "--ownership-range-min",
        type=float,
        default=0.05,
        help="Lower bound for boolean ownership signals (agent_initiated/user_triggered) fraction",
    )
    parser.add_argument(
        "--ownership-range-max",
        type=float,
        default=0.95,
        help="Upper bound for boolean ownership signals (agent_initiated/user_triggered) fraction",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]
    for fp in files:
        with fp.open() as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    return records


def span(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return max(vals) - min(vals)


def validate(records: List[Dict], args: argparse.Namespace) -> List[str]:
    failures: List[str] = []
    if len(records) < args.min_records:
        failures.append(
            f"Insufficient records: {len(records)} < min_records={args.min_records}"
        )

    # Label presence
    missing_keys = set()
    for rec in records:
        lbl = rec.get("labels", {})
        missing_keys.update([k for k in EXPECTED_LABEL_KEYS if k not in lbl])
    if missing_keys:
        failures.append(f"Missing label keys: {sorted(missing_keys)}")

    # Collect numeric ranges and ownership fractions
    numeric_ranges: Dict[str, List[float]] = defaultdict(list)
    ownership = {"agent_initiated": 0, "user_triggered": 0}
    for rec in records:
        lbl = rec.get("labels", {})
        for k, v in lbl.items():
            if isinstance(v, (int, float)):
                numeric_ranges[k].append(float(v))
        for key in ownership:
            ownership[key] += 1 if lbl.get(key) else 0

    # ΔΦ span
    delta_phi_span = span(numeric_ranges.get("delta_phi", []))
    if delta_phi_span < args.min_delta_phi_span:
        failures.append(
            f"delta_phi collapsed: span {delta_phi_span:.3f} < {args.min_delta_phi_span}"
        )

    # Valence negative coverage
    valence_vals = numeric_ranges.get("valence", [])
    if valence_vals and min(valence_vals) > args.min_valence_floor:
        failures.append(
            f"valence never reaches negative/low range: min {min(valence_vals):.3f} > floor {args.min_valence_floor}"
        )

    # Arousal high coverage
    arousal_vals = numeric_ranges.get("arousal", [])
    if arousal_vals and max(arousal_vals) < args.min_arousal_ceiling:
        failures.append(
            f"arousal never reaches high intensity: max {max(arousal_vals):.3f} < ceiling {args.min_arousal_ceiling}"
        )

    # Regime mass coverage
    regime_totals: Dict[str, float] = {k: 0.0 for k in REGIME_KEYS}
    for rec in records:
        lbl = rec.get("labels", {})
        for rk in REGIME_KEYS:
            regime_totals[rk] += float(lbl.get(rk, 0.0))
    regime_avgs = {k: v / max(len(records), 1) for k, v in regime_totals.items()}
    low_regimes = [k for k, avg in regime_avgs.items() if avg < args.min_regime_mass]
    if low_regimes:
        failures.append(
            f"regime mass collapsed for: {', '.join(low_regimes)} (avg<{args.min_regime_mass})"
        )

    # Ownership diversity
    for key, count in ownership.items():
        frac = count / max(len(records), 1)
        if not (args.ownership_range_min <= frac <= args.ownership_range_max):
            failures.append(
                f"{key} fraction collapsed: {frac:.3f} not in "
                f"[{args.ownership_range_min}, {args.ownership_range_max}]"
            )

    return failures


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        sys.stderr.write(f"Path not found: {path}\n")
        sys.exit(1)

    records = load_records(path)
    failures = validate(records, args)

    if failures:
        sys.stderr.write("Validation FAILED:\n")
        for item in failures:
            sys.stderr.write(f"- {item}\n")
        sys.exit(1)

    print("Validation PASSED")


if __name__ == "__main__":
    main()
