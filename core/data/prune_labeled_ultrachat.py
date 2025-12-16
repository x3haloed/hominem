#!/usr/bin/env python3
"""
Prune labeled UltraChat trajectories by removing low-signal ("boring") examples.
Heuristics keep examples with emotional intensity, conflict/crisis/boundary play,
ownership variety, or meaningful ΔΦ / reward. Optionally sample a small fraction
of boring rows to preserve distributional sanity.

Usage:
  python core/data/prune_labeled_ultrachat.py \
      --input data/processed_datasets_unified/ultrachat_trajectories/shard_00000.jsonl \
      --output data/processed_datasets_unified/ultrachat_trajectories/shard_00000_pruned.jsonl \
      --boring-sample-prob 0.05
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune low-signal labeled trajectories")
    parser.add_argument("--input", help="Input labeled shard (.jsonl)")
    parser.add_argument("--input-dir", help="Input directory of labeled shards (.jsonl)")
    parser.add_argument("--output", help="Output pruned shard (.jsonl)")
    parser.add_argument("--output-dir", help="Output directory for pruned shards (.jsonl)")
    parser.add_argument(
        "--boring-sample-prob",
        type=float,
        default=0.05,
        help="Probability to retain a boring row for coverage",
    )
    # Thresholds for keeping rows
    parser.add_argument("--min_arousal_keep", type=float, default=0.6)
    parser.add_argument("--min_abs_valence_keep", type=float, default=0.4)
    parser.add_argument("--min_predictive_discrepancy_keep", type=float, default=0.4)
    parser.add_argument("--min_delta_phi_keep", type=float, default=0.1)
    parser.add_argument("--min_reward_intensity_keep", type=float, default=1.0)
    parser.add_argument("--min_regime_spice", type=float, default=0.2)
    parser.add_argument("--keep_play", action="store_true", help="Also keep play if above regime spice")
    return parser.parse_args()


def is_spicy(lbl: Dict[str, Any], args: argparse.Namespace) -> bool:
    # Intensity / valence / surprise
    if lbl.get("arousal", 0.0) > args.min_arousal_keep:
        return True
    if abs(lbl.get("valence", 0.0)) > args.min_abs_valence_keep:
        return True
    if lbl.get("predictive_discrepancy", 0.0) > args.min_predictive_discrepancy_keep:
        return True
    # Regimes
    regimes = [
        ("regime_conflict", True),
        ("regime_crisis", True),
        ("regime_boundary", True),
        ("regime_play", args.keep_play),
    ]
    for rk, enabled in regimes:
        if enabled and lbl.get(rk, 0.0) >= args.min_regime_spice:
            return True
    # ΔΦ / reward
    if abs(lbl.get("delta_phi", 0.0)) > args.min_delta_phi_keep:
        return True
    if lbl.get("reward_intensity", 0.0) > args.min_reward_intensity_keep:
        return True
    # Ownership variety
    if lbl.get("agent_initiated") or lbl.get("commitment_active"):
        return True
    return False


def main() -> None:
    args = parse_args()
    if not args.input and not args.input_dir:
        raise SystemExit("Provide --input or --input-dir")
    if args.input_dir and not args.output_dir:
        raise SystemExit("When using --input-dir, provide --output-dir")
    if args.input and not args.output:
        raise SystemExit("When using --input, provide --output")

    random.seed(0)
    total_kept = 0
    total_dropped = 0

    def process_file(in_path: Path, out_path: Path) -> None:
        nonlocal total_kept, total_dropped
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        dropped = 0
        with in_path.open() as fin, out_path.open("w") as fout:
            for line in fin:
                if not line.strip():
                    continue
                obj = json.loads(line)
                labels = obj.get("labels", {})
                if is_spicy(labels, args):
                    fout.write(line)
                    kept += 1
                else:
                    if random.random() < args.boring_sample_prob:
                        fout.write(line)
                        kept += 1
                    else:
                        dropped += 1
        total_kept += kept
        total_dropped += dropped
        print(f"[{in_path.name}] kept {kept}, dropped {dropped} -> {out_path}")

    if args.input:
        process_file(Path(args.input), Path(args.output))
    else:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        files: List[Path] = sorted(in_dir.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No .jsonl files in {in_dir}")
        for fp in files:
            process_file(fp, out_dir / fp.name)

    print(f"Total kept {total_kept}, dropped {total_dropped}")


if __name__ == "__main__":
    main()
