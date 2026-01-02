#!/usr/bin/env python3
"""
Run a small sweep over reward-manifold training hyperparameters.

This executes train_reward_manifold.py for each (warmup_ratio, max_grad_norm)
pair and records eval JSON summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _parse_floats(values: List[str]) -> List[float]:
    out = []
    for v in values:
        out.append(float(v))
    return out


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


def _read_mean_correlation(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    overall = data.get("overall", {})
    return overall.get("mean_correlation")


def _run_command(args: List[str]) -> None:
    print("▶️ ", " ".join(args))
    subprocess.check_call(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep reward-manifold training hyperparams.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/labeled/reward_samples_lowered_k.jsonl"],
    )
    parser.add_argument("--preset", default="bert_base")
    parser.add_argument("--output-root", default="artifacts/reward_manifold_sweep")
    parser.add_argument("--eval-root", default="artifacts")
    parser.add_argument("--warmup-steps", nargs="+", default=["50", "100", "150"])
    parser.add_argument("--max-grad-norms", nargs="+", default=["0.8", "1.0", "1.2"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    warmups = _parse_floats(args.warmup_steps)
    grads = _parse_floats(args.max_grad_norms)

    output_root = Path(args.output_root)
    eval_root = Path(args.eval_root)
    output_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[str, float | None]] = []

    for warmup in warmups:
        for grad in grads:
            tag = f"{args.preset}_wu{_tag(warmup)}_gn{_tag(grad)}"
            out_dir = output_root / tag
            eval_path = eval_root / f"reward_manifold_{tag}_eval.json"

            if args.skip_existing and eval_path.exists():
                mean_corr = _read_mean_correlation(eval_path)
                results.append((tag, mean_corr))
                print(f"⏭️  Skipping existing {eval_path}")
                continue

            cmd = [
                sys.executable,
                "-m",
                "core.lora_trainer.train_reward_manifold",
                "--inputs",
                *args.inputs,
                "--preset",
                args.preset,
                "--output-dir",
                str(out_dir),
                "--eval-json",
                str(eval_path),
                "--warmup-steps",
                str(int(warmup)),
                "--max-grad-norm",
                str(grad),
                "--save-steps",
                str(args.save_steps),
                "--logging-steps",
                str(args.logging_steps),
            ]
            _run_command(cmd)
            mean_corr = _read_mean_correlation(eval_path)
            results.append((tag, mean_corr))

    results.sort(key=lambda r: (r[1] is None, -(r[1] or 0.0)))
    print("\n📊 Sweep results (sorted by mean correlation):")
    for tag, corr in results:
        corr_str = "n/a" if corr is None else f"{corr:.4f}"
        print(f"- {tag}: mean_correlation={corr_str}")


if __name__ == "__main__":
    main()
