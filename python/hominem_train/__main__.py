"""Entry point for hominem_train."""

from __future__ import annotations

import argparse
import sys


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hominem_train entry point")
    parser.add_argument(
        "command",
        nargs="?",
        help="Training command (manifold, regime, reward)",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    _load_dotenv()
    parser = _build_parser()
    ns = parser.parse_args()

    command = (ns.command or "").strip().lower()
    if command == "manifold":
        from hominem_train.manifold_train import main as manifold_main

        manifold_main(ns.args)
        return
    if command == "regime":
        from hominem_train.regime_train import main as regime_main

        regime_main(ns.args)
        return
    if command == "reward":
        from hominem_train.reward_train import main as reward_main

        reward_main(ns.args)
        return

    raise SystemExit(
        "Usage: python -m hominem_train <command> [args]\n"
        "Commands: manifold, regime, reward\n"
        "Examples:\n"
        "  python -m hominem_train manifold --dataset-path data/manifold.jsonl\n"
        "  python -m hominem_train regime --dataset-path data/regime.jsonl\n"
        "  python -m hominem_train reward --dataset-path data/reward.jsonl\n"
    )


if __name__ == "__main__":
    main()
