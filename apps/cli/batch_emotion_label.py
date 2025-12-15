#!/usr/bin/env python3
"""
Unified-theory batch labeler (clean rewrite).

Goals:
- Feed real multi-turn conversation slices to the unified-theory auto-labeler.
- Capture the full label object (manifold, self-fractions, anchors, regimes, Φ/ΔΦ).
- Emit re-entrant shards under data/processed_datasets_unified.
- Be robust to partial failures: validate, retry per-turn, persist progress incrementally.

Legacy compatibility, old schemas, and partial label sets are intentionally ignored.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.serve.emotion_engine import EmotionEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset specification (only keeping sources that can provide multi-turn context)
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    hf_path: Optional[str]  # None for local files
    local_path: Optional[str]  # Path relative to repo root for local files
    split: Optional[str]  # None for local files
    target_use: Sequence[str]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    # Stanford SHP contains a `history` field; although often a single block, we preserve it verbatim.
    "stanford_shp": DatasetSpec(
        name="stanford_shp",
        hf_path="stanfordnlp/SHP",
        local_path=None,
        split="train",
        target_use=("phi_training", "regime_classifier"),
    ),
    # Ultrachat trajectories - local JSONL file
    "ultrachat_trajectories": DatasetSpec(
        name="ultrachat_trajectories",
        hf_path=None,
        local_path="data/processed_datasets_unified/ultrachat_trajectories.jsonl",
        split=None,
        target_use=("phi_training", "regime_classifier", "emotion_manifold"),
    ),
    # Synthetic ultrachat-style trajectories (unlabeled conversations) generated locally.
    "ultrachat_synthetic_trajectories": DatasetSpec(
        name="ultrachat_synthetic_trajectories",
        hf_path=None,
        local_path="data/processed_datasets_unified/ultrachat_trajectories_synthetic.jsonl",
        split=None,
        target_use=("phi_training", "regime_classifier", "emotion_manifold"),
    ),
}


# ---------------------------------------------------------------------------
# Progress + shard management (re-entrant)
# ---------------------------------------------------------------------------


class ProgressTracker:
    def __init__(self, progress_path: Path):
        self.progress_path = progress_path
        self.data = {"next_record_index": 0, "shard_index": 0, "records_in_shard": 0}
        if progress_path.exists():
            try:
                self.data.update(json.loads(progress_path.read_text()))
            except json.JSONDecodeError:
                print(f"⚠️ Progress file {progress_path} corrupt, restarting from 0.")

    def update(self, next_record_index: int, shard_index: int, records_in_shard: int) -> None:
        self.data.update(
            {
                "next_record_index": next_record_index,
                "shard_index": shard_index,
                "records_in_shard": records_in_shard,
            }
        )
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.progress_path)


class ShardWriter:
    def __init__(self, dataset_dir: Path, shard_size: int, shard_index: int, records_in_shard: int):
        self.dataset_dir = dataset_dir
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.shard_index = shard_index
        self.records_in_shard = records_in_shard
        self.file = None
        self._open_current_shard()

    def _open_current_shard(self) -> None:
        if self.file:
            self.file.close()
        shard_path = self.dataset_dir / f"shard_{self.shard_index:05d}.jsonl"
        mode = "a" if shard_path.exists() else "w"
        self.file = shard_path.open(mode, encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        json.dump(record, self.file, ensure_ascii=False)
        self.file.write("\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.records_in_shard += 1
        if self.records_in_shard >= self.shard_size:
            self.shard_index += 1
            self.records_in_shard = 0
            self._open_current_shard()

    def close(self) -> None:
        if self.file:
            self.file.close()


def cleanup_partial_outputs(dataset_dir: Path, next_record_index: int) -> None:
    """Trim any records with record_index >= next_record_index (incomplete writes)."""
    shard_paths = sorted(dataset_dir.glob("shard_*.jsonl"))
    if not shard_paths:
        return
    for shard_path in reversed(shard_paths):
        lines = shard_path.read_text().splitlines()
        if not lines:
            shard_path.unlink(missing_ok=True)
            continue
        keep: List[str] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_index", 0) < next_record_index:
                keep.append(line)
        if len(keep) == len(lines):
            break
        shard_path.write_text("\n".join(keep) + ("\n" if keep else ""))
        if keep:
            break
        shard_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------


def _build_history_from_shp(history_raw: Any) -> List[Dict[str, str]]:
    """
    Preserve SHP history as best as available.
    - If list: alternate user/assistant roles per element.
    - If string: wrap as a single user turn (SHP often stores one blob).
    """
    history: List[Dict[str, str]] = []
    if isinstance(history_raw, list):
        for i, text in enumerate(history_raw):
            if isinstance(text, str) and text.strip():
                role = "user" if i % 2 == 0 else "assistant"
                history.append({"role": role, "content": text.strip()})
    elif isinstance(history_raw, str) and history_raw.strip():
        history.append({"role": "user", "content": history_raw.strip()})
    return history


def generate_stanford_shp(record_idx: int, record: Dict[str, Any], spec: DatasetSpec) -> List[Dict[str, Any]]:
    history = _build_history_from_shp(record.get("history"))
    if not history:
        history = [{"role": "user", "content": record.get("post_id", "SHP prompt")}]

    metadata = {
        "post_id": record.get("post_id"),
        "domain": record.get("domain"),
        "labels": record.get("labels"),
        "upvote_ratio": record.get("upvote_ratio"),
    }

    turns: List[Dict[str, Any]] = []
    for variant_key, variant_label in (("human_ref_A", "A"), ("human_ref_B", "B")):
        response = (record.get(variant_key, "") or "").strip()
        if not response:
            continue
        pair_type = "preferred" if (record.get("labels") == (0 if variant_label == "A" else 1)) else "alternative"
        turns.append(
            {
                "turn_id": f"{spec.name}-{record_idx}-{variant_label}",
                "history": history,
                "target": {"role": "assistant", "content": response},
                "previous_phi": None,
                "metadata": {**metadata, "variant": variant_label, "pair_type": pair_type},
                "target_use": spec.target_use,
                "record_index": record_idx,
            }
        )
    return turns


def generate_ultrachat_trajectories(record_idx: int, record: Dict[str, Any], spec: DatasetSpec) -> List[Dict[str, Any]]:
    """Generate multiple training turns from ultrachat trajectories.

    Creates training samples where conversation history starts with assistant message
    and ends with user message (assistant is "prompting" the user).
    """
    full_conversation = record.get("full_conversation", [])

    # Fallback: accept already-trimmed history/target pairs (e.g., synthetic trajectories)
    if not full_conversation:
        history = record.get("history") or []
        target = record.get("target") or {}
        if history and target:
            metadata = record.get("metadata", {})
            return [
                {
                    "turn_id": record.get("turn_id") or f"{spec.name}-{record_idx}-0",
                    "history": history,
                    "target": target,
                    "previous_phi": None,
                    "metadata": metadata,
                    "target_use": spec.target_use,
                    "record_index": record_idx,
                }
            ]
        return []

    # Track only the final valid training turn so we emit a single record per conversation.
    last_turn = None

    # Generate training samples from different points in the conversation
    # We want history to start with assistant and end with user before the target
    for i in range(1, len(full_conversation) - 1):  # Start from 1 to skip initial user, end before last turn
        # Check if we have a valid pattern: assistant -> user -> assistant -> ... -> user (before target)
        # The history should end with a user message, and target should be assistant response

        current_turn = full_conversation[i]
        next_turn = full_conversation[i + 1]

        # History should end with user message
        if current_turn.get("role") != "user":
            continue

        # Target should be assistant response
        if next_turn.get("role") != "assistant":
            continue

        # Build history: from start up to current user message
        # Ensure history starts with assistant (skip initial user if present)
        history_start_idx = 0
        if full_conversation[0].get("role") == "user":
            history_start_idx = 1  # Skip initial user message

        # Ensure we have at least one assistant->user exchange
        if i < history_start_idx + 2:
            continue

        history = full_conversation[history_start_idx:i+1]  # Up to and including current user message

        # Verify history starts with assistant and ends with user
        if not history or history[0].get("role") != "assistant" or history[-1].get("role") != "user":
            continue

        # Ensure we have meaningful conversation depth (at least 3 turns)
        if len(history) < 3:
            continue

        metadata = {
            "conversation_id": record.get("conversation_id"),
            "history_length": len(history),
            "total_turns": len(full_conversation),
            "source": record.get("source", "ultrachat_multiturn"),
            "target_role": next_turn.get("role"),
            "sample_position": i,  # Which turn in the conversation this sample represents
        }

        turn_id = f"ultrachat-{record_idx}-{i}"

        last_turn = {
            "turn_id": turn_id,
            "history": history,
            "target": next_turn,
            "previous_phi": None,
            "metadata": metadata,
            "target_use": spec.target_use,
            "record_index": record_idx,
        }

    return [last_turn] if last_turn else []


GENERATOR_BY_DATASET = {
    "stanford_shp": generate_stanford_shp,
    "ultrachat_trajectories": generate_ultrachat_trajectories,
    "ultrachat_synthetic_trajectories": generate_ultrachat_trajectories,
}


# ---------------------------------------------------------------------------
# Labeling core
# ---------------------------------------------------------------------------


def build_output_record(turn: Dict[str, Any], labels: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": turn.get("dataset"),
        "record_index": turn.get("record_index"),
        "turn_id": turn.get("turn_id"),
        "pair_type": turn.get("metadata", {}).get("pair_type"),
        "history": turn.get("history"),
        "target": turn.get("target"),
        "metadata": turn.get("metadata"),
        "target_use": turn.get("target_use"),
        "labels": labels,
    }


async def process_dataset(
    dataset_key: str,
    spec: DatasetSpec,
    args: argparse.Namespace,
    engine: EmotionEngine,
) -> None:
    if spec.local_path:
        print(f"\n🚀 Dataset: {dataset_key} (local: {spec.local_path})")
        # Load local JSONL file
        import json
        records = []
        with open(spec.local_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        total_records = len(records)
        print(f"   Total rows available: {total_records}")
    else:
        print(f"\n🚀 Dataset: {dataset_key} ({spec.hf_path} :: {spec.split})")
        dataset = load_dataset(spec.hf_path, split=args.split or spec.split)
        records = dataset
        total_records = len(dataset)
        print(f"   Total rows available: {total_records}")

    dataset_dir = Path(args.output_dir) / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(dataset_dir / "progress.json")
    cleanup_partial_outputs(dataset_dir, progress.data["next_record_index"])

    writer = ShardWriter(
        dataset_dir=dataset_dir,
        shard_size=args.records_per_shard,
        shard_index=progress.data["shard_index"],
        records_in_shard=progress.data["records_in_shard"],
    )

    generator = GENERATOR_BY_DATASET[dataset_key]

    start_index = progress.data["next_record_index"]
    max_records = args.max_records or total_records
    processed = 0
    commit_index = start_index

    pending_conversations: List[Dict[str, Any]] = []
    record_queue: List[Dict[str, Any]] = []

    async def flush_batches(force: bool = False) -> None:
        nonlocal commit_index
        while pending_conversations and (force or len(pending_conversations) >= args.conversations_per_call):
            batch_size = min(len(pending_conversations), args.conversations_per_call)
            batch = pending_conversations[:batch_size]

            try:
                batch_result = await engine.label_conversation_turns_batch(batch)
            except Exception as e:
                print(f"⚠️ Batch request failed, retrying per-turn: {e}")
                batch_result = []
                for idx, turn in enumerate(batch):
                    labels = await engine.label_conversation_turn(
                        conversation_history=turn.get("history", []),
                        target_message=turn["target"],
                        previous_phi=turn.get("previous_phi")
                    )
                    batch_result.append({"conversation_index": idx, "labels": labels})

            for result in batch_result:
                conv = batch[result["conversation_index"]]
                writer.write(build_output_record(conv, result["labels"]))

            del pending_conversations[:batch_size]

            # Advance commit_index per fully consumed records
            remaining = batch_size
            while record_queue and remaining > 0:
                head = record_queue[0]
                consume = min(head["remaining_turns"], remaining)
                head["remaining_turns"] -= consume
                remaining -= consume
                if head["remaining_turns"] == 0:
                    commit_index = head["record_index"] + 1
                    record_queue.pop(0)
                    progress.update(commit_index, writer.shard_index, writer.records_in_shard)

    try:
        for record_index in range(start_index, total_records):
            if processed >= max_records:
                break

            record = records[record_index]
            turns = generator(record_index, record, spec)
            # Attach dataset name for output
            for t in turns:
                t["dataset"] = dataset_key

            if not turns:
                processed += 1
                commit_index = record_index + 1
                progress.update(commit_index, writer.shard_index, writer.records_in_shard)
                continue

            record_queue.append({"record_index": record_index, "remaining_turns": len(turns)})
            pending_conversations.extend(turns)

            await flush_batches(force=False)
            processed += 1
            if processed % 100 == 0:
                print(f"   … queued {processed} records (last idx={record_index})")

        await flush_batches(force=True)
    finally:
        writer.close()

    print(f"✅ Finished dataset {dataset_key}: {processed} new records labeled.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified-theory batch labeler (multi-turn).")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset key to process. Can be specified multiple times. Defaults to all.",
    )
    parser.add_argument("--split", type=str, help="Optional split override (defaults per dataset).")
    parser.add_argument("--output-dir", type=str, default="data/processed_datasets_unified", help="Output root directory.")
    parser.add_argument("--config", type=str, default="config/inference.toml", help="Emotion engine config path.")
    parser.add_argument("--records-per-shard", type=int, default=2000, help="Max records per JSONL shard.")
    parser.add_argument(
        "--conversations-per-call",
        type=int,
        default=8,
        help="Number of conversation payloads (history+target) to send per LLM request.",
    )
    parser.add_argument("--max-records", type=int, help="Limit how many records to process per dataset.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    engine = EmotionEngine(args.config)
    dataset_keys = args.dataset or list(DATASET_SPECS.keys())

    try:
        for key in dataset_keys:
            spec = DATASET_SPECS[key]
            await process_dataset(key, spec, args, engine)
    finally:
        await engine.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")
