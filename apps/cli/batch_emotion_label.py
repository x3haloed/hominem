#!/usr/bin/env python3
"""
Unified Theory batch data labeler.

Rewrites the old conversation-only labeler into a dataset-oriented pipeline that:
  • Pulls raw corpora (SetFit/emotion, GoEmotions, Dahoas/rm-static, Stanford SHP, …)
  • Sends the required utterances/pairs through the EmotionEngine
  • Emits sharded JSONL artifacts that are immediately usable for the
    training workloads described in docs/unified_theory.md and
    docs/UNIFIED_THEORY_ENGINEERING_SPEC.md.

Characteristics:
  • Re-entrant per dataset: crash-safe via per-dataset progress tracking.
  • Output shards are capped (default 2k records) to keep file sizes manageable.
  • Uses the existing EmotionEngine batch endpoint for throughput while still
    guaranteeing deterministic resumption semantics.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

from dotenv import load_dotenv

load_dotenv()

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.serve.emotion_engine import EmotionEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------

GO_EMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


@dataclass
class DatasetSpec:
    """Configuration describing how to turn a source dataset into label-ready pairs."""

    name: str
    hf_path: str
    split: str
    generator: str
    target_use: Sequence[str]
    text_field: Optional[str] = None
    prompt_field: Optional[str] = None
    response_variants: Optional[List[Dict[str, Any]]] = None
    metadata_fields: Optional[Sequence[str]] = None
    context_fields: Optional[Sequence[str]] = None
    speaker_role: str = "user"
    respondent_role: str = "assistant"
    context_prefix: str = ""


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "setfit_emotion": DatasetSpec(
        name="setfit_emotion",
        hf_path="SetFit/emotion",
        split="train",
        generator="single_text",
        text_field="text",
        metadata_fields=("label", "label_text"),
        context_fields=("label_text",),
        context_prefix="SetFit/emotion",
        speaker_role="system",
        respondent_role="user",
        target_use=("emotion_manifold",),
    ),
    "go_emotions": DatasetSpec(
        name="go_emotions",
        hf_path="google-research-datasets/go_emotions",
        split="train",
        generator="single_text_go",
        text_field="text",
        metadata_fields=("labels",),
        context_prefix="GoEmotions",
        speaker_role="system",
        respondent_role="user",
        target_use=("emotion_manifold",),
    ),
    "dahoas_rm_static": DatasetSpec(
        name="dahoas_rm_static",
        hf_path="Dahoas/rm-static",
        split="train",
        generator="prompt_response",
        prompt_field="prompt",
        response_variants=[
            {"field": "chosen", "pair_type": "preferred", "target_use": ("phi_training", "regime_classifier")},
            {"field": "rejected", "pair_type": "rejected", "target_use": ("phi_training",)},
        ],
        metadata_fields=("response",),
        context_prefix="Dahoas/rm-static",
        speaker_role="user",
        respondent_role="assistant",
        target_use=("phi_training",),
    ),
    "stanford_shp": DatasetSpec(
        name="stanford_shp",
        hf_path="stanfordnlp/SHP",
        split="train",
        generator="shp_pairs",
        metadata_fields=("post_id", "domain", "labels", "upvote_ratio"),
        context_prefix="StanfordSHP",
        speaker_role="user",
        respondent_role="assistant",
        target_use=("phi_training", "regime_classifier"),
    ),
}


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Handles crash-safe progress bookkeeping."""

    def __init__(self, progress_path: Path):
        self.progress_path = progress_path
        self.data = {
            "next_record_index": 0,
            "shard_index": 0,
            "records_in_shard": 0,
        }
        if progress_path.exists():
            try:
                self.data.update(json.loads(progress_path.read_text()))
            except json.JSONDecodeError:
                print(f"⚠️  Progress file {progress_path} is corrupt, restarting from 0.")

    def update(self, next_record_index: int, shard_index: int, records_in_shard: int) -> None:
        self.data.update(
            {
                "next_record_index": next_record_index,
                "shard_index": shard_index,
                "records_in_shard": records_in_shard,
            }
        )
        tmp_path = self.progress_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2))
        tmp_path.replace(self.progress_path)


class ShardWriter:
    """Manages sharded JSONL output with bounded file sizes."""

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


# ---------------------------------------------------------------------------
# Pair generators
# ---------------------------------------------------------------------------

def _base_metadata(record: Dict[str, Any], fields: Optional[Sequence[str]]) -> Dict[str, Any]:
    return {field: record.get(field) for field in (fields or [])}


def generate_single_text_pairs(record_idx: int, record: Dict[str, Any], spec: DatasetSpec) -> List[Dict[str, Any]]:
    text = (record.get(spec.text_field or "", "") or "").strip()
    if not text:
        return []

    metadata = _base_metadata(record, spec.metadata_fields)
    context_parts = [spec.context_prefix] if spec.context_prefix else []
    for field in spec.context_fields or []:
        value = metadata.get(field)
        if value not in (None, ""):
            context_parts.append(f"{field}={value}")
    context = " | ".join(context_parts) if context_parts else None

    return [
        {
            "pair_id": f"{spec.name}-{record_idx}",
            "pair_type": "utterance",
            "speaker_role": spec.speaker_role,
            "respondent_role": spec.respondent_role,
            "speaker_message": "How are you feeling right now?",
            "respondent_message": text,
            "context": context,
            "metadata": metadata,
            "target_use": spec.target_use,
        }
    ]


def generate_single_text_pairs_go(record_idx: int, record: Dict[str, Any], spec: DatasetSpec) -> List[Dict[str, Any]]:
    pairs = generate_single_text_pairs(record_idx, record, spec)
    if not pairs:
        return pairs

    metadata = pairs[0]["metadata"] or {}
    label_ids = metadata.get("labels") or []
    label_names = [GO_EMOTIONS_LABELS[idx] for idx in label_ids if idx < len(GO_EMOTIONS_LABELS)]
    pairs[0]["context"] = f"{spec.context_prefix} | tags={','.join(label_names) or 'unknown'}"
    pairs[0]["metadata"]["label_names"] = label_names
    return pairs


def generate_prompt_response_pairs(record_idx: int, record: Dict[str, Any], spec: DatasetSpec) -> List[Dict[str, Any]]:
    prompt = (record.get(spec.prompt_field or "", "") or "").strip()
    if not prompt:
        return []

    metadata = _base_metadata(record, spec.metadata_fields)
    pairs: List[Dict[str, Any]] = []
    for variant in spec.response_variants or []:
        response = (record.get(variant["field"], "") or "").strip()
        if not response:
            continue
        pair_targets = tuple(variant.get("target_use") or spec.target_use)
        pairs.append(
            {
                "pair_id": f"{spec.name}-{record_idx}-{variant['field']}",
                "pair_type": variant.get("pair_type", variant["field"]),
                "speaker_role": spec.speaker_role,
                "respondent_role": spec.respondent_role,
                "speaker_message": prompt,
                "respondent_message": response,
                "context": spec.context_prefix,
                "metadata": {**metadata, "variant": variant["field"]},
                "target_use": pair_targets,
            }
        )
    return pairs


def generate_shp_pairs(record_idx: int, record: Dict[str, Any], spec: DatasetSpec) -> List[Dict[str, Any]]:
    history = record.get("history")
    if isinstance(history, list):
        speaker = "\n".join(history)
    else:
        speaker = history or ""
    if not speaker:
        speaker = record.get("post_id", "SHP prompt")

    metadata = _base_metadata(record, spec.metadata_fields)
    pairs: List[Dict[str, Any]] = []

    for variant_key, variant_label in (("human_ref_A", "A"), ("human_ref_B", "B")):
        response = (record.get(variant_key, "") or "").strip()
        if not response:
            continue
        pair_type = "preferred" if (record.get("labels") == (0 if variant_label == "A" else 1)) else "alternative"
        pairs.append(
            {
                "pair_id": f"{spec.name}-{record_idx}-{variant_label}",
                "pair_type": pair_type,
                "speaker_role": spec.speaker_role,
                "respondent_role": spec.respondent_role,
                "speaker_message": speaker,
                "respondent_message": response,
                "context": spec.context_prefix,
                "metadata": {**metadata, "variant": variant_label},
                "target_use": spec.target_use,
            }
        )
    return pairs


PAIR_GENERATORS = {
    "single_text": generate_single_text_pairs,
    "single_text_go": generate_single_text_pairs_go,
    "prompt_response": generate_prompt_response_pairs,
    "shp_pairs": generate_shp_pairs,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cleanup_partial_outputs(dataset_dir: Path, next_record_index: int) -> None:
    """Trim any trailing records whose indices are ≥ next_record_index (incomplete writes)."""
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
            break  # nothing to trim
        shard_path.write_text("\n".join(keep) + ("\n" if keep else ""))
        if keep:
            break
        shard_path.unlink(missing_ok=True)


def build_output_record(dataset_name: str, record_idx: int, pair: Dict[str, Any], labels: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "record_index": record_idx,
        "pair_id": pair["pair_id"],
        "pair_type": pair.get("pair_type"),
        "speaker_role": pair.get("speaker_role"),
        "respondent_role": pair.get("respondent_role"),
        "speaker_message": pair.get("speaker_message"),
        "respondent_message": pair.get("respondent_message"),
        "context": pair.get("context"),
        "metadata": pair.get("metadata"),
        "target_use": pair.get("target_use"),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

async def process_dataset(
    dataset_key: str,
    spec: DatasetSpec,
    args: argparse.Namespace,
    engine: EmotionEngine,
) -> None:
    print(f"\n🚀 Dataset: {dataset_key} ({spec.hf_path} :: {spec.split})")

    dataset = load_dataset(spec.hf_path, split=args.split or spec.split)
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

    start_index = progress.data["next_record_index"]
    max_records = args.max_records or total_records
    processed = 0
    commit_index = start_index

    pair_generator = PAIR_GENERATORS.get(spec.generator)
    if not pair_generator:
        raise ValueError(f"No pair generator registered for '{spec.generator}'")

    pending_pairs: List[Dict[str, Any]] = []
    pending_entries: List[Dict[str, Any]] = []
    record_queue: List[Dict[str, Any]] = []

    async def flush_batches(force: bool = False) -> None:
        nonlocal commit_index
        while pending_pairs and (force or len(pending_pairs) >= args.pairs_per_call):
            batch_size = min(len(pending_pairs), args.pairs_per_call)
            batch_pairs = pending_pairs[:batch_size]
            batch_entries = pending_entries[:batch_size]

            batch_result = await engine.label_message_pairs_batch(batch_pairs)

            for result in batch_result:
                entry = batch_entries[result["pair_index"]]
                writer.write(build_output_record(dataset_key, entry["record_index"], entry["pair"], result["labels"]))

            del pending_pairs[:batch_size]
            del pending_entries[:batch_size]

            remaining = batch_size
            while record_queue and remaining > 0:
                head = record_queue[0]
                consume = min(head["remaining_pairs"], remaining)
                head["remaining_pairs"] -= consume
                remaining -= consume
                if head["remaining_pairs"] == 0:
                    commit_index = head["record_index"] + 1
                    record_queue.pop(0)
                    progress.update(commit_index, writer.shard_index, writer.records_in_shard)

    try:
        for record_index in range(start_index, total_records):
            if processed >= max_records:
                break
            record = dataset[record_index]
            pairs = pair_generator(record_index, record, spec)

            if not pairs:
                processed += 1
                commit_index = record_index + 1
                progress.update(commit_index, writer.shard_index, writer.records_in_shard)
                continue

            record_queue.append({"record_index": record_index, "remaining_pairs": len(pairs)})
            for pair in pairs:
                pending_pairs.append(pair)
                pending_entries.append({"record_index": record_index, "pair": pair})

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
    parser = argparse.ArgumentParser(description="Batch label external datasets for Unified Theory training.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset key to process. Can be specified multiple times. Defaults to all.",
    )
    parser.add_argument("--split", type=str, help="Optional split override (defaults per dataset).")
    parser.add_argument("--output-dir", type=str, default="data/processed_datasets", help="Output root directory.")
    parser.add_argument("--config", type=str, default="config/inference.toml", help="Emotion engine config path.")
    parser.add_argument("--records-per-shard", type=int, default=2000, help="Max records per JSONL shard.")
    parser.add_argument("--pairs-per-call", type=int, default=10, help="Number of pairs to send per LLM request.")
    parser.add_argument("--max-records", type=int, help="Limit how many records to process per dataset.")
    parser.add_argument("--resume", action="store_true", help="(Deprecated) kept for CLI compatibility.")
    parser.add_argument("--batch-size", type=int, default=8, help="Legacy arg retained (ignored).")
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
        print("\n⏹️  Interrupted by user")
