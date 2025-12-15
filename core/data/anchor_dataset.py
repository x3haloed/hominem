from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


ANCHOR_TARGETS = ("survival", "belonging", "control", "emotional_health")


@dataclass
class AnchorRecord:
    dataset: str
    record_index: int
    pair_id: str
    pair_type: str
    speaker_message: str
    respondent_message: str
    metadata: Dict[str, Any]
    target_use: Sequence[str]
    labels: Dict[str, float]
    anchors: Dict[str, float]
    phi_value: float


def _iter_dataset_dirs(root: Path, datasets: Optional[Sequence[str]]) -> Iterable[Path]:
    if datasets:
        for name in datasets:
            path = root / name
            if path.is_dir():
                yield path
    else:
        for path in sorted(root.iterdir()):
            if path.is_dir() and path.name != "__pycache__":
                yield path


def iter_anchor_records(
    root: Path,
    *,
    datasets: Optional[Sequence[str]] = None,
    target_use_filter: Optional[str] = None,
) -> Iterator[AnchorRecord]:
    """
    Stream AnchorRecord objects from enriched JSONL shards.

    Args:
        root: Directory containing dataset folders with shard_XXXXX.jsonl files.
        datasets: Optional subset of dataset names to include.
        target_use_filter: If provided, only yield records whose target_use contains this tag.
    """
    for dataset_dir in _iter_dataset_dirs(root, datasets):
        for shard_path in sorted(dataset_dir.glob("shard_*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as shard_file:
                for line in shard_file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if target_use_filter and target_use_filter not in record.get("target_use", []):
                        continue
                    phi = record.get("phi", {})
                    yield AnchorRecord(
                        dataset=record.get("dataset", dataset_dir.name),
                        record_index=record.get("record_index", -1),
                        pair_id=record.get("pair_id", ""),
                        pair_type=record.get("pair_type", ""),
                        speaker_message=record.get("speaker_message", ""),
                        respondent_message=record.get("respondent_message", ""),
                        metadata=record.get("metadata") or {},
                        target_use=record.get("target_use") or [],
                        labels=record.get("labels") or {},
                        anchors=record.get("anchors") or {},
                        phi_value=float(phi.get("value", 0.0)),
                    )


def load_anchor_records(
    root: Path,
    *,
    datasets: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[AnchorRecord]:
    """Materialize anchor records into memory (useful for small experiments)."""
    records: List[AnchorRecord] = []
    for record in iter_anchor_records(root, datasets=datasets):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


class AnchorTorchDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Torch dataset that exposes enriched anchor targets for Unified Theory training.

    Each item tokenizes the speaker/respondent pair into a single sequence and
    returns regression targets for the four anchors plus φ value.
    """

    def __init__(
        self,
        records: Sequence[AnchorRecord],
        tokenizer,
        *,
        max_length: int = 768,
        label_mean: Optional[torch.Tensor] = None,
        label_std: Optional[torch.Tensor] = None,
    ) -> None:
        self._records = list(records)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._label_mean = label_mean
        self._label_std = label_std

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self._records[idx]
        speaker = record.speaker_message or ""
        respondent = record.respondent_message or ""
        text = f"Speaker: {speaker}\nRespondent: {respondent}"

        encoded = self._tokenizer(
            text,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_values = [float(record.anchors.get(target, 0.0)) for target in ANCHOR_TARGETS]
        label_values.append(record.phi_value)
        labels = torch.tensor(label_values, dtype=torch.float32)

        if self._label_mean is not None and self._label_std is not None:
            labels = (labels - self._label_mean) / self._label_std

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }

