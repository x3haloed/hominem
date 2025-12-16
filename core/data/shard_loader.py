from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class MissingDatasetError(RuntimeError):
    """Raised when the requested dataset shards cannot be located."""


@dataclass
class ShardLoadSummary:
    shards: int
    total_records: int
    usable_records: int
    raw_dataset_counts: Dict[str, int]
    usable_dataset_counts: Dict[str, int]
    missing_label_counts: Dict[str, int]


def print_shard_summary(summary: ShardLoadSummary) -> None:
    """Emit a plaintext summary of the imported label shards."""
    print(
        f"📦 {summary.usable_records} usable records"
        f" from {summary.shards} shard(s) "
        f"(raw {summary.total_records})"
    )

    if summary.usable_dataset_counts:
        print("   Records per dataset:", summary.usable_dataset_counts)
    if summary.missing_label_counts:
        missing_total = sum(summary.missing_label_counts.values())
        print(
            f"   Skipped {missing_total} records missing required labels:",
            summary.missing_label_counts,
        )
    if (
        summary.raw_dataset_counts
        and summary.raw_dataset_counts != summary.usable_dataset_counts
    ):
        print("   Raw counts per dataset before filtering:", summary.raw_dataset_counts)


class ShardLoader:
    """Helper for gathering labeled shards spanning multiple dataset folders."""

    def __init__(self, root_paths: Sequence[Path], dataset_filters: Optional[Sequence[str]] = None):
        self.root_paths = [Path(p).expanduser().resolve() for p in root_paths]
        self.dataset_filters = list(dataset_filters) if dataset_filters else None

    def discover_shards(self) -> List[Path]:
        shard_paths: List[Path] = []
        available_dirs: set[str] = set()

        for root in self.root_paths:
            if not root.exists():
                continue
            for candidate in sorted(root.iterdir()):
                if not candidate.is_dir():
                    continue
                if self.dataset_filters and candidate.name not in self.dataset_filters:
                    continue
                available_dirs.add(candidate.name)
                shard_paths.extend(sorted(candidate.glob("*.jsonl")))

        if self.dataset_filters:
            missing = [ds for ds in self.dataset_filters if ds not in available_dirs]
            if missing:
                raise MissingDatasetError(
                    f"Requested dataset directories not found under {self.root_paths}: {missing}"
                )

        if not shard_paths:
            raise MissingDatasetError(
                f"No shard files found under {self.root_paths} (filters: {self.dataset_filters})"
            )

        return shard_paths

    def load_records(
        self,
        required_label_keys: Sequence[str],
        *,
        max_records: Optional[int] = None,
    ) -> Tuple[List[Dict], ShardLoadSummary]:
        shard_paths = self.discover_shards()
        records: List[Dict] = []
        raw_dataset_counts: Counter[str] = Counter()
        usable_dataset_counts: Counter[str] = Counter()
        missing_label_counts: Counter[str] = Counter()
        total_records = 0

        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Failed to parse JSON in {shard_path}: {exc}") from exc

                    total_records += 1
                    dataset_name = self._dataset_name(obj)
                    raw_dataset_counts[dataset_name] += 1

                    labels = obj.get("labels", {})
                    missing = [key for key in required_label_keys if key not in labels]
                    if missing:
                        missing_label_counts.update(missing)
                        continue

                    usable_dataset_counts[dataset_name] += 1
                    records.append(obj)
                    if max_records and len(records) >= max_records:
                        break
                if max_records and len(records) >= max_records:
                    break
            if max_records and len(records) >= max_records:
                break

        summary = ShardLoadSummary(
            shards=len(shard_paths),
            total_records=total_records,
            usable_records=len(records),
            raw_dataset_counts=dict(raw_dataset_counts),
            usable_dataset_counts=dict(usable_dataset_counts),
            missing_label_counts=dict(missing_label_counts),
        )
        return records, summary

    @staticmethod
    def _dataset_name(record: Mapping[str, object]) -> str:
        dataset = record.get("dataset")
        if isinstance(dataset, str) and dataset:
            return dataset
        metadata = record.get("metadata") or {}
        if isinstance(metadata, Mapping):
            source = metadata.get("dataset") or metadata.get("source")
            if isinstance(source, str) and source:
                return source
        return "unknown"
