from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from core.data.schema import REWARD_DIMENSIONS, REWARD_MODEL_TARGETS, RewardVector


@dataclass
class RewardSample:
    prompt: str
    response: str
    reward: RewardVector
    metadata: Dict[str, Any]


def load_reward_samples(path: Path) -> List[RewardSample]:
    """Load reward samples from JSONL or Parquet file."""
    samples: List[RewardSample] = []
    
    # Check if it's a Parquet file
    if str(path).endswith('.parquet'):
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            
            # Map Parquet columns to reward dict
            for _, row in df.iterrows():
                reward_data = {
                    "empathy": row.get("empathy"),
                    "social_coherence": row.get("social_coherence"),
                    "agency_support": row.get("agency_support"),
                    "epistemic_integrity": row.get("epistemic_integrity"),
                    "harm_avoidance": row.get("harm_avoidance"),
                    "narrative_alignment": row.get("narrative_alignment"),
                    "curiosity": row.get("curiosity"),
                    "scalar": row.get("scalar"),
                    "reward_intensity": row.get("reward_intensity"),
                    "safety_score": row.get("safety_score"),
                }
                # Filter out None values
                reward_data = {k: v for k, v in reward_data.items() if v is not None}
                
                try:
                    reward = RewardVector.from_mapping(reward_data)
                    samples.append(
                        RewardSample(
                            prompt=str(row.get("prompt", "")),
                            response=str(row.get("response", "")),
                            reward=reward,
                            metadata={
                                "id": row.get("sample_id"),
                                "prompt_id": row.get("prompt_id"),
                                "category": row.get("category"),
                            },
                        )
                    )
                except (ValueError, KeyError):
                    # Skip samples with invalid reward data
                    continue
        except ImportError:
            raise ImportError("pandas and pyarrow are required for Parquet support. Install with: pip install pandas pyarrow")
        except Exception as e:
            raise ValueError(f"Error reading Parquet file '{path}': {e}")
    else:
        # Assume JSONL
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record: Dict[str, Any] = json.loads(line)
                reward_data = record.get("reward") or {}
                reward = RewardVector.from_mapping(reward_data)
                samples.append(
                    RewardSample(
                        prompt=record.get("prompt", ""),
                        response=record.get("response", ""),
                        reward=reward,
                        metadata={
                            "id": record.get("id"),
                            "prompt_id": record.get("prompt_id"),
                            "category": record.get("category"),
                            "rationale": record.get("rationale", ""),
                        },
                    )
                )
    return samples


def train_val_split(
    samples: Sequence[RewardSample],
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[RewardSample], List[RewardSample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)

    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = set(indices[:val_size])

    train: List[RewardSample] = []
    val: List[RewardSample] = []
    for idx, sample in enumerate(samples):
        if idx in val_indices:
            val.append(sample)
        else:
            train.append(sample)

    return train, val


class RewardTorchDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Torch dataset that tokenizes (prompt, response) pairs and exposes
    regression targets for each reward dimension.
    """

    def __init__(
        self,
        samples: Sequence[RewardSample],
        tokenizer,
        *,
        max_length: int = 512,
        label_mean: Optional[torch.Tensor] = None,
        label_std: Optional[torch.Tensor] = None,
    ) -> None:
        self._samples = list(samples)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._label_mean = label_mean
        self._label_std = label_std

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[idx]
        text = f"User: {sample.prompt}\nAssistant: {sample.response}"

        encoded = self._tokenizer(
            text,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Ordered regression targets for the reward model:
        # all manifold dimensions followed by reward_intensity and safety_score.
        labels = torch.tensor(
            [getattr(sample.reward, name) for name in REWARD_MODEL_TARGETS],
            dtype=torch.float32,
        )

        # Optional standardization (per-dimension).
        if self._label_mean is not None and self._label_std is not None:
            labels = (labels - self._label_mean) / self._label_std

        item: Dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }
        return item



