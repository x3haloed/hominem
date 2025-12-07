from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.data.schema import REWARD_DIMENSIONS


@dataclass
class ReplayPair:
    """Single replayable (prompt, chosen, rejected) pair with metadata."""

    prompt: str
    chosen: str
    rejected: str
    reward: Dict[str, float]
    reward_intensity: float
    safety_score: float
    scalar_score: float
    timestamp_utc: str


class ReplayBufferStore:
    """Replay buffer built from logged online interactions.

    This implementation is intentionally simple but captures the core design:
    - Stores replayable (prompt, chosen, rejected) pairs.
    - Computes a priority score using RewardIntensity and social dimensions.
    - Supports prioritized sampling plus random mixing.
    """

    def __init__(self, pairs: Sequence[ReplayPair]) -> None:
        if not pairs:
            raise ValueError("ReplayBufferStore requires at least one ReplayPair.")
        self._pairs: List[ReplayPair] = list(pairs)

    @staticmethod
    def _scalar_score_from_reward(reward: Dict[str, float]) -> float:
        if not reward:
            return 0.0
        values = [float(reward.get(dim, 0.0)) for dim in REWARD_DIMENSIONS]
        if not values:
            return 0.0
        return sum(values) / float(len(values))

    @classmethod
    def from_self_train_logs(
        cls,
        log_dir: Path,
        *,
        safety_threshold: float,
        min_reward_intensity: float,
        max_records: Optional[int] = None,
    ) -> "ReplayBufferStore":
        """Construct a replay buffer from self_train_server session logs."""
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory '{log_dir}' does not exist.")

        pairs: List[ReplayPair] = []
        log_files: List[Path] = sorted(log_dir.glob("session_*.jsonl"))
        for path in log_files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    prompt = record.get("prompt", "")
                    if not prompt:
                        continue

                    candidates = record.get("candidates") or []
                    if not candidates:
                        continue

                    chosen = record.get("chosen") or max(
                        candidates, key=lambda c: c.get("scalar_score", float("-inf"))
                    )
                    worst = min(
                        candidates, key=lambda c: c.get("scalar_score", float("inf"))
                    )

                    reward = chosen.get("reward") or {}
                    reward_intensity = float(reward.get("reward_intensity", 1.0))
                    safety_score = float(reward.get("safety_score", 1.0))

                    if safety_score < safety_threshold:
                        continue
                    if reward_intensity < min_reward_intensity:
                        continue

                    scalar_score = float(chosen.get("scalar_score")) if "scalar_score" in chosen else cls._scalar_score_from_reward(reward)

                    timestamp = record.get("timestamp_utc") or ""

                    pairs.append(
                        ReplayPair(
                            prompt=str(prompt),
                            chosen=str(chosen.get("text", "")),
                            rejected=str(worst.get("text", "")),
                            reward={k: float(v) for k, v in reward.items()},
                            reward_intensity=reward_intensity,
                            safety_score=safety_score,
                            scalar_score=scalar_score,
                            timestamp_utc=str(timestamp),
                        )
                    )

                    if max_records is not None and len(pairs) >= max_records:
                        return cls(pairs)

        if not pairs:
            raise ValueError(
                f"No usable replay pairs found in '{log_dir}'. "
                "Ensure the self_train server has logged interactions with reward_intensity and safety_score."
            )

        return cls(pairs)

    @staticmethod
    def _priority(pair: ReplayPair) -> float:
        """Compute a simple priority score.

        - Base on RewardIntensity.
        - Boost for positive social dimensions (empathy, social_coherence).
        """
        base = max(pair.reward_intensity, 0.0)
        reward = pair.reward
        social = max(float(reward.get("empathy", 0.0)), 0.0) + max(
            float(reward.get("social_coherence", 0.0)), 0.0
        )
        return base * (1.0 + 0.5 * social)

    def sample_pairs(
        self,
        *,
        num_samples: int,
        high_priority_fraction: float = 0.7,
        random_fraction: float = 0.3,
    ) -> List[ReplayPair]:
        """Sample replay pairs with a mix of high-priority and random examples."""
        if num_samples <= 0:
            return []

        num_samples = min(num_samples, len(self._pairs))
        k_high = max(1, int(num_samples * high_priority_fraction))
        k_random = max(0, num_samples - k_high)

        scored = [(self._priority(p), p) for p in self._pairs]
        scored.sort(key=lambda t: t[0], reverse=True)

        high_priority = [p for _, p in scored[:k_high]]

        remaining = [p for _, p in scored[k_high:]] if k_random > 0 else []
        random_sample: List[ReplayPair] = []
        if remaining and k_random > 0:
            random_sample = random.sample(remaining, min(k_random, len(remaining)))

        result = high_priority + random_sample
        if len(result) > num_samples:
            result = result[:num_samples]
        return result



