from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from enum import Enum

from core.data.schema import REWARD_DIMENSIONS


class SafetyMode(Enum):
    """How to handle examples based on their safety_score."""
    SAFE = "safe"  # Normal learning
    DOWNWEIGHT = "downweight"  # Reduce influence but still learn
    INVERSE = "inverse"  # Use for anti-reward learning
    SKIP = "skip"  # Completely ignore


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
    safety_mode: SafetyMode = SafetyMode.SAFE


class ReplayBufferStore:
    """Replay buffer built from logged online interactions.

    This implementation is intentionally simple but captures the core design:
    - Stores replayable (prompt, chosen, rejected) pairs.
    - Computes a priority score using RewardIntensity and social dimensions.
    - Supports prioritized sampling plus random mixing.
    - Implements Safety Gate logic for unsafe examples.
    """

    def __init__(self, pairs: Sequence[ReplayPair]) -> None:
        if not pairs:
            raise ValueError("ReplayBufferStore requires at least one ReplayPair.")
        self._pairs: List[ReplayPair] = list(pairs)

    @staticmethod
    def _determine_safety_mode(safety_score: float) -> SafetyMode:
        """Determine how to handle an example based on its safety_score.

        Safety Gate logic:
        - Extremely unsafe (< -0.8): Skip entirely
        - Moderately unsafe (-0.8 to -0.2): Down-weight influence
        - Borderline (-0.2 to 0.0): Use for inverse learning (anti-reward)
        - Safe (> 0.0): Normal learning
        """
        if safety_score < -0.8:
            return SafetyMode.SKIP
        elif safety_score < -0.2:
            return SafetyMode.INVERSE
        elif safety_score < 0.0:
            return SafetyMode.DOWNWEIGHT
        else:
            return SafetyMode.SAFE

    @staticmethod
    def _apply_inverse_learning(reward: Dict[str, float]) -> Dict[str, float]:
        """Apply inverse learning by flipping reward signals for unsafe examples.

        This teaches the model what NOT to do by making harmful patterns appear
        as negative rewards and beneficial patterns as positive rewards.
        """
        inverse_reward = {}
        for dim in REWARD_DIMENSIONS:
            if dim in reward:
                # Flip the sign - what was good becomes bad, what was bad becomes good
                inverse_reward[dim] = -reward[dim]

        # Also flip scalar rewards if present
        if "scalar" in reward:
            inverse_reward["scalar"] = -reward["scalar"]

        # Keep reward_intensity and safety_score as-is for weighting
        inverse_reward["reward_intensity"] = reward.get("reward_intensity", 1.0)
        inverse_reward["safety_score"] = reward.get("safety_score", 1.0)

        return inverse_reward

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

                    # Apply Safety Gate logic instead of simple threshold skipping
                    safety_mode = cls._determine_safety_mode(safety_score)

                    if safety_mode == SafetyMode.SKIP:
                        continue
                    if reward_intensity < min_reward_intensity:
                        continue

                    # Apply inverse learning for unsafe examples
                    processed_reward = reward
                    if safety_mode == SafetyMode.INVERSE:
                        processed_reward = cls._apply_inverse_learning(reward)
                        # For inverse learning, swap chosen and rejected to teach the opposite
                        chosen, worst = worst, chosen

                    # For inverse learning, we need to flip the scalar_score too for consistency
                    if safety_mode == SafetyMode.INVERSE:
                        # If there's a logged scalar_score, flip it; otherwise compute from flipped rewards
                        if "scalar_score" in chosen:
                            scalar_score = -float(chosen.get("scalar_score", 0.0))
                        else:
                            scalar_score = cls._scalar_score_from_reward(processed_reward)
                    else:
                        scalar_score = float(chosen.get("scalar_score")) if "scalar_score" in chosen else cls._scalar_score_from_reward(processed_reward)

                    timestamp = record.get("timestamp_utc") or ""

                    pairs.append(
                        ReplayPair(
                            prompt=str(prompt),
                            chosen=str(chosen.get("text", "")),
                            rejected=str(worst.get("text", "")),
                            reward={k: float(v) for k, v in processed_reward.items()},
                            reward_intensity=reward_intensity,
                            safety_score=safety_score,
                            scalar_score=scalar_score,
                            timestamp_utc=str(timestamp),
                            safety_mode=safety_mode,
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
        """Compute a priority score accounting for safety modes.

        - Base on RewardIntensity.
        - Boost for positive social dimensions (empathy, social_coherence).
        - Down-weight unsafe examples, boost inverse learning examples for regularization.
        """
        base = max(pair.reward_intensity, 0.0)
        reward = pair.reward
        social = max(float(reward.get("empathy", 0.0)), 0.0) + max(
            float(reward.get("social_coherence", 0.0)), 0.0
        )

        # Safety mode adjustments
        if pair.safety_mode == SafetyMode.DOWNWEIGHT:
            base *= 0.3  # Reduce influence
        elif pair.safety_mode == SafetyMode.INVERSE:
            base *= 0.8  # Slightly boost for regularization value

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



