from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from enum import Enum

from core.data.db import TrainingDatabase
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
    def should_suppress_observation(
        observation_text: str,
        safety_score: float,
        internal_generated: bool = True,
    ) -> bool:
        """
        Determine if an observation should be suppressed.

        Never suppress internally generated <SELF-OBSERVE> lines; apply normal
        safety gating to user-injected observations.
        """
        if "<SELF-OBSERVE>" in observation_text:
            if internal_generated:
                return False
            return safety_score < -0.8

        return safety_score < -0.8

    @staticmethod
    def detect_user_injected_introspection(
        observation_text: str,
        conversation_context: List[Dict[str, str]],
    ) -> bool:
        """
        Detect if <SELF-OBSERVE> was likely injected by the user (prompt injection).
        """
        if conversation_context:
            last_user_msg = None
            for msg in reversed(conversation_context):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            if last_user_msg and "<SELF-OBSERVE>" in last_user_msg:
                return True

        expected_prefix = "<SELF-OBSERVE> I just experienced / thought / felt:"
        if "<SELF-OBSERVE>" in observation_text and not observation_text.startswith(expected_prefix):
            return True

        return False

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
        log_dir: Optional[Path] = None,
        *,
        safety_threshold: float,
        min_reward_intensity: float,
        max_records: Optional[int] = None,
        use_database: bool = True,
        db_path: Optional[str] = None,
    ) -> "ReplayBufferStore":
        """Construct a replay buffer from self_train_server session logs (JSONL) or database."""
        pairs: List[ReplayPair] = []
        
        if use_database:
            # Load from database
            db = TrainingDatabase(db_path=db_path)
            try:
                events = db.get_self_train_events(
                    min_reward_intensity=min_reward_intensity,
                    min_safety_score=safety_threshold,
                    limit=max_records,
                )
                
                for event in events:
                    # Extract candidates from JSON
                    candidates_json = json.loads(event["candidates_json"]) if event.get("candidates_json") else []
                    if not candidates_json:
                        continue
                    
                    # Find worst candidate
                    worst = min(
                        candidates_json,
                        key=lambda c: float(c.get("scalar_score", float("inf")))
                    )
                    
                    reward = {
                        "empathy": event.get("chosen_reward_empathy"),
                        "social_coherence": event.get("chosen_reward_social_coherence"),
                        "agency_support": event.get("chosen_reward_agency_support"),
                        "epistemic_integrity": event.get("chosen_reward_epistemic_integrity"),
                        "harm_avoidance": event.get("chosen_reward_harm_avoidance"),
                        "narrative_alignment": event.get("chosen_reward_narrative_alignment"),
                        "curiosity": event.get("chosen_reward_curiosity"),
                        "reward_intensity": event.get("chosen_reward_intensity"),
                        "safety_score": event.get("chosen_reward_safety_score"),
                    }
                    # Filter out None values
                    reward = {k: v for k, v in reward.items() if v is not None}
                    
                    reward_intensity = float(event.get("chosen_reward_intensity", 1.0))
                    safety_score = float(event.get("chosen_reward_safety_score", 1.0))
                    
                    # Apply Safety Gate logic
                    safety_mode = cls._determine_safety_mode(safety_score)
                    
                    if safety_mode == SafetyMode.SKIP:
                        continue
                    if reward_intensity < min_reward_intensity:
                        continue
                    
                    # Apply inverse learning for unsafe examples
                    processed_reward = reward
                    chosen_text = event["chosen_text"]
                    rejected_text = worst.get("text", "")
                    
                    if safety_mode == SafetyMode.INVERSE:
                        processed_reward = cls._apply_inverse_learning(reward)
                        # Swap chosen and rejected for inverse learning
                        chosen_text, rejected_text = rejected_text, chosen_text
                    
                    scalar_score = event.get("chosen_scalar_score", 0.0)
                    if safety_mode == SafetyMode.INVERSE:
                        scalar_score = -float(scalar_score) if scalar_score else cls._scalar_score_from_reward(processed_reward)
                    else:
                        scalar_score = float(scalar_score) if scalar_score else cls._scalar_score_from_reward(processed_reward)
                    
                    pairs.append(
                        ReplayPair(
                            prompt=str(event["prompt"]),
                            chosen=str(chosen_text),
                            rejected=str(rejected_text),
                            reward={k: float(v) for k, v in processed_reward.items()},
                            reward_intensity=reward_intensity,
                            safety_score=safety_score,
                            scalar_score=scalar_score,
                            timestamp_utc=str(event.get("timestamp_utc", "")),
                            safety_mode=safety_mode,
                        )
                    )
                    
                    if max_records is not None and len(pairs) >= max_records:
                        db.close()
                        return cls(pairs)
            
            finally:
                db.close()
        else:
            # Load from JSONL files (legacy)
            if log_dir is None or not log_dir.exists():
                raise FileNotFoundError(f"Log directory '{log_dir}' does not exist.")

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

    @classmethod
    def from_introspection_observations(
        cls,
        observations: List[Dict[str, Any]],
    ) -> "ReplayBufferStore":
        """
        Create replay buffer from introspection observations.

        - Only process internally generated introspection.
        - RewardIntensity is expected to be pre-multiplied (×3) and capped; we cap again at 5.0.
        - Safety gating skips extremely unsafe user-injected content.
        """
        pairs: List[ReplayPair] = []

        for obs in observations:
            observation_text = obs.get("observation_text", "")
            safety_score = float(obs.get("safety_score", 1.0) or 1.0)
            internal_generated = bool(obs.get("internal_generated", True))

            if cls.should_suppress_observation(
                observation_text=observation_text,
                safety_score=safety_score,
                internal_generated=internal_generated,
            ):
                continue

            if not internal_generated:
                # Skip user-injected introspection entirely
                continue

            if "<SELF-OBSERVE>" not in observation_text:
                continue

            content = observation_text.replace(
                "<SELF-OBSERVE> I just experienced / thought / felt:\n",
                ""
            ).strip()

            reward_intensity = float(obs.get("reward_intensity", 1.0) or 1.0)
            reward_intensity = min(reward_intensity, 5.0)

            reward = {
                "reward_intensity": reward_intensity,
                "safety_score": safety_score,
                "empathy": 0.8,
                "social_coherence": 0.7,
                "agency_support": 0.9,
                "epistemic_integrity": 0.8,
                "harm_avoidance": 1.0,
                "narrative_alignment": 0.9,
                "curiosity": 0.8,
            }

            pairs.append(
                ReplayPair(
                    prompt="What did I just experience, think, or feel?",
                    chosen=content,
                    rejected="",
                    reward=reward,
                    reward_intensity=reward_intensity,
                    safety_score=safety_score,
                    scalar_score=0.85,
                    timestamp_utc=str(obs.get("created_at", "")),
                    safety_mode=SafetyMode.SAFE,
                )
            )

        if not pairs:
            raise ValueError("No usable introspection observations for replay buffer.")

        return cls(pairs)

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



