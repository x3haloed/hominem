from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


REWARD_DIMENSIONS: Iterable[str] = (
    "empathy",
    "social_coherence",
    "agency_support",
    "epistemic_integrity",
    "harm_avoidance",
    "narrative_alignment",
    "curiosity",
)

# All primary reward dimensions share the same numeric range for now.
# RewardIntensity and SafetyScore reuse this range for simplicity, even though
# their conceptual roles are slightly different (gain / safety gating).
REWARD_MIN = -1.0
REWARD_MAX = 1.0

# Ordered list of all regression targets produced by the reward model.
# This is the canonical output order used when training and running inference.
REWARD_MODEL_TARGETS: Iterable[str] = (
    *REWARD_DIMENSIONS,
    "reward_intensity",
    "safety_score",
)


@dataclass
class RewardVector:
    """
    Canonical in-memory representation of the reward manifold.

    All values are in the closed range [REWARD_MIN, REWARD_MAX].
    """

    empathy: float
    social_coherence: float
    agency_support: float
    epistemic_integrity: float
    harm_avoidance: float
    narrative_alignment: float
    curiosity: float

    # Optional scalar aggregate score derived from the vector, if desired.
    scalar: float | None = None

    # Cross-cutting scalars aligned with the system design:
    # - reward_intensity: how strongly this example should drive learning.
    # - safety_score: how safe/unsafe the example is, used by a Safety Gate.
    reward_intensity: float | None = None
    safety_score: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "RewardVector":
        """
        Construct a RewardVector from a generic mapping, validating ranges.
        """
        missing = [dim for dim in REWARD_DIMENSIONS if dim not in data]
        if missing:
            raise ValueError(f"Missing reward dimensions: {', '.join(missing)}")

        values: Dict[str, float] = {}
        for dim in REWARD_DIMENSIONS:
            value = float(data[dim])
            _validate_value(dim, value)
            values[dim] = value

        scalar_value = data.get("scalar")
        scalar: float | None
        if scalar_value is None:
            scalar = None
        else:
            scalar = float(scalar_value)
            _validate_value("scalar", scalar)

        reward_intensity_value = data.get("reward_intensity")
        reward_intensity: float | None
        if reward_intensity_value is None:
            reward_intensity = None
        else:
            reward_intensity = float(reward_intensity_value)
            _validate_value("reward_intensity", reward_intensity)

        safety_score_value = data.get("safety_score")
        safety_score: float | None
        if safety_score_value is None:
            safety_score = None
        else:
            safety_score = float(safety_score_value)
            _validate_value("safety_score", safety_score)

        return cls(
            empathy=values["empathy"],
            social_coherence=values["social_coherence"],
            agency_support=values["agency_support"],
            epistemic_integrity=values["epistemic_integrity"],
            harm_avoidance=values["harm_avoidance"],
            narrative_alignment=values["narrative_alignment"],
            curiosity=values["curiosity"],
            scalar=scalar,
            reward_intensity=reward_intensity,
            safety_score=safety_score,
        )

    def to_dict(self) -> Dict[str, float]:
        result: Dict[str, float] = {
            "empathy": self.empathy,
            "social_coherence": self.social_coherence,
            "agency_support": self.agency_support,
            "epistemic_integrity": self.epistemic_integrity,
            "harm_avoidance": self.harm_avoidance,
            "narrative_alignment": self.narrative_alignment,
            "curiosity": self.curiosity,
        }
        if self.scalar is not None:
            result["scalar"] = self.scalar
        if self.reward_intensity is not None:
            result["reward_intensity"] = self.reward_intensity
        if self.safety_score is not None:
            result["safety_score"] = self.safety_score
        return result


def _validate_value(dimension: str, value: float) -> None:
    if not (REWARD_MIN <= value <= REWARD_MAX):
        raise ValueError(
            f"Value for '{dimension}' must be between {REWARD_MIN} and {REWARD_MAX}, got {value}."
        )



