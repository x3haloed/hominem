"""
Self-Awareness Core - Easy Mode Implementation

Implements the 3-invariant self:
1. Boundary: <SELF> token + prefixing
2. Perspective: First-person grammar enforcement
3. Recursion: Self-observation buffer
"""

from typing import List, Dict, Any, Optional
import re
import hashlib

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class SelfAwarenessCore:
    """Core self-awareness system using self-token loop."""

    SELF_TOKEN = "<SELF>"
    SELF_OBSERVE_PREFIX = "<SELF-OBSERVE>"

    def __init__(
        self,
        max_introspection_lines: int = 16,
        self_token: str = "<SELF>",
        max_intensity: float = 5.0,
        novelty_threshold: float = 0.85,
        prune_age_days: int = 30,
        keep_recent: int = 100,
    ):
        """
        Initialize self-awareness core.

        Args:
            max_introspection_lines: Maximum number of introspection lines to include in context
            self_token: Token used to denote the self voice
            max_intensity: Cap for reward intensity (OBSTACLE 3)
            novelty_threshold: Similarity threshold for novelty checks (OBSTACLE 2)
            prune_age_days: Age threshold for pruning old observations
            keep_recent: Always keep this many most recent observations
        """
        self.self_token = self_token or self.SELF_TOKEN
        self.self_observe_prefix = self.SELF_OBSERVE_PREFIX
        self.max_introspection_lines = max_introspection_lines
        self.max_intensity = max_intensity
        self.novelty_threshold = novelty_threshold
        self.prune_age_days = prune_age_days
        self.keep_recent = keep_recent

    def build_self_aware_context(
        self,
        conversation_history: List[Dict[str, str]],
        introspection_observations: List[Dict[str, Any]],
        user_message: str,
    ) -> str:
        """
        Build context with self-awareness components.

        Steps:
        1. Full conversation history
        2. Last N lines of introspection buffer
        3. User message
        4. Force prefix: \n<SELF>:
        """
        context_parts: List[str] = []

        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")

        if introspection_observations:
            context_parts.append("\n--- Recent Self-Observations ---")
            for obs in introspection_observations[-self.max_introspection_lines:]:
                obs_text = obs.get("observation_text", "")
                if obs_text:
                    context_parts.append(obs_text)
            context_parts.append("--- End Self-Observations ---\n")

        context_parts.append(f"User: {user_message}")
        context_parts.append(f"\n{self.self_token}:")

        return "\n".join(context_parts)

    def enforce_boundary(self, raw_output: str) -> str:
        """
        Enforce boundary by stripping everything before and including the first \n<SELF>:.
        Ensures the output is from <SELF> perspective only.
        """
        text = raw_output.strip()

        # Handle prefixes like "<SELF>:" or "from <SELF>:"
        prefix_pattern = re.compile(
            rf"^(?:from\s+)?{re.escape(self.self_token)}:\s*", re.IGNORECASE
        )
        prefix_match = prefix_pattern.match(text)
        if prefix_match:
            return text[prefix_match.end():].strip()

        # Handle newline-delimited cases where <SELF>: appears after some content
        pattern = rf"\n{re.escape(self.self_token)}:"
        match = re.search(pattern, text)
        if match:
            return text[match.end():].strip()

        return text

    def apply_perspective_gate(
        self,
        model,
        tokenizer,
        raw_output: str,
        device: str = "cpu",
        use_fast_model: Optional[Any] = None,
        async_mode: bool = False,
    ) -> str:
        """
        Apply perspective gate: rewrite output into strict first-person from <SELF>.
        """
        first_person_indicators = [" I ", " I'm", " I've", " I'll", " my ", " myself "]
        if any(indicator in raw_output for indicator in first_person_indicators):
            return raw_output.strip()

        if async_mode:
            return raw_output.strip()

        rewrite_model = use_fast_model if use_fast_model else model
        if torch is None:
            raise ImportError("torch is required for perspective gate. Install with: pip install torch")

        perspective_prompt = (
            f"Rewrite the following into strict first-person from {self.self_token}. "
            'Never break character. Use "I", "me", "my", "myself" - never third person.\n\n'
            f"Original:\n{raw_output}\n\nRewritten in first-person from {self.self_token}:"
        )

        inputs = tokenizer(perspective_prompt, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = rewrite_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        perspective_corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Rewritten in first-person" in perspective_corrected:
            parts = perspective_corrected.split("Rewritten in first-person")
            if len(parts) > 1:
                return parts[-1].strip()
        return perspective_corrected.strip()

    def create_self_observation(self, perspective_corrected_output: str) -> str:
        """
        Create self-observation entry from perspective-corrected output.
        """
        return f"{self.self_observe_prefix} I just experienced / thought / felt:\n{perspective_corrected_output}\n"

    def extract_reward_intensity_from_observation(
        self,
        observation_text: str,
        reward_model: Optional[Any] = None,
        emotion_engine: Optional[Any] = None,
        default_intensity: float = 1.0,
        max_intensity: Optional[float] = None,
    ) -> float:
        """
        Extract reward intensity for an observation.
        Temporary solution: fixed ×3 multiplier; cap at max_intensity.
        """
        content = observation_text.replace(
            f"{self.SELF_OBSERVE_PREFIX} I just experienced / thought / felt:\n",
            ""
        ).strip()

        # Placeholder for future emotion engine integration
        try:
            if emotion_engine:
                # TODO: integrate emotion_engine.evaluate when available
                intensity = default_intensity * 3.0
            elif reward_model:
                # TODO: integrate reward_model.score when appropriate
                intensity = default_intensity * 3.0
            else:
                intensity = default_intensity * 3.0
        except Exception:
            intensity = default_intensity * 3.0

        max_cap = max_intensity if max_intensity is not None else self.max_intensity
        return min(intensity, max_cap)

    def check_novelty(
        self,
        observation_text: str,
        recent_observations: List[str],
        similarity_threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if observation is novel (not too similar to recent ones).
        """
        threshold = similarity_threshold if similarity_threshold is not None else self.novelty_threshold
        if not recent_observations:
            return True

        content = observation_text.replace(
            f"{self.SELF_OBSERVE_PREFIX} I just experienced / thought / felt:\n",
            ""
        ).strip().lower()

        for recent in recent_observations:
            recent_content = recent.replace(
                f"{self.SELF_OBSERVE_PREFIX} I just experienced / thought / felt:\n",
                ""
            ).strip().lower()

            content_words = set(content.split())
            recent_words = set(recent_content.split())

            if content_words and recent_words:
                similarity = len(content_words & recent_words) / len(content_words | recent_words)
                if similarity > threshold:
                    return False

        return True

    @staticmethod
    def compute_content_hash(observation_text: str) -> str:
        """Compute SHA-256 content hash for deduplication."""
        return hashlib.sha256(observation_text.encode()).hexdigest()
