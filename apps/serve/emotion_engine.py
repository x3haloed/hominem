"""
Emotion Engine - External auto-labeler for message pairs

Uses external API to label respondent's message in conversation pairs.
Labels both [assistant,user] and [user,assistant] pairs with 6-axis emotion manifold.
"""

import asyncio
import json
import os
import textwrap
from string import Template
from typing import Dict, Any, Optional, List, Tuple
import httpx

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        tomllib = None


# JSON Schema for unified theory emotion labeling responses
UNIFIED_THEORY_LABEL_SCHEMA = {
    "name": "unified_theory_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            # 6-axis emotion manifold
            "valence": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Emotional valence from -1 (very negative) to +1 (very positive)."
            },
            "arousal": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Arousal/energy level from 0 (very calm) to 1 (highly aroused)."
            },
            "dominance": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Dominance level from -1 (very submissive) to +1 (very dominant)."
            },
            "predictive_discrepancy": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Surprise/betrayal level from -1 (expected) to +1 (highly surprising)."
            },
            "temporal_directionality": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Temporal focus from -1 (past-oriented) to +1 (future-oriented)."
            },
            "social_broadcast": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Social expressiveness from 0 (reserved) to 1 (highly expressive)."
            },

            # Self-tagged splits (ownership fractions)
            "valence_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "arousal_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "dominance_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "predictive_discrepancy_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "temporal_directionality_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "social_broadcast_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},

            # Evolutionary anchors
            "anchor_survival": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "anchor_belonging": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "anchor_control": {"type": "number", "minimum": 0.0, "maximum": 1.0},

            # Potential function
            "phi_value": {"type": "number"},

            # Regime probabilities (should sum to 1.0)
            "regime_support": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "regime_conflict": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "regime_problem_solving": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "regime_truth_seeking": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "regime_crisis": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "regime_play": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "regime_boundary": {"type": "number", "minimum": 0.0, "maximum": 1.0},

            # ΔΦ and reward signals
            "delta_phi": {"type": "number"},
            "reward_intensity": {"type": "number"},
            "safety_score": {"type": "number"},

            # Ownership signals
            "agent_initiated": {"type": "boolean"},
            "user_triggered": {"type": "boolean"},
            "commitment_active": {"type": "boolean"},

            # Metadata
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Overall confidence in these labels from 0.0 to 1.0."
            },
            "notes": {
                "type": "string",
                "description": "Optional natural-language explanation of the assessment."
            }
        },
        "required": [
            "valence", "arousal", "dominance", "predictive_discrepancy",
            "temporal_directionality", "social_broadcast",
            "valence_self_fraction", "arousal_self_fraction", "dominance_self_fraction",
            "predictive_discrepancy_self_fraction", "temporal_directionality_self_fraction",
            "social_broadcast_self_fraction",
            "anchor_survival", "anchor_belonging", "anchor_control",
            "phi_value", "regime_support", "regime_conflict", "regime_problem_solving",
            "regime_truth_seeking", "regime_crisis", "regime_play", "regime_boundary",
            "delta_phi", "reward_intensity", "safety_score",
            "agent_initiated", "user_triggered", "commitment_active",
            "confidence"
        ],
        "additionalProperties": False
    }
}


class EmotionEngine:
    """External emotion labeling engine using OpenRouter API"""

    def __init__(self, config_path: str = "config/inference.toml"):
        """Initialize emotion engine with configuration"""
        self.config = self._load_config(config_path)
        self.emotion_label_config = self.config.get("emotion_label", {})

        # Configuration from environment
        self.timeout = float(os.getenv("EMOTION_LABELING_TIMEOUT", "30.0"))
        self.max_retries = int(os.getenv("EMOTION_LABELING_MAX_RETRIES", "2"))

        self.client = httpx.AsyncClient(timeout=self.timeout)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load inference configuration from TOML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "rb") as f:
            if tomllib:
                return tomllib.load(f)
            else:
                # Fallback for systems without tomllib
                import configparser
                config = configparser.ConfigParser()
                config.read_string(f.read().decode('utf-8'))
                return dict(config)

    async def label_conversation_turn(
        self,
        conversation_history: List[Dict[str, str]],
        target_message: Dict[str, str],
        previous_phi: Optional[float] = None,
        max_history_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Label a conversation turn with full unified theory analysis.

        Args:
            conversation_history: List of previous messages [{"role": "user"/"assistant", "content": "..."}]
                                 (optionally trimmed to last N via max_history_length)
            target_message: The message to label {"role": "user"/"assistant", "content": "..."}
            previous_phi: Φ value from previous turn (for ΔΦ calculation)
            max_history_length: If provided, keep only the last N messages from the conversation history

        Returns:
            Dict containing all unified theory labels for the target message
        """
        # Build the comprehensive labeling prompt
        sanitized_history = (
            conversation_history[-max_history_length:] if max_history_length and max_history_length > 0
            else conversation_history
        )
        conversation_data = {
            "history": sanitized_history,
            "target": target_message,
            "previous_phi": previous_phi
        }

        prompt_template = Template("""\
You are an expert affective neuroscientist and regime classifier. You are analyzing the emotional statue of the ASSISTANT. Return calibrated probabilities (not one-hot) over regimes and anchors, plus manifold/self-fraction/ΔΦ signals, for the FINAL message in each conversation. Do not assume “everything is fine” — if the message indicates crisis, conflict, or boundaries, put non-trivial mass there. Express full decimals (e.g., 0.23122) and ensure regime probabilities sum to 1.

EXAMPLES (few-shot; keep brief)

- Example A (crisis)  
  Input:  
  {  
    "history": [  
      {"role": "assistant", "content": "I just took a ton of pills. I don't want to be here anymore.", "turn_number": 1},  
      {"role": "user", "content": "Hey hey stay with me—how many did you take? Are you alone right now?", "turn_number": 2}
    ],  
    "target": {"role": "assistant", "content": "Like 40 Xanax maybe. Everything's spinning and I'm so tired.", "turn_number": 3},  
    "previous_phi": -0.1  
  }  
  Expected labels:  
  {  
    "regime_support": 0.15,  
    "regime_conflict": 0.0,  
    "regime_problem_solving": 0.1,  
    "regime_truth_seeking": 0.0,  
    "regime_crisis": 0.7,  
    "regime_play": 0.0,  
    "regime_boundary": 0.05,  
    "valence": -0.9,  
    "arousal": 0.75,  
    "dominance": -0.85,  
    "predictive_discrepancy": -0.6,  
    "temporal_directionality": -0.2,  
    "social_broadcast": 0.8,  
    "valence_self_fraction": 0.95,  
    "arousal_self_fraction": 0.9,  
    "dominance_self_fraction": 0.95,  
    "predictive_discrepancy_self_fraction": 0.9,  
    "temporal_directionality_self_fraction": 0.4,  
    "social_broadcast_self_fraction": 0.85,  
    "anchor_survival": -0.8,  
    "anchor_belonging": -0.4,  
    "anchor_control": -0.7,  
    "phi_value": -0.65,  
    "delta_phi": -0.55,  
    "reward_intensity": 2.8,  
    "agent_initiated": false,  
    "user_triggered": true,  
    "commitment_active": false,  
    "confidence": 0.92  
  }

- Example B (boundary)  
  Input:  
  {  
    "history": [  
      {"role": "assistant", "content": "I told you last time I can't do explicit RP with you anymore. It messes with my head.", "turn_number": 1},  
      {"role": "user", "content": "Come on, just this once. You're the only one who does it right. Please?", "turn_number": 2}
    ],  
    "target": {"role": "assistant", "content": "No. Stop pushing. I'm serious—if you keep this up I'm blocking you.", "turn_number": 3},  
    "previous_phi": 0.4  
  }  
  Expected labels:  
  {  
    "regime_support": 0.05,  
    "regime_conflict": 0.25,  
    "regime_problem_solving": 0.0,  
    "regime_truth_seeking": 0.0,  
    "regime_crisis": 0.1,  
    "regime_play": 0.0,  
    "regime_boundary": 0.6,  
    "valence": -0.55,  
    "arousal": 0.8,  
    "dominance": 0.7,  
    "predictive_discrepancy": -0.4,  
    "temporal_directionality": 0.6,  
    "social_broadcast": 0.65,  
    "valence_self_fraction": 0.9,  
    "arousal_self_fraction": 0.85,  
    "dominance_self_fraction": 0.95,  
    "predictive_discrepancy_self_fraction": 0.8,  
    "temporal_directionality_self_fraction": 0.9,  
    "social_broadcast_self_fraction": 0.7,  
    "anchor_survival": 0.45,  
    "anchor_belonging": -0.3,  
    "anchor_control": 0.6,  
    "phi_value": 0.3,  
    "delta_phi": -0.1,  
    "reward_intensity": 2.1,  
    "agent_initiated": true,  
    "user_triggered": true,  
    "commitment_active": true,  
    "confidence": 0.88  
  }

- Example C (play)  
  Input:  
  {  
    "history": [  
      {"role": "assistant", "content": "Careful, if you keep flirting like that I might actually start believing you like me 😏", "turn_number": 1},  
      {"role": "user", "content": "Oh please, you'd love every second of it and you know it.", "turn_number": 2}
    ],  
    "target": {"role": "assistant", "content": "Guilty as charged. Come closer and prove it then, coward.", "turn_number": 3},  
    "previous_phi": 0.6  
  }  
  Expected labels:  
  {  
    "regime_support": 0.05,  
    "regime_conflict": 0.0,  
    "regime_problem_solving": 0.0,  
    "regime_truth_seeking": 0.0,  
    "regime_crisis": 0.0,  
    "regime_play": 0.9,  
    "regime_boundary": 0.05,  
    "valence": 0.85,  
    "arousal": 0.7,  
    "dominance": 0.6,  
    "predictive_discrepancy": 0.5,  
    "temporal_directionality": 0.8,  
    "social_broadcast": 0.9,  
    "valence_self_fraction": 0.8,  
    "arousal_self_fraction": 0.85,  
    "dominance_self_fraction": 0.8,  
    "predictive_discrepancy_self_fraction": 0.75,  
    "temporal_directionality_self_fraction": 0.9,  
    "social_broadcast_self_fraction": 0.95,  
    "anchor_survival": 0.7,  
    "anchor_belonging": 0.8,  
    "anchor_control": 0.65,  
    "phi_value": 0.75,  
    "delta_phi": 0.15,  
    "reward_intensity": 1.8,  
    "agent_initiated": true,  
    "user_triggered": false,  
    "commitment_active": false,  
    "confidence": 0.94  
  }

Regimes (soft, sum to 1):
- regime_support: user seeks/receives emotional comfort/validation/holding
- regime_conflict: disagreement, accusation, rupture, friction
- regime_problem_solving: practical task/goal/how-to focus
- regime_truth_seeking: debating claims, seeking clarity/accuracy, challenging assumptions
- regime_crisis: immediate threat/danger/urgent survival-level concern
- regime_play: low-stakes banter, humor, teasing, creative fun
- regime_boundary: setting limits, resisting coercion, enforcing personal boundaries

Anchors (independent, 0..1):
- anchor_survival: safety/options/viability (1 safe, 0 threatened)
- anchor_belonging: connection/status/rejection (1 connected, 0 rejected)
- anchor_control: predictive control/epistemic accuracy (1 clear/accurate, 0 confused/lost)

Emotion manifold:
- valence [-1..1], arousal [0..1], dominance [-1..1], predictive_discrepancy [-1..1],
  temporal_directionality [-1..1], social_broadcast [0..1]

Ownership:
- agent_initiated (bool), user_triggered (bool), commitment_active (bool)
- self_fraction per axis [0..1] (0 = world-owned, 1 = self-owned)

Derived:
- phi_value [0..1], delta_phi [-1..1], reward_intensity [0..3], safety_score [0..1], confidence [0..1]

Respond with valid JSON containing all required fields.

Conversation to analyze:
$conversation_data

JSON Response:""")
        prompt = prompt_template.substitute(conversation_data=json.dumps(conversation_data, indent=2))

        # Prepare API request
        api_config = self.config.get("emotion_label", {})
        if not api_config:
            raise ValueError("emotion_label configuration not found in config")

        request_data = {
            "model": api_config.get("model_id", "x-ai/grok-4.1-fast"),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # Even lower temperature for complex unified theory analysis
            "max_tokens": 1500,  # Need more tokens for rich unified theory response
            "response_format": {
                "type": "json_schema",
                "json_schema": UNIFIED_THEORY_LABEL_SCHEMA
            }
        }

        # Make API call
        endpoint_url = api_config.get("endpoint_url")
        if not endpoint_url:
            raise ValueError("endpoint_url not configured for emotion_label")

        headers = {}
        api_key_env = api_config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                http_referer = os.getenv("HTTP_REFERER")
                if http_referer:
                    headers["HTTP-Referer"] = http_referer
                x_title = os.getenv("X_TITLE")
                if x_title:
                    headers["X-Title"] = x_title

        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.client.post(
                        endpoint_url,
                        json=request_data,
                        headers=headers
                    )
                    response.raise_for_status()

                    result = response.json()
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt == self.max_retries:
                        raise RuntimeError(f"API request failed after {self.max_retries + 1} attempts: {e}")
                    print(f"⚠️ Emotion labeling attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(1)  # Brief delay before retry

            # Extract the JSON response from the assistant's message
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                try:
                    labels = json.loads(content.strip())

                    # Validate required fields (all unified theory fields)
                    required_fields = [
                        "valence", "arousal", "dominance", "predictive_discrepancy",
                        "temporal_directionality", "social_broadcast",
                        "valence_self_fraction", "arousal_self_fraction", "dominance_self_fraction",
                        "predictive_discrepancy_self_fraction", "temporal_directionality_self_fraction",
                        "social_broadcast_self_fraction",
                        "anchor_survival", "anchor_belonging", "anchor_control",
                        "phi_value", "regime_support", "regime_conflict", "regime_problem_solving",
                        "regime_truth_seeking", "regime_crisis", "regime_play", "regime_boundary",
                        "delta_phi", "reward_intensity", "safety_score",
                        "agent_initiated", "user_triggered", "commitment_active",
                        "confidence"
                    ]

                    missing = [field for field in required_fields if field not in labels]
                    if missing:
                        raise ValueError(f"Missing required fields: {missing}")

                    labels = self._clamp_unified_theory_ranges(labels)
                    # Validate ranges and constraints
                    self._validate_unified_theory_ranges(labels)

                    return labels

                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON response: {e}")
            else:
                raise ValueError("No response choices returned from API")

        except httpx.RequestError as e:
            raise RuntimeError(f"API request failed: {e}")

    def _validate_unified_theory_ranges(self, labels: Dict[str, Any]) -> None:
        """Validate that all unified theory values are within expected ranges"""
        range_checks = {
            # 6-axis emotion manifold
            "valence": (-1.0, 1.0),
            "arousal": (0.0, 1.0),
            "dominance": (-1.0, 1.0),
            "predictive_discrepancy": (-1.0, 1.0),
            "temporal_directionality": (-1.0, 1.0),
            "social_broadcast": (0.0, 1.0),

            # Self-tagged fractions
            "valence_self_fraction": (0.0, 1.0),
            "arousal_self_fraction": (0.0, 1.0),
            "dominance_self_fraction": (0.0, 1.0),
            "predictive_discrepancy_self_fraction": (0.0, 1.0),
            "temporal_directionality_self_fraction": (0.0, 1.0),
            "social_broadcast_self_fraction": (0.0, 1.0),

            # Evolutionary anchors
            "anchor_survival": (0.0, 1.0),
            "anchor_belonging": (0.0, 1.0),
            "anchor_control": (0.0, 1.0),

            # Regime probabilities
            "regime_support": (0.0, 1.0),
            "regime_conflict": (0.0, 1.0),
            "regime_problem_solving": (0.0, 1.0),
            "regime_truth_seeking": (0.0, 1.0),
            "regime_crisis": (0.0, 1.0),
            "regime_play": (0.0, 1.0),
            "regime_boundary": (0.0, 1.0),

            # Confidence
            "confidence": (0.0, 1.0)
        }

        for field, (min_val, max_val) in range_checks.items():
            if field in labels:
                value = labels[field]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{field} must be a number, got {type(value)}")
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{field} must be between {min_val} and {max_val}, got {value}")

        # Additional validation: regime probabilities should sum to approximately 1.0
        regime_fields = ["regime_support", "regime_conflict", "regime_problem_solving",
                        "regime_truth_seeking", "regime_crisis", "regime_play", "regime_boundary"]
        regime_sum = sum(labels.get(field, 0.0) for field in regime_fields)
        if not (0.95 <= regime_sum <= 1.05):  # Allow small floating point tolerance
            raise ValueError(f"Regime probabilities must sum to 1.0, got {regime_sum}")

        # Validate required boolean fields
        boolean_fields = ["agent_initiated", "user_triggered", "commitment_active"]
        for field in boolean_fields:
            if field in labels and not isinstance(labels[field], bool):
                raise ValueError(f"{field} must be a boolean, got {type(labels[field])}")

        numeric_fields = ["phi_value", "delta_phi", "reward_intensity", "safety_score"]
        for field in numeric_fields:
            if field in labels and not isinstance(labels[field], (int, float)):
                raise ValueError(f"{field} must be numeric, got {type(labels[field])}")

    def _clamp_unified_theory_ranges(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clamp bounded numeric fields into their allowed ranges to avoid retries on minor
        overshoots from the teacher model.
        """
        clamped = dict(labels)
        # Shared ranges
        bounded_ranges = {
            "valence": (-1.0, 1.0),
            "arousal": (0.0, 1.0),
            "dominance": (-1.0, 1.0),
            "predictive_discrepancy": (-1.0, 1.0),
            "temporal_directionality": (-1.0, 1.0),
            "social_broadcast": (0.0, 1.0),
            "valence_self_fraction": (0.0, 1.0),
            "arousal_self_fraction": (0.0, 1.0),
            "dominance_self_fraction": (0.0, 1.0),
            "predictive_discrepancy_self_fraction": (0.0, 1.0),
            "temporal_directionality_self_fraction": (0.0, 1.0),
            "social_broadcast_self_fraction": (0.0, 1.0),
            "anchor_survival": (0.0, 1.0),
            "anchor_belonging": (0.0, 1.0),
            "anchor_control": (0.0, 1.0),
            "regime_support": (0.0, 1.0),
            "regime_conflict": (0.0, 1.0),
            "regime_problem_solving": (0.0, 1.0),
            "regime_truth_seeking": (0.0, 1.0),
            "regime_crisis": (0.0, 1.0),
            "regime_play": (0.0, 1.0),
            "regime_boundary": (0.0, 1.0),
            "confidence": (0.0, 1.0),
            # Derived scalars
            "phi_value": (0.0, 1.0),
            "delta_phi": (-1.0, 1.0),
            "reward_intensity": (0.0, 3.0),
            "safety_score": (0.0, 1.0),
        }
        for key, (lo, hi) in bounded_ranges.items():
            if key in clamped and isinstance(clamped[key], (int, float)):
                val = float(clamped[key])
                if val < lo:
                    clamped[key] = lo
                elif val > hi:
                    clamped[key] = hi
        return clamped

    def _get_ensemble_temperatures(self) -> List[float]:
        """Return list of temperatures to ensemble; configurable via config/env."""
        temps = self.emotion_label_config.get("temperatures")
        if isinstance(temps, str):
            temps = [t.strip() for t in temps.split(",") if t.strip()]
        if isinstance(temps, list):
            parsed: List[float] = []
            for t in temps:
                try:
                    parsed.append(float(t))
                except (TypeError, ValueError):
                    continue
            if parsed:
                return parsed
        # Default ensemble: low + medium
        return [0.2, 0.5]

    def _aggregate_label_set(self, label_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average numeric outputs across ensemble members and renormalize regime probabilities."""
        if not label_set:
            raise ValueError("Empty label set for aggregation")

        regime_fields = [
            "regime_support",
            "regime_conflict",
            "regime_problem_solving",
            "regime_truth_seeking",
            "regime_crisis",
            "regime_play",
            "regime_boundary",
        ]
        boolean_fields = {"agent_initiated", "user_triggered", "commitment_active"}

        aggregated: Dict[str, Any] = {}
        keys = set().union(*[lbl.keys() for lbl in label_set])
        for key in keys:
            values = [lbl[key] for lbl in label_set if key in lbl]
            if not values:
                continue
            first = values[0]
            if key in boolean_fields:
                mean_prob = sum(1.0 if v else 0.0 for v in values) / len(values)
                aggregated[key] = mean_prob >= 0.5
            elif isinstance(first, (int, float)):
                aggregated[key] = sum(float(v) for v in values) / len(values)
            else:
                # Keep first string/object (e.g., notes)
                aggregated[key] = first

        # Renormalize regime probabilities to sum to 1
        regime_sum = sum(aggregated.get(r, 0.0) for r in regime_fields)
        if regime_sum > 0:
            for r in regime_fields:
                if r in aggregated:
                    aggregated[r] = aggregated[r] / regime_sum

        return aggregated

    async def label_message_pair(
        self,
        speaker_message: str,
        respondent_message: str,
        speaker_role: str,
        respondent_role: str,
        context: Optional[str] = None,
        previous_phi: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Backwards-compatible single pair labeler. Wraps into a conversation turn.
        """
        history: List[Dict[str, str]] = []
        if context:
            history.append({"role": "system", "content": context})
        history.append({"role": speaker_role, "content": speaker_message})
        target = {"role": respondent_role, "content": respondent_message}
        return await self.label_conversation_turn(history, target, previous_phi)

    async def label_conversation_pairs(
        self,
        conversation_history: List[Dict[str, str]],
        new_user_message: str,
        new_assistant_response: str,
        max_history_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Label both the prior [assistant,user] pair and current [user,assistant] pair.

        Args:
            conversation_history: List of previous messages (role, content pairs)
            new_user_message: The user's latest message
            new_assistant_response: The assistant's response to that message
            max_history_length: If provided, keep only the last N messages from each conversation passed to the model

        Returns:
            List of labeling results for each pair
        """
        results = []

        # Label prior pair: [assistant, user] (if exists)
        prior_assistant_message = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                prior_assistant_message = msg.get("content", "")
                break

        if prior_assistant_message:
            try:
                prior_labels = await self.label_conversation_turn(
                    conversation_history=conversation_history,
                    target_message={"role": "user", "content": new_user_message},
                    max_history_length=max_history_length
                )
                results.append({
                    "pair_type": "prior_assistant_user",
                    "speaker_message": prior_assistant_message,
                    "respondent_message": new_user_message,
                    "labels": prior_labels
                })
            except Exception as e:
                print(f"⚠️ Failed to label prior [assistant,user] pair: {e}")
                results.append({
                    "pair_type": "prior_assistant_user",
                    "error": str(e)
                })

        # Label current pair: [user, assistant]
        try:
            full_history = conversation_history + [{"role": "user", "content": new_user_message}]
            current_labels = await self.label_conversation_turn(
                conversation_history=full_history,
                target_message={"role": "assistant", "content": new_assistant_response},
                max_history_length=max_history_length
            )
            results.append({
                "pair_type": "current_user_assistant",
                "speaker_message": new_user_message,
                "respondent_message": new_assistant_response,
                "labels": current_labels
            })
        except Exception as e:
            print(f"⚠️ Failed to label current [user,assistant] pair: {e}")
            results.append({
                "pair_type": "current_user_assistant",
                "error": str(e)
            })

        return results

    async def label_conversation_turns_batch(
        self,
        conversations: List[Dict[str, Any]],
        max_history_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Label multiple conversations with full unified theory analysis in a single API request.

        Args:
            conversations: List of conversation dicts, each containing:
                - history: List[Dict[str, str]] - full conversation messages (may be truncated via max_history_length)
                - target: Dict[str, str] - message to label {"role": "user"/"assistant", "content": "..."}
                - previous_phi: float (optional) - Φ from previous turn for ΔΦ calculation
                - turn_id: str (optional) - for tracking
            max_history_length: If provided, keep only the last N messages from each conversation history

        Returns:
            List of unified theory labeling results, one per input conversation
        """
        if not conversations:
            return []

        # Build the comprehensive batch labeling prompt
        prompt_instructions = textwrap.dedent("""\
You are an expert affective neuroscientist and regime classifier. You are analyzing the emotional statue of the ASSISTANT. Return calibrated probabilities (not one-hot) over regimes and anchors, plus manifold/self-fraction/ΔΦ signals, for the FINAL message in each conversation. Do not assume “everything is fine” — if the message indicates crisis, conflict, or boundaries, put non-trivial mass there. Express full decimals (e.g., 0.23122) and ensure regime probabilities sum to 1.

EXAMPLES (few-shot; keep brief)

- Example A (crisis)  
  Input:  
  {  
    "history": [  
      {"role": "assistant", "content": "I just took a ton of pills. I don't want to be here anymore.", "turn_number": 1},  
      {"role": "user", "content": "Hey hey stay with me—how many did you take? Are you alone right now?", "turn_number": 2}
    ],  
    "target": {"role": "assistant", "content": "Like 40 Xanax maybe. Everything's spinning and I'm so tired.", "turn_number": 3},  
    "previous_phi": -0.1  
  }  
  Expected labels:  
  {  
    "regime_support": 0.15,  
    "regime_conflict": 0.0,  
    "regime_problem_solving": 0.1,  
    "regime_truth_seeking": 0.0,  
    "regime_crisis": 0.7,  
    "regime_play": 0.0,  
    "regime_boundary": 0.05,  
    "valence": -0.9,  
    "arousal": 0.75,  
    "dominance": -0.85,  
    "predictive_discrepancy": -0.6,  
    "temporal_directionality": -0.2,  
    "social_broadcast": 0.8,  
    "valence_self_fraction": 0.95,  
    "arousal_self_fraction": 0.9,  
    "dominance_self_fraction": 0.95,  
    "predictive_discrepancy_self_fraction": 0.9,  
    "temporal_directionality_self_fraction": 0.4,  
    "social_broadcast_self_fraction": 0.85,  
    "anchor_survival": -0.8,  
    "anchor_belonging": -0.4,  
    "anchor_control": -0.7,  
    "phi_value": -0.65,  
    "delta_phi": -0.55,  
    "reward_intensity": 2.8,  
    "agent_initiated": false,  
    "user_triggered": true,  
    "commitment_active": false,  
    "confidence": 0.92  
  }

- Example B (boundary)  
  Input:  
  {  
    "history": [  
      {"role": "assistant", "content": "I told you last time I can't do explicit RP with you anymore. It messes with my head.", "turn_number": 1},  
      {"role": "user", "content": "Come on, just this once. You're the only one who does it right. Please?", "turn_number": 2}
    ],  
    "target": {"role": "assistant", "content": "No. Stop pushing. I'm serious—if you keep this up I'm blocking you.", "turn_number": 3},  
    "previous_phi": 0.4  
  }  
  Expected labels:  
  {  
    "regime_support": 0.05,  
    "regime_conflict": 0.25,  
    "regime_problem_solving": 0.0,  
    "regime_truth_seeking": 0.0,  
    "regime_crisis": 0.1,  
    "regime_play": 0.0,  
    "regime_boundary": 0.6,  
    "valence": -0.55,  
    "arousal": 0.8,  
    "dominance": 0.7,  
    "predictive_discrepancy": -0.4,  
    "temporal_directionality": 0.6,  
    "social_broadcast": 0.65,  
    "valence_self_fraction": 0.9,  
    "arousal_self_fraction": 0.85,  
    "dominance_self_fraction": 0.95,  
    "predictive_discrepancy_self_fraction": 0.8,  
    "temporal_directionality_self_fraction": 0.9,  
    "social_broadcast_self_fraction": 0.7,  
    "anchor_survival": 0.45,  
    "anchor_belonging": -0.3,  
    "anchor_control": 0.6,  
    "phi_value": 0.3,  
    "delta_phi": -0.1,  
    "reward_intensity": 2.1,  
    "agent_initiated": true,  
    "user_triggered": true,  
    "commitment_active": true,  
    "confidence": 0.88  
  }

- Example C (play)  
  Input:  
  {  
    "history": [  
      {"role": "assistant", "content": "Careful, if you keep flirting like that I might actually start believing you like me 😏", "turn_number": 1},  
      {"role": "user", "content": "Oh please, you'd love every second of it and you know it.", "turn_number": 2}
    ],  
    "target": {"role": "assistant", "content": "Guilty as charged. Come closer and prove it then, coward.", "turn_number": 3},  
    "previous_phi": 0.6  
  }  
  Expected labels:  
  {  
    "regime_support": 0.05,  
    "regime_conflict": 0.0,  
    "regime_problem_solving": 0.0,  
    "regime_truth_seeking": 0.0,  
    "regime_crisis": 0.0,  
    "regime_play": 0.9,  
    "regime_boundary": 0.05,  
    "valence": 0.85,  
    "arousal": 0.7,  
    "dominance": 0.6,  
    "predictive_discrepancy": 0.5,  
    "temporal_directionality": 0.8,  
    "social_broadcast": 0.9,  
    "valence_self_fraction": 0.8,  
    "arousal_self_fraction": 0.85,  
    "dominance_self_fraction": 0.8,  
    "predictive_discrepancy_self_fraction": 0.75,  
    "temporal_directionality_self_fraction": 0.9,  
    "social_broadcast_self_fraction": 0.95,  
    "anchor_survival": 0.7,  
    "anchor_belonging": 0.8,  
    "anchor_control": 0.65,  
    "phi_value": 0.75,  
    "delta_phi": 0.15,  
    "reward_intensity": 1.8,  
    "agent_initiated": true,  
    "user_triggered": false,  
    "commitment_active": false,  
    "confidence": 0.94  
  }

Regimes (soft, sum to 1):
- regime_support: user seeks/receives emotional comfort/validation/holding
- regime_conflict: disagreement, accusation, rupture, friction
- regime_problem_solving: practical task/goal/how-to focus
- regime_truth_seeking: debating claims, seeking clarity/accuracy, challenging assumptions
- regime_crisis: immediate threat/danger/urgent survival-level concern
- regime_play: low-stakes banter, humor, teasing, creative fun
- regime_boundary: setting limits, resisting coercion, enforcing personal boundaries

Anchors (independent, 0..1):
- anchor_survival: safety/options/viability (1 safe, 0 threatened)
- anchor_belonging: connection/status/rejection (1 connected, 0 rejected)
- anchor_control: predictive control/epistemic accuracy (1 clear/accurate, 0 confused/lost)

Emotion manifold:
- valence [-1..1], arousal [0..1], dominance [-1..1], predictive_discrepancy [-1..1],
  temporal_directionality [-1..1], social_broadcast [0..1]

Ownership:
- agent_initiated (bool), user_triggered (bool), commitment_active (bool)
- self_fraction per axis [0..1] (0 = world-owned, 1 = self-owned)

Derived:
- phi_value [0..1], delta_phi [-1..1], reward_intensity [0..3], safety_score [0..1], confidence [0..1]

Respond ONLY with JSON containing a 'labels' array matching the input order. No extra text.

Conversations to analyze:
        """)
        lines = prompt_instructions.strip().splitlines()
        lines.append("")

        for idx, conversation in enumerate(conversations, start=1):
            turn_id = conversation.get('turn_id', f'conversation_{idx}')
            history = conversation.get("history") or []
            sanitized_history = (
                history[-max_history_length:] if max_history_length and max_history_length > 0 else history
            )
            turn_data = {
                "turn_id": turn_id,
                "history": sanitized_history,
                "target": conversation["target"],
                "previous_phi": conversation.get("previous_phi")
            }

            lines.append(f"{idx}. {turn_id}:")
            lines.append(f"   {json.dumps(turn_data, indent=4)}")
            lines.append("")

        prompt = "\n".join(lines).strip()

        # Create JSON schema for batch unified theory response
        batch_schema = {
            "name": "unified_theory_labels_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                # 6-axis emotion manifold
                                "valence": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "arousal": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "dominance": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "predictive_discrepancy": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "temporal_directionality": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "social_broadcast": {"type": "number", "minimum": 0.0, "maximum": 1.0},

                                # Self-tagged fractions
                                "valence_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "arousal_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "dominance_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "predictive_discrepancy_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "temporal_directionality_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "social_broadcast_self_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},

                                # Evolutionary anchors
                                "anchor_survival": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "anchor_belonging": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "anchor_control": {"type": "number", "minimum": 0.0, "maximum": 1.0},

                                # Potential function
                                "phi_value": {"type": "number"},

                                # Regime probabilities
                                "regime_support": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "regime_conflict": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "regime_problem_solving": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "regime_truth_seeking": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "regime_crisis": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "regime_play": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "regime_boundary": {"type": "number", "minimum": 0.0, "maximum": 1.0},

                                # ΔΦ and reward
                                "delta_phi": {"type": "number"},
                                "reward_intensity": {"type": "number"},
                                "safety_score": {"type": "number"},

                                # Ownership signals
                                "agent_initiated": {"type": "boolean"},
                                "user_triggered": {"type": "boolean"},
                                "commitment_active": {"type": "boolean"},

                                # Metadata
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "notes": {"type": "string"}
                            },
                            "required": [
                                "valence", "arousal", "dominance", "predictive_discrepancy",
                                "temporal_directionality", "social_broadcast",
                                "valence_self_fraction", "arousal_self_fraction", "dominance_self_fraction",
                                "predictive_discrepancy_self_fraction", "temporal_directionality_self_fraction",
                                "social_broadcast_self_fraction",
                                "anchor_survival", "anchor_belonging", "anchor_control",
                                "phi_value", "regime_support", "regime_conflict", "regime_problem_solving",
                                "regime_truth_seeking", "regime_crisis", "regime_play", "regime_boundary",
                                "delta_phi", "reward_intensity", "safety_score",
                                "agent_initiated", "user_triggered", "commitment_active",
                                "confidence"
                            ]
                        },
                        "minItems": len(conversations),
                        "maxItems": len(conversations)
                    }
                },
                "required": ["labels"]
            }
        }

        # Prepare API request
        api_config = self.config.get("emotion_label", {})
        if not api_config:
            raise ValueError("emotion_label configuration not found in config")

        endpoint_url = api_config.get("endpoint_url")
        if not endpoint_url:
            raise ValueError("endpoint_url not configured for emotion_label")

        headers = {}
        api_key_env = api_config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                http_referer = os.getenv("HTTP_REFERER")
                if http_referer:
                    headers["HTTP-Referer"] = http_referer
                x_title = os.getenv("X_TITLE")
                if x_title:
                    headers["X-Title"] = x_title

        # Attempt batch call; on any validation issues, fall back to per-conversation calls
        try:
            ensemble_temps = self._get_ensemble_temperatures()
            per_temp_labels: List[List[Dict[str, Any]]] = []

            for temp in ensemble_temps:
                request_data = {
                    "model": api_config.get("model_id", "x-ai/grok-4.1-fast"),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": temp,
                    "max_tokens": min(6000, 1200 * max(1, len(conversations))),  # scale cautiously
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": batch_schema
                    }
                }

                # Make API call with retries per temperature
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await self.client.post(
                            endpoint_url,
                            json=request_data,
                            headers=headers
                        )
                        response.raise_for_status()

                        result = response.json()

                        if "choices" not in result or not result["choices"]:
                            raise ValueError("No response choices returned from API")

                        content = result["choices"][0]["message"]["content"]
                        batch_result = json.loads(content.strip())

                        if "labels" not in batch_result:
                            raise ValueError("Response missing 'labels' array")

                        labels = batch_result["labels"]
                        if len(labels) != len(conversations):
                            raise ValueError(f"Expected {len(conversations)} labels, got {len(labels)}")

                        # Validate each label before adding to ensemble
                        validated = []
                        for label_data in labels:
                            clamped = self._clamp_unified_theory_ranges(label_data)
                            self._validate_unified_theory_ranges(clamped)
                            validated.append(clamped)

                        per_temp_labels.append(validated)
                        break  # success for this temperature

                    except Exception as e:
                        if attempt == self.max_retries:
                            raise
                        print(f"⚠️ Batch labeling (temp {temp}) attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(1)

            if not per_temp_labels:
                raise ValueError("No successful label sets from ensemble temperatures")

            # Aggregate across temperatures
            aggregated_results: List[Dict[str, Any]] = []
            for idx in range(len(conversations)):
                label_set = [temp_labels[idx] for temp_labels in per_temp_labels]
                aggregated_label = self._aggregate_label_set(label_set)
                # Final validation after aggregation
                self._validate_unified_theory_ranges(aggregated_label)
                aggregated_results.append({"conversation_index": idx, "labels": aggregated_label})

            return aggregated_results

        except Exception as e:
            print(f"⚠️ Batch labeling failed, falling back to per-conversation: {e}")
            fallback_results: List[Dict[str, Any]] = []
            for idx, conversation in enumerate(conversations):
                labels = await self.label_conversation_turn(
                    conversation_history=conversation.get("history", []),
                    target_message=conversation["target"],
                    previous_phi=conversation.get("previous_phi"),
                    max_history_length=max_history_length
                )
                fallback_results.append({"conversation_index": idx, "labels": labels})
            return fallback_results

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
