"""
Emotion Engine - External auto-labeler for message pairs

Uses external API to label respondent's message in conversation pairs.
Labels both [assistant,user] and [user,assistant] pairs with 6-axis emotion manifold.
"""

import asyncio
import json
import os
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
            "anchor_survival": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "anchor_belonging": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "anchor_control": {"type": "number", "minimum": -1.0, "maximum": 1.0},

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
        previous_phi: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Label a conversation turn with full unified theory analysis.

        Args:
            conversation_history: List of previous messages [{"role": "user"/"assistant", "content": "..."}]
                                 (at least last 2-3 turns for ownership decisions)
            target_message: The message to label {"role": "user"/"assistant", "content": "..."}
            previous_phi: Φ value from previous turn (for ΔΦ calculation)

        Returns:
            Dict containing all unified theory labels for the target message
        """
        # Build the comprehensive labeling prompt
        sanitized_history = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
        conversation_data = {
            "history": sanitized_history,
            "target": target_message,
            "previous_phi": previous_phi
        }

        prompt = f"""Analyze the target message in this conversation using the Unified Theory of Artificial Mind framework.

Use the following analysis framework:

1. **6-Axis Emotion Manifold** (ranges specified):
   - valence: [-1, 1] (pleasure/pain hedonic tone)
   - arousal: [0, 1] (physiological mobilization/energy)
   - dominance: [-1, 1] (perceived control: I act on it ↔ it acts on me)
   - predictive_discrepancy: [-1, 1] (signed surprise: positive = better than expected)
   - temporal_directionality: [-1, 1] (prospect/reflection: -1 past-oriented, +1 future-oriented)
   - social_broadcast: [0, 1] (internalized audience/display preparation)

2. **Self-Tagged Ownership** (0-1 fractions owned by agent vs world):
   - Examine conversation history to determine if emotions originated from agent actions/commitments
   - agent_initiated: true if emotional trigger from agent's prior output/actions
   - user_triggered: true if direct response to user's input/behavior
   - commitment_active: true if relates to ongoing agent prospect/future commitment
   - For each axis, assign self_fraction based on ownership signals

3. **Evolutionary Anchors** [-1, 1] (slow error evaluation):
   - Survival/Resource Integrity: agency support, harm minimization, resource preservation
   - Social Belonging/Status: empathy correctness, social coherence, narrative alignment
   - Predictive Control/Epistemic Accuracy: epistemic integrity, curiosity resolution, surprise reduction

4. **Regime Classification** (7 probabilities summing to 1.0):
   - support, conflict, problem_solving, truth_seeking, crisis, play, boundary

5. **Potential Function Φ** (composite evaluation using regime weights)

6. **ΔΦ and Reward** (change from previous Φ, intensity calculation)

Respond with valid JSON containing all required fields.

Conversation to analyze:
{json.dumps(conversation_data, indent=2)}

JSON Response:"""

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
            "anchor_survival": (-1.0, 1.0),
            "anchor_belonging": (-1.0, 1.0),
            "anchor_control": (-1.0, 1.0),

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
        new_assistant_response: str
    ) -> List[Dict[str, Any]]:
        """
        Label both the prior [assistant,user] pair and current [user,assistant] pair.

        Args:
            conversation_history: List of previous messages (role, content pairs)
            new_user_message: The user's latest message
            new_assistant_response: The assistant's response to that message

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
                    conversation_history=conversation_history[-3:],
                    target_message={"role": "user", "content": new_user_message}
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
                conversation_history=full_history[-3:],
                target_message={"role": "assistant", "content": new_assistant_response}
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
        conversation_turns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Label multiple conversation turns with full unified theory analysis in a single API request.

        Args:
            conversation_turns: List of turn dicts, each containing:
                - history: List[Dict[str, str]] - previous messages (at least 2-3 turns)
                - target: Dict[str, str] - message to label {"role": "user"/"assistant", "content": "..."}
                - previous_phi: float (optional) - Φ from previous turn for ΔΦ calculation
                - turn_id: str (optional) - for tracking

        Returns:
            List of unified theory labeling results, one per input turn
        """
        if not conversation_turns:
            return []

        # Build the comprehensive batch labeling prompt
        lines = [
            "Analyze each conversation turn using the Unified Theory of Artificial Mind framework.",
            "For each turn, examine the conversation history (at least the last 2-3 turns) to determine ownership and context.",
            "",
            "Use the following analysis framework for EACH turn:",
            "",
            "1. **6-Axis Emotion Manifold** (ranges [-1,1] except arousal/social_broadcast [0,1]):",
            "   - valence: hedonic tone (pleasure ↔ pain)",
            "   - arousal: physiological mobilization/energy [0,1]",
            "   - dominance: perceived control (-1 submissive ↔ +1 dominant)",
            "   - predictive_discrepancy: signed surprise (-1 expected ↔ +1 highly surprising)",
            "   - temporal_directionality: -1 past-oriented ↔ +1 future-oriented",
            "   - social_broadcast: display preparation [0,1]",
            "",
            "2. **Self-Tagged Ownership** (0-1 fractions owned by agent):",
            "   - agent_initiated: true if emotional trigger from agent's prior actions/commitments",
            "   - user_triggered: true if direct response to user's input/behavior",
            "   - commitment_active: true if relates to ongoing agent prospect/future commitment",
            "   - Assign self_fraction per axis based on these ownership signals",
            "",
            "3. **Evolutionary Anchors** [-1,1] (slow error evaluation):",
            "   - anchor_survival: agency support, harm minimization, resource preservation",
            "   - anchor_belonging: empathy correctness, social coherence, narrative alignment",
            "   - anchor_control: epistemic integrity, curiosity resolution, surprise reduction",
            "",
            "4. **Regime Classification** (7 probabilities summing to 1.0):",
            "   support, conflict, problem_solving, truth_seeking, crisis, play, boundary",
            "",
            "5. **Potential Function Φ** (composite evaluation using regime weights)",
            "",
            "6. **ΔΦ and Reward** (change from previous Φ, intensity calculation)",
            "",
            "Respond with valid JSON containing a 'labels' array, where each element corresponds to the input turns in order.",
            "",
            "Conversation turns to analyze:"
        ]

        for idx, turn in enumerate(conversation_turns, start=1):
            turn_id = turn.get('turn_id', f'turn_{idx}')
            history = turn.get("history") or []
            sanitized_history = history[-3:] if len(history) >= 3 else history
            turn_data = {
                "turn_id": turn_id,
                "history": sanitized_history,  # Last 3 turns max
                "target": turn["target"],
                "previous_phi": turn.get("previous_phi")
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
                                "anchor_survival": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "anchor_belonging": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                                "anchor_control": {"type": "number", "minimum": -1.0, "maximum": 1.0},

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
                        "minItems": len(conversation_turns),
                        "maxItems": len(conversation_turns)
                    }
                },
                "required": ["labels"]
            }
        }

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
            "temperature": 0.2,
            "max_tokens": min(6000, 1200 * max(1, len(conversation_turns))),  # scale cautiously
            "response_format": {
                "type": "json_schema",
                "json_schema": batch_schema
            }
        }

        # Make API call with retries
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

        # Attempt batch call; on any validation issues, fall back to per-turn calls
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

                    # Extract the JSON response
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        batch_result = json.loads(content.strip())

                        # Validate response structure
                        if "labels" not in batch_result:
                            raise ValueError("Response missing 'labels' array")

                        labels = batch_result["labels"]
                        if len(labels) != len(conversation_turns):
                            raise ValueError(f"Expected {len(conversation_turns)} labels, got {len(labels)}")

                        # Validate each label
                        validated_results = []
                        for i, label_data in enumerate(labels):
                            self._validate_unified_theory_ranges(label_data)
                            validated_results.append({
                                "pair_index": i,
                                "labels": label_data
                            })

                        return validated_results

                    raise ValueError("No response choices returned from API")

                except Exception as e:
                    if attempt == self.max_retries:
                        raise
                    print(f"⚠️ Batch emotion labeling attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(1)

        except Exception as e:
            print(f"⚠️ Batch labeling failed, falling back to per-turn: {e}")
            fallback_results: List[Dict[str, Any]] = []
            for idx, turn in enumerate(conversation_turns):
                labels = await self.label_conversation_turn(
                    conversation_history=turn.get("history", []),
                    target_message=turn["target"],
                    previous_phi=turn.get("previous_phi")
                )
                fallback_results.append({"pair_index": idx, "labels": labels})
            return fallback_results

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
