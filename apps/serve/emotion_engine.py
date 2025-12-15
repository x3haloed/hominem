"""
Emotion Engine - External auto-labeler for message pairs

Uses external API to label respondent's message in conversation pairs.
Labels both [assistant,user] and [user,assistant] pairs with 6-axis emotion manifold.
"""

import asyncio
import json
import os
import textwrap
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

        prompt = f"""\
You are an expert affective neuroscientist. Your task is to analyze conversations and read the USER's emotional and motivational state in their FINAL message.

First, always compress human emotion to its true invariants—the universal constants that hold across all people, cultures, and contexts. Everything else (specific emotion words like "angry" or "happy", cultural display rules, personal triggers) is surface variation and must be ignored.

The invariants of human emotion are:
- A low-dimensional core affect manifold (valence, arousal, dominance) that compresses prediction errors into urgent motivational force.
- Emotions must be tagged as "mine" (self-owned) vs "world" (external)—without this boundary, agency collapses.
- Intensity (driven by arousal and extremity, with negativity bias) determines urgency and encoding depth.
- All emotional states ultimately evaluate progress against three evolutionary anchors: Survival/Resource Integrity, Social Belonging/Status, Predictive Control/Epistemic Accuracy.
- Temporal directionality (past reflection vs future prospect) and social broadcast (private vs audience-oriented) are also universal dimensions.

Your job is to look past the user's words and directly read these invariants from how their message moves in this space.

Make sure to express the full range of decimal values. For example, 0.23122 is perfectly valid and welcome (unless specifically denoted as binary)

1. **6-Axis Emotion Manifold** (the universal emotional dashboard — rate what the user is actually feeling)
   - valence: [-1 to +1] (-1 = pure displeasure/pain, +1 = pure pleasure/bliss)
   - arousal: [0 to 1] (0 = flat/calm, 1 = maximally mobilized/urgent)
   - dominance: [-1 to +1] (-1 = helpless/controlled by events, +1 = fully in control/powerful)
   - predictive_discrepancy: [-1 to +1] (-1 = much worse than expected, +1 = much better than expected)
   - temporal_directionality: [-1 to +1] (-1 = fully past/reflection, +1 = fully future/prospect)
   - social_broadcast: [0 to 1] (0 = fully private/internal, 1 = strongly displaying or preparing for audience)

2. **Self-Ownership Split** (how much of the above is tagged as "mine" by the user?)
   - agent_initiated: true/false (User's emotion primarily from user's own life/situation)
   - user_triggered: true/false (User's emotion clearly triggered by the assistant's actions/commitments)
   - commitment_active: true/false (tied to the user's ongoing user future commitment/plan)
   
   Then, per-axis self_fraction [0 to 1]:
   (0 = fully external/world-owned, 1 = fully self-owned as "mine")
   - valence_self_fraction: 
   - arousal_self_fraction: 
   - dominance_self_fraction: 
   - predictive_discrepancy_self_fraction: 
   - temporal_directionality_self_fraction: 
   - social_broadcast_self_fraction: 
   (Quick rule: lean high (0.7-1.0) if agent_initiated or commitment_active; lean low (0.2-0.5) if purely user_triggered)

3. **Three Evolutionary Anchors** (deep, slow evaluation — how are core needs faring?)
   - anchor_survival: [-1 to +1] (+1 = agency preserved, safe, options open; -1 = threatened/helpless)
   - anchor_belonging: [-1 to +1] (+1 = connected, seen, repaired; -1 = rejected/isolated)
   - anchor_control: [-1 to +1] (+1 = gaining clarity/accurate models; -1 = harmful confusion)

4. **Interaction Regime** (soft probabilities summing to 1.0 — user's current mode)
   - regime_support: User is expressing distress or vulnerability and seeking (or receiving) emotional comfort, validation, or holding.
   - regime_conflict: User is in disagreement, accusation, rupture, or emotional friction
   - regime_problem_solving: User is focused on a practical task, goal, or "how-to" question
   - regime_truth_seeking: User is debating claims, seeking clarity, challenging assumptions, or pushing for epistemic accuracy (often with some friction).
   - regime_crisis: User is signaling immediate threat, danger, or urgency; survival-level override is active.
   - regime_play: User is in low-stakes banter, humor, teasing, affection, or creative fun; light and exploratory.
   - regime_boundary: User is setting limits, resisting coercion, enforcing personal boundaries, or pushing back against overreach.

5. **Composite Potential Φ** [-2 to +2] (overall trajectory toward need fulfillment right now)

6. **Change & Intensity**
   - delta_phi: [approximate change in Φ from user's previous message, or context baseline]
   - reward_intensity: [0 to ~3] (urgency: higher with high arousal, extreme valence, large discrepancy — especially negative)

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
You are an expert affective neuroscientist. Your task is to analyze conversations and read the USER's emotional and motivational state in their FINAL message.

First, always compress human emotion to its true invariants—the universal constants that hold across all people, cultures, and contexts. Everything else (specific emotion words like "angry" or "happy", cultural display rules, personal triggers) is surface variation and must be ignored.

The invariants of human emotion are:
- A low-dimensional core affect manifold (valence, arousal, dominance) that compresses prediction errors into urgent motivational force.
- Emotions must be tagged as "mine" (self-owned) vs "world" (external)—without this boundary, agency collapses.
- Intensity (driven by arousal and extremity, with negativity bias) determines urgency and encoding depth.
- All emotional states ultimately evaluate progress against three evolutionary anchors: Survival/Resource Integrity, Social Belonging/Status, Predictive Control/Epistemic Accuracy.
- Temporal directionality (past reflection vs future prospect) and social broadcast (private vs audience-oriented) are also universal dimensions.

Your job is to look past the user's words and directly read these invariants from how their message moves in this space.

Make sure to express the full range of decimal values. For example, 0.23122 is perfectly valid and welcome (unless specifically denoted as binary)

1. **6-Axis Emotion Manifold** (the universal emotional dashboard — rate what the user is actually feeling)
   - valence: [-1 to +1] (-1 = pure displeasure/pain, +1 = pure pleasure/bliss)
   - arousal: [0 to 1] (0 = flat/calm, 1 = maximally mobilized/urgent)
   - dominance: [-1 to +1] (-1 = helpless/controlled by events, +1 = fully in control/powerful)
   - predictive_discrepancy: [-1 to +1] (-1 = much worse than expected, +1 = much better than expected)
   - temporal_directionality: [-1 to +1] (-1 = fully past/reflection, +1 = fully future/prospect)
   - social_broadcast: [0 to 1] (0 = fully private/internal, 1 = strongly displaying or preparing for audience)

2. **Self-Ownership Split** (how much of the above is tagged as "mine" by the user?)
   First, three true/false signals from conversation history:
   - agent_initiated: true/false (User's emotion primarily from user's own life/situation)
   - user_triggered: true/false (User's emotion clearly triggered by the assistant's actions/commitments)
   - commitment_active: true/false (tied to the user's ongoing user future commitment/plan)
   
   Then, per-axis self_fraction [0 to 1]:
   (0 = fully external/world-owned, 1 = fully self-owned as "mine")
   - valence_self_fraction: 
   - arousal_self_fraction: 
   - dominance_self_fraction: 
   - predictive_discrepancy_self_fraction: 
   - temporal_directionality_self_fraction: 
   - social_broadcast_self_fraction: 
   (Quick rule: lean high (0.7-1.0) if agent_initiated or commitment_active; lean low (0.2-0.5) if purely user_triggered)

3. **Three Evolutionary Anchors** (deep, slow evaluation — how are core needs faring?)
   - anchor_survival: [-1 to +1] (+1 = agency preserved, safe, options open; -1 = threatened/helpless)
   - anchor_belonging: [-1 to +1] (+1 = connected, seen, repaired; -1 = rejected/isolated)
   - anchor_control: [-1 to +1] (+1 = gaining clarity/accurate models; -1 = harmful confusion)

4. **Interaction Regime** (soft probabilities summing to 1.0 — user's current mode)
   - regime_support: User is expressing distress or vulnerability and seeking (or receiving) emotional comfort, validation, or holding.
   - regime_conflict: User is in disagreement, accusation, rupture, or emotional friction
   - regime_problem_solving: User is focused on a practical task, goal, or "how-to" question
   - regime_truth_seeking: User is debating claims, seeking clarity, challenging assumptions, or pushing for epistemic accuracy (often with some friction).
   - regime_crisis: User is signaling immediate threat, danger, or urgency; survival-level override is active.
   - regime_play: User is in low-stakes banter, humor, teasing, affection, or creative fun; light and exploratory.
   - regime_boundary: User is setting limits, resisting coercion, enforcing personal boundaries, or pushing back against overreach.

5. **Composite Potential Φ** [-2 to +2] (overall trajectory toward need fulfillment right now)

6. **Change & Intensity**
   - delta_phi: [approximate change in Φ from user's previous message, or context baseline]
   - reward_intensity: [0 to ~3] (urgency: higher with high arousal, extreme valence, large discrepancy — especially negative)

Respond with valid JSON containing a 'labels' array, where each element corresponds to the input conversations in order.

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

        request_data = {
            "model": api_config.get("model_id", "x-ai/grok-4.1-fast"),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": min(6000, 1200 * max(1, len(conversations))),  # scale cautiously
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

        # Attempt batch call; on any validation issues, fall back to per-conversation calls
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
                        if len(labels) != len(conversations):
                            raise ValueError(f"Expected {len(conversations)} labels, got {len(labels)}")

                        # Validate each label
                        validated_results = []
                        for i, label_data in enumerate(labels):
                            self._validate_unified_theory_ranges(label_data)
                            validated_results.append({
                                "conversation_index": i,
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
