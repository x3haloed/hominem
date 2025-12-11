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


# JSON Schema for emotion labeling responses
EMOTION_LABEL_SCHEMA = {
    "name": "emotion_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "valence": {
                "type": "number",
                "minimum": -2.0,
                "maximum": 2.0,
                "description": "Emotional valence from -2 (very negative) to +2 (very positive)."
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
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in these labels from 0.0 to 1.0."
            },
            "notes": {
                "type": "string",
                "description": "Optional natural-language explanation of the emotion assessment."
            }
        },
        "required": [
            "valence", "arousal", "dominance",
            "predictive_discrepancy", "temporal_directionality", "social_broadcast",
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

    async def label_message_pair(
        self,
        speaker_message: str,
        respondent_message: str,
        speaker_role: str,
        respondent_role: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Label the respondent's message in a message pair using external API.

        Args:
            speaker_message: Message from the speaker (initiates the pair)
            respondent_message: Message to be labeled (respondent's reply)
            speaker_role: Role of speaker ('user' or 'assistant')
            respondent_role: Role of respondent ('user' or 'assistant')
            context: Optional conversation context

        Returns:
            Dict containing emotion labels for the respondent's message
        """
        # Build the labeling prompt
        pair_data = {
            "speaker": speaker_message,
            "speaker_role": speaker_role,
            "respondent": respondent_message,
            "respondent_role": respondent_role
        }

        if context:
            pair_data["context"] = context

        prompt = f"""Please label the respondent's message in the following conversation pair with emotion dimensions. Use the 6-axis emotion manifold:

Dimensions (ranges specified):
- valence: positive (+2) to negative (-2) emotional valence
- arousal: high energy/arousal (0 = calm, 1 = highly aroused)
- dominance: dominant/confident (+1) to submissive/passive (-1)
- predictive_discrepancy: surprised/betrayed (+1) to expected/predictable (-1)
- temporal_directionality: future-oriented/prospect (+1) to past-oriented/reflection (-1)
- social_broadcast: socially expressive/outward (0 = reserved, 1 = highly expressive)

Also provide:
- confidence: 0.0 to 1.0 (how confident you are in these labels)
- notes: brief explanation of your reasoning

Respond with valid JSON only.

Pair to label:
{json.dumps(pair_data, indent=2)}

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
            "temperature": 0.3,  # Lower temperature for more consistent labeling
            "max_tokens": 500,
            "response_format": {
                "type": "json_schema",
                "json_schema": EMOTION_LABEL_SCHEMA
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
                headers["HTTP-Referer"] = os.getenv("HTTP_REFERER", "")
                headers["X-Title"] = os.getenv("X_TITLE", "")

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

                    # Validate required fields
                    required_fields = [
                        "valence", "arousal", "dominance",
                        "predictive_discrepancy", "temporal_directionality", "social_broadcast",
                        "confidence"
                    ]

                    missing = [field for field in required_fields if field not in labels]
                    if missing:
                        raise ValueError(f"Missing required fields: {missing}")

                    # Validate ranges
                    self._validate_emotion_ranges(labels)

                    return labels

                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON response: {e}")
            else:
                raise ValueError("No response choices returned from API")

        except httpx.RequestError as e:
            raise RuntimeError(f"API request failed: {e}")

    def _validate_emotion_ranges(self, labels: Dict[str, Any]) -> None:
        """Validate that emotion values are within expected ranges"""
        range_checks = {
            "valence": (-2.0, 2.0),
            "arousal": (0.0, 1.0),
            "dominance": (-1.0, 1.0),
            "predictive_discrepancy": (-1.0, 1.0),
            "temporal_directionality": (-1.0, 1.0),
            "social_broadcast": (0.0, 1.0),
            "confidence": (0.0, 1.0)
        }

        for field, (min_val, max_val) in range_checks.items():
            if field in labels:
                value = labels[field]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{field} must be a number, got {type(value)}")
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{field} must be between {min_val} and {max_val}, got {value}")

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

        # Find the last assistant message before the current user message
        prior_assistant_message = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                prior_assistant_message = msg.get("content", "")
                break

        # Label prior pair: [assistant, user] (if exists)
        if prior_assistant_message:
            try:
                prior_labels = await self.label_message_pair(
                    speaker_message=prior_assistant_message,
                    respondent_message=new_user_message,
                    speaker_role="assistant",
                    respondent_role="user",
                    context="Conversation between AI assistant and human user"
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
            current_labels = await self.label_message_pair(
                speaker_message=new_user_message,
                respondent_message=new_assistant_response,
                speaker_role="user",
                respondent_role="assistant",
                context="Conversation between human user and AI assistant"
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

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()