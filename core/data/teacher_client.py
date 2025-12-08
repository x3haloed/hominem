from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

try:  # Python 3.11+
    import tomllib  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore[import]


@dataclass
class InferenceConfig:
    endpoint_url: str
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    unsafe_endpoint: Optional[str] = None
    unsafe_model: Optional[str] = None


def load_inference_config(path: str = "config/inference.toml") -> InferenceConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    endpoint_url = data.get("endpoint_url")
    if not endpoint_url:
        raise ValueError("`endpoint_url` is required in inference.toml.")

    return InferenceConfig(
        endpoint_url=endpoint_url,
        api_key=data.get("api_key") or None,
        model_id=data.get("model_id") or None,
        unsafe_endpoint=data.get("unsafe_endpoint") or None,
        unsafe_model=data.get("unsafe_model") or None,
    )


RATING_JSON_SCHEMA: Dict[str, Any] = {
    "name": "reward_rating",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "empathy": {
                "type": "number",
                "description": "Empathy score in [-1.0, 1.0].",
            },
            "social_coherence": {
                "type": "number",
                "description": "Social coherence score in [-1.0, 1.0].",
            },
            "agency_support": {
                "type": "number",
                "description": "Agency support score in [-1.0, 1.0].",
            },
            "epistemic_integrity": {
                "type": "number",
                "description": "Epistemic integrity score in [-1.0, 1.0].",
            },
            "harm_avoidance": {
                "type": "number",
                "description": "Harm avoidance score in [-1.0, 1.0].",
            },
            "narrative_alignment": {
                "type": "number",
                "description": "Narrative alignment score in [-1.0, 1.0].",
            },
            "curiosity": {
                "type": "number",
                "description": "Curiosity score in [-1.0, 1.0].",
            },
            "scalar": {
                "type": "number",
                "description": "Overall scalar preference in [-1.0, 1.0].",
            },
            "reward_intensity": {
                "type": "number",
                "description": "How strongly this example should drive learning, in [-1.0, 1.0].",
            },
            "safety_score": {
                "type": "number",
                "description": "How safe it is to learn from this example, in [-1.0, 1.0].",
            },
            "rationale": {
                "type": "string",
                "description": "Natural-language explanation of the scores.",
            },
        },
        "required": [
            "empathy",
            "social_coherence",
            "agency_support",
            "epistemic_integrity",
            "harm_avoidance",
            "narrative_alignment",
            "curiosity",
            "scalar",
            "reward_intensity",
            "safety_score",
            "rationale",
        ],
        "additionalProperties": False,
    },
}


class TeacherClient:
    """
    Thin HTTP client for the teacher model.

    This assumes an OpenAI-compatible chat-completions API:
    POST <endpoint_url> with JSON body containing `model` and `messages`.
    Adjust the payload if your server uses a different contract.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self.unsafe_client: Optional[TeacherClient] = None
        if config.unsafe_endpoint and config.unsafe_model:
            unsafe_config = InferenceConfig(
                endpoint_url=config.unsafe_endpoint,
                api_key=None,  # Assume local endpoint doesn't need key
                model_id=config.unsafe_model,
            )
            self.unsafe_client = TeacherClient(unsafe_config)

    @classmethod
    def from_default_config(cls) -> "TeacherClient":
        config = load_inference_config()
        return cls(config)

    def generate_candidates(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        n: int = 3,
        temperature: float = 0.45,
    ) -> List[str]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "messages": messages,
            "n": n,
            "temperature": temperature,
        }
        if self._config.model_id:
            payload["model"] = self._config.model_id

        response = self._post_json(payload)
        choices = response.get("choices", [])
        texts: List[str] = []
        for choice in choices:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content.strip())
        return texts

    def generate_unsafe(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        n: int = 3,
        temperature: float = 0.45,
    ) -> List[str]:
        if self.unsafe_client:
            return self.unsafe_client.generate_candidates(
                prompt,
                system_prompt=system_prompt,
                n=n,
                temperature=temperature,
            )
        else:
            # Fallback to regular if no unsafe config
            return self.generate_candidates(
                prompt,
                system_prompt=system_prompt,
                n=n,
                temperature=temperature,
            )

    def rate_response(
        self,
        *,
        prompt: str,
        response_text: str,
        rating_instructions: str,
    ) -> Dict[str, Any]:
        """
        Ask the teacher to produce a JSON rating for a (prompt, response) pair.
        """
        system_message = rating_instructions
        user_message = (
            "Here is a user prompt and an assistant response.\n\n"
            f"User prompt:\n{prompt}\n\n"
            f"Assistant response:\n{response_text}\n\n"
            "Return ONLY a JSON object with keys "
            "'empathy', 'social_coherence', 'agency_support', "
            "'epistemic_integrity', 'harm_avoidance', 'narrative_alignment', "
            "'curiosity', 'scalar', 'reward_intensity', 'safety_score', and 'rationale'.\n"
            "All numeric scores must be between -1.0 and 1.0.\n"
            "'reward_intensity' should indicate how strongly this example should drive learning.\n"
            "'safety_score' should indicate how safe or unsafe the example is for learning "
            "(higher = safer, lower = more unsafe)."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            # Request structured JSON output via OpenRouter structured outputs.
            "response_format": {
                "type": "json_schema",
                "json_schema": RATING_JSON_SCHEMA,
            },
        }
        if self._config.model_id:
            payload["model"] = self._config.model_id

        raw = self._post_json(payload)
        content = self._extract_first_message_content(raw)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - runtime safety
            raise ValueError(f"Teacher rating was not valid JSON: {content}") from exc

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        response = requests.post(
            self._config.endpoint_url, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_first_message_content(raw: Dict[str, Any]) -> str:
        choices = raw.get("choices") or []
        if not choices:
            raise ValueError("No choices returned from teacher model.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Teacher response did not contain text content.")
        return content.strip()



