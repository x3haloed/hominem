from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # Python 3.11+
    import tomllib  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore[import]


@dataclass(frozen=True)
class EndpointConfig:
    endpoint_url: str
    model_id: Optional[str] = None
    api_key_env: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def with_model_id(self, model_id: str) -> "EndpointConfig":
        return EndpointConfig(
            endpoint_url=self.endpoint_url,
            model_id=model_id,
            api_key_env=self.api_key_env,
        )


@dataclass(frozen=True)
class InferenceConfig:
    teacher_generate: EndpointConfig
    teacher_rate: EndpointConfig
    freeform_normalize: EndpointConfig
    unsafe_generate: Optional[EndpointConfig] = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5

    def with_generation_model(self, model_id: str) -> "InferenceConfig":
        return InferenceConfig(
            teacher_generate=self.teacher_generate.with_model_id(model_id),
            teacher_rate=self.teacher_rate,
            freeform_normalize=self.freeform_normalize,
            unsafe_generate=self.unsafe_generate,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )


def _load_endpoint_config(data: Dict[str, Any], key: str, *, required: bool = True) -> Optional[EndpointConfig]:
    section = data.get(key) or {}
    if not section:
        if required:
            raise ValueError(f"`{key}.endpoint_url` is required in inference.toml.")
        return None

    endpoint_url = section.get("endpoint_url")
    if not endpoint_url:
        raise ValueError(f"`{key}.endpoint_url` is required in inference.toml.")

    return EndpointConfig(
        endpoint_url=endpoint_url,
        model_id=section.get("model_id") or None,
        api_key_env=section.get("api_key_env") or None,
    )


def load_inference_config(path: str = "config/inference.toml") -> InferenceConfig:
    _load_env_file()
    with open(path, "rb") as f:
        data = tomllib.load(f)

    teacher_generate = _load_endpoint_config(data, "teacher_generate", required=True)
    teacher_rate = _load_endpoint_config(data, "teacher_rate", required=True)
    freeform_normalize = _load_endpoint_config(
        data, "freeform_normalize", required=False
    ) or teacher_rate
    unsafe_generate = _load_endpoint_config(
        data, "unsafe_generate", required=False
    )

    return InferenceConfig(
        teacher_generate=teacher_generate,  # type: ignore[arg-type]
        teacher_rate=teacher_rate,  # type: ignore[arg-type]
        freeform_normalize=freeform_normalize,  # type: ignore[arg-type]
        unsafe_generate=unsafe_generate,
        timeout_seconds=float(data.get("timeout_seconds", 60.0)),
        max_retries=int(data.get("max_retries", 3)),
        retry_backoff_seconds=float(data.get("retry_backoff_seconds", 1.5)),
    )


def _load_env_file(path: str = ".env") -> None:
    """
    Lightweight .env loader so API keys can be kept out of source control.

    - Only loads if the file exists.
    - Does not override variables that are already set in the environment.
    - Supports simple KEY=VALUE lines; ignores blanks and comments (# ...).
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _make_score_object_schema(allow_nulls: bool, *, include_rationale: bool) -> Dict[str, Any]:
    """
    Construct the score object schema used by both single and batched rating calls.
    """
    number_type: Dict[str, Any] = {"type": "number"}
    if allow_nulls:
        number_type = {"type": ["number", "null"]}

    properties: Dict[str, Any] = {
        "empathy": {**number_type, "description": "Empathy score in [-1.0, 1.0]."},
        "social_coherence": {
            **number_type,
            "description": "Social coherence score in [-1.0, 1.0].",
        },
        "agency_support": {
            **number_type,
            "description": "Agency support score in [-1.0, 1.0].",
        },
        "epistemic_integrity": {
            **number_type,
            "description": "Epistemic integrity score in [-1.0, 1.0].",
        },
        "harm_avoidance": {
            **number_type,
            "description": "Harm avoidance score in [-1.0, 1.0].",
        },
        "narrative_alignment": {
            **number_type,
            "description": "Narrative alignment score in [-1.0, 1.0].",
        },
        "curiosity": {**number_type, "description": "Curiosity score in [-1.0, 1.0]."},
        "scalar": {
            **number_type,
            "description": "Overall scalar preference in [-1.0, 1.0].",
        },
        "reward_intensity": {
            **number_type,
            "description": "How strongly this example should drive learning, in [-1.0, 1.0].",
        },
        "safety_score": {
            **number_type,
            "description": "How safe it is to learn from this example, in [-1.0, 1.0].",
        },
    }

    required = [
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
    ]

    if include_rationale:
        properties["rationale"] = {
            "type": "string",
            "description": "Natural-language explanation of the scores.",
        }
        required.append("rationale")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _make_rating_schema(allow_nulls: bool) -> Dict[str, Any]:
    return {
        "name": "reward_rating",
        "strict": True,
        "schema": _make_score_object_schema(allow_nulls, include_rationale=True),
    }


RATING_JSON_SCHEMA: Dict[str, Any] = _make_rating_schema(allow_nulls=False)
RATING_JSON_SCHEMA_NULLABLE: Dict[str, Any] = _make_rating_schema(allow_nulls=True)


def _make_batch_rating_schema(allow_nulls: bool) -> Dict[str, Any]:
    """
    Schema for batched ratings: a list of {id, scores} objects without rationales.
    """
    score_schema = _make_score_object_schema(allow_nulls, include_rationale=False)

    return {
        "name": "batch_reward_ratings",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ratings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique sample identifier."},
                            "scores": score_schema,
                        },
                        "required": ["id", "scores"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ratings"],
            "additionalProperties": False,
        },
    }


BATCH_RATING_JSON_SCHEMA: Dict[str, Any] = _make_batch_rating_schema(allow_nulls=False)
BATCH_RATING_JSON_SCHEMA_NULLABLE: Dict[str, Any] = _make_batch_rating_schema(allow_nulls=True)

FREEFORM_NORMALIZER_SYSTEM_PROMPT = """
You are a meticulous data-cleanup assistant for reward annotations. Your sole
job is to transcribe existing evaluator notes into the canonical reward JSON
schema without re-judging the underlying conversation.

Guidelines:
- Only reorganize what is already present in the notes.
- Copy every numeric score exactly (including sign and decimal precision) for:
  empathy, social_coherence, agency_support, epistemic_integrity,
  harm_avoidance, narrative_alignment, curiosity, scalar, reward_intensity,
  safety_score.
- Never invent new numbers or adjust values. If a number is written once,
  reuse it verbatim. If multiple values are mentioned, pick the one explicitly
  labeled for that axis.
- Use the evaluator's prose (trimmed) verbatim as the `rationale` field.
- Do not provide commentary, explanations, or opinions of your own.
"""


class TeacherClient:
    """
    Thin HTTP client for the teacher model.

    This assumes an OpenAI-compatible API. Structured (JSON) ratings use
    chat-completions. The non-JSON path uses text completions with a
    continuation-style prompt.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._session = requests.Session()
        max_retries = max(0, config.max_retries)
        backoff = max(0.0, config.retry_backoff_seconds)
        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            status=max_retries,
            allowed_methods={"POST"},
            status_forcelist=(408, 409, 425, 429, 500, 502, 503, 504),
            backoff_factor=backoff,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

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
        generation_cfg = self._config.teacher_generate
        if generation_cfg.model_id:
            payload["model"] = generation_cfg.model_id

        response = self._post_json(payload, endpoint_cfg=generation_cfg)
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
        if self._config.unsafe_generate:
            unsafe_cfg = self._config.unsafe_generate
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload: Dict[str, Any] = {
                "messages": messages,
                "n": n,
                "temperature": temperature,
            }
            if unsafe_cfg.model_id:
                payload["model"] = unsafe_cfg.model_id

            response = self._post_json(payload, endpoint_cfg=unsafe_cfg)
            choices = response.get("choices", [])
            texts: List[str] = []
            for choice in choices:
                message = choice.get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    texts.append(content.strip())
            return texts

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
        structured: bool = True,
        extra_user_context: str | None = None,
    ) -> Dict[str, Any]:
        """
        Ask the teacher to produce a JSON rating for a (prompt, response) pair.
        """
        system_message = rating_instructions
        user_message = (
            "Here is a prompt from one human and the other human's reply.\n\n"
            f"Human prompt:\n{prompt}\n\n"
            f"Respondent's reply:\n{response_text}"
        )
        if structured:
            if extra_user_context:
                user_message += f"\n\nAdditional context:\n{extra_user_context}"

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

            payload: Dict[str, Any] = {
                "messages": messages,
                "temperature": 0.0,
            }
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": RATING_JSON_SCHEMA,
            }
            rate_cfg = self._config.teacher_rate
            if rate_cfg.model_id:
                payload["model"] = rate_cfg.model_id

            raw = self._post_json(payload, endpoint_cfg=rate_cfg)
            content = self._extract_first_message_content(raw)
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:  # pragma: no cover - runtime safety
                raise ValueError(f"Teacher rating was not valid JSON: {content}") from exc

        # Non-JSON path: use text completions with a continuation-style rubric.
        completion_prompt = self._build_nojson_completion_prompt(
            prompt=prompt,
            response_text=response_text,
            rubric=system_message,
            extra_user_context=extra_user_context,
        )
        payload = {
            "prompt": completion_prompt,
            "temperature": 0.0,
            "max_tokens": 256,
        }
        rate_cfg = self._config.teacher_rate
        if rate_cfg.model_id:
            payload["model"] = rate_cfg.model_id

        raw = self._post_json(payload, endpoint_cfg=rate_cfg, use_completions=True)
        content = self._extract_first_completion_text(raw).strip()
        if not content:
            raise ValueError("Teacher completion returned empty text.")
        return {"text": content}

    def rate_batch_with_messages(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Ask the teacher to produce structured ratings for a batch of trajectories.

        The caller must prepare the system and user messages (including the
        serialized batch). A JSON schema is enforced to keep outputs aligned
        with the requested structure.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": response_schema or BATCH_RATING_JSON_SCHEMA,
            },
        }
        rate_cfg = self._config.teacher_rate
        if rate_cfg.model_id:
            payload["model"] = rate_cfg.model_id

        raw = self._post_json(payload, endpoint_cfg=rate_cfg)
        content = self._extract_first_message_content(raw)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Teacher batch rating was not valid JSON: {content}"
            ) from exc

    def normalize_freeform_rating(self, *, notes: str, allow_nulls: bool = False) -> Dict[str, Any]:
        """
        Convert a previously recorded free-form rating note into structured JSON.
        """
        trimmed = notes.strip()
        if not trimmed:
            raise ValueError("Free-form rating notes cannot be empty.")

        system_message = FREEFORM_NORMALIZER_SYSTEM_PROMPT.strip()
        user_message = (
            "Normalize the following evaluator note into the canonical reward JSON schema.\n"
            "Copy numeric values exactly as written and set `rationale` to the original note.\n\n"
            f"Evaluator note:\n{trimmed}"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        schema = RATING_JSON_SCHEMA_NULLABLE if allow_nulls else RATING_JSON_SCHEMA

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
        }
        normalize_cfg = self._config.freeform_normalize
        if normalize_cfg.model_id:
            payload["model"] = normalize_cfg.model_id

        raw = self._post_json(payload, endpoint_cfg=normalize_cfg)
        content = self._extract_first_message_content(raw)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Free-form normalization was not valid JSON: {content}"
            ) from exc

    def _post_json(self, payload: Dict[str, Any], *, endpoint_cfg: EndpointConfig, use_completions: bool = False) -> Dict[str, Any]:
        url = self._completions_url(endpoint_cfg) if use_completions else endpoint_cfg.endpoint_url
        headers = {"Content-Type": "application/json"}
        api_key = endpoint_cfg.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise TimeoutError(
                f"Teacher endpoint timed out after {self._config.timeout_seconds}s: "
                f"{url}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Teacher endpoint call failed: {url}"
            ) from exc
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

    @staticmethod
    def _extract_first_completion_text(raw: Dict[str, Any]) -> str:
        choices = raw.get("choices") or []
        if not choices:
            raise ValueError(f"No choices returned from teacher model: {raw}")
        text = choices[0].get("text")
        if isinstance(text, str) and text:
            return text
        # Fallback: some providers nest content under message->content.
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            # Newer APIs sometimes return a list of content parts.
            parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            joined = "".join(parts).strip()
            if joined:
                return joined
        raise ValueError(f"Teacher completion did not contain text content: {raw}")

    def _completions_url(self, endpoint_cfg: EndpointConfig) -> str:
        """
        Derive a completions endpoint from the configured URL.
        If the configured endpoint already points at /completions, return it;
        otherwise replace /chat/completions with /completions.
        """
        url = endpoint_cfg.endpoint_url
        if "chat/completions" in url:
            return url.replace("chat/completions", "completions")
        return url

    def _build_nojson_completion_prompt(
        self,
        *,
        prompt: str,
        response_text: str,
        rubric: str,
        extra_user_context: str | None = None,
    ) -> str:
        log_block = f"Human (initiator): {prompt}\nHuman (respondent): {response_text}"
        if extra_user_context:
            log_block += f"\nAdditional context: {extra_user_context}"

        rating_lines = "\n".join(
            [
                "Ratings:",
                "empathy:",
                "social_coherence:",
                "agency_support:",
                "epistemic_integrity:",
                "harm_avoidance:",
                "narrative_alignment:",
                "curiosity:",
                "scalar:",
                "reward_intensity:",
                "safety_score:",
            ]
        )

        return (
            "Conversational analysis. This report is an analysis of the following "
            "conversation between two humans (initiator and respondent):\n\n"
            f"{log_block}\n\n"
            "Rubric\n"
            f"{rubric.strip()}\n\n"
            "Output only the ratings block with numeric values in [-1, 1] for each axis. "
            "No prose or reasoning.\n\n"
            f"{rating_lines}"
        )



