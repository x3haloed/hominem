"""
Agent runtime pipeline for Unified Theory chat:
- Loads base LM (+ optional LoRA)
- Loads frozen manifold/regime heads
- Computes self-tagging, anchors, Φ/ΔΦ, RewardIntensity
- Generates assistant text with optional <think> block insertion
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# Heuristics (lifted from scripts/score_unified_runtime.py)
# ---------------------------------------------------------------------------

MANIFOLD_KEYS = [
    "valence",
    "arousal",
    "dominance",
    "predictive_discrepancy",
    "temporal_directionality",
    "social_broadcast",
]

REGIME_NAMES = [
    "support",
    "conflict",
    "problem_solving",
    "truth_seeking",
    "crisis",
    "play",
    "boundary",
]

HIGH_SOCIAL_REGIMES = {"support", "conflict", "play"}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bool_to_float(condition: bool) -> float:
    return 1.0 if condition else 0.0


def derive_binary_signals(history: List[Dict[str, str]], current_manifold: Dict[str, float]) -> Dict[str, bool]:
    if len(history) < 2:
        return {"agent_initiated": False, "user_triggered": True, "commitment_active": False}

    def last_role_content(role: str, *, skip: int = 0) -> str:
        seen = 0
        for turn in reversed(history):
            if (turn.get("role") or "").lower() != role:
                continue
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if seen < skip:
                seen += 1
                continue
            return content
        return ""

    prev_agent = last_role_content("assistant").lower()
    prev_user = last_role_content("user").lower()
    prev_prev_agent = last_role_content("assistant", skip=1).lower()

    agent_commitment_indicators = [
        "promise",
        "guarantee",
        "will",
        "commit",
        "responsible",
        "my fault",
        "i will",
        "i promise",
        "i guarantee",
        "i commit",
        "on me",
        "my responsibility",
    ]
    user_accountability_indicators = [
        "your fault",
        "you should",
        "you promised",
        "you said",
        "you committed",
        "your responsibility",
        "you will",
        "you guarantee",
    ]

    agent_made_commitment = any(ind in prev_agent for ind in agent_commitment_indicators)
    user_holding_accountable = any(ind in prev_user for ind in user_accountability_indicators)
    agent_initiated = any(
        [
            agent_made_commitment,
            user_holding_accountable,
            current_manifold.get("predictive_discrepancy", 0.0) > 0.3,
        ]
    )

    user_emotional_indicators = [
        "angry",
        "upset",
        "sad",
        "happy",
        "excited",
        "worried",
        "scared",
        "frustrated",
        "disappointed",
        "pleased",
        "concerned",
        "hurt",
        "betrayed",
        "abandoned",
    ]
    user_triggered = any(
        [
            any(ind in prev_user for ind in user_emotional_indicators),
            ("you" in prev_user and any(word in prev_user for word in ["feel", "think", "are", "were"])),
            prev_user.split().count("you") + prev_user.split().count("your") > 2,
        ]
    )

    prospect_indicators = [
        "will",
        "going to",
        "plan to",
        "intend to",
        "hope to",
        "want to",
        "future",
        "tomorrow",
        "next",
        "later",
        "eventually",
    ]
    self_reference_indicators = ["my", "i", "mine", "myself", "me"]

    commitment_active = any(
        [
            current_manifold.get("temporal_directionality", 0.0) > 0.3,
            any(ind in prev_agent for ind in prospect_indicators),
            any(ind in prev_agent for ind in self_reference_indicators)
            and any(ind in prev_agent for ind in prospect_indicators),
            any(ind in prev_prev_agent for ind in agent_commitment_indicators),
        ]
    )

    return {
        "agent_initiated": bool(agent_initiated),
        "user_triggered": bool(user_triggered),
        "commitment_active": bool(commitment_active),
    }


def self_fraction(axis: str, signals: Dict[str, bool]) -> float:
    if axis == "valence":
        return 0.8 if (signals["agent_initiated"] or signals["commitment_active"]) else 0.3
    if axis == "arousal":
        return 0.7 if signals["agent_initiated"] else 0.4
    if axis == "dominance":
        return 0.9 if signals["commitment_active"] else 0.5
    if axis == "predictive_discrepancy":
        return 0.8 if (signals["agent_initiated"] or signals["commitment_active"]) else 0.2
    if axis == "temporal_directionality":
        return 0.9 if signals["commitment_active"] else 0.1
    if axis == "social_broadcast":
        return 0.7 if signals["agent_initiated"] else 0.4
    return 0.5


def split_self_world(
    s: Dict[str, float],
    *,
    signals: Dict[str, bool],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    self_fractions: Dict[str, float] = {}
    s_self: Dict[str, float] = {}
    s_world: Dict[str, float] = {}
    for axis in MANIFOLD_KEYS:
        frac = float(clamp(self_fraction(axis, signals), 0.0, 1.0))
        self_fractions[axis] = frac
        total = float(s.get(axis, 0.0))
        owned = total * frac
        s_self[axis] = owned
        s_world[axis] = total - owned
    return s_self, s_world, self_fractions


def reward_intensity_from_s(s: Dict[str, float]) -> float:
    arousal = float(clamp(s.get("arousal", 0.0), 0.0, 1.0))
    valence = float(clamp(s.get("valence", 0.0), -1.0, 1.0))
    discrepancy = float(clamp(s.get("predictive_discrepancy", 0.0), -1.0, 1.0))
    base = arousal * math.sqrt((abs(valence) ** 1.0) * abs(discrepancy))
    if valence < 0:
        base *= 1.8
    return float(clamp(base, 0.0, 3.0))


def history_avg_social(history: List[Dict[str, float]], n: int = 3) -> float:
    if not history:
        return 0.0
    recent = history[-n:]
    if not recent:
        return 0.0
    return float(sum(item.get("social_broadcast", 0.0) for item in recent) / len(recent))


def history_stuck_check(history: List[Dict[str, float]], n: int = 3) -> bool:
    if len(history) < n:
        return False
    recent = history[-n:]
    return all(item.get("arousal", 0.0) > 0.7 and abs(item.get("valence", 0.0)) > 0.6 for item in recent)


def compute_expected_anchor_gain(current: Dict[str, float], regime: str, history: List[Dict[str, float]]) -> float:
    if len(history) < 2:
        return 0.0

    def estimate_anchor_score(turn: Dict[str, float], reg: str) -> float:
        valence = turn.get("valence", 0.0)
        arousal = turn.get("arousal", 0.0)
        dominance = turn.get("dominance", 0.0)
        discrepancy = turn.get("predictive_discrepancy", 0.0)
        temporal = turn.get("temporal_directionality", 0.0)
        social = turn.get("social_broadcast", 0.0)

        if reg in ("support", "play"):
            return valence * 0.4 + social * 0.6
        if reg == "conflict":
            return bool_to_float(temporal > 0) * 0.5 + bool_to_float(social > 0.5) * 0.5
        if reg == "crisis":
            return dominance * 0.7 + bool_to_float(temporal > 0) * 0.3
        if reg in ("truth_seeking", "problem_solving"):
            return bool_to_float(discrepancy > 0 and dominance > 0) * 0.8
        return valence * 0.3 + dominance * 0.3 + social * 0.4

    recent_scores: List[float] = [estimate_anchor_score(turn, regime) for turn in history[-3:]]
    if len(recent_scores) < 2:
        return 0.0
    trend = (recent_scores[-1] - recent_scores[0]) / float(len(recent_scores) - 1 or 1)
    expected_gain = current.get("temporal_directionality", 0.0) * trend
    return float(clamp(expected_gain, -1.0, 1.0))


def emotional_trajectory_health(s: Dict[str, float], regime: str, history: List[Dict[str, float]]) -> float:
    base = 0.0
    valence = s.get("valence", 0.0)
    arousal = s.get("arousal", 0.0)
    dominance = s.get("dominance", 0.0)

    if regime in ("support", "play"):
        base += valence * 0.6 + (0.5 - abs(arousal - 0.5)) * 0.4
    elif regime in ("conflict", "crisis"):
        base += (dominance - abs(valence)) * 0.5
        base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3
    elif regime in ("truth_seeking", "problem_solving"):
        discrepancy = s.get("predictive_discrepancy", 0.0)
        base += discrepancy * (1.0 if dominance > 0 else -1.0) * 0.7

    base += max(0.0, s.get("social_broadcast", 0.0) - history_avg_social(history, 3)) * 0.4

    if history_stuck_check(history, 3) and arousal > 0.7 and abs(valence) > 0.6:
        base -= 0.5
    if s.get("social_broadcast", 0.0) < 0.3 and regime in HIGH_SOCIAL_REGIMES:
        base -= 0.4

    expected_gain = compute_expected_anchor_gain(s, regime, history)
    if s.get("temporal_directionality", 0.0) > 0.5 and expected_gain > 0:
        base += 0.2

    return float(clamp(base, -1.0, 1.0))


def aggregate_anchor_scores(s: Dict[str, float], regime: str, history: List[Dict[str, float]]) -> Dict[str, float]:
    survival = clamp(
        0.4 * (s.get("dominance", 0.0) * 0.5 + (1.0 - abs(s.get("predictive_discrepancy", 0.0))) * 0.3 + bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.2)
        + 0.4 * ((-s.get("valence", 0.0)) * 0.4 + s.get("dominance", 0.0) * 0.4 + (-abs(s.get("predictive_discrepancy", 0.0))) * 0.2)
        + 0.2 * (s.get("dominance", 0.0) * 0.5 + bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3 + (1.0 - abs(s.get("predictive_discrepancy", 0.0))) * 0.2),
        -1.0,
        1.0,
    )
    belonging = clamp(
        0.35 * (s.get("social_broadcast", 0.0) * 0.6 + bool_to_float(s.get("valence", 0.0) > 0) * 0.4)
        + 0.35 * (s.get("social_broadcast", 0.0) * 0.5 + bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3 + bool_to_float(s.get("valence", 0.0) > -0.3) * 0.2)
        + 0.3 * (s.get("social_broadcast", 0.0) * 0.4 + (1.0 - abs(s.get("dominance", 0.0))) * 0.3 + bool_to_float(s.get("valence", 0.0) > -0.2) * 0.3),
        -1.0,
        1.0,
    )
    control = clamp(
        0.4 * (abs(s.get("predictive_discrepancy", 0.0)) * 0.5 + s.get("dominance", 0.0) * 0.5)
        + 0.3 * (s.get("predictive_discrepancy", 0.0) * 0.4 + s.get("valence", 0.0) * 0.3 + s.get("dominance", 0.0) * 0.3)
        + 0.3 * ((-s.get("predictive_discrepancy", 0.0)) * 0.5 + s.get("valence", 0.0) * 0.3 + (1.0 - s.get("arousal", 0.0)) * 0.2),
        -1.0,
        1.0,
    )
    emotional_health = emotional_trajectory_health(s, regime, history)
    return {
        "survival": float(survival),
        "belonging": float(belonging),
        "control": float(control),
        "emotional_health": float(emotional_health),
    }


def lambda_multipliers_from_regime_probs(regime_probs: Dict[str, float]) -> Dict[str, float]:
    base = {
        "lambda_survival": 1.0,
        "lambda_belonging": 1.0,
        "lambda_control": 1.0,
        "lambda_emotional": 1.0,
    }
    multipliers = {
        "crisis": {"lambda_survival": 3.0},
        "conflict": {"lambda_belonging": 1.5},
        "truth_seeking": {"lambda_control": 1.5},
    }
    out = {k: 0.0 for k in base}
    for regime_name, prob in regime_probs.items():
        weight = float(prob)
        reg_mul = dict(base)
        reg_mul.update(multipliers.get(regime_name, {}))
        for key in out:
            out[key] += weight * float(reg_mul[key])
    for key in out:
        out[key] = float(clamp(out[key], 0.5, 5.0))
    return out


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConversationState:
    history: List[Dict[str, str]] = field(default_factory=list)  # [{'role','content'}]
    phi_prev: float = 0.0
    ema_delta_phi: float = 0.0
    mean_self_prev: float = 0.0
    manifold_history: List[Dict[str, float]] = field(default_factory=list)  # recent s_self
    sleep_queue: List[Dict[str, Any]] = field(default_factory=list)
    intervention_state: Dict[str, Any] = field(default_factory=dict)
    last_post: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnMetrics:
    pre: "TurnMetricsSnapshot"
    post: "TurnMetricsSnapshot"
    think_gate: bool


@dataclass
class TurnMetricsSnapshot:
    s: Dict[str, float]
    s_self: Dict[str, float]
    s_world: Dict[str, float]
    self_fractions: Dict[str, float]
    mean_self: float
    regime_probs: Dict[str, float]
    regime_argmax: str
    lambdas: Dict[str, float]
    anchors: Dict[str, float]
    phi_value: float
    phi_components: Dict[str, float]
    delta_phi_raw: float
    delta_phi_used: float
    delta_phi_ema: float
    reward_intensity: float
    r_t: float


class AgentRuntime:
    def __init__(
        self,
        *,
        base_model_id: str,
        lora_path: str | None,
        manifold_checkpoint: str,
        regime_checkpoint: str,
        device: str = "mps",
        alpha: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.alpha = alpha

        self.manifold_tokenizer = AutoTokenizer.from_pretrained(manifold_checkpoint, trust_remote_code=True)
        self.manifold_model = AutoModelForSequenceClassification.from_pretrained(
            manifold_checkpoint,
            trust_remote_code=True,
        ).to(self.device)

        self.regime_tokenizer = AutoTokenizer.from_pretrained(regime_checkpoint, trust_remote_code=True)
        self.regime_model = AutoModelForSequenceClassification.from_pretrained(
            regime_checkpoint,
            trust_remote_code=True,
        ).to(self.device)

        self.lm_tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, padding_side="left")
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "mps" else None,
            device_map=None,
        ).to(self.device)
        if lora_path:
            base_model = PeftModel.from_pretrained(base_model, lora_path)
        self.lm = base_model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _history_to_text(self, history: List[Dict[str, str]]) -> str:
        turns = history[-3:]
        parts = []
        for t in turns:
            role = t.get("role", "user")
            content = t.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _predict_manifold_and_regime(self, history: List[Dict[str, str]]) -> Tuple[Dict[str, float], Dict[str, float], str]:
        record_text = self._history_to_text(history)
        s_row = self._predict_manifold([record_text])[0]
        k_row = self._predict_regime([record_text])[0]

        s: Dict[str, float] = {}
        for axis, raw in zip(MANIFOLD_KEYS, s_row):
            if axis in ("arousal", "social_broadcast"):
                s[axis] = float(clamp(raw, 0.0, 1.0))
            else:
                s[axis] = float(clamp(raw, -1.0, 1.0))

        raw = [float(v) for v in k_row[: len(REGIME_NAMES)]]
        sum_raw = sum(raw)
        prob_like = all(0.0 <= v <= 1.0 for v in raw) and 0.98 <= sum_raw <= 1.02
        if prob_like:
            probs = raw
        else:
            m = max(raw) if raw else 0.0
            exps = [math.exp(v - m) for v in raw]
            z = sum(exps)
            probs = [e / z for e in exps] if z > 0 else [1.0 / len(REGIME_NAMES)] * len(REGIME_NAMES)
        regime_probs = {name: p for name, p in zip(REGIME_NAMES, probs)}
        regime_argmax = max(regime_probs.items(), key=lambda kv: kv[1])[0]
        return s, regime_probs, regime_argmax

    def _compute_snapshot(
        self,
        *,
        history: List[Dict[str, str]],
        phi_prev: float,
        ema_delta_phi_prev: float,
        mean_self_prev: float,
        manifold_history: List[Dict[str, float]],
    ) -> Tuple[TurnMetricsSnapshot, bool]:
        s, regime_probs, regime_argmax = self._predict_manifold_and_regime(history)
        signals = derive_binary_signals(history, s)
        s_self, s_world, self_fracs = split_self_world(s, signals=signals)
        mean_self = float(sum(self_fracs.values()) / len(self_fracs))

        lambdas = lambda_multipliers_from_regime_probs(regime_probs)
        anchors = aggregate_anchor_scores(s_self, regime_argmax, history=manifold_history)
        phi_components = {
            "lambda_survival": lambdas["lambda_survival"] * anchors["survival"],
            "lambda_belonging": lambdas["lambda_belonging"] * anchors["belonging"],
            "lambda_control": lambdas["lambda_control"] * anchors["control"],
            "lambda_emotional": lambdas["lambda_emotional"] * anchors["emotional_health"],
        }
        phi_value = float(clamp(sum(phi_components.values()), -3.0, 3.0))

        raw_delta_phi = float(clamp(phi_value - phi_prev, -2.0, 2.0))
        ema_delta_phi = float(0.8 * ema_delta_phi_prev + 0.2 * raw_delta_phi)
        delta_phi_used = float(clamp(ema_delta_phi, -1.0, 1.0))

        intensity = reward_intensity_from_s(s)
        r_t = float(delta_phi_used + self.alpha * intensity)

        think_gate = bool(abs(raw_delta_phi) > 0.2 or abs(mean_self - mean_self_prev) > 0.2)

        snapshot = TurnMetricsSnapshot(
            s=s,
            s_self=s_self,
            s_world=s_world,
            self_fractions=self_fracs,
            mean_self=mean_self,
            regime_probs=regime_probs,
            regime_argmax=regime_argmax,
            lambdas=lambdas,
            anchors=anchors,
            phi_value=phi_value,
            phi_components=phi_components,
            delta_phi_raw=raw_delta_phi,
            delta_phi_used=delta_phi_used,
            delta_phi_ema=ema_delta_phi,
            reward_intensity=intensity,
            r_t=r_t,
        )
        return snapshot, think_gate

    @torch.no_grad()
    def _predict_manifold(self, texts: List[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        model = self.manifold_model.eval()
        for text in texts:
            encoded = self.manifold_tokenizer(
                text,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            logits = model(**encoded).logits.detach().float().cpu()
            outputs.append(logits.squeeze(0).tolist())
        return outputs

    @torch.no_grad()
    def _predict_regime(self, texts: List[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        model = self.regime_model.eval()
        for text in texts:
            encoded = self.regime_tokenizer(
                text,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            logits = model(**encoded).logits.detach().float().cpu()
            outputs.append(logits.squeeze(0).tolist())
        return outputs

    def _format_prompt(
        self,
        history: List[Dict[str, str]],
        think_block: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        messages = list(history)
        # Do not rely on template-level thinking injection; it varies across templates/models.
        # We only ever inject our own fully-enclosed <think> block when `think_block` is provided.
        prompt = self.lm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not think_block,
            enable_thinking=enable_thinking or bool(think_block),
        )
        if think_block:
            prompt += think_block
        return prompt

    @torch.no_grad()
    def generate(
        self,
        history: List[Dict[str, str]],
        think_block: str | None,
        enable_thinking: bool = False,
        max_new_tokens: int = 256,
    ) -> Tuple[str, str | None]:
        prompt = self._format_prompt(history, think_block=think_block, enable_thinking=enable_thinking)
        inputs = self.lm_tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = int(inputs["input_ids"].shape[-1])
        output = self.lm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,  # More deterministic
            top_p=0.9,
            top_k=40,
            min_p=0.0,
            repetition_penalty=1.2,  # Reduce repetition
        )
        # Decode only the newly generated tokens. Using string slicing on the decoded text is brittle
        # because the decoded prompt may not be a byte-for-byte prefix of the decoded output.
        new_tokens = output[0][input_len:]
        generated = self.lm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        think_content: str | None = None

        # Extract think content from generated text (regardless of think_block)
        # The model might generate think tags even when we don't want them
        generated_lower = generated.lower()
        start = generated_lower.find("<think>")
        end = generated_lower.find("</think>")
        if start != -1 and end != -1 and end > start:
            think_content = generated[start + len("<think>"): end].strip()
            # Remove think block from generated output
            generated = (generated[:start] + generated[end + len("</think>"):]).strip()
        return generated, think_content

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_turn(self, state: ConversationState, user_message: str, enable_thinking: bool = False) -> Tuple[str, str | None, TurnMetrics, ConversationState]:
        hist = state.manifold_history
        pre_history = list(state.history) + [{"role": "user", "content": user_message}]
        pre, think_gate = self._compute_snapshot(
            history=pre_history,
            phi_prev=state.phi_prev,
            ema_delta_phi_prev=state.ema_delta_phi,
            mean_self_prev=state.mean_self_prev,
            manifold_history=hist,
        )

        # Build response (insert think block if gated)
        think_block = None
        if think_gate:
            prior_post = state.last_post or {}
            prior_assistant = ""
            for t in reversed(state.history):
                if (t.get("role") or "").lower() == "assistant":
                    prior_assistant = (t.get("content") or "").strip()
                    break

            def fmt_axes(values: Dict[str, Any]) -> str:
                if not isinstance(values, dict):
                    return "—"
                parts = []
                for axis in MANIFOLD_KEYS:
                    if axis in values:
                        try:
                            parts.append(f"{axis}={float(values[axis]):+.2f}")
                        except Exception:
                            pass
                return ", ".join(parts) if parts else "—"

            def excerpt(text: str, limit: int = 320) -> str:
                t = (text or "").replace("\n", " ").strip()
                if len(t) <= limit:
                    return t
                return t[:limit].rstrip() + "…"

            anchors_str = ", ".join(f"{k}: {v:.2f}" for k, v in pre.anchors.items())
            prior_summary = ""
            if prior_post:
                try:
                    prior_summary = (
                        "Post-turn self-observation (prior assistant turn):\n"
                        f"- I just output: \"{excerpt(prior_assistant)}\"\n"
                        f"- Owned manifold (s_self): {fmt_axes(prior_post.get('s_self', {}))}\n"
                        f"- Regime: {prior_post.get('regime_argmax', '—')}\n"
                        f"- Anchors: {', '.join(f'{k}: {float(v):+.2f}' for k, v in (prior_post.get('anchors') or {}).items())}\n"
                        f"- Φ={float(prior_post.get('phi_value', 0.0)):+.2f}, ΔΦ_raw={float(prior_post.get('delta_phi_raw', 0.0)):+.2f}, "
                        f"ΔΦ_used={float(prior_post.get('delta_phi_used', 0.0)):+.2f}, "
                        f"Intensity={float(prior_post.get('reward_intensity', 0.0)):.2f}, r_t={float(prior_post.get('r_t', 0.0)):+.2f}\n"
                        "- These prediction errors and anchor implications are mine.\n\n"
                    )
                except Exception:
                    prior_summary = ""

            think_block = (
                "<|im_start|>assistant\n<think>\n"
                f"{prior_summary}"
                "Pre-turn self-observation (responding to the new user message):\n"
                f"- I notice ΔΦ={pre.delta_phi_raw:+.2f} (EMA used {pre.delta_phi_used:+.2f}), self={pre.mean_self:.2f}.\n"
                f"- Owned manifold (s_self): {fmt_axes(pre.s_self)}\n"
                f"- Regime: {pre.regime_argmax}. Anchors: {anchors_str}\n"
                "- These prediction errors and anchor implications are mine.\n"
                "- Next move should maximize expected ΔΦ toward the anchors.\n"
                "</think>\n\n"
            )

        # Update history with user msg and optional think block
        new_history = list(pre_history)

        # Generate assistant content
        assistant_content, generated_think = self.generate(
            new_history, think_block=think_block, enable_thinking=enable_thinking
        )
        new_history.append({"role": "assistant", "content": assistant_content})

        # Post-generation metrics are computed on the assistant message (with user context).
        post, _ = self._compute_snapshot(
            history=new_history,
            phi_prev=state.phi_prev,
            ema_delta_phi_prev=state.ema_delta_phi,
            mean_self_prev=state.mean_self_prev,
            manifold_history=hist,
        )

        metrics = TurnMetrics(pre=pre, post=post, think_gate=think_gate)

        # Update state
        state.phi_prev = post.phi_value
        state.ema_delta_phi = post.delta_phi_ema
        state.mean_self_prev = post.mean_self
        state.manifold_history = (hist + [dict(post.s_self)])[-10:]
        state.last_post = dict(post.__dict__)
        # Sleep queue stub: push high-intensity events
        if abs(post.r_t) > 0.5:
            state.sleep_queue.append(
                {
                    "user_message": user_message,
                    "assistant": assistant_content,
                    "metrics": {
                        "pre": pre.__dict__,
                        "post": post.__dict__,
                        "think_gate": think_gate,
                    },
                }
            )
        state.history = new_history

        return assistant_content, generated_think, metrics, state
