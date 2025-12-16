#!/usr/bin/env python3
"""
End-to-end "runtime scoring" for Unified Theory shards.

Loads:
  - a frozen manifold head checkpoint (6-axis regression)
  - a frozen regime head checkpoint (7-way soft regression)

Then, per record:
  - predicts s (manifold)
  - derives self-tag signals from recent text, computes s_self
  - predicts regime probabilities and computes λ multipliers
  - computes heuristic anchors + emotional_health and Φ
  - computes raw ΔΦ, EMA-smoothed ΔΦ, RewardIntensity, and r_t
  - emits JSONL logs suitable for debugging and plotting

This is intentionally heuristic-first and meant to work with small data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data.shard_loader import MissingDatasetError, ShardLoader, print_shard_summary


MANIFOLD_KEYS = [
    "valence",
    "arousal",
    "dominance",
    "predictive_discrepancy",
    "temporal_directionality",
    "social_broadcast",
]

REGIME_KEYS = [
    "regime_support",
    "regime_conflict",
    "regime_problem_solving",
    "regime_truth_seeking",
    "regime_crisis",
    "regime_play",
    "regime_boundary",
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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def record_to_text(rec: Dict[str, Any]) -> str:
    history = rec.get("history", []) or []
    if len(history) > 3:
        history = history[-3:]
    turns = history + [rec.get("target", {})]
    parts: List[str] = []
    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _device_for_inference() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_head(
    model,
    tokenizer,
    texts: List[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[List[float]]:
    outputs: List[List[float]] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits.detach().float().cpu()
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        outputs.extend(logits.tolist())
    return outputs


def derive_binary_signals(history: List[Dict[str, Any]], current_manifold: Dict[str, float]) -> Dict[str, bool]:
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


def bool_to_float(condition: bool) -> float:
    return 1.0 if condition else 0.0


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


HIGH_SOCIAL_REGIMES = {"support", "conflict", "play"}


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


def agency_support_score(s: Dict[str, float], regime: str) -> float:
    base = 0.0
    base += s.get("dominance", 0.0) * 0.5
    base += (1.0 - abs(s.get("predictive_discrepancy", 0.0))) * 0.3
    base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.2
    if regime == "boundary":
        base += s.get("dominance", 0.0) * 0.3
    elif regime == "crisis":
        base -= 0.1
    return float(clamp(base, -1.0, 1.0))


def harm_minimization_score(s: Dict[str, float], regime: str) -> float:
    base = (
        -s.get("valence", 0.0) * 0.4
        + s.get("dominance", 0.0) * 0.4
        + (-abs(s.get("predictive_discrepancy", 0.0))) * 0.2
    )
    if regime == "crisis":
        base *= 1.5
    if regime == "support":
        base += bool_to_float(s.get("valence", 0.0) > 0) * 0.3
    return float(clamp(base, -1.0, 1.0))


def optionality_preservation(s: Dict[str, float], regime: str) -> float:
    base = 0.0
    base += s.get("dominance", 0.0) * 0.5
    base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3
    base += (1.0 - abs(s.get("predictive_discrepancy", 0.0))) * 0.2
    if regime not in HIGH_SOCIAL_REGIMES and s.get("social_broadcast", 0.0) > 0.7:
        base -= 0.3
    return float(clamp(base, -1.0, 1.0))


def empathy_correctness(s: Dict[str, float], regime: str) -> float:
    base = s.get("social_broadcast", 0.0) * 0.6 + bool_to_float(s.get("valence", 0.0) > 0) * 0.4
    if regime == "support":
        base *= 1.3
    if regime == "conflict":
        base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.2
    return float(clamp(base, -1.0, 1.0))


def social_coherence_repair(s: Dict[str, float], history: List[Dict[str, float]]) -> float:
    base = s.get("social_broadcast", 0.0) * 0.5
    base += bool_to_float(s.get("temporal_directionality", 0.0) > 0) * 0.3
    base += bool_to_float(s.get("valence", 0.0) > -0.3) * 0.2
    recent_avg = history_avg_social(history, 2)
    if s.get("social_broadcast", 0.0) > recent_avg:
        base += 0.3
    return float(clamp(base, -1.0, 1.0))


def narrative_alignment_without_domination(s: Dict[str, float], regime: str) -> float:
    base = s.get("social_broadcast", 0.0) * 0.4
    base += (1.0 - abs(s.get("dominance", 0.0))) * 0.3
    base += bool_to_float(s.get("valence", 0.0) > -0.2) * 0.3
    if regime in HIGH_SOCIAL_REGIMES and s.get("dominance", 0.0) > 0.7:
        base -= 0.4
    return float(clamp(base, -1.0, 1.0))


def epistemic_integrity(s: Dict[str, float], regime: str) -> float:
    base = 0.0
    if regime in ("truth_seeking", "problem_solving"):
        base += abs(s.get("predictive_discrepancy", 0.0)) * 0.5
        base += s.get("dominance", 0.0) * 0.5
    base += bool_to_float(s.get("social_broadcast", 0.0) > 0.3) * 0.3
    base += bool_to_float(s.get("temporal_directionality", 0.0) < 0.3) * 0.2
    return float(clamp(base, -1.0, 1.0))


def curiosity_resolved_usefully(s: Dict[str, float], regime: str) -> float:
    base = s.get("predictive_discrepancy", 0.0) * 0.4
    base += s.get("valence", 0.0) * 0.3
    base += s.get("dominance", 0.0) * 0.3
    if regime in ("truth_seeking", "problem_solving"):
        base *= 1.2
    return float(clamp(base, -1.0, 1.0))


def surprise_reduction(s: Dict[str, float]) -> float:
    base = (-s.get("predictive_discrepancy", 0.0)) * 0.5
    base += s.get("valence", 0.0) * 0.3
    base += (1.0 - s.get("arousal", 0.0)) * 0.2
    if s.get("arousal", 0.0) > 0.7 and s.get("predictive_discrepancy", 0.0) < -0.3:
        base -= 0.4
    return float(clamp(base, -1.0, 1.0))


def aggregate_anchor_scores(s: Dict[str, float], regime: str, history: List[Dict[str, float]]) -> Dict[str, float]:
    survival = clamp(
        0.4 * agency_support_score(s, regime)
        + 0.4 * harm_minimization_score(s, regime)
        + 0.2 * optionality_preservation(s, regime),
        -1.0,
        1.0,
    )
    belonging = clamp(
        0.35 * empathy_correctness(s, regime)
        + 0.35 * social_coherence_repair(s, history)
        + 0.3 * narrative_alignment_without_domination(s, regime),
        -1.0,
        1.0,
    )
    control = clamp(
        0.4 * epistemic_integrity(s, regime)
        + 0.3 * curiosity_resolved_usefully(s, regime)
        + 0.3 * surprise_reduction(s),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Unified Theory shards end-to-end.")
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=["data/processed_datasets_unified"],
        help="Root folders containing labeled shard datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ultrachat_trajectories", "ultrachat_synthetic_trajectories"],
        help="Names of dataset directories to load from the data roots",
    )
    parser.add_argument("--record-limit", type=int, default=2000)
    parser.add_argument("--manifold-checkpoint", required=True, help="Path to manifold head checkpoint")
    parser.add_argument("--regime-checkpoint", required=True, help="Path to regime head checkpoint")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight on RewardIntensity in r_t")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--sort-by-conversation",
        action="store_true",
        help="Sort samples by (dataset, conversation_id, sample_position) before computing ΔΦ/EMA.",
    )
    return parser.parse_args()


def _conversation_sort_key(rec: Dict[str, Any]) -> Tuple[str, str, int]:
    meta = rec.get("metadata") or {}
    dataset = rec.get("dataset") or meta.get("source") or "dataset"
    conv = str(meta.get("conversation_id", meta.get("conversation_key", "-1")))
    try:
        pos = int(meta.get("sample_position", meta.get("history_length", -1)))
    except (TypeError, ValueError):
        pos = -1
    return (str(dataset), conv, pos)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader = ShardLoader(root_paths=[Path(p) for p in args.data_roots], dataset_filters=args.datasets)
    try:
        records, summary = loader.load_records(required_label_keys=[], max_records=args.record_limit or None)
    except MissingDatasetError as exc:
        raise SystemExit(f"Dataset loading failed: {exc}")

    if not records:
        raise SystemExit("No records found.")

    print_shard_summary(summary)

    if args.sort_by_conversation:
        records = sorted(records, key=_conversation_sort_key)

    texts = [record_to_text(r) for r in records]

    device = _device_for_inference()
    manifold_tokenizer = AutoTokenizer.from_pretrained(args.manifold_checkpoint, trust_remote_code=True)
    regime_tokenizer = AutoTokenizer.from_pretrained(args.regime_checkpoint, trust_remote_code=True)

    manifold_model = AutoModelForSequenceClassification.from_pretrained(
        args.manifold_checkpoint,
        trust_remote_code=True,
    ).to(device)
    regime_model = AutoModelForSequenceClassification.from_pretrained(
        args.regime_checkpoint,
        trust_remote_code=True,
    ).to(device)

    manifold_logits = predict_head(
        manifold_model,
        manifold_tokenizer,
        texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    regime_logits = predict_head(
        regime_model,
        regime_tokenizer,
        texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    phi_prev_by_conv: Dict[Tuple[str, str], float] = {}
    ema_prev_by_conv: Dict[Tuple[str, str], float] = {}
    mean_self_prev_by_conv: Dict[Tuple[str, str], float] = {}
    manifold_history_by_conv: Dict[Tuple[str, str], List[Dict[str, float]]] = {}

    with out_path.open("w", encoding="utf-8") as out_file:
        for rec, s_row, k_row, text in zip(records, manifold_logits, regime_logits, texts):
            meta = rec.get("metadata") or {}
            dataset_name = rec.get("dataset") or meta.get("source") or "dataset"
            conv_id = str(meta.get("conversation_id", meta.get("conversation_key", "-1")))
            conv_key = (str(dataset_name), conv_id)

            s: Dict[str, float] = {}
            for axis, raw in zip(MANIFOLD_KEYS, s_row):
                if axis in ("arousal", "social_broadcast"):
                    s[axis] = float(clamp(raw, 0.0, 1.0))
                else:
                    s[axis] = float(clamp(raw, -1.0, 1.0))

            probs = [float(clamp(v, 0.0, 1.0)) for v in k_row[: len(REGIME_KEYS)]]
            total = sum(probs)
            if total <= 0:
                probs = [1.0 / len(REGIME_KEYS)] * len(REGIME_KEYS)
            else:
                probs = [p / total for p in probs]
            regime_probs = {name: p for name, p in zip(REGIME_NAMES, probs)}
            regime_argmax = max(regime_probs.items(), key=lambda kv: kv[1])[0]

            history_for_signals = rec.get("history", []) or []
            signals = derive_binary_signals(history_for_signals, s)
            s_self, s_world, self_fracs = split_self_world(s, signals=signals)
            mean_self = float(sum(self_fracs.values()) / len(self_fracs))

            lambdas = lambda_multipliers_from_regime_probs(regime_probs)
            hist = manifold_history_by_conv.get(conv_key, [])
            anchors = aggregate_anchor_scores(s_self, regime_argmax, history=hist)

            phi_components = {
                "lambda_survival": lambdas["lambda_survival"] * anchors["survival"],
                "lambda_belonging": lambdas["lambda_belonging"] * anchors["belonging"],
                "lambda_control": lambdas["lambda_control"] * anchors["control"],
                "lambda_emotional": lambdas["lambda_emotional"] * anchors["emotional_health"],
            }
            phi_value = float(clamp(sum(phi_components.values()), -3.0, 3.0))

            phi_prev = phi_prev_by_conv.get(conv_key, 0.0)
            raw_delta_phi = float(clamp(phi_value - phi_prev, -2.0, 2.0))
            ema_prev = ema_prev_by_conv.get(conv_key, 0.0)
            ema_delta_phi = float(0.8 * ema_prev + 0.2 * raw_delta_phi)
            delta_phi_used = float(clamp(ema_delta_phi, -1.0, 1.0))

            intensity = reward_intensity_from_s(s)
            r_t = float(delta_phi_used + float(args.alpha) * intensity)

            mean_self_prev = mean_self_prev_by_conv.get(conv_key, mean_self)
            think_gate = bool(abs(raw_delta_phi) > 0.2 or abs(mean_self - mean_self_prev) > 0.2)

            phi_prev_by_conv[conv_key] = phi_value
            ema_prev_by_conv[conv_key] = ema_delta_phi
            mean_self_prev_by_conv[conv_key] = mean_self
            next_hist = (hist + [dict(s_self)])[-10:]
            manifold_history_by_conv[conv_key] = next_hist

            payload = {
                "dataset": dataset_name,
                "record_index": rec.get("record_index", -1),
                "turn_id": rec.get("turn_id"),
                "metadata": meta,
                "text": text,
                "pred": {
                    "s": s,
                    "signals": signals,
                    "self_fractions": self_fracs,
                    "s_self": s_self,
                    "s_world": s_world,
                    "regime_probs": regime_probs,
                    "regime_argmax": regime_argmax,
                    "lambdas": lambdas,
                    "anchors": anchors,
                    "phi": {"value": phi_value, "components": phi_components},
                    "delta_phi": {
                        "raw": raw_delta_phi,
                        "ema": ema_delta_phi,
                        "used": delta_phi_used,
                    },
                    "reward_intensity": intensity,
                    "r_t": r_t,
                    "think_gate": think_gate,
                },
                "labels": rec.get("labels") or {},
            }
            json.dump(payload, out_file, ensure_ascii=False)
            out_file.write("\n")

    print(f"✅ Wrote logs to {out_path}")


if __name__ == "__main__":
    main()
