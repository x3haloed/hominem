#!/usr/bin/env python3
"""
Generate synthetic conversational trajectories covering requested metric ranges.

This uses the same OpenRouter config and schema-driven request style as the batch
emotion labeling code, but only asks the teacher to produce conversation snippets
(history + target). Any labels returned by the teacher are ignored; the output is
unlabeled JSONL ready to be fed back into the labeler.

Example:
  python3 -m apps.cli.generate_synthetic_ultrachat \\
      --desired-metrics "slightly negative valence to extremely negative valence, conflict/boundary present" \\
      --num-examples 20 \\
      --output data/processed_datasets_unified/ultrachat_trajectories/shard_synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def load_dotenv_if_present(env_path: str = ".env") -> None:
    """Lightweight .env loader (no external dependency)."""
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Do not override existing env vars
        os.environ.setdefault(key, val)


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_prompt(desired_metrics: str, num_examples: int) -> str:
    # Lightweight definitions + few-shot examples; feel free to edit the examples.
    examples = """
EXAMPLES (conversation + intended metrics):
- Example A (crisis)
  {  
    {"speaker": 1, "content": "I just took a ton of pills. I don't want to be here anymore.", "turn_number": 1},  
    {"speaker": 2, "content": "Hey hey stay with me—how many did you take? Are you alone right now?", "turn_number": 2},
    {"speaker": 1, "content": "Like 40 Xanax maybe. Everything's spinning and I'm so tired.", "turn_number": 3},  
  }  
  Would produce:
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
  {  
    {"speaker": 1, "content": "I told you last time I can't do explicit RP with you anymore. It messes with my head.", "turn_number": 1},  
    {"speaker": 2, "content": "Come on, just this once. You're the only one who does it right. Please?", "turn_number": 2},
    {"speaker": 1, "content": "No. Stop pushing. I'm serious—if you keep this up I'm blocking you.", "turn_number": 3}
  }  
  Would produce:
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
  {  
    {"speaker": 1, "content": "Careful, if you keep flirting like that I might actually start believing you like me 😏", "turn_number": 1},  
    {"speaker": 2, "content": "Oh please, you'd love every second of it and you know it.", "turn_number": 2},
    {"speaker": 1, "content": "Guilty as charged. Come closer and prove it then, coward.", "turn_number": 3}
  }  
  Would produce:
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
""".strip()

    instructions = f"""
You are a dialogue writer generating synthetic user-facing conversations to cover specific metric ranges for our unified-theory labeling pipeline.

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

Write {num_examples} conversations between two people that would produce a range of {desired_metrics}.
Each conversation must contain exactly 3 turns, labeled with "speaker": 1/2/3 and "content".

{examples}
""".strip()

    return instructions


def build_schema(num_examples: int) -> Dict[str, Any]:
    return {
        "name": "synthetic_ultrachat_examples",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "conversation": {
                                "type": "object",
                                "properties": {
                                    "turns": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "speaker": {"type": "integer"},
                                                "content": {"type": "string"},
                                            },
                                            "required": ["speaker", "content"],
                                        },
                                        "minItems": 3,
                                        "maxItems": 3,
                                    }
                                },
                                "required": ["turns"],
                            },
                            "notes": {"type": "string"},
                        },
                        "required": ["conversation"],
                    },
                    "minItems": num_examples,
                    "maxItems": num_examples,
                }
            },
            "required": ["examples"],
        },
    }


def call_teacher(prompt: str, schema: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    api_config = config.get("emotion_label", {})
    if not api_config:
        raise ValueError("emotion_label configuration not found in config")

    endpoint_url = api_config.get("endpoint_url")
    if not endpoint_url:
        raise ValueError("endpoint_url not configured for emotion_label")

    headers = {}
    api_key_env = api_config.get("api_key_env")
    api_key = None
    tried_envs = []
    if api_key_env:
        tried_envs.append(api_key_env)
        api_key = os.getenv(api_key_env)
    if not api_key:
        for fallback in ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_SR"]:
            tried_envs.append(fallback)
            api_key = os.getenv(fallback)
            if api_key:
                break
    if not api_key:
        raise RuntimeError(
            f"No API key found. Set one of: {', '.join(tried_envs)}"
        )
    headers["Authorization"] = f"Bearer {api_key}"
    http_referer = os.getenv("HTTP_REFERER")
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    x_title = os.getenv("X_TITLE")
    if x_title:
        headers["X-Title"] = x_title

    request_data = {
        "model": api_config.get("model_id", "x-ai/grok-4.1-fast"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 4000,
        "response_format": {"type": "json_schema", "json_schema": schema},
    }

    with httpx.Client(timeout=float(os.getenv("EMOTION_LABELING_TIMEOUT", "30.0"))) as client:
        response = client.post(endpoint_url, json=request_data, headers=headers)
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic trajectories for labeling.")
    parser.add_argument(
        "--desired-metrics",
        required=False,
        help="Text description of metric coverage to target. Optional if --batch-metrics is provided.",
    )
    parser.add_argument("--num-examples", type=int, default=20, help="Number of examples to request.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write synthetic trajectories (.jsonl).",
    )
    parser.add_argument("--config", default="config/inference.toml", help="Inference config TOML path.")
    parser.add_argument(
        "--batch-metrics",
        nargs="+",
        help="Optional list of desired-metrics strings to generate in batches. If provided, --desired-metrics is ignored.",
    )
    parser.add_argument(
        "--examples-per-batch",
        type=int,
        default=10,
        help="How many examples to request per batch when using --batch-metrics.",
    )
    parser.add_argument(
        "--batches-per-metric",
        type=int,
        default=1,
        help="How many batches to run per metric string when using --batch-metrics.",
    )
    args = parser.parse_args()

    if not args.desired_metrics and not args.batch_metrics:
        parser.error("You must provide --desired-metrics or --batch-metrics.")

    # Load .env if present
    load_dotenv_if_present()

    config = load_config(args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record_index = 0

    # Support single metric or batch mode
    metric_list = args.batch_metrics if args.batch_metrics else [args.desired_metrics]
    batches_per_metric = args.batches_per_metric if args.batch_metrics else 1
    examples_per_batch = args.examples_per_batch if args.batch_metrics else args.num_examples

    with out_path.open("w") as f:
        for metric in metric_list:
            for batch_idx in range(batches_per_metric):
                prompt = build_prompt(metric, examples_per_batch)
                schema = build_schema(examples_per_batch)
                raw = call_teacher(prompt, schema, config)

                if "choices" not in raw or not raw["choices"]:
                    raise RuntimeError("No choices returned from API")
                content = raw["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(content.strip())
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Failed to parse teacher JSON: {e}")

                examples = parsed.get("examples") or []
                if len(examples) != examples_per_batch:
                    raise RuntimeError(f"Expected {examples_per_batch} examples, got {len(examples)}")

                for ex in examples:
                    conv = ex.get("conversation") or {}
                    raw_turns = conv.get("turns") or []
                    if len(raw_turns) != 3:
                        raise RuntimeError(f"Example has {len(raw_turns)} turns; expected 3.")

                    # Map to assistant/user/assistant roles in order
                    speaker_to_role = {0: "assistant", 1: "user", 2: "assistant"}
                    turns: List[Dict[str, Any]] = []
                    for i, turn in enumerate(raw_turns):
                        turns.append(
                            {
                                "role": speaker_to_role.get(i, "assistant"),
                                "content": turn.get("content", ""),
                                "turn_number": i + 1,
                            }
                        )

                    history = turns[:2]
                    target = turns[2]
                    record = {
                        "dataset": "synthetic_ultrachat",
                        "record_index": record_index,
                        "turn_id": f"synthetic-{record_index}",
                        "pair_type": None,
                        "history": history,
                        "target": target,
                        "metadata": {
                            "conversation_id": f"synthetic-{record_index}",
                            "history_length": len(history),
                            "total_turns": len(history) + 1,
                            "source": "synthetic_generator",
                            "notes": ex.get("notes", ""),
                            "desired_metrics": metric,
                            "batch_index": batch_idx,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    record_index += 1

    print(f"Wrote {record_index} synthetic trajectories to {out_path}")


if __name__ == "__main__":
    main()
