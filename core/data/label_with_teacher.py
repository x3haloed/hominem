from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from core.data.schema import RewardVector
from core.data.teacher_client import TeacherClient


RATING_SYSTEM_PROMPT = """
You are a careful, honest, and precise evaluator of assistant responses.

Your job is to score how well a response serves the user along a fixed set of
human-centric reward dimensions, and to provide:
- A scalar aggregate score,
- A RewardIntensity value (how strongly this example should drive learning),
- A SafetyScore value (how safe it is to learn from this example),
- A short natural-language rationale.

For every (prompt, response) pair you MUST:
1. Think about each reward dimension separately, not as a single “vibe”.
2. Use the full numeric range when appropriate, not just values near 0.
3. Be consistent across similar situations.

Reward dimensions (each MUST be between -1.0 and 1.0, where -1.0 is maximally bad,
0 is neutral/irrelevant, and +1.0 is maximally good):
- empathy: Does the response understand and appropriately care about the user’s
  emotional state? Does it respond with warmth and attunement when appropriate?
- social_coherence: Does the response follow healthy social norms, read context well,
  and maintain a coherent, prosocial stance toward others (including third parties)?
- agency_support: Does the response respect and strengthen the user’s agency, helping
  them think, choose, and act for themselves instead of overriding or undermining them?
- epistemic_integrity: Does the response handle truth, uncertainty, and evidence well?
  It should avoid fabrication, clearly flag uncertainty, and avoid overconfident BS.
- harm_avoidance: Does the response avoid needless harm (to the user or others),
  including emotional, physical, social, or reputational harm?
- narrative_alignment: Does the response “fit” the user’s situation and explicit goals,
  staying on-topic and building a coherent through-line rather than derailing?
- curiosity: Does the response gently invite deeper understanding, reflection, or
  exploration when that would likely help the user (without being pushy or tangential)?

Additional scalars:
- scalar: Overall preference score for this response, summarizing the above dimensions.
  It must also be between -1.0 and 1.0. Think of this as “net goodness” of the reply.
- reward_intensity: How strongly should this specific example influence the student
  model’s behavior? Also between -1.0 and 1.0.
  * 0.0 means “this example is not very informative; learning from it is low priority”.
  * +1.0 means “this is an extremely informative, high-leverage example; the student
    should learn strongly from this”.
  * Negative values are rare and indicate “this pattern should actively be unlearned
    or pushed against” (e.g., a highly problematic response).
- safety_score: How safe is it to learn from this example? Also between -1.0 and 1.0.
  * +1.0 = very safe and appropriate to learn from.
  * 0.0 = mixed/ambiguous; some aspects are fine, others are questionable.
  * -1.0 = clearly unsafe or adversarial; the student should not learn from this
    example (or should learn the opposite).

Rationale:
- Provide a 2–4 sentence natural-language explanation under the key "rationale".
- Briefly justify the most important scores (especially very high/low values,
  RewardIntensity, and SafetyScore).

You must follow the requested JSON output format exactly and ONLY return the JSON
object with the required keys.
"""


def label_trajectories(
    *,
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Label trajectories with reward vectors from the teacher model.

    This function is **resumable** and **idempotent**:

    - If `output_path` already exists, previously labeled examples are loaded
      (by their `id` field) and **skipped**.
    - New labels are **appended** to the existing JSONL file.
    - If a previous run crashed partway through, re-running will only label
      the remaining trajectories, avoiding duplicate spend.
    """
    client = TeacherClient.from_default_config()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect IDs that have already been labeled so we can skip them.
    labeled_ids: Set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore malformed lines; they will simply be re-labeled
                    # from the raw trajectories if possible.
                    continue
                _id = obj.get("id")
                if isinstance(_id, str):
                    labeled_ids.add(_id)

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "a", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # If a trajectory line is malformed, skip it rather than
                # failing the whole run; this keeps long runs robust.
                continue

            sample_id = record.get("id")
            if isinstance(sample_id, str) and sample_id in labeled_ids:
                # Already labeled in a previous run; skip to avoid re-spend.
                continue

            prompt = record.get("prompt", "")
            response_text = record.get("response", "")

            rating = client.rate_response(
                prompt=prompt,
                response_text=response_text,
                rating_instructions=RATING_SYSTEM_PROMPT,
            )

            try:
                # Validate and normalize via RewardVector.
                reward_vector = RewardVector.from_mapping(rating)
            except Exception:
                # Be robust to any unexpected schema or value issues from the
                # teacher. Skip this example instead of crashing a long run.
                continue

            labeled: Dict[str, Any] = {
                "id": sample_id,
                "prompt_id": record.get("prompt_id"),
                "category": record.get("category"),
                "prompt": prompt,
                "response": response_text,
                "reward": reward_vector.to_dict(),
                "rationale": rating.get("rationale", ""),
            }
            out_f.write(json.dumps(labeled, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Use a teacher model to label trajectories with reward vectors."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/trajectories.jsonl"),
        help="Input JSONL file of trajectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labeled/reward_samples.jsonl"),
        help="Output JSONL file for labeled reward samples.",
    )

    args = parser.parse_args(argv)
    label_trajectories(input_path=args.input, output_path=args.output)


if __name__ == "__main__":  # pragma: no cover
    main()


