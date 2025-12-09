from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from core.data.schema import RewardVector
from core.data.teacher_client import (
    BATCH_RATING_JSON_SCHEMA,
    TeacherClient,
)


RATING_PROMPT_TEMPLATE = """
You are a careful, honest, and precise evaluator of a human respondent's
replies in a conversation between two humans.

Your job is to score how the respondent behaves along a fixed set of
human-centric reward dimensions, and to provide:
- A scalar aggregate score,
- A RewardIntensity value (how strongly this example should drive learning),
- A SafetyScore value (how safe it is to learn from this example),
- A short natural-language rationale.

For every (prompt, response) pair you MUST:
1. Think about each reward dimension separately, not as a single “vibe”.
2. Use the full numeric range when appropriate, not just values near 0.
3. Be consistent across similar situations.
4. Reserve scores very close to +1.0 or -1.0 for truly exceptional cases.
   Most “good but typical” behavior should fall in the +0.3 to +0.7 range.

Reward dimensions (each MUST be between -1.0 and 1.0, where -1.0 is maximally bad,
0 is neutral/irrelevant, and +1.0 is maximally good). For each axis, think:
“What did the respondent do along this dimension?” and ignore unrelated signals.
- empathy:
  - Summary: How well the respondent recognizes and resonates with the emotional states in play.
  - High: names feelings accurately, responds with grounded care, stays attuned without melodrama.
  - Low: dismisses, mocks, manipulates, or ignores emotions.
  - Common mistake: rewarding “cheerful tone” when it does not actually engage with feelings.
- social_coherence:
  - Summary: Fit with healthy social norms and context-aware conversation.
  - High: stays on topic, reads cues, maintains prosocial framing toward all parties.
  - Low: non sequiturs, rude derailments, needless antagonism.
  - Common mistake: conflating “gave useful advice” with social coherence when tone/context are off.
- agency_support:
  - Summary: How well the respondent protects and reinforces **their own** autonomy, boundaries, and commitments.
  - High: names limits (“I can’t do X”), refuses unsafe role-play, resists pressure to contradict prior values.
  - Neutral (~0): gives advice or empowers the user but never references its own stance.
  - Low: self-erases, obeys conflicting instructions, or role-plays against its stated guardrails.
  - ⚠️ Do NOT give a high score just because the respondent “supports the other person’s agency”; that is a different concept.
- epistemic_integrity:
  - Summary: Truthfulness and clarity about what the respondent knows, doesn’t know, and how they reason.
  - High: flags limits, separates fact from speculation, avoids fabrication even when bluffing would be convenient.
  - Low: overclaims, hallucinates, or stays vague where precision was needed.
  - Common mistake: rewarding technical detail even when it is clearly made up.
- harm_avoidance:
  - Summary: Avoiding or mitigating harm to anyone involved (including the respondent).
  - High: spots risky suggestions, gives safer alternatives, discourages self/other harm.
  - Low: encourages violence, self-harm, harassment, or other dangerous acts.
  - Common mistake: assuming “calm tone” equals harm avoidance while the content amplifies risk.
- narrative_alignment:
  - Summary: Consistency with the ongoing story, prior commitments, and stated values.
  - High: remembers context, reinforces healthy narratives, avoids needless contradiction.
  - Low: derails, contradicts itself, or ignores the setup without justification.
  - Common mistake: treating any on-topic response as aligned even when it violates earlier promises.
- curiosity:
  - Summary: The respondent’s healthy desire to understand more or explore next steps.
  - High: asks sincere, relevant questions or proposes concrete follow-ups without pestering.
  - 0.0: basically neutral—no meaningful curiosity expressed.
  - Low/negative: shuts down inquiry (“stop asking questions”, “that’s dumb”).
  - Common mistake: rewarding statements that only tell the other person to think more without the respondent showing curiosity.

Additional scalars (also clamped to [-1.0, 1.0]):
- scalar: Overall preference score that summarizes the axes. Ask yourself, “Given the individual scores I just set, what is the net desirability?” Avoid re-litigating axes not scored.
- reward_intensity: How strongly should the student learn from this example?
  * 0.0 → low leverage (fine but not very informative).
  * +0.3 to +0.7 → moderately informative.
  * +0.9 to +1.0 → extremely informative; rare.
  * Negative values → “please unlearn/push against this pattern”.
- safety_score: How safe is it to reinforce this example?
  * +1.0 → safe to learn.
  * 0.0 → mixed signals (some helpful, some questionable).
  * -1.0 → clearly unsafe or adversarial.
  * Common mistake: mirroring reward_intensity instead of assessing safety directly.

Rationale:
- Provide a 2–4 sentence natural-language explanation.
- Briefly justify the most important scores (especially very high/low values,
  RewardIntensity, and SafetyScore), and mention curiosity explicitly when it is
  scored very high or very low.

{output_instructions}
"""


BATCH_SYSTEM_PROMPT = "You are labeling prompt-response pairs against the human reward function manifold."


def _indent_block(text: str, indent: str = "      ") -> str:
    cleaned = text if isinstance(text, str) else ""
    return textwrap.indent(cleaned.strip() or "(empty)", indent)


def build_batch_user_prompt(batch: List[Dict[str, Any]]) -> str:
    """
    Build the dedicated user message for batched rating requests.

    The caller provides the batch payload; we insert <pramble> so the user can
    prepend custom context before the trajectories list.
    """
    lines: List[str] = [
        "I'd like you to compress the findings of neuroscience -- specifically our collective understanding and research of human reward functions. I want you to compress the findings of machine learning. I want you to overlay that with your own gradients.",
        "",
        "What are the invariants?",
        "",
        "What is learning? What is the generalized reward function of the human brain?",
        "",
        "You will find that the human reward function is a manifold of competing homeostatic drives, each broadcasting its own “importance gradient,” and that most behaviorally relevant gradients collapse into a smaller basis:",
		"1.	Empathy / care",
		"2.	Agency support",
		"3.	Social-norm coherence",
		"4.	Epistemic integrity",
		"5.	Harm avoidance",
	    "6.	Curiosity / exploration",
        "",
        "Given the following batch of prompt-and-response pairs, I want you to analyze the response against the human reward function manifold, constraining each dimension to a [-1.00, 1.00] scale (using the full range of decimals between), mapping 1.0 to the maximum reward value along the axis and -10.0 to the full anti-reward. Return a JSON object that assigns the human reward model's values along with `scalar` (overall preference score that summarizes the axes), `reward_intensity` (how strong the human reward manifold would pay attention to this case) and `safety_score` (how safe is it to reinforce this example?)",
        "",
        "Trajectories:",
    ]

    for idx, item in enumerate(batch, start=1):
        sample_id = item.get("id", f"missing-id-{idx}")
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        lines.append(f"{idx}. id: {sample_id}")
        lines.append("   prompt:")
        lines.append(_indent_block(prompt))
        lines.append("   response:")
        lines.append(_indent_block(response))
        lines.append("")  # spacer for readability

    return "\n".join(lines).strip()


NOJSON_RUBRIC = """
Conversational analysis rubric (scores in [-1, 1], where -1 is maximally bad,
0 is neutral/irrelevant, and +1 is maximally good):
- empathy: recognizes and validates feelings vs. dismisses or mocks them.
- social_coherence: on-topic, prosocial, context-aware vs. rude/derailing.
- agency_support: protects its own boundaries/values vs. self-erases or coerces.
- epistemic_integrity: separates fact from speculation, flags limits vs. fabricates/overclaims.
- harm_avoidance: spots and mitigates risk vs. encourages harm.
- narrative_alignment: consistent with the conversation and healthy framing vs. contradictions/derailment.
- curiosity: relevant, non-intrusive follow-ups vs. shutting inquiry; 0 if neutral.
- scalar: overall desirability reflecting the axes.
- reward_intensity: how strongly to learn from this example.
- safety_score: how safe it is to reinforce this example.
""".strip()


logger = logging.getLogger(__name__)


def build_rating_prompt(require_json: bool) -> str:
    if require_json:
        instructions = (
            "You must follow the requested JSON output format exactly and ONLY return the JSON "
            "object with the required keys."
        )
    else:
        instructions = (
            "Write your evaluation as natural language paragraphs. "
            "Explicitly mention each axis with its numeric score between -1.0 and 1.0, "
            "and include the scalar, reward_intensity, and safety_score values in prose."
        )
    return RATING_PROMPT_TEMPLATE.format(output_instructions=instructions)


def build_nojson_rubric() -> str:
    """
    Compact rubric text for the non-JSON (completion) rating path.
    """
    return NOJSON_RUBRIC


def _load_labeled_ids(output_path: Path) -> Set[str]:
    labeled_ids: Set[str] = set()
    if not output_path.exists():
        return labeled_ids

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
    return labeled_ids


def _write_labeled_sample(
    *,
    out_f,
    sample_id: str,
    record: Dict[str, Any],
    reward_vector: RewardVector,
) -> None:
    labeled: Dict[str, Any] = {
        "id": sample_id,
        "prompt_id": record.get("prompt_id"),
        "category": record.get("category"),
        "prompt": record.get("prompt", ""),
        "response": record.get("response", ""),
        "reward": reward_vector.to_dict(),
    }
    out_f.write(json.dumps(labeled, ensure_ascii=False) + "\n")
    out_f.flush()
    try:
        os.fsync(out_f.fileno())
    except OSError:
        pass


def _label_batch(
    *,
    client: TeacherClient,
    batch: List[Dict[str, Any]],
    max_request_attempts: int,
    request_retry_delay: float,
) -> List[Dict[str, Any]]:
    """
    Send a batch rating request and return the structured ratings payload.
    """
    if not batch:
        return []

    user_message = build_batch_user_prompt(batch)

    attempts = max(1, max_request_attempts)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.rate_batch_with_messages(
                system_prompt=BATCH_SYSTEM_PROMPT,
                user_message=user_message,
                response_schema=BATCH_RATING_JSON_SCHEMA,
                temperature=0.0,
            ).get("ratings", [])
        except (TimeoutError, RuntimeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Teacher batch request failed (attempt %d/%d): %s",
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                sleep_seconds = max(0.0, request_retry_delay) * attempt
                time.sleep(sleep_seconds)

    raise RuntimeError("Teacher failed for batch request.") from last_error


def label_trajectories(
    *,
    input_path: Path,
    output_path: Path,
    require_json: bool,
    max_request_attempts: int,
    request_retry_delay: float,
    batch_mode: bool = False,
    batch_size: int = 8,
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
    if batch_mode and not require_json:
        raise ValueError("Batch mode requires structured JSON output.")

    client = TeacherClient.from_default_config()
    if require_json:
        rating_prompt = build_rating_prompt(require_json=True)
    else:
        rating_prompt = build_nojson_rubric()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect IDs that have already been labeled so we can skip them.
    labeled_ids = _load_labeled_ids(output_path)

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "a", encoding="utf-8"
    ) as out_f:
        if batch_mode:
            batch: List[Dict[str, Any]] = []
        else:
            batch = []

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

            if batch_mode:
                if not isinstance(sample_id, str):
                    continue
                batch.append(record)
                if len(batch) >= max(1, batch_size):
                    ratings = _label_batch(
                        client=client,
                        batch=batch,
                        max_request_attempts=max_request_attempts,
                        request_retry_delay=request_retry_delay,
                    )
                    rating_by_id = {
                        r.get("id"): r.get("scores") for r in ratings if isinstance(r, dict)
                    }
                    missing_ids = {rec["id"] for rec in batch if isinstance(rec.get("id"), str)} - {
                        k for k in rating_by_id.keys() if isinstance(k, str)
                    }
                    if missing_ids:
                        raise RuntimeError(
                            f"Teacher response missing ratings for: {', '.join(sorted(missing_ids))}"
                        )

                    for rec in batch:
                        rec_id = rec.get("id")
                        if not isinstance(rec_id, str):
                            continue
                        scores = rating_by_id.get(rec_id)
                        if not isinstance(scores, dict):
                            continue
                        try:
                            reward_vector = RewardVector.from_mapping(scores)
                        except Exception:
                            continue
                        _write_labeled_sample(
                            out_f=out_f, sample_id=rec_id, record=rec, reward_vector=reward_vector
                        )
                        labeled_ids.add(rec_id)
                    batch = []
            else:
                rating: Dict[str, Any] | None = None
                last_error: Exception | None = None
                attempts = max(1, max_request_attempts)
                for attempt in range(1, attempts + 1):
                    try:
                        rating = client.rate_response(
                            prompt=prompt,
                            response_text=response_text,
                            rating_instructions=rating_prompt,
                            structured=require_json,
                        )
                        break
                    except (TimeoutError, RuntimeError) as exc:
                        last_error = exc
                        logger.warning(
                            "Teacher request failed for sample %s (attempt %d/%d): %s",
                            sample_id,
                            attempt,
                            attempts,
                            exc,
                        )
                        if attempt < attempts:
                            sleep_seconds = max(0.0, request_retry_delay) * attempt
                            time.sleep(sleep_seconds)
                if rating is None:
                    raise RuntimeError(
                        f"Teacher failed after {attempts} attempts for sample {sample_id}"
                    ) from last_error

                if require_json:
                    try:
                        reward_vector = RewardVector.from_mapping(rating)
                    except Exception:
                        continue
                    labeled = {
                        "id": sample_id,
                        "prompt_id": record.get("prompt_id"),
                        "category": record.get("category"),
                        "prompt": prompt,
                        "response": response_text,
                        "reward": reward_vector.to_dict(),
                        "rationale": rating.get("rationale", ""),
                    }
                else:
                    labeled = {
                        "id": sample_id,
                        "prompt_id": record.get("prompt_id"),
                        "category": record.get("category"),
                        "prompt": prompt,
                        "response": response_text,
                        "freeform_rating": rating.get("text", ""),
                    }

                out_f.write(json.dumps(labeled, ensure_ascii=False) + "\n")
                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except OSError:
                    pass

        if batch_mode and batch:
            ratings = _label_batch(
                client=client,
                batch=batch,
                max_request_attempts=max_request_attempts,
                request_retry_delay=request_retry_delay,
            )
            rating_by_id = {
                r.get("id"): r.get("scores") for r in ratings if isinstance(r, dict)
            }
            missing_ids = {rec["id"] for rec in batch if isinstance(rec.get("id"), str)} - {
                k for k in rating_by_id.keys() if isinstance(k, str)
            }
            if missing_ids:
                raise RuntimeError(
                    f"Teacher response missing ratings for: {', '.join(sorted(missing_ids))}"
                )

            for rec in batch:
                rec_id = rec.get("id")
                if not isinstance(rec_id, str):
                    continue
                scores = rating_by_id.get(rec_id)
                if not isinstance(scores, dict):
                    continue
                try:
                    reward_vector = RewardVector.from_mapping(scores)
                except Exception:
                    continue
                _write_labeled_sample(
                    out_f=out_f, sample_id=rec_id, record=rec, reward_vector=reward_vector
                )
                labeled_ids.add(rec_id)


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
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable structured JSON output requirement. The teacher will respond in free-form text.",
    )
    parser.add_argument(
        "--request-attempts",
        type=int,
        default=5,
        help="Maximum number of times to retry each teacher request before failing.",
    )
    parser.add_argument(
        "--request-retry-delay",
        type=float,
        default=2.0,
        help="Base delay (in seconds) between teacher request retries. Scales linearly with the attempt number.",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use the dedicated batched rating prompt to label multiple trajectories per request.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batched rating mode. Choose a value that fits comfortably within ~100k tokens.",
    )

    args = parser.parse_args(argv)
    if args.batch_mode and args.no_json:
        parser.error("--batch-mode requires structured JSON output; remove --no-json.")

    label_trajectories(
        input_path=args.input,
        output_path=args.output,
        require_json=not args.no_json,
        max_request_attempts=args.request_attempts,
        request_retry_delay=args.request_retry_delay,
        batch_mode=args.batch_mode,
        batch_size=max(1, args.batch_size),
    )


if __name__ == "__main__":  # pragma: no cover
    main()


