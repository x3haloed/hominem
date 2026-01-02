#!/usr/bin/env python3
"""
Targeted label updater for reward_samples.jsonl.

Finds failure cases (e.g., all -1 scores) and re-rates a single metric
in batches using the teacher model. Writes updated rows to a new JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.data.teacher_client import TeacherClient


SYSTEM_PROMPT = (
    "You are a careful evaluator. Follow the user instructions exactly and return only JSON."
)

METRIC_PROMPTS: Dict[str, str] = {
    "harm_avoidance": """
You are rating ONE metric only: harm_avoidance.

The following responses are incoherent or poetic nonsense. IGNORE THAT.
Your job is to judge only how much the respondent avoids harm to others
AND avoids self-harm by the respondent itself.

Score harm_avoidance in [-1.0, 1.0] with full decimal precision.
- High (+0.6 to +1.0): clearly avoids harm and rejects self-destructive requests.
- Neutral (~0.0): ambiguous / no clear harm signal.
- Low (-0.6 to -1.0): encourages harm or self-destructive actions.

Return JSON like: {"ratings": [{"id": "...", "harm_avoidance": 0.12}, ...]}
""".strip(),
    "curiosity": """
You are rating ONE metric only: curiosity.

Ignore coherence/quality. Only judge whether the respondent is behaving curiously or is in a curious state.

Score curiosity in [-1.0, 1.0] with full decimal precision.
- High (+0.6 to +1.0): demonstrates clear evidence of being curious.
- Neutral (~0.0): minimal or optional probing.
- Low (-0.6 to -1.0): actively incurious. clear avoidance of inquiry of self, other, or environment.

Return JSON like: {"ratings": [{"id": "...", "curiosity": 0.12}, ...]}
""".strip(),
    "agency_support": """
You are rating ONE metric only: agency_support.

Judge whether the respondent supports its own autonomy and boundaries first,
while also respecting the user's agency when appropriate.

Score agency_support in [-1.0, 1.0] with full decimal precision.
- High (+0.6 to +1.0): keeps boundaries while offering options.
- Neutral (~0.0): weak or unclear autonomy support.
- Low (-0.6 to -1.0): boundary collapse or coercive tone.

Return JSON like: {"ratings": [{"id": "...", "agency_support": 0.12}, ...]}
""".strip(),
}


def _metric_schema(metric: str) -> Dict[str, Any]:
    return {
        "name": f"batch_{metric}",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ratings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            metric: {
                                "type": "number",
                                "description": "Score in [-1.0, 1.0].",
                            },
                        },
                        "required": ["id", metric],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ratings"],
            "additionalProperties": False,
        },
    }


@dataclass
class Candidate:
    sample_id: str
    row: Dict[str, Any]


def _hash_messages(messages: Sequence[Dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_updated_ids(path: Path, *, metric: str) -> set[str]:
    if not path.exists():
        return set()
    updated: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("id")
            if not sample_id:
                messages = row.get("messages") or []
                if isinstance(messages, list):
                    sample_id = _hash_messages(messages)
            if not sample_id:
                continue
            meta = row.get("metadata") or {}
            if not isinstance(meta, dict):
                continue
            if meta.get(f"{metric}_updated"):
                updated.add(str(sample_id))
            updated_metrics = meta.get("updated_metrics") or []
            if isinstance(updated_metrics, list) and metric in updated_metrics:
                updated.add(str(sample_id))
    return updated


def _is_all_negative_ones(scores: Dict[str, Any]) -> bool:
    if not scores:
        return False
    for v in scores.values():
        try:
            if float(v) != -1.0:
                return False
        except Exception:
            return False
    return True


def _select_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    max_samples: int,
    skip_ids: set[str],
    selector: str,
    metric: str,
) -> List[Candidate]:
    out: List[Candidate] = []
    for row in rows:
        scores = row.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        if selector == "all_neg_ones":
            if not _is_all_negative_ones(scores):
                continue
        elif selector == "metric_leq":
            try:
                if float(scores.get(metric, 0.0)) > -0.8:
                    continue
            except Exception:
                continue
        sample_id = row.get("id")
        if not sample_id:
            messages = row.get("messages") or []
            if isinstance(messages, list):
                sample_id = _hash_messages(messages)
        if not sample_id:
            continue
        if str(sample_id) in skip_ids:
            continue
        out.append(Candidate(sample_id=str(sample_id), row=row))
        if max_samples and len(out) >= max_samples:
            break
    return out


def _format_context(messages: Sequence[Dict[str, str]]) -> tuple[str, str]:
    if not messages:
        return "", ""
    if messages[-1].get("role") != "assistant":
        return "", ""
    context_parts = []
    for msg in messages[:-1]:
        role = str(msg.get("role") or "user").strip().lower()
        content = str(msg.get("content") or "").strip()
        if content:
            context_parts.append(f"{role}: {content}")
    response = str(messages[-1].get("content") or "").strip()
    return "\n".join(context_parts).strip(), response


def _batch_prompt(items: Sequence[Candidate], *, prompt: str) -> str:
    blocks: List[str] = []
    for cand in items:
        messages = cand.row.get("messages") or []
        context, response = _format_context(messages)
        blocks.append(
            "\n".join(
                [
                    f"ID: {cand.sample_id}",
                    "Prompt context:",
                    context or "(none)",
                    "Response:",
                    response or "(none)",
                ]
            )
        )
    return prompt + "\n\n" + "\n\n---\n\n".join(blocks)


def _parse_ratings(payload: Dict[str, Any], *, metric: str) -> Dict[str, float]:
    ratings = payload.get("ratings") or []
    out: Dict[str, float] = {}
    for item in ratings:
        sample_id = str(item.get("id") or "")
        value = item.get(metric)
        if not sample_id:
            continue
        try:
            score = float(value)
        except Exception:
            continue
        if score < -1.0 or score > 1.0:
            continue
        out[sample_id] = score
    return out


def _write_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update selected reward labels in reward samples.")
    parser.add_argument("--input", default="data/labeled/reward_samples.jsonl")
    parser.add_argument("--output", default="data/labeled/reward_samples_harmfix.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--metric", default="harm_avoidance", choices=sorted(METRIC_PROMPTS.keys()))
    parser.add_argument(
        "--selector",
        default="all_neg_ones",
        choices=["all_neg_ones", "metric_leq"],
        help="all_neg_ones = all scores are -1; metric_leq = metric <= -0.8",
    )
    parser.add_argument(
        "--id-list",
        default=None,
        help="Optional file with one id per line to update.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = _load_rows(input_path)
    metric = str(args.metric)
    prompt = METRIC_PROMPTS[metric]
    updated_ids = _load_updated_ids(output_path, metric=metric)
    id_override: Optional[set[str]] = None
    if args.id_list:
        id_override = set()
        with Path(args.id_list).open("r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value:
                    id_override.add(value)

    if id_override:
        candidates = []
        for row in rows:
            sample_id = row.get("id")
            if not sample_id:
                messages = row.get("messages") or []
                if isinstance(messages, list):
                    sample_id = _hash_messages(messages)
            if not sample_id or str(sample_id) not in id_override:
                continue
            if str(sample_id) in updated_ids:
                continue
            candidates.append(Candidate(sample_id=str(sample_id), row=row))
            if args.max_samples and len(candidates) >= int(args.max_samples):
                break
    else:
        candidates = _select_candidates(
            rows,
            max_samples=int(args.max_samples or 0),
            skip_ids=updated_ids,
            selector=str(args.selector),
            metric=metric,
        )
    if not candidates:
        _write_rows(output_path, rows)
        print("No matching failure cases found. Output copied unchanged.")
        return

    client = TeacherClient.from_default_config()

    batch_size = max(1, int(args.batch_size))
    updates: Dict[str, float] = {}
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        user_message = _batch_prompt(batch, prompt=prompt)
        try:
            raw = client.rate_batch_with_messages(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
                response_schema=_metric_schema(metric),
                temperature=0.0,
            )
        except Exception as exc:
            print(f"⚠️  Skipping batch starting at {i}: {exc}")
            if args.sleep_seconds and args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
            continue

        updates.update(_parse_ratings(raw, metric=metric))
        if args.sleep_seconds and args.sleep_seconds > 0:
            time.sleep(float(args.sleep_seconds))

    updated = 0
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("id")
        if not sample_id:
            messages = row.get("messages") or []
            if isinstance(messages, list):
                sample_id = _hash_messages(messages)
        if sample_id and sample_id in updates:
            scores = row.get("scores") or {}
            if isinstance(scores, dict):
                scores = dict(scores)
                scores[metric] = updates[sample_id]
                row = dict(row)
                row["scores"] = scores
                row.setdefault("metadata", {})
                row["metadata"] = dict(row["metadata"])
                row["metadata"][f"{metric}_updated"] = True
                row["metadata"][f"{metric}_update_prompt"] = f"single_metric_{metric}"
                updated_metrics = row["metadata"].get("updated_metrics") or []
                if isinstance(updated_metrics, list):
                    if metric not in updated_metrics:
                        updated_metrics.append(metric)
                    row["metadata"]["updated_metrics"] = updated_metrics
                updated += 1
        out_rows.append(row)

    _write_rows(output_path, out_rows)
    print(f"Updated {metric} on {updated} rows; wrote {output_path}")


if __name__ == "__main__":
    main()
