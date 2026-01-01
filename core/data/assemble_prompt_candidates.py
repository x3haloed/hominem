#!/usr/bin/env python3
"""
Assemble prompt/response candidates for reward labeling.

Sources (priority order):
1) Canonical conversations DB (messages table).
2) Synthetic shard JSONL files.
3) HF datasets (optional, used to pad to a target size).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence


def _default_db_path() -> str:
    return os.path.join(str(Path.home()), "Documents", "hominem", "conversations.db")


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _hash_messages(messages: Sequence[Dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list):
        return None
    out: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not role or not content:
            continue
        out.append({"role": role, "content": content})
    if not out:
        return None
    if out[0]["role"] == "assistant":
        out = out[1:]
    return out or None


def _messages_from_prompt_completion(prompt: str, completion: str) -> Optional[List[Dict[str, str]]]:
    prompt = prompt.strip()
    completion = completion.strip()
    if not prompt or not completion:
        return None
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def _extract_messages_from_record(record: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    if "messages" in record:
        return _normalize_messages(record.get("messages"))
    if "prompt" in record and "completion" in record:
        return _messages_from_prompt_completion(record.get("prompt", ""), record.get("completion", ""))
    if "instruction" in record and "output" in record:
        prompt = record.get("instruction") or ""
        inp = record.get("input") or ""
        if inp:
            prompt = f"{prompt}\n{inp}".strip()
        return _messages_from_prompt_completion(prompt, record.get("output", ""))
    if "history_json" in record and "assistant" in record:
        messages = _normalize_messages(record.get("history_json"))
        if not messages:
            return None
        response = str(record.get("assistant") or "").strip()
        if not response:
            return None
        return messages + [{"role": "assistant", "content": response}]
    if "history" in record and "assistant" in record:
        messages = _normalize_messages(record.get("history"))
        if not messages:
            return None
        response = str(record.get("assistant") or "").strip()
        if not response:
            return None
        return messages + [{"role": "assistant", "content": response}]
    if "history" in record and "target" in record:
        messages = _normalize_messages(record.get("history"))
        if not messages:
            return None
        target = record.get("target")
        if isinstance(target, dict):
            role = str(target.get("role") or "assistant").strip().lower()
            content = str(target.get("content") or "").strip()
            if content:
                return messages + [{"role": role or "assistant", "content": content}]
    if "response" in record and "user_message" in record:
        return _messages_from_prompt_completion(record.get("user_message", ""), record.get("response", ""))
    return None


@dataclass
class Candidate:
    messages: List[Dict[str, str]]
    source: str
    metadata: Dict[str, Any]


def _read_db_candidates(
    *,
    db_path: str,
    history_turns: int,
    min_chars: int,
) -> List[Candidate]:
    if history_turns <= 0:
        history_turns = 1
    max_messages = history_turns * 2
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    candidates: List[Candidate] = []
    try:
        cur = con.execute(
            "SELECT conversation_id, role, content, id FROM messages ORDER BY conversation_id, id ASC"
        )
        current_id = None
        history: List[Dict[str, str]] = []
        for row in cur.fetchall():
            convo_id = row["conversation_id"]
            if current_id is None or convo_id != current_id:
                current_id = convo_id
                history = []
            role = str(row["role"] or "").lower()
            content = str(row["content"] or "").strip()
            if not role or not content:
                continue
            history.append({"role": role, "content": content})
            if role != "assistant":
                continue
            if len(content) < min_chars:
                continue
            messages = history[-max_messages:]
            if messages and messages[0]["role"] == "assistant":
                messages = messages[1:]
            if not messages or messages[-1]["role"] != "assistant":
                continue
            candidates.append(
                Candidate(
                    messages=list(messages),
                    source="canonical_db",
                    metadata={
                        "conversation_id": convo_id,
                        "message_id": int(row["id"]),
                    },
                )
            )
    finally:
        con.close()
    return candidates


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                record["_line"] = line_num
                yield record


def _read_synthetic_candidates(
    *,
    roots: Sequence[str],
    include_glob: str,
    min_chars: int,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.glob(include_glob):
            if not path.is_file():
                continue
            for record in _iter_jsonl_records(path):
                messages = _extract_messages_from_record(record)
                if not messages:
                    continue
                if len(messages[-1].get("content", "")) < min_chars:
                    continue
                candidates.append(
                    Candidate(
                        messages=messages,
                        source="synthetic",
                        metadata={
                            "path": str(path),
                            "line": record.get("_line"),
                        },
                    )
                )
    return candidates


def _read_hf_candidates(
    *,
    dataset_names: Sequence[str],
    split: str,
    limit: int,
    seed: int,
    min_chars: int,
) -> List[Candidate]:
    if not dataset_names or limit <= 0:
        return []
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise SystemExit(f"datasets not available: {exc}") from exc

    rng = random.Random(int(seed))
    out: List[Candidate] = []
    for name in dataset_names:
        ds = load_dataset(name, split=split)
        if len(ds) == 0:
            continue
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            if len(out) >= limit:
                break
            record = ds[idx]
            if not isinstance(record, dict):
                continue
            messages = _extract_messages_from_record(record)
            if not messages:
                continue
            if len(messages[-1].get("content", "")) < min_chars:
                continue
            out.append(
                Candidate(
                    messages=messages,
                    source=f"hf:{name}",
                    metadata={"dataset": name, "index": idx},
                )
            )
        if len(out) >= limit:
            break
    return out


def _dedupe(candidates: Sequence[Candidate]) -> List[Candidate]:
    seen: set[str] = set()
    out: List[Candidate] = []
    for cand in candidates:
        key = _hash_messages(cand.messages)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _write_jsonl(candidates: Sequence[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cand in candidates:
            record = {
                "messages": cand.messages,
                "images": [],
                "source": cand.source,
                "metadata": cand.metadata,
                "images_placeholder": True,
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble prompt/response candidates for reward labeling.")
    parser.add_argument("--output", default="data/exports/reward_candidates.jsonl")
    parser.add_argument("--db-path", default=_default_db_path())
    parser.add_argument("--history-turns", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--no-dedupe", action="store_true")

    parser.add_argument(
        "--synthetic-root",
        action="append",
        default=[],
        help="Root folders containing synthetic JSONL shards (repeatable).",
    )
    parser.add_argument("--synthetic-glob", default="**/*.jsonl")

    parser.add_argument(
        "--hf-dataset",
        action="append",
        default=[],
        help="HF dataset names to sample from (repeatable).",
    )
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-limit", type=int, default=0, help="Max HF samples to add (0 = none).")
    parser.add_argument(
        "--target-total",
        type=int,
        default=0,
        help="If set, pad with HF samples until reaching this total.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    db_candidates = _read_db_candidates(
        db_path=str(args.db_path),
        history_turns=int(args.history_turns),
        min_chars=int(args.min_chars),
    )
    synthetic_candidates = _read_synthetic_candidates(
        roots=[str(p) for p in args.synthetic_root],
        include_glob=str(args.synthetic_glob),
        min_chars=int(args.min_chars),
    )

    combined: List[Candidate] = []
    combined.extend(db_candidates)
    combined.extend(synthetic_candidates)

    if args.target_total and args.target_total > 0:
        hf_limit = max(0, int(args.target_total) - len(combined))
        hf_limit = _clamp(hf_limit, 0, int(args.hf_limit) if args.hf_limit else hf_limit)
    else:
        hf_limit = int(args.hf_limit or 0)

    hf_candidates = _read_hf_candidates(
        dataset_names=[str(n) for n in args.hf_dataset],
        split=str(args.hf_split),
        limit=hf_limit,
        seed=int(args.seed),
        min_chars=int(args.min_chars),
    )
    combined.extend(hf_candidates)

    if not args.no_dedupe:
        combined = _dedupe(combined)

    output_path = Path(args.output)
    _write_jsonl(combined, output_path)

    print(
        f"Wrote {len(combined)} candidates to {output_path} "
        f"(db={len(db_candidates)}, synthetic={len(synthetic_candidates)}, hf={len(hf_candidates)})"
    )


if __name__ == "__main__":
    main()
