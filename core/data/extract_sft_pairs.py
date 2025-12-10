"""
Extraction pipeline for SFT (supervised fine-tuning) pairs.

Implements the Phase 5.5 dual-channel memory extraction:
- Pair every assistant turn with the most recent user turn.
- Deduplicate multiple assistant drafts to the same user turn using cosine
  similarity (>0.92), keeping the highest-quality response (reward intensity
  preferred).
- Optional context inclusion to reduce verbatim memorization risk.
- Optional introspection inclusion (off by default) with strict gating and
  content-hash deduplication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.data.db import (
    TrainingDatabase,
    MAX_INSTRUCTION_LENGTH,
    MAX_RESPONSE_LENGTH,
)
from apps.serve.database import DatabaseManager

SIMILARITY_THRESHOLD = 0.92


@dataclass
class SFTPair:
    """In-memory representation of an SFT training pair."""

    instruction: str
    response: str
    source: str  # conversation | manual | correction | knowledge_update | synthetic
    conversation_id: Optional[str] = None
    message_id: Optional[int] = None
    message_index: Optional[int] = None
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _text_vector(text: str) -> Tuple[Counter, float]:
    tokens = re.findall(r"\w+", text.lower())
    counts = Counter(tokens)
    norm = sum(value * value for value in counts.values()) ** 0.5
    return counts, norm


def _cosine(a: Tuple[Counter, float], b: Tuple[Counter, float]) -> float:
    va, na = a
    vb, nb = b
    if na == 0 or nb == 0:
        return 0.0
    dot = sum(value * vb.get(token, 0) for token, value in va.items())
    return dot / (na * nb)


def _reward_intensity(message: Dict[str, Any]) -> Optional[float]:
    emotion = message.get("emotion_labels") or {}
    auto_labels = emotion.get("auto") or {}
    intensity = auto_labels.get("reward_intensity")
    if intensity is None:
        return None
    try:
        return float(intensity)
    except (TypeError, ValueError):
        return None


def _confidence_from_reward(intensity: Optional[float]) -> Optional[float]:
    if intensity is None:
        return None
    # Map roughly from [-1, 1] to [0, 1] with clamping.
    scaled = (intensity + 1.0) / 2.0
    return max(0.0, min(1.0, scaled))


def _better_response(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True if response a is preferred over b."""
    a_reward = _reward_intensity(a)
    b_reward = _reward_intensity(b)
    if a_reward is not None and b_reward is not None:
        if a_reward != b_reward:
            return a_reward > b_reward
    # Fallback: prefer longer content (proxy for information)
    return len(a.get("content", "")) > len(b.get("content", ""))


def _deduplicate_assistant_messages(
    messages: List[Dict[str, Any]],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Deduplicate assistant messages that answer the same user turn."""
    kept: List[Dict[str, Any]] = []
    vectors: List[Tuple[Counter, float]] = []

    for message in messages:
        vector = _text_vector(message.get("content", ""))
        best_idx = None
        best_sim = 0.0

        for idx, kept_vector in enumerate(vectors):
            sim = _cosine(vector, kept_vector)
            if sim > similarity_threshold and sim > best_sim:
                best_idx = idx
                best_sim = sim

        if best_idx is None:
            kept.append(message)
            vectors.append(vector)
            continue

        if _better_response(message, kept[best_idx]):
            kept[best_idx] = message
            vectors[best_idx] = vector

    return kept


def _build_instruction(
    messages: List[Dict[str, Any]],
    user_index: int,
    include_context: bool,
) -> str:
    user_message = messages[user_index]
    if not include_context:
        return user_message["content"]

    context_lines: List[str] = []
    for msg in messages[: user_index + 1]:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        context_lines.append(f"{prefix}: {msg['content']}")
    return "Previous conversation:\n" + "\n".join(context_lines)


def _cap_lengths(instruction: str, response: str) -> Tuple[str, str]:
    """Apply max-length limits to instruction/response."""
    if len(instruction) > MAX_INSTRUCTION_LENGTH:
        instruction = instruction[:MAX_INSTRUCTION_LENGTH]
    if len(response) > MAX_RESPONSE_LENGTH:
        response = response[:MAX_RESPONSE_LENGTH]
    return instruction, response


def extract_sft_pairs_from_conversation(
    conversation: Dict[str, Any],
    include_context: bool = True,
    include_introspection: Optional[bool] = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[SFTPair]:
    """Extract all (user, assistant) pairs from a conversation."""
    messages = conversation.get("messages") or []
    conversation_id = conversation.get("conversation_id")
    pairs: List[SFTPair] = []

    last_user_pos: Optional[int] = None
    assistant_buffer: List[Dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal assistant_buffer
        if last_user_pos is None or not assistant_buffer:
            assistant_buffer = []
            return

        deduped = _deduplicate_assistant_messages(
            assistant_buffer,
            similarity_threshold=similarity_threshold,
        )
        instruction = _build_instruction(
            messages, last_user_pos, include_context=include_context
        )
        for assistant_msg in deduped:
            instruction_capped, response_capped = _cap_lengths(
                instruction, assistant_msg.get("content", "")
            )
            reward_intensity = _reward_intensity(assistant_msg)
            confidence = _confidence_from_reward(reward_intensity)
            metadata = {
                "source_message_created_at": assistant_msg.get("created_at"),
                "reward_intensity": reward_intensity,
                "include_context": include_context,
                "dedup_similarity_threshold": similarity_threshold,
            }
            pairs.append(
                SFTPair(
                    instruction=instruction_capped,
                    response=response_capped,
                    source="conversation",
                    conversation_id=conversation_id,
                    message_id=assistant_msg.get("id"),
                    message_index=assistant_msg.get("message_index"),
                    extraction_method="conversation_pairing",
                    confidence=confidence,
                    category=None,
                    metadata=metadata,
                )
            )
        assistant_buffer = []

    for idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            flush_buffer()
            last_user_pos = idx
            continue
        if msg.get("role") == "assistant":
            if last_user_pos is None:
                continue  # No preceding user message to pair with
            assistant_buffer.append(msg)
    flush_buffer()

    if include_introspection and conversation.get("introspection_observations"):
        seen_hashes = {
            hashlib.sha256(pair.response.encode()).hexdigest() for pair in pairs
        }
        for obs in conversation["introspection_observations"]:
            if not obs.get("internal_generated", True):
                continue
            obs_text = obs.get("observation_text") or ""
            if not obs_text:
                continue
            content = obs_text.replace(
                "<SELF-OBSERVE> I just experienced / thought / felt:\n", ""
            ).strip()
            if not content:
                continue
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            instr_capped, resp_capped = _cap_lengths(
                "What did I just experience, think, or feel?",
                content,
            )
            pairs.append(
                SFTPair(
                    instruction=instr_capped,
                    response=resp_capped,
                    source="conversation",
                    conversation_id=conversation_id,
                    message_id=obs.get("id"),
                    message_index=obs.get("observation_index"),
                    extraction_method="introspection",
                    confidence=0.35,  # Downweighted (OBSTACLE 4)
                    category="introspection",
                    metadata={
                        "is_introspection": True,
                        "content_hash": content_hash,
                        "reward_intensity": obs.get("reward_intensity"),
                        "safety_score": obs.get("safety_score"),
                    },
                )
            )

    return pairs


def _conversation_ids(
    db: TrainingDatabase, all_conversations: bool, since: Optional[str]
) -> Iterable[str]:
    if not all_conversations:
        return []
    query = "SELECT conversation_id FROM conversations WHERE is_active = TRUE"
    params: List[Any] = []
    if since:
        query += " AND created_at >= ?"
        params.append(since)
    cursor = db.connection.execute(query, params)
    return [row[0] for row in cursor.fetchall()]


def extract_and_store_sft_pairs(
    db: TrainingDatabase,
    conversation_id: Optional[str] = None,
    all_conversations: bool = False,
    since: Optional[str] = None,
    include_context: bool = True,
    skip_existing: bool = True,
    include_introspection: Optional[bool] = None,
    export_jsonl: Optional[str] = None,
) -> int:
    """Extract conversational pairs and persist them."""
    if not conversation_id and not all_conversations:
        raise ValueError("Must provide conversation_id or set all_conversations=True.")
    if conversation_id and all_conversations:
        raise ValueError("Use either conversation_id or all_conversations, not both.")

    db_manager = DatabaseManager(db.db_path)
    extracted_pairs: List[SFTPair] = []

    target_ids: Iterable[str]
    if conversation_id:
        target_ids = [conversation_id]
    else:
        target_ids = _conversation_ids(db, all_conversations=True, since=since)

    from core.data.admin_flags import get_flag  # late import to avoid cycles
    effective_include_introspection = (
        include_introspection
        if include_introspection is not None
        else bool(get_flag("include_introspection", False))
    )

    for cid in target_ids:
        conversation = db_manager.get_conversation(
            cid, include_introspection=effective_include_introspection
        )
        if not conversation:
            continue
        conversation_pairs = extract_sft_pairs_from_conversation(
            conversation,
            include_context=include_context,
            include_introspection=effective_include_introspection,
        )
        for pair in conversation_pairs:
            if skip_existing and pair.message_id is not None:
                existing = db.connection.execute(
                    "SELECT 1 FROM sft_pairs WHERE message_id = ? LIMIT 1",
                    (pair.message_id,),
                ).fetchone()
                if existing:
                    continue
            extracted_pairs.append(pair)

    if export_jsonl:
        with open(export_jsonl, "w", encoding="utf-8") as f:
            for pair in extracted_pairs:
                f.write(json.dumps(asdict(pair)) + "\n")

    for pair in extracted_pairs:
        db.insert_sft_pair(
            instruction=pair.instruction,
            response=pair.response,
            source=pair.source,
            conversation_id=pair.conversation_id,
            message_id=pair.message_id,
            message_index=pair.message_index,
            extraction_method=pair.extraction_method,
            confidence=pair.confidence,
            category=pair.category,
            metadata=pair.metadata,
            extracted_at=datetime.utcnow().isoformat(),
        )

    return len(extracted_pairs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SFT pairs from conversations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--conversation-id", type=str, help="Conversation UUID to extract")
    group.add_argument(
        "--all-conversations",
        action="store_true",
        help="Extract all available conversations",
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Only process conversations created at or after this ISO timestamp",
    )
    parser.add_argument(
        "--include-context",
        dest="include_context",
        action="store_true",
        default=True,
        help="Include conversation context in the instruction (default: on)",
    )
    parser.add_argument(
        "--no-include-context",
        dest="include_context",
        action="store_false",
        help="Disable conversation context in the instruction",
    )
    parser.add_argument(
        "--include-introspection",
        dest="include_introspection",
        action="store_true",
        help="Include internal introspection observations (overrides admin flag)",
    )
    parser.add_argument(
        "--exclude-introspection",
        dest="include_introspection",
        action="store_false",
        help="Exclude introspection (overrides admin flag)",
    )
    parser.add_argument(
        "--use-admin-introspection-flag",
        action="store_true",
        help="Use admin flag (config/admin_flags.json) for introspection inclusion when no explicit include/exclude flag is set.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip message_ids that already exist in sft_pairs",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the SQLite database (defaults to TrainingDatabase path)",
    )
    parser.add_argument(
        "--export-jsonl",
        type=str,
        help="Optional path to export extracted pairs as JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    db = TrainingDatabase(db_path=args.db_path)
    include_introspection = args.include_introspection
    if args.use_admin_introspection_flag and include_introspection is None:
        include_introspection = None  # triggers admin flag lookup
    count = extract_and_store_sft_pairs(
        db=db,
        conversation_id=args.conversation_id,
        all_conversations=args.all_conversations,
        since=args.since,
        include_context=args.include_context,
        skip_existing=args.skip_existing,
        include_introspection=include_introspection,
        export_jsonl=args.export_jsonl,
    )
    print(f"Extracted and stored {count} SFT pairs.")


if __name__ == "__main__":
    main()
