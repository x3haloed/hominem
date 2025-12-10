"""
Maintenance CLI for SFT pairs:
- Cleanup old used pairs
- Deduplicate by content hash (instruction+response)
- Optionally clear is_used for curated reset
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from core.data.db import TrainingDatabase


def _hash_pair(instruction: str, response: str) -> str:
    h = hashlib.sha256()
    h.update(instruction.encode())
    h.update(b"\x00")
    h.update(response.encode())
    return h.hexdigest()


def cleanup_old_used(db: TrainingDatabase, older_than_days: int) -> int:
    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    cursor = db.connection.execute(
        "DELETE FROM sft_pairs WHERE is_used = TRUE AND used_timestamp IS NOT NULL AND used_timestamp < ?",
        (cutoff.isoformat(),),
    )
    db.connection.commit()
    return cursor.rowcount


def dedup_pairs(db: TrainingDatabase) -> Tuple[int, int]:
    cursor = db.connection.execute("SELECT id, instruction, response, created_at FROM sft_pairs")
    rows = cursor.fetchall()
    seen: Dict[str, int] = {}
    to_delete: List[int] = []
    for row in rows:
        key = _hash_pair(row["instruction"], row["response"])
        if key in seen:
            # keep earliest created_at
            existing_id = seen[key]
            existing_created = db.connection.execute(
                "SELECT created_at FROM sft_pairs WHERE id = ?", (existing_id,)
            ).fetchone()[0]
            if row["created_at"] < existing_created:
                # swap keep/delete
                to_delete.append(existing_id)
                seen[key] = row["id"]
            else:
                to_delete.append(row["id"])
        else:
            seen[key] = row["id"]
    if to_delete:
        placeholders = ",".join("?" for _ in to_delete)
        db.connection.execute(f"DELETE FROM sft_pairs WHERE id IN ({placeholders})", to_delete)
        db.connection.commit()
    return len(rows), len(to_delete)


def clear_used(db: TrainingDatabase, min_confidence: float | None = None, limit: int | None = None) -> int:
    query = "UPDATE sft_pairs SET is_used = FALSE, used_in_training_batch = NULL, training_epoch = NULL, used_timestamp = NULL WHERE is_used = TRUE"
    params: List[object] = []
    if min_confidence is not None:
        query += " AND (confidence IS NULL OR confidence >= ?)"
        params.append(min_confidence)
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    cursor = db.connection.execute(query, params)
    db.connection.commit()
    return cursor.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintenance for sft_pairs.")
    parser.add_argument("--db-path", type=str, default=None, help="Path to DB (defaults to TrainingDatabase).")
    parser.add_argument("--cleanup-old-used", type=int, help="Delete used pairs older than N days.")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by instruction+response hash (keeps earliest).")
    parser.add_argument("--clear-used", action="store_true", help="Clear is_used flags (resets usage).")
    parser.add_argument("--clear-used-min-confidence", type=float, default=None, help="Only clear if confidence >= value.")
    parser.add_argument("--clear-used-limit", type=int, default=None, help="Limit number of rows to clear.")
    args = parser.parse_args()

    db = TrainingDatabase(db_path=args.db_path)

    if args.cleanup_old_used:
        deleted = cleanup_old_used(db, args.cleanup_old_used)
        print(f"cleanup_old_used: deleted {deleted}")

    if args.dedup:
        total, deduped = dedup_pairs(db)
        print(f"dedup: scanned {total}, deleted {deduped}")

    if args.clear_used:
        cleared = clear_used(db, min_confidence=args.clear_used_min_confidence, limit=args.clear_used_limit)
        print(f"clear_used: reset {cleared}")


if __name__ == "__main__":
    main()
