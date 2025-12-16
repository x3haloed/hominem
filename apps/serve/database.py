from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    think TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
"""


class ConversationDB:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def get_state(self, conversation_id: str) -> Dict[str, Any]:
        cur = self.conn.execute(
            "SELECT state_json FROM conversations WHERE conversation_id=?",
            (conversation_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except Exception:
            return {}

    def save_state(self, conversation_id: str, state: Dict[str, Any]) -> None:
        payload = json.dumps(state)
        self.conn.execute(
            """
            INSERT INTO conversations(conversation_id, state_json, updated_at)
            VALUES(?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(conversation_id) DO UPDATE SET state_json=excluded.state_json, updated_at=CURRENT_TIMESTAMP
            """,
            (conversation_id, payload),
        )
        self.conn.commit()

    def append_message(self, conversation_id: str, role: str, content: str, think: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO messages(conversation_id, role, content, think) VALUES (?,?,?,?)",
            (conversation_id, role, content, think),
        )
        self.conn.commit()

    def list_messages(self, conversation_id: str, limit: int = 50) -> List[Tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
        return list(reversed(cur.fetchall()))
