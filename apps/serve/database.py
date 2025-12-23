from __future__ import annotations

import json
import sqlite3
import threading
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

CREATE TABLE IF NOT EXISTS sleep_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant TEXT NOT NULL,
    think TEXT,
    history_json TEXT,
    metrics_json TEXT,
    r_t REAL,
    reward_intensity REAL,
    delta_phi_used REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    used INTEGER DEFAULT 0,
    used_at DATETIME,
    used_in_run TEXT
);

-- Counterfactual replay scaffolding (sleep-time candidate generation + preference training).
CREATE TABLE IF NOT EXISTS sleep_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sleep_event_id INTEGER NOT NULL,
    candidate_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    model_id TEXT,
    adapter_path TEXT,
    temperature REAL,
    top_p REAL,
    max_new_tokens INTEGER,
    seed INTEGER,
    q_resp REAL,
    r_t REAL,
    score REAL,
    safety_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sleep_event_id) REFERENCES sleep_events(id)
);

CREATE INDEX IF NOT EXISTS idx_sleep_candidates_event ON sleep_candidates(sleep_event_id);

CREATE TABLE IF NOT EXISTS preference_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sleep_event_id INTEGER,
    chosen_candidate_id INTEGER NOT NULL,
    rejected_candidate_id INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    weight REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sleep_event_id) REFERENCES sleep_events(id),
    FOREIGN KEY (chosen_candidate_id) REFERENCES sleep_candidates(id),
    FOREIGN KEY (rejected_candidate_id) REFERENCES sleep_candidates(id)
);

CREATE INDEX IF NOT EXISTS idx_preference_pairs_event ON preference_pairs(sleep_event_id);

CREATE TABLE IF NOT EXISTS reward_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sleep_candidate_id INTEGER NOT NULL,
    q_resp REAL,
    unsafe INTEGER DEFAULT 0,
    notes TEXT,
    labeled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sleep_candidate_id) REFERENCES sleep_candidates(id)
);

CREATE INDEX IF NOT EXISTS idx_reward_labels_candidate ON reward_labels(sleep_candidate_id);
"""


class ConversationDB:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync endpoints in a threadpool, so DB access may occur from
        # threads other than the one that created the connection.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def get_state(self, conversation_id: str) -> Dict[str, Any]:
        with self._lock:
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
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO conversations(conversation_id, state_json, updated_at)
                VALUES(?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_id) DO UPDATE SET state_json=excluded.state_json, updated_at=CURRENT_TIMESTAMP
                """,
                (conversation_id, payload),
            )
            self.conn.commit()

    def append_message(self, conversation_id: str, role: str, content: str, think: str | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages(conversation_id, role, content, think) VALUES (?,?,?,?)",
                (conversation_id, role, content, think),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def list_messages(self, conversation_id: str, limit: int = 50) -> List[Tuple[str, str]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
            return list(reversed(cur.fetchall()))

    def insert_sleep_event(
        self,
        *,
        conversation_id: str,
        user_message: str,
        assistant: str,
        think: str | None,
        history: List[Dict[str, str]] | None,
        metrics: Dict[str, Any] | None,
        r_t: float | None,
        reward_intensity: float | None,
        delta_phi_used: float | None,
    ) -> int:
        history_json = json.dumps(history) if history is not None else None
        metrics_json = json.dumps(metrics) if metrics is not None else None
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO sleep_events
                (conversation_id, user_message, assistant, think, history_json, metrics_json,
                 r_t, reward_intensity, delta_phi_used)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    conversation_id,
                    user_message,
                    assistant,
                    think,
                    history_json,
                    metrics_json,
                    r_t,
                    reward_intensity,
                    delta_phi_used,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def list_sleep_events(
        self,
        *,
        only_unused: bool = True,
        limit: int = 1000,
        min_r_t: float | None = None,
        min_reward_intensity: float | None = None,
        require_positive_r_t: bool = True,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM sleep_events WHERE 1=1"
        params: List[Any] = []
        if only_unused:
            query += " AND used=0"
        if min_r_t is not None:
            if require_positive_r_t:
                query += " AND r_t >= ?"
                params.append(float(min_r_t))
            else:
                query += " AND abs(r_t) >= ?"
                params.append(float(min_r_t))
        if min_reward_intensity is not None:
            query += " AND reward_intensity >= ?"
            params.append(float(min_reward_intensity))
        query += " ORDER BY created_at ASC"
        query += " LIMIT ?"
        params.append(int(limit))
        with self._lock:
            cur = self.conn.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
        for row in rows:
            for key in ("history_json", "metrics_json"):
                if row.get(key):
                    try:
                        row[key] = json.loads(row[key])
                    except Exception:
                        pass
        return rows

    def mark_sleep_events_used(self, *, event_ids: List[int], run_id: str) -> None:
        if not event_ids:
            return
        placeholders = ",".join(["?"] * len(event_ids))
        with self._lock:
            self.conn.execute(
                f"""
                UPDATE sleep_events
                SET used=1, used_at=CURRENT_TIMESTAMP, used_in_run=?
                WHERE id IN ({placeholders})
                """,
                (run_id, *event_ids),
            )
            self.conn.commit()
