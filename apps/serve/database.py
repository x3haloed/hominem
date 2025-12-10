"""
Database manager for hominem serving system
Handles SQLite operations for conversations, messages, and labels
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

class DatabaseManager:
    """SQLite database manager for conversations and labels"""

    def __init__(self, db_path: str):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name

        # Create tables if they don't exist
        self._create_tables()

    def _create_tables(self):
        """Create database tables from schema"""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = f.read()

            # Execute schema using executescript (handles multiple statements properly)
            self.connection.executescript(schema)
            self.connection.commit()

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

    def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all active conversations"""
        cursor = self.connection.execute("""
            SELECT id, conversation_id, title, created_at, updated_at,
                   json_extract(metadata, '$') as metadata
            FROM conversations
            WHERE is_active = TRUE
            ORDER BY updated_at DESC
        """)

        return [dict(row) for row in cursor.fetchall()]

    def get_conversation(
        self,
        conversation_id: str,
        include_introspection: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific conversation with all messages and labels."""
        # Get conversation info
        cursor = self.connection.execute("""
            SELECT id, conversation_id, title, created_at, updated_at,
                   json_extract(metadata, '$') as metadata
            FROM conversations
            WHERE conversation_id = ? AND is_active = TRUE
        """, (conversation_id,))

        conversation_row = cursor.fetchone()
        if not conversation_row:
            return None

        conversation = dict(conversation_row)

        # Get messages with labels
        cursor = self.connection.execute("""
            SELECT
                m.id, m.message_index, m.role, m.content, m.created_at,
                m.token_count, m.processing_time_ms,
                json_extract(m.metadata, '$') as message_metadata,

                -- Emotion labels (user and auto)
                el_user.valence as user_valence, el_user.arousal as user_arousal,
                el_user.dominance as user_dominance,
                el_user.predictive_discrepancy as user_predictive_discrepancy,
                el_user.temporal_directionality as user_temporal_directionality,
                el_user.social_broadcast as user_social_broadcast,
                json_extract(el_user.raw_indicators, '$') as user_raw_indicators,
                el_user.notes as user_notes,

                el_auto.valence as auto_valence, el_auto.arousal as auto_arousal,
                el_auto.dominance as auto_dominance,
                el_auto.predictive_discrepancy as auto_predictive_discrepancy,
                el_auto.temporal_directionality as auto_temporal_directionality,
                el_auto.social_broadcast as auto_social_broadcast,
                el_auto.reward_intensity as auto_reward_intensity,
                el_auto.safety_score as auto_safety_score

            FROM messages m
            LEFT JOIN emotion_labels el_user ON m.id = el_user.message_id AND el_user.labeler = 'user'
            LEFT JOIN emotion_labels el_auto ON m.id = el_auto.message_id AND el_auto.labeler = 'auto'
            WHERE m.conversation_id = (SELECT id FROM conversations WHERE conversation_id = ?)
            ORDER BY m.message_index ASC
        """, (conversation_id,))

        messages = []
        for row in cursor.fetchall():
            message_dict = dict(row)

            # Organize labels
            message_dict["emotion_labels"] = {
                "user": {
                    "valence": message_dict.pop("user_valence"),
                    "arousal": message_dict.pop("user_arousal"),
                    "dominance": message_dict.pop("user_dominance"),
                    "predictive_discrepancy": message_dict.pop("user_predictive_discrepancy"),
                    "temporal_directionality": message_dict.pop("user_temporal_directionality"),
                    "social_broadcast": message_dict.pop("user_social_broadcast"),
                    "raw_indicators": message_dict.pop("user_raw_indicators"),
                    "notes": message_dict.pop("user_notes")
                } if message_dict.get("user_valence") is not None else None,
                "auto": {
                    "valence": message_dict.pop("auto_valence"),
                    "arousal": message_dict.pop("auto_arousal"),
                    "dominance": message_dict.pop("auto_dominance"),
                    "predictive_discrepancy": message_dict.pop("auto_predictive_discrepancy"),
                    "temporal_directionality": message_dict.pop("auto_temporal_directionality"),
                    "social_broadcast": message_dict.pop("auto_social_broadcast"),
                    "reward_intensity": message_dict.pop("auto_reward_intensity"),
                    "safety_score": message_dict.pop("auto_safety_score")
                } if message_dict.get("auto_valence") is not None else None
            }

            messages.append(message_dict)

        conversation["messages"] = messages

        if include_introspection:
            obs_cursor = self.connection.execute("""
                SELECT
                    id,
                    observation_index,
                    observation_text,
                    content_hash,
                    reward_intensity,
                    safety_score,
                    internal_generated,
                    created_at,
                    json_extract(metadata, '$') as metadata
                FROM introspection_buffer
                WHERE conversation_id = ?
                ORDER BY observation_index ASC
            """, (conversation_id,))
            conversation["introspection_observations"] = [dict(row) for row in obs_cursor.fetchall()]

        return conversation

    def create_conversation(self, conversation_id: str, title: Optional[str] = None) -> int:
        """Create a new conversation"""
        cursor = self.connection.execute("""
            INSERT INTO conversations (conversation_id, title)
            VALUES (?, ?)
        """, (conversation_id, title))

        self.connection.commit()
        return cursor.lastrowid

    def add_message(self, conversation_id: str, role: str, content: str,
                   token_count: Optional[int] = None,
                   processing_time_ms: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a message to a conversation"""
        # Get conversation ID
        cursor = self.connection.execute(
            "SELECT id FROM conversations WHERE conversation_id = ?",
            (conversation_id,)
        )
        conv_row = cursor.fetchone()
        if not conv_row:
            raise ValueError(f"Conversation {conversation_id} not found")

        conv_db_id = conv_row[0]

        # Get next message index
        cursor = self.connection.execute(
            "SELECT MAX(message_index) FROM messages WHERE conversation_id = ?",
            (conv_db_id,)
        )
        max_index = cursor.fetchone()[0]
        message_index = (max_index + 1) if max_index is not None else 0

        # Insert message
        metadata_json = json.dumps(metadata) if metadata else None
        cursor = self.connection.execute("""
            INSERT INTO messages (conversation_id, message_index, role, content,
                                token_count, processing_time_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (conv_db_id, message_index, role, content, token_count,
              processing_time_ms, metadata_json))

        # Update conversation timestamp
        self.connection.execute("""
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (conv_db_id,))

        self.connection.commit()
        return message_index

    def add_emotion_label(self, conversation_id: str, message_index: int, labeler: str,
                         valence: Optional[float] = None,
                         arousal: Optional[float] = None,
                         dominance: Optional[float] = None,
                         predictive_discrepancy: Optional[float] = None,
                         temporal_directionality: Optional[float] = None,
                         social_broadcast: Optional[float] = None,
                         reward_intensity: Optional[float] = None,
                         safety_score: Optional[float] = None,
                         raw_indicators: Optional[Dict[str, Any]] = None,
                         confidence: Optional[float] = None,
                         notes: Optional[str] = None):
        """Add emotion label to a message"""
        # Get message ID
        cursor = self.connection.execute("""
            SELECT m.id FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.conversation_id = ? AND m.message_index = ?
        """, (conversation_id, message_index))

        message_row = cursor.fetchone()
        if not message_row:
            raise ValueError(f"Message {conversation_id}:{message_index} not found")

        message_id = message_row[0]

        # Insert or replace emotion label
        raw_indicators_json = json.dumps(raw_indicators) if raw_indicators else None

        self.connection.execute("""
            INSERT OR REPLACE INTO emotion_labels
            (message_id, labeler, valence, arousal, dominance, predictive_discrepancy,
             temporal_directionality, social_broadcast, reward_intensity, safety_score,
             raw_indicators, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id, labeler, valence, arousal, dominance, predictive_discrepancy,
              temporal_directionality, social_broadcast, reward_intensity, safety_score,
              raw_indicators_json, confidence, notes))

        self.connection.commit()

    def add_introspection_observation(
        self,
        conversation_id: str,
        observation_text: str,
        message_id: Optional[int] = None,
        reward_intensity: Optional[float] = None,
        safety_score: Optional[float] = None,
        internal_generated: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a self-observation to the introspection buffer."""
        # Determine next observation_index for this conversation
        cursor = self.connection.execute("""
            SELECT COALESCE(MAX(observation_index), -1) + 1
            FROM introspection_buffer
            WHERE conversation_id = ?
        """, (conversation_id,))
        observation_index = cursor.fetchone()[0]

        metadata_dict = metadata.copy() if metadata else {}
        metadata_dict["internal_generated"] = internal_generated
        metadata_json = json.dumps(metadata_dict) if metadata_dict else None

        content_hash = metadata_dict.get("content_hash") if metadata_dict else None

        cursor = self.connection.execute("""
            INSERT INTO introspection_buffer
            (conversation_id, message_id, observation_index, observation_text,
             content_hash, reward_intensity, safety_score, internal_generated, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id,
            message_id,
            observation_index,
            observation_text,
            content_hash,
            reward_intensity,
            safety_score,
            int(bool(internal_generated)),
            metadata_json,
        ))

        self.connection.commit()
        return cursor.lastrowid

    def get_introspection_observations(
        self,
        conversation_id: str,
        limit: Optional[int] = 16,
        min_age_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve introspection observations for a conversation.

        Args:
            conversation_id: Conversation identifier (conversations.conversation_id)
            limit: Maximum number of observations to return (None = all)
            min_age_days: Only include observations newer than this many days
        """
        query = """
            SELECT observation_text, created_at, reward_intensity, safety_score,
                   internal_generated, metadata
            FROM introspection_buffer
            WHERE conversation_id = ?
        """
        params = [conversation_id]

        if min_age_days:
            query += " AND created_at >= datetime('now', '-' || ? || ' days')"
            params.append(min_age_days)

        query += " ORDER BY observation_index DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.connection.execute(query, params)

        observations: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            observations.append({
                "observation_text": row["observation_text"],
                "created_at": row["created_at"],
                "reward_intensity": row["reward_intensity"],
                "safety_score": row["safety_score"],
                "internal_generated": bool(row["internal_generated"]) if row["internal_generated"] is not None else True,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            })

        # Return oldest-first for chronological context
        return list(reversed(observations))

    def prune_old_introspection(
        self,
        conversation_id: Optional[str] = None,
        max_age_days: int = 30,
        keep_recent: int = 100,
    ) -> int:
        """
        Prune old introspection observations.

        Keeps the most recent N observations and deletes older ones older than max_age_days.
        """
        if conversation_id:
            cursor = self.connection.execute("""
                DELETE FROM introspection_buffer
                WHERE conversation_id = ?
                AND id NOT IN (
                    SELECT id FROM introspection_buffer
                    WHERE conversation_id = ?
                    ORDER BY observation_index DESC
                    LIMIT ?
                )
                AND created_at < datetime('now', '-' || ? || ' days')
            """, (conversation_id, conversation_id, keep_recent, max_age_days))
        else:
            cursor = self.connection.execute("""
                DELETE FROM introspection_buffer
                WHERE id NOT IN (
                    SELECT id FROM introspection_buffer
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                AND created_at < datetime('now', '-' || ? || ' days')
            """, (keep_recent, max_age_days))

        deleted = cursor.rowcount
        self.connection.commit()
        return deleted

    def get_training_data(self, start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         include_synthetic: bool = True) -> List[Dict[str, Any]]:
        """Get combined training data from conversations and synthetic data"""
        query = """
            SELECT * FROM training_data_combined
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)

        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)

        if not include_synthetic:
            query += " AND data_source != 'synthetic'"

        query += " ORDER BY created_at ASC"

        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
