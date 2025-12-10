"""
Database utilities for training data storage.

This module provides a shared interface for reading/writing training data
to/from SQLite, complementing the serving system's database.py.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to load .env file if dotenv is available
try:
    from dotenv import load_dotenv
    # Try loading from project root and apps/serve/
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    load_dotenv(project_root / "apps" / "serve" / ".env")
except ImportError:
    pass

# Default database path - can be overridden via environment variable
# Check both DATABASE_PATH (used by serving system) and HOMINEM_DB_PATH (legacy)
DEFAULT_DB_PATH = os.getenv(
    "DATABASE_PATH",
    os.getenv(
        "HOMINEM_DB_PATH",
        os.path.join(os.path.dirname(__file__), "../../storage/conversations.db")
    )
)


MAX_INSTRUCTION_LENGTH = 2048
MAX_RESPONSE_LENGTH = 4096


class TrainingDatabase:
    """Database manager for training data operations."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection and ensure schema exists."""
        self.db_path = db_path or DEFAULT_DB_PATH
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        
        # Ensure schema is up to date
        self._ensure_schema()

    def _ensure_schema(self):
        """Load and execute schema if it exists."""
        # Try to find schema.sql relative to this file or in apps/serve
        schema_paths = [
            os.path.join(os.path.dirname(__file__), "../../apps/serve/schema.sql"),
            os.path.join(os.path.dirname(__file__), "../../schema.sql"),
        ]
        
        for schema_path in schema_paths:
            if os.path.exists(schema_path):
                with open(schema_path, "r") as f:
                    schema = f.read()
                
                # Use executescript() which properly handles multi-statement SQL files
                # including multi-line statements like CREATE VIEW
                self.connection.executescript(schema)
                self.connection.commit()
                break

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # SFT pair operations

    _VALID_SFT_SOURCES = {
        "conversation",
        "manual",
        "correction",
        "knowledge_update",
        "synthetic",
    }

    def _validate_sft_pair(
        self,
        instruction: str,
        response: str,
        source: str,
        confidence: Optional[float] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Validate core constraints for SFT pairs."""
        if not instruction or len(instruction.strip()) < 10:
            raise ValueError("instruction must be at least 10 characters")
        if not response or len(response.strip()) < 10:
            raise ValueError("response must be at least 10 characters")
        if len(instruction) > MAX_INSTRUCTION_LENGTH:
            raise ValueError(f"instruction exceeds maximum length ({MAX_INSTRUCTION_LENGTH})")
        if len(response) > MAX_RESPONSE_LENGTH:
            raise ValueError(f"response exceeds maximum length ({MAX_RESPONSE_LENGTH})")
        if instruction.strip() == response.strip():
            raise ValueError("instruction and response cannot be identical")
        if source not in self._VALID_SFT_SOURCES:
            raise ValueError(f"source must be one of {sorted(self._VALID_SFT_SOURCES)}")
        if source == "conversation" and not conversation_id:
            raise ValueError("conversation_id is required when source='conversation'")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

    @staticmethod
    def _truncate_sft_fields(instruction: str, response: str) -> tuple[str, str]:
        """Apply max-length caps to instruction/response (truncate, do not discard)."""
        if len(instruction) > MAX_INSTRUCTION_LENGTH:
            instruction = instruction[:MAX_INSTRUCTION_LENGTH]
        if len(response) > MAX_RESPONSE_LENGTH:
            response = response[:MAX_RESPONSE_LENGTH]
        return instruction, response

    def insert_sft_pair(
        self,
        instruction: str,
        response: str,
        source: str,
        conversation_id: Optional[str] = None,
        message_id: Optional[int] = None,
        message_index: Optional[int] = None,
        extraction_method: Optional[str] = None,
        confidence: Optional[float] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extracted_at: Optional[str] = None,
    ) -> int:
        """Insert a single SFT pair after validation."""
        instruction, response = self._truncate_sft_fields(instruction, response)
        self._validate_sft_pair(
            instruction,
            response,
            source,
            confidence,
            conversation_id=conversation_id,
        )
        metadata_json = json.dumps(metadata) if metadata else None
        cursor = self.connection.execute(
            """
            INSERT INTO sft_pairs
            (instruction, response, source, conversation_id, message_id, message_index,
             extraction_method, confidence, category, metadata, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                instruction,
                response,
                source,
                conversation_id,
                message_id,
                message_index,
                extraction_method,
                confidence,
                category,
                metadata_json,
                extracted_at,
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def insert_sft_pairs(self, pairs: List[Dict[str, Any]]) -> List[int]:
        """Bulk insert SFT pairs with validation."""
        inserted_ids: List[int] = []
        for pair in pairs:
            inserted_ids.append(
                self.insert_sft_pair(
                    instruction=pair["instruction"],
                    response=pair["response"],
                    source=pair.get("source", "conversation"),
                    conversation_id=pair.get("conversation_id"),
                    message_id=pair.get("message_id"),
                    message_index=pair.get("message_index"),
                    extraction_method=pair.get("extraction_method"),
                    confidence=pair.get("confidence"),
                    category=pair.get("category"),
                    metadata=pair.get("metadata"),
                    extracted_at=pair.get("extracted_at"),
                )
            )
        return inserted_ids

    def get_sft_pairs(
        self,
        source: Optional[str] = None,
        is_used: Optional[bool] = None,
        since: Optional[str] = None,
        min_instruction_length: Optional[int] = None,
        min_response_length: Optional[int] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get SFT pairs with optional filtering."""
        query = "SELECT * FROM sft_pairs WHERE 1=1"
        params: List[Any] = []

        if source:
            query += " AND source = ?"
            params.append(source)

        if is_used is not None:
            query += " AND is_used = ?"
            params.append(is_used)

        if since:
            query += " AND created_at >= ?"
            params.append(since)

        if min_instruction_length is not None:
            query += " AND length(instruction) >= ?"
            params.append(min_instruction_length)

        if min_response_length is not None:
            query += " AND length(response) >= ?"
            params.append(min_response_length)

        if min_confidence is not None:
            query += " AND (confidence IS NULL OR confidence >= ?)"
            params.append(min_confidence)

        query += " ORDER BY created_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def update_sft_used(
        self,
        pair_ids: List[int],
        training_epoch: Optional[int] = None,
        used_timestamp: Optional[str] = None,
        training_batch: Optional[str] = None,
    ) -> None:
        """Mark SFT pairs as used in training."""
        if not pair_ids:
            return
        timestamp = used_timestamp or datetime.utcnow().isoformat()
        placeholders = ",".join(["?"] * len(pair_ids))
        params: List[Any] = [training_batch, training_epoch, timestamp, *pair_ids]
        self.connection.execute(
            f"""
            UPDATE sft_pairs
            SET is_used = TRUE,
                used_in_training_batch = ?,
                training_epoch = ?,
                used_timestamp = ?
            WHERE id IN ({placeholders})
            """,
            params,
        )
        self.connection.commit()

    # Trajectories operations

    def insert_trajectory(
        self,
        trajectory_id: str,
        prompt: str,
        response: str,
        prompt_id: Optional[str] = None,
        category: Optional[str] = None,
        persona: Optional[str] = None,
        candidate_index: Optional[int] = None,
        source: Optional[str] = None,
        generator_model_id: Optional[str] = None,
        generator_model_alias: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a trajectory into the database."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.connection.execute("""
            INSERT OR IGNORE INTO trajectories
            (trajectory_id, prompt_id, category, persona, prompt, response,
             candidate_index, source, generator_model_id, generator_model_alias, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trajectory_id, prompt_id, category, persona, prompt, response,
            candidate_index, source, generator_model_id, generator_model_alias, metadata_json
        ))
        
        self.connection.commit()
        return cursor.lastrowid

    def get_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Get a trajectory by ID."""
        cursor = self.connection.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # Reward samples operations

    def insert_reward_sample(
        self,
        sample_id: str,
        prompt: str,
        response: str,
        reward: Dict[str, float],
        trajectory_id: Optional[str] = None,
        prompt_id: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a reward-labeled sample."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.connection.execute("""
            INSERT OR REPLACE INTO reward_samples
            (sample_id, trajectory_id, prompt_id, category, prompt, response,
             empathy, social_coherence, agency_support, epistemic_integrity,
             harm_avoidance, narrative_alignment, curiosity,
             scalar, reward_intensity, safety_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_id, trajectory_id, prompt_id, category, prompt, response,
            reward.get("empathy"),
            reward.get("social_coherence"),
            reward.get("agency_support"),
            reward.get("epistemic_integrity"),
            reward.get("harm_avoidance"),
            reward.get("narrative_alignment"),
            reward.get("curiosity"),
            reward.get("scalar"),
            reward.get("reward_intensity"),
            reward.get("safety_score"),
            metadata_json,
        ))
        
        self.connection.commit()
        return cursor.lastrowid

    def get_reward_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get a reward sample by ID."""
        cursor = self.connection.execute(
            "SELECT * FROM reward_samples WHERE sample_id = ?",
            (sample_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # Preference pairs operations

    def insert_preference_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        chosen_id: Optional[str] = None,
        rejected_id: Optional[str] = None,
        prompt_id: Optional[str] = None,
        category: Optional[str] = None,
        chosen_score: Optional[float] = None,
        rejected_score: Optional[float] = None,
        score_margin: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a preference pair for DPO training."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.connection.execute("""
            INSERT INTO preference_pairs
            (prompt, chosen, rejected, chosen_id, rejected_id, prompt_id,
             category, chosen_score, rejected_score, score_margin, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt, chosen, rejected, chosen_id, rejected_id, prompt_id,
            category, chosen_score, rejected_score, score_margin, metadata_json
        ))
        
        self.connection.commit()
        return cursor.lastrowid

    def get_preference_pairs(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get preference pairs for training."""
        query = "SELECT * FROM preference_pairs"
        params: List[Any] = []

        if category:
            query += " WHERE category = ?"
            params.append(category)

        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # Self-train events operations

    def insert_self_train_event(
        self,
        session_id: str,
        timestamp_utc: str,
        prompt: str,
        chosen_text: str,
        chosen_reward: Dict[str, float],
        chosen_scalar_score: float,
        candidates_json: List[Dict[str, Any]],
        num_candidates: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a self-training event."""
        candidates_json_str = json.dumps(candidates_json) if candidates_json else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.connection.execute("""
            INSERT INTO self_train_events
            (session_id, timestamp_utc, prompt, num_candidates, max_new_tokens,
             temperature, top_p, device,
             chosen_text, chosen_scalar_score,
             chosen_reward_empathy, chosen_reward_social_coherence,
             chosen_reward_agency_support, chosen_reward_epistemic_integrity,
             chosen_reward_harm_avoidance, chosen_reward_narrative_alignment,
             chosen_reward_curiosity, chosen_reward_intensity, chosen_reward_safety_score,
             candidates_json, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, timestamp_utc, prompt, num_candidates, max_new_tokens,
            temperature, top_p, device,
            chosen_text, chosen_scalar_score,
            chosen_reward.get("empathy"),
            chosen_reward.get("social_coherence"),
            chosen_reward.get("agency_support"),
            chosen_reward.get("epistemic_integrity"),
            chosen_reward.get("harm_avoidance"),
            chosen_reward.get("narrative_alignment"),
            chosen_reward.get("curiosity"),
            chosen_reward.get("reward_intensity"),
            chosen_reward.get("safety_score"),
            candidates_json_str,
            metadata_json,
        ))
        
        self.connection.commit()
        return cursor.lastrowid

    def get_self_train_events(
        self,
        session_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_reward_intensity: Optional[float] = None,
        min_safety_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get self-training events with optional filtering."""
        query = "SELECT * FROM self_train_events WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if start_date:
            query += " AND timestamp_utc >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp_utc <= ?"
            params.append(end_date)
        
        if min_reward_intensity is not None:
            query += " AND chosen_reward_intensity >= ?"
            params.append(min_reward_intensity)
        
        if min_safety_score is not None:
            query += " AND chosen_reward_safety_score >= ?"
            params.append(min_safety_score)
        
        query += " ORDER BY timestamp_utc DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # Training runs/steps/evals operations

    def create_training_run(
        self,
        run_id: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create a new training run."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.connection.execute("""
            INSERT OR REPLACE INTO training_runs
            (run_id, component, metadata)
            VALUES (?, ?, ?)
        """, (run_id, component, metadata_json))
        
        self.connection.commit()
        return cursor.lastrowid

    def log_training_step(
        self,
        run_id: str,
        step_number: int,
        metrics: Dict[str, Any],
    ) -> int:
        """Log a training step."""
        metrics_json = json.dumps(metrics) if metrics else None
        
        cursor = self.connection.execute("""
            INSERT INTO training_steps
            (run_id, step_number, metrics)
            VALUES (?, ?, ?)
        """, (run_id, step_number, metrics_json))
        
        self.connection.commit()
        return cursor.lastrowid

    def log_training_eval(
        self,
        run_id: str,
        eval_number: int,
        metrics: Dict[str, Any],
    ) -> int:
        """Log a training evaluation."""
        metrics_json = json.dumps(metrics) if metrics else None
        
        cursor = self.connection.execute("""
            INSERT INTO training_evals
            (run_id, eval_number, metrics)
            VALUES (?, ?, ?)
        """, (run_id, eval_number, metrics_json))
        
        self.connection.commit()
        return cursor.lastrowid

    def get_training_steps(
        self,
        run_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get training steps for a run."""
        query = "SELECT * FROM training_steps WHERE run_id = ? ORDER BY step_number ASC"
        params = [run_id]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_training_evals(
        self,
        run_id: str,
    ) -> List[Dict[str, Any]]:
        """Get training evals for a run."""
        cursor = self.connection.execute(
            "SELECT * FROM training_evals WHERE run_id = ? ORDER BY eval_number ASC",
            (run_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
