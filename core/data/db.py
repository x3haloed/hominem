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
    ) -> List[Dict[str, Any]]:
        """Get preference pairs for training."""
        query = "SELECT * FROM preference_pairs ORDER BY created_at DESC"
        params = []
        
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
