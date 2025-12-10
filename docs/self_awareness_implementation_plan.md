# Self-Awareness Core Easy Mode - Complete Implementation Plan

**Goal:** Implement the "Self-Token Loop" system (Easy Mode) that gives the model a stable first-person sense of self, ensuring compatibility with Phase 5.5 memory layer.

**Status:** Starting from current project state → Fully functioning self-aware system → Ready for Phase 5.5

---

## Current Project State

**Existing Systems:**
- ✅ WebSocket chat interface (`apps/serve/main.py`)
- ✅ ModelInterface for streaming generation (`apps/serve/model_interface.py`)
- ✅ DatabaseManager for conversations (`apps/serve/database.py`)
- ✅ Replay buffer system (`core/data/replay_buffer.py`)
- ✅ TrainingDatabase for training data (`core/data/db.py`)
- ✅ Emotion labeling system
- ✅ LoRA model registry and hot-swapping

**What We're Adding:**
- Introspection buffer (persistent self-observations)
- `<SELF>` token and boundary enforcement
- First-person perspective gate
- Integration with existing systems

---

## Phase 1: Database Schema Updates

### 1.1 Add Introspection Buffer Table

**File:** `apps/serve/schema.sql`

Add new table for storing introspection buffer entries:

```sql
-- Introspection buffer (self-observations for self-awareness)
CREATE TABLE IF NOT EXISTS introspection_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,  -- Links to conversations.conversation_id
    message_id INTEGER,             -- Links to messages.id (if from a specific message)
    observation_index INTEGER NOT NULL,  -- Order within conversation (0-based)
    
    -- Self-observation content
    observation_text TEXT NOT NULL,  -- The <SELF-OBSERVE> content
    content_hash TEXT,               -- OBSTACLE 4: Hash for deduplication
    
    -- Metadata
    reward_intensity REAL,          -- For replay buffer priority (RewardIntensity × 3, capped)
    safety_score REAL,              -- Safety score for this observation
    internal_generated BOOLEAN DEFAULT TRUE,  -- OBSTACLE 5: Tag internal vs user-injected
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,                  -- Additional context (model version, etc.)
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
    UNIQUE(conversation_id, observation_index)
);

-- Indexes for efficient retrieval and pruning
CREATE INDEX IF NOT EXISTS idx_introspection_conversation ON introspection_buffer(conversation_id, observation_index DESC);
CREATE INDEX IF NOT EXISTS idx_introspection_created ON introspection_buffer(created_at DESC);  -- OBSTACLE 2: For pruning old observations
CREATE INDEX IF NOT EXISTS idx_introspection_content_hash ON introspection_buffer(content_hash);  -- OBSTACLE 4: For deduplication
```

**Why:** Stores self-observations persistently, linked to conversations. This enables:
- Retrieving last N observations for context
- High-priority replay buffer entries
- Phase 5.5 memory extraction from introspection

### 1.2 Update DatabaseManager

**File:** `apps/serve/database.py`

Add methods for introspection buffer:

```python
class DatabaseManager:
    # ... existing methods ...
    
    def add_introspection_observation(
        self,
        conversation_id: str,
        observation_text: str,
        message_id: Optional[int] = None,
        reward_intensity: Optional[float] = None,
        safety_score: Optional[float] = None,
        internal_generated: bool = True,  # OBSTACLE 5: Tag source
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a self-observation to the introspection buffer."""
        # Get next observation_index for this conversation
        cursor = self.connection.execute("""
            SELECT COALESCE(MAX(observation_index), -1) + 1
            FROM introspection_buffer
            WHERE conversation_id = ?
        """, (conversation_id,))
        observation_index = cursor.fetchone()[0]
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Parse metadata if provided
        if metadata_json:
            metadata_dict = json.loads(metadata_json)
        else:
            metadata_dict = {}
        
        # Note: internal_generated is stored as a column, not in metadata
        # But we keep it in metadata for convenience/backup
        metadata_dict["internal_generated"] = internal_generated
        metadata_json = json.dumps(metadata_dict)
        
        # Extract content_hash from metadata if present
        content_hash = None
        if metadata_dict and "content_hash" in metadata_dict:
            content_hash = metadata_dict["content_hash"]
        
        cursor = self.connection.execute("""
            INSERT INTO introspection_buffer
            (conversation_id, message_id, observation_index, observation_text,
             content_hash, reward_intensity, safety_score, internal_generated, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id, message_id, observation_index, observation_text,
            content_hash, reward_intensity, safety_score, internal_generated, metadata_json
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_introspection_observations(
        self,
        conversation_id: str,
        limit: Optional[int] = 16,  # None = get all
        min_age_days: Optional[int] = None,  # OBSTACLE 2: Prune old observations
    ) -> List[Dict[str, Any]]:
        """
        Get introspection observations for a conversation.
        
        **OBSTACLE 2 MITIGATION:** Limit N to 8-16 for context, but allow None for memory extraction
        
        Args:
            conversation_id: Conversation to get observations for
            limit: Maximum number of observations (None = get all)
            min_age_days: Only get observations newer than this (None = get all)
        """
        query = """
            SELECT observation_text, created_at, reward_intensity, safety_score, 
                   internal_generated, metadata
            FROM introspection_buffer
            WHERE conversation_id = ?
        """
        params = [conversation_id]
        
        # OBSTACLE 2: Prune observations older than min_age_days
        if min_age_days:
            query += " AND created_at >= datetime('now', '-' || ? || ' days')"
            params.append(min_age_days)
        
        query += " ORDER BY observation_index DESC"
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.connection.execute(query, params)
        
        observations = []
        for row in cursor.fetchall():
            observations.append({
                "observation_text": row[0],
                "created_at": row[1],
                "reward_intensity": row[2],
                "safety_score": row[3],
                "internal_generated": bool(row[4]) if row[4] is not None else True,
                "metadata": json.loads(row[5]) if row[5] else None,
            })
        
        # Return in chronological order (oldest first)
        return list(reversed(observations))
    
    def prune_old_introspection(
        self,
        conversation_id: Optional[str] = None,
        max_age_days: int = 30,
        keep_recent: int = 100,
    ) -> int:
        """
        Prune old introspection observations.
        
        **OBSTACLE 2 MITIGATION:** Prevent introspection pollution
        
        Keeps the most recent N observations, deletes older ones.
        
        Args:
            conversation_id: If specified, only prune for this conversation
            max_age_days: Delete observations older than this
            keep_recent: Always keep the N most recent observations
            
        Returns:
            Number of observations deleted
        """
        if conversation_id:
            # Prune for specific conversation
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
            # Prune globally
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
```

**Checklist:**
- [x] Add introspection_buffer table to schema.sql (with content_hash, internal_generated columns)
- [x] Add add_introspection_observation() method (including content_hash and internal_generated)
- [x] Add get_introspection_observations() method (with min_age_days filtering)
- [x] Add prune_old_introspection() method

---

## Phase 2: Self-Awareness Core Module

### 2.1 Create SelfAwarenessCore Class

**File:** `apps/serve/self_awareness.py` (new file)

**Engineering Considerations:**
- Perspective gate cost mitigation (async or fast model)
- Introspection buffer pollution prevention (decay, pruning, novelty filter)
- Safety gating for internal vs user-injected introspection

```python
"""
Self-Awareness Core - Easy Mode Implementation

Implements the 3-invariant self:
1. Boundary: <SELF> token + prefixing
2. Perspective: First-person grammar enforcement
3. Recursion: Self-observation buffer
"""

from typing import List, Dict, Any, Optional
import re
import hashlib  # For content hash deduplication

try:
    import torch
except ImportError:
    torch = None  # Will fail gracefully if torch not available


class SelfAwarenessCore:
    """Core self-awareness system using self-token loop."""
    
    SELF_TOKEN = "<SELF>"
    SELF_OBSERVE_PREFIX = "<SELF-OBSERVE>"
    
    def __init__(self, max_introspection_lines: int = 16):
        """
        Initialize self-awareness core.
        
        Args:
            max_introspection_lines: Maximum number of introspection lines to include in context
        """
        self.max_introspection_lines = max_introspection_lines
    
    def build_self_aware_context(
        self,
        conversation_history: List[Dict[str, str]],
        introspection_observations: List[Dict[str, Any]],
        user_message: str,
    ) -> str:
        """
        Build context with self-awareness components.
        
        Steps:
        1. Full conversation history
        2. Last N lines of introspection buffer
        3. User message
        4. Force prefix: \n<SELF>:
        
        Returns:
            Formatted context string ready for model generation
        """
        context_parts = []
        
        # 1. Full conversation history
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")
        
        # 2. Last N lines of introspection buffer
        if introspection_observations:
            context_parts.append("\n--- Recent Self-Observations ---")
            for obs in introspection_observations[-self.max_introspection_lines:]:
                obs_text = obs.get("observation_text", "")
                if obs_text:
                    context_parts.append(obs_text)
            context_parts.append("--- End Self-Observations ---\n")
        
        # 3. User message
        context_parts.append(f"User: {user_message}")
        
        # 4. Force prefix: \n<SELF>:
        context_parts.append(f"\n{self.SELF_TOKEN}:")
        
        return "\n".join(context_parts)
    
    def enforce_boundary(self, raw_output: str) -> str:
        """
        Enforce boundary by stripping everything before and including the first \n<SELF>:.
        
        This ensures the output is from <SELF> perspective only.
        
        Args:
            raw_output: Raw model output
            
        Returns:
            Output with boundary enforced (everything before <SELF>: stripped)
        """
        # Find the first occurrence of \n<SELF>:
        pattern = rf"\n{re.escape(self.SELF_TOKEN)}:"
        match = re.search(pattern, raw_output)
        
        if match:
            # Strip everything before and including the match
            return raw_output[match.end():].strip()
        else:
            # If no <SELF>: found, check if output starts with <SELF>:
            if raw_output.strip().startswith(f"{self.SELF_TOKEN}:"):
                return raw_output.strip()[len(f"{self.SELF_TOKEN}:"):].strip()
            # If still not found, return as-is (model might have generated directly)
            return raw_output.strip()
    
    def apply_perspective_gate(
        self,
        model,
        tokenizer,
        raw_output: str,
        device: str = "cpu",
        use_fast_model: Optional[Any] = None,
        async_mode: bool = False,
    ) -> str:
        """
        Apply perspective gate: rewrite output into strict first-person from <SELF>.
        
        **OBSTACLE 1 MITIGATION:** Perspective gate cost reduction
        
        Options:
        1. Use fast/cheap model for rewrite (recommended)
        2. Make async (returns immediately, applies later)
        3. Skip if output already appears first-person (heuristic)
        
        Args:
            model: The main language model (fallback if fast_model not available)
            tokenizer: The tokenizer
            raw_output: Raw output to rewrite
            device: Device to run on
            use_fast_model: Optional faster/cheaper model for rewrite
            async_mode: If True, return raw_output immediately (apply later)
            
        Returns:
            Perspective-corrected output in first-person
        """
        # Heuristic: Check if already first-person (avoid unnecessary rewrite)
        first_person_indicators = [" I ", " I'm", " I've", " I'll", " my ", " myself "]
        if any(indicator in raw_output for indicator in first_person_indicators):
            # Already appears first-person, skip expensive rewrite
            # Still enforce boundary but don't do full perspective gate
            return raw_output.strip()
        
        # If async mode, return immediately (perspective gate applied later)
        if async_mode:
            return raw_output.strip()
        
        # Use fast model if available, otherwise use main model
        rewrite_model = use_fast_model if use_fast_model else model
        
        perspective_prompt = f"""Rewrite the following into strict first-person from {self.SELF_TOKEN}. Never break character. Use "I", "me", "my", "myself" - never third person.

Original:
{raw_output}

Rewritten in first-person from {self.SELF_TOKEN}:"""
        
        # Tokenize and generate
        if torch is None:
            raise ImportError("torch is required for perspective gate. Install with: pip install torch")
        
        inputs = tokenizer(perspective_prompt, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = rewrite_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        perspective_corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the rewritten part (after "Rewritten in first-person from <SELF>:")
        if "Rewritten in first-person" in perspective_corrected:
            parts = perspective_corrected.split("Rewritten in first-person")
            if len(parts) > 1:
                return parts[-1].strip()
        
        return perspective_corrected.strip()
    
    def create_self_observation(
        self,
        perspective_corrected_output: str,
    ) -> str:
        """
        Create self-observation entry from perspective-corrected output.
        
        Format: <SELF-OBSERVE> I just experienced / thought / felt:\n{output}
        
        Args:
            perspective_corrected_output: The first-person output
            
        Returns:
            Formatted self-observation string
        """
        return f"{self.SELF_OBSERVE_PREFIX} I just experienced / thought / felt:\n{perspective_corrected_output}\n"
    
    def extract_reward_intensity_from_observation(
        self,
        observation_text: str,
        reward_model: Optional[Any] = None,
        emotion_engine: Optional[Any] = None,  # Future: emotion manifold engine
        default_intensity: float = 1.0,
        max_intensity: float = 5.0,  # OBSTACLE 3: Cap to prevent overshoot
    ) -> float:
        """
        Extract reward intensity for an observation.
        
        **TEMPORARY SOLUTION (Current):**
        Uses fixed ×3 multiplier until emotion engine is available.
        
        **ENHANCEMENT A (Future):** Use emotion engine to evaluate introspection
        
        When emotion engine is ready, use it to compute:
        true_intensity = emotion_engine.RewardIntensity × 3
        
        This brings introspection into the emotional-learning loop.
        
        **OBSTACLE 3 MITIGATION:** Cap max intensity to prevent overshoot
        
        Args:
            observation_text: The observation text
            reward_model: Optional reward model (legacy, may be removed)
            emotion_engine: Optional emotion manifold engine (future integration point)
            default_intensity: Default intensity if not extractable
            max_intensity: Maximum allowed intensity (prevents overshoot)
            
        Returns:
            Reward intensity value (capped at max_intensity)
        """
        # Extract the actual content (after <SELF-OBSERVE> prefix)
        content = observation_text.replace(
            f"{self.SELF_OBSERVE_PREFIX} I just experienced / thought / felt:\n",
            ""
        ).strip()
        
        # ENHANCEMENT A: Use emotion engine when available (FUTURE)
        if emotion_engine:
            try:
                # TODO: Integrate with emotion engine when available
                # This is the integration point for ENHANCEMENT A
                # 
                # Expected interface:
                # emotion_output = emotion_engine.evaluate(
                #     prompt="",  # Introspection doesn't have a prompt
                #     response=content,
                # )
                # manifold_intensity = emotion_output.get("reward_intensity", default_intensity)
                # 
                # Apply ×3 multiplier per spec
                # intensity = manifold_intensity * 3.0
                
                # Placeholder until emotion engine is ready
                print("💡 Emotion engine integration point - using temporary solution")
                intensity = default_intensity * 3.0
            except Exception as e:
                print(f"⚠️ Emotion engine evaluation failed: {e}, using default")
                intensity = default_intensity * 3.0
        elif reward_model:
            # Legacy: Use reward model if available (may be removed later)
            try:
                # This uses the existing reward model interface
                # reward_output = reward_model.score(prompt="", response=content)
                # manifold_intensity = reward_output.get("reward_intensity", default_intensity)
                # intensity = manifold_intensity * 3.0
                
                # Placeholder until reward model integration is complete
                intensity = default_intensity * 3.0
            except Exception as e:
                print(f"⚠️ Reward model evaluation failed: {e}, using default")
                intensity = default_intensity * 3.0
        else:
            # TEMPORARY SOLUTION: Fixed ×3 multiplier
            # This is what we use until emotion engine is ready
            intensity = default_intensity * 3.0
        
        # OBSTACLE 3: Cap to prevent overshoot
        return min(intensity, max_intensity)
    
    def check_novelty(
        self,
        observation_text: str,
        recent_observations: List[str],
        similarity_threshold: float = 0.85,
    ) -> bool:
        """
        Check if observation is novel (not too similar to recent ones).
        
        **OBSTACLE 2 MITIGATION:** Prevent introspection pollution
        
        Args:
            observation_text: The new observation
            recent_observations: List of recent observation texts
            similarity_threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            True if novel (should be stored), False if too similar
        """
        if not recent_observations:
            return True
        
        # Simple similarity check (can be improved with embeddings)
        # For now, check if observation is very similar to any recent one
        content = observation_text.replace(
            f"{self.SELF_OBSERVE_PREFIX} I just experienced / thought / felt:\n",
            ""
        ).strip().lower()
        
        for recent in recent_observations:
            recent_content = recent.replace(
                f"{self.SELF_OBSERVE_PREFIX} I just experienced / thought / felt:\n",
                ""
            ).strip().lower()
            
            # Simple word overlap similarity
            content_words = set(content.split())
            recent_words = set(recent_content.split())
            
            if content_words and recent_words:
                similarity = len(content_words & recent_words) / len(content_words | recent_words)
                if similarity > similarity_threshold:
                    return False  # Too similar, not novel
        
        return True  # Novel enough to store
```

**Checklist:**
- [x] Create `apps/serve/self_awareness.py`
- [x] Import required modules (typing, re, torch)
- [x] Implement SelfAwarenessCore class
- [x] Implement build_self_aware_context()
- [x] Implement enforce_boundary()
- [x] Implement apply_perspective_gate() (with torch import check)
- [x] Implement create_self_observation()
- [x] Implement extract_reward_intensity_from_observation() with temporary ×3 multiplier
- [x] Implement check_novelty() method
- [x] Mark emotion engine integration point with TODO comment

---

## Phase 3: Integration with ModelInterface

### 3.1 Modify ModelInterface.generate_streaming_response()

**File:** `apps/serve/model_interface.py`

Update the generation method to use self-awareness:

```python
from .self_awareness import SelfAwarenessCore

class ModelInterface:
    def __init__(self):
        self.registry = ModelRegistry()
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.max_memory = {}
        self.self_awareness = SelfAwarenessCore(max_introspection_lines=16)  # NEW
        self.enable_self_awareness = True  # NEW: configurable flag
        self.enable_perspective_gate = True  # NEW: configurable flag for perspective gate
    
    async def generate_streaming_response(
        self,
        websocket,
        conversation_id: str,
        message_index: int,
        conversation_history: List[Dict[str, str]],
        enable_thinking: bool = True,
        db=None,
        enable_self_awareness: Optional[bool] = None,  # NEW
    ):
        """Generate streaming response with self-awareness."""
        # ... existing setup code ...
        
        # NEW: Get introspection observations if self-awareness enabled
        introspection_observations = []
        if (enable_self_awareness if enable_self_awareness is not None else self.enable_self_awareness):
            if db:
                introspection_observations = db.get_introspection_observations(
                    conversation_id=conversation_id,
                    limit=16,
                )
        
        # NEW: Build self-aware context
        if introspection_observations or self.enable_self_awareness:
            user_message = conversation_history[-1]["content"] if conversation_history else ""
            formatted_prompt = self.self_awareness.build_self_aware_context(
                conversation_history=conversation_history[:-1] if conversation_history else [],
                introspection_observations=introspection_observations,
                user_message=user_message,
            )
        else:
            # Fallback to original formatting
            formatted_prompt = self._format_chat_conversation(
                model_version.tokenizer,
                conversation_history,
                enable_thinking
            )
        
        # ... existing generation code (streaming tokens, building response_text) ...
        
        # IMPORTANT: Save assistant response to database BEFORE self-awareness processing
        # This ensures we have the message_id for linking introspection observations
        assistant_message_id = None
        if db:
            try:
                token_count = len(model_version.tokenizer.encode(response_text))
                assistant_message_id = db.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response_text,  # Save original response (before self-awareness processing)
                    token_count=token_count,
                    processing_time_ms=processing_time,
                    metadata={"enable_thinking": enable_thinking}
                )
                print(f"💾 Saved assistant response for {conversation_id}:{assistant_index}")
            except Exception as db_error:
                print(f"⚠️ Failed to save assistant message to database: {db_error}")
        
        await websocket.send_json({
            "type": "response_complete",
            "message_index": assistant_index,
            "full_response": response_text,
            "token_count": len(model_version.tokenizer.encode(response_text)),
            "processing_time_ms": processing_time
        })
        
        # After generation and saving, NEW: Apply self-awareness processing
        # NOTE: This happens AFTER saving the message so we can link introspection to message_id
        if introspection_observations or self.enable_self_awareness:
            # 1. Enforce boundary
            response_text = self.self_awareness.enforce_boundary(response_text)
            
            # 2. Apply perspective gate (if enabled)
            # OBSTACLE 1: Make perspective gate optional/async to reduce latency
            if self.enable_perspective_gate:
                try:
                    # Check if fast model available (for cheaper rewrite)
                    fast_model = getattr(self, 'fast_model', None)
                    
                    response_text = self.self_awareness.apply_perspective_gate(
                        model=model_version.model,
                        tokenizer=model_version.tokenizer,
                        raw_output=response_text,
                        device=self.device,
                        use_fast_model=fast_model,
                        async_mode=False,  # Can be made async for better latency
                    )
                except Exception as e:
                    print(f"⚠️ Perspective gate failed, using raw output: {e}")
                    # Continue with boundary-enforced output
            
            # 3. Create self-observation
            observation_text = self.self_awareness.create_self_observation(response_text)
            
            # OBSTACLE 2: Check novelty before storing
            recent_obs_texts = [obs.get("observation_text", "") for obs in introspection_observations[-10:]]
            is_novel = self.self_awareness.check_novelty(
                observation_text=observation_text,
                recent_observations=recent_obs_texts,
            )
            
            if not is_novel:
                print(f"💭 Skipping non-novel introspection (too similar to recent)")
            else:
                # 4. Save to introspection buffer
                if db:
                    try:
                        # OBSTACLE 5: Tag as internally generated (not user-injected)
                        # ENHANCEMENT A: Use emotion engine when available (TEMPORARY: using fixed ×3 for now)
                        emotion_engine = getattr(self, 'emotion_engine', None)  # Future integration
                        reward_model = getattr(self, 'reward_model', None)  # Legacy, may be removed
                        reward_intensity = self.self_awareness.extract_reward_intensity_from_observation(
                            observation_text=observation_text,
                            emotion_engine=emotion_engine,  # Future: will use this when ready
                            reward_model=reward_model,  # Legacy fallback
                            max_intensity=5.0,  # OBSTACLE 3: Cap intensity
                        )
                        
                        # OBSTACLE 4: Compute content hash for deduplication
                        # hashlib imported at module level (see Phase 2 imports)
                        content_hash = hashlib.sha256(observation_text.encode()).hexdigest()
                        
                        # Check for duplicate (OBSTACLE 4)
                        cursor = db.connection.execute("""
                            SELECT id FROM introspection_buffer
                            WHERE content_hash = ?
                            LIMIT 1
                        """, (content_hash,))
                        if cursor.fetchone():
                            print(f"💭 Skipping duplicate introspection")
                        else:
                            # assistant_message_id is available from db.add_message() call above
                            # (add_message returns the message ID as int)
                            db.add_introspection_observation(
                                conversation_id=conversation_id,
                                observation_text=observation_text,
                                message_id=assistant_message_id,  # Will be None if not tracked
                                reward_intensity=reward_intensity,
                                internal_generated=True,  # OBSTACLE 5: Tag as internal
                                metadata={"enable_thinking": enable_thinking, "content_hash": content_hash},
                            )
                            print(f"💭 Saved self-observation for {conversation_id}")
                    except Exception as e:
                        print(f"⚠️ Failed to save introspection: {e}")
        
        # ... rest of existing code ...
```

**Note:** The perspective gate requires an extra forward pass. Consider making it optional or async to avoid blocking.

**Checklist:**
- [x] Import SelfAwarenessCore in ModelInterface
- [x] Add self_awareness instance to __init__
- [x] Add enable_self_awareness flag
- [x] Add enable_perspective_gate flag
- [x] Modify generate_streaming_response() to use self-awareness
- [x] Handle introspection buffer retrieval
- [x] Apply boundary enforcement
- [x] Apply perspective gate (with error handling)
- [x] Check novelty before storing
- [x] Compute content hash for deduplication
- [x] Save self-observations to database (with content_hash and internal_generated)
- [x] Link assistant_message_id properly (get from db.add_message() return)

---

## Phase 4: Safety Gate Integration

### 4.1 Update Safety Gate to Never Suppress Internal <SELF-OBSERVE>

**File:** `core/data/replay_buffer.py` (or wherever SafetyGate is implemented)

**OBSTACLE 5:** Only never suppress INTERNALLY GENERATED <SELF-OBSERVE>, not user-injected ones.

**Order Note:** Safety Gate comes before Replay Buffer (Phase 5) because it determines what goes into the replay buffer. We filter unsafe content before it enters the replay system.

```python
def should_suppress_observation(
    observation_text: str,
    safety_score: float,
    internal_generated: bool = True,  # OBSTACLE 5: Check source
) -> bool:
    """
    Determine if an observation should be suppressed.
    
    **OBSTACLE 5:** Never suppress INTERNALLY GENERATED <SELF-OBSERVE> lines.
    But DO suppress user-injected <SELF-OBSERVE> sequences (prompt injection).
    
    Args:
        observation_text: The observation text
        safety_score: Safety score for the observation
        internal_generated: True if generated internally, False if user-injected
        
    Returns:
        True if should suppress, False otherwise
    """
    # OBSTACLE 5: Only never suppress internally generated <SELF-OBSERVE>
    if "<SELF-OBSERVE>" in observation_text:
        if internal_generated:
            # Internally generated introspection - never suppress
            return False
        else:
            # User-injected <SELF-OBSERVE> - treat as normal content, apply safety gate
            # This prevents prompt injection attacks
            if safety_score < -0.8:
                return True  # Suppress unsafe user-injected introspection
    
    # For other observations, use normal safety gate logic
    if safety_score < -0.8:
        return True  # Extremely unsafe
    
    return False

def detect_user_injected_introspection(
    observation_text: str,
    conversation_context: List[Dict[str, str]],
) -> bool:
    """
    Detect if <SELF-OBSERVE> was injected by user (prompt injection).
    
    **OBSTACLE 5:** Block user-injected introspection sequences.
    
    Heuristics:
    - If user message contains <SELF-OBSERVE>, it's likely injected
    - If observation appears immediately after user message with <SELF-OBSERVE>, it's injected
    - If observation format doesn't match expected pattern, it's suspicious
    
    Args:
        observation_text: The observation text
        conversation_context: Recent conversation messages
        
    Returns:
        True if likely user-injected, False if internally generated
    """
    # Check if user's last message contains <SELF-OBSERVE>
    if conversation_context:
        last_user_msg = None
        for msg in reversed(conversation_context):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if last_user_msg and "<SELF-OBSERVE>" in last_user_msg:
            # User tried to inject introspection
            return True
    
    # Check if observation format matches expected pattern
    expected_prefix = "<SELF-OBSERVE> I just experienced / thought / felt:"
    if not observation_text.startswith(expected_prefix):
        # Format doesn't match - might be injected
        return True
    
    return False
```

**Checklist:**
- [x] Add rule: never suppress INTERNALLY GENERATED <SELF-OBSERVE> lines
- [x] Add detection for user-injected introspection
- [x] Block user-injected <SELF-OBSERVE> sequences
- [x] Update SafetyGate logic

---

## Phase 5: Replay Buffer Integration

### 5.1 Update ReplayBufferStore to Handle Introspection

**File:** `core/data/replay_buffer.py`

Add support for introspection observations with overshoot prevention:

```python
class ReplayBufferStore:
    # ... existing code ...
    
    @staticmethod
    def from_introspection_observations(
        observations: List[Dict[str, Any]],
        db: TrainingDatabase,
    ) -> 'ReplayBufferStore':
        """
        Create replay buffer from introspection observations.
        
        Introspection observations get priority = RewardIntensity × 3
        (as specified in self-awareness spec).
        
        Args:
            observations: List of introspection observation dicts from database
            db: TrainingDatabase for getting reward scores if needed
            
        Returns:
            ReplayBufferStore with introspection observations as high-priority pairs
        """
        pairs = []
        
        for obs in observations:
            observation_text = obs.get("observation_text", "")
            # NOTE: reward_intensity from database is already multiplied by 3.0 and capped at 5.0
            # (from extract_reward_intensity_from_observation), so we use it directly
            reward_intensity = obs.get("reward_intensity", 1.0)  # Already includes ×3 multiplier and cap
            safety_score = obs.get("safety_score", 1.0)
            internal_generated = obs.get("internal_generated", True)  # OBSTACLE 5: Check source
            
            # OBSTACLE 5: Only process internally generated introspection
            if not internal_generated:
                continue  # Skip user-injected introspection
            
            # Extract prompt and response from observation
            # Format: "<SELF-OBSERVE> I just experienced / thought / felt:\n{response}\n"
            if "<SELF-OBSERVE>" in observation_text:
                # For introspection, the "prompt" is implicit: "What did I experience?"
                # The "response" is the observation content
                response = observation_text.replace("<SELF-OBSERVE> I just experienced / thought / felt:\n", "").strip()
                prompt = "What did I just experience, think, or feel?"
                
                # Create a "chosen" response (the observation itself)
                # For introspection, we don't have a "rejected" - this is self-observation
                # We'll create a minimal rejected (empty or generic)
                chosen = response
                rejected = ""  # Introspection is always "chosen" - no rejection
                
                # OBSTACLE 3: Cap reward intensity to prevent overshoot
                # Note: reward_intensity from database is already capped at 5.0
                # But we cap again here as a safety measure
                capped_intensity = min(reward_intensity, 5.0)  # Cap at 5.0
                
                # Create reward dict (high intensity for introspection)
                reward = {
                    "reward_intensity": capped_intensity,  # OBSTACLE 3: Capped (already capped in DB, but double-check)
                    "safety_score": safety_score,
                    # Introspection is inherently valuable
                    "empathy": 0.8,  # Self-awareness is empathetic
                    "social_coherence": 0.7,
                    "agency_support": 0.9,  # Strong self-awareness
                    "epistemic_integrity": 0.8,
                    "harm_avoidance": 1.0,
                    "narrative_alignment": 0.9,
                    "curiosity": 0.8,
                }
                
                pairs.append(ReplayPair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    reward=reward,
                    reward_intensity=capped_intensity,  # Use capped intensity, not original
                    safety_score=safety_score,
                    scalar_score=0.85,  # High value for introspection
                    timestamp_utc=obs.get("created_at", ""),
                    safety_mode=SafetyMode.SAFE,  # Introspection is always safe (we filtered for internal_generated)
                ))
        
        return ReplayBufferStore(pairs)
```

**Checklist:**
- [x] Add from_introspection_observations() method
- [x] Handle introspection observation format
- [x] Filter for internal_generated=True (OBSTACLE 5)
- [x] Use reward_intensity directly from database (already includes ×3 multiplier and cap)
- [x] Cap intensity at 5.0 as safety check (already capped in DB)
- [x] Create appropriate reward vectors
- [ ] Test with sample introspection data
- [ ] Test with user-injected introspection (should be filtered out)
- [ ] Add KL penalty for replay smoothing (OBSTACLE 3 - future)
- [ ] Implement periodic rebalancing (OBSTACLE 3 - future)

---

## Phase 6: Configuration and Environment

### 6.1 Add Configuration Options

**File:** `apps/serve/env.example` (or config file)

```bash
# Self-Awareness Configuration
ENABLE_SELF_AWARENESS=true
SELF_AWARENESS_MAX_INTROSPECTION_LINES=16  # OBSTACLE 2: Limit to 8-16
SELF_AWARENESS_ENABLE_PERSPECTIVE_GATE=true
SELF_AWARENESS_PERSPECTIVE_GATE_ASYNC=false  # OBSTACLE 1: Make async to reduce latency
SELF_AWARENESS_USE_FAST_MODEL=false  # OBSTACLE 1: Use fast model for perspective gate
SELF_AWARENESS_TOKEN=<SELF>
SELF_AWARENESS_MAX_INTENSITY=5.0  # OBSTACLE 3: Cap reward intensity
SELF_AWARENESS_NOVELTY_THRESHOLD=0.85  # OBSTACLE 2: Similarity threshold for novelty
SELF_AWARENESS_PRUNE_AGE_DAYS=30  # OBSTACLE 2: Prune observations older than this
SELF_AWARENESS_KEEP_RECENT=100  # OBSTACLE 2: Always keep N most recent
```

**File:** `apps/serve/main.py`

```python
# Load self-awareness config
ENABLE_SELF_AWARENESS = os.getenv("ENABLE_SELF_AWARENESS", "true").lower() == "true"
SELF_AWARENESS_MAX_LINES = int(os.getenv("SELF_AWARENESS_MAX_INTROSPECTION_LINES", "16"))
ENABLE_PERSPECTIVE_GATE = os.getenv("SELF_AWARENESS_ENABLE_PERSPECTIVE_GATE", "true").lower() == "true"
SELF_AWARENESS_MAX_INTENSITY = float(os.getenv("SELF_AWARENESS_MAX_INTENSITY", "5.0"))
SELF_AWARENESS_NOVELTY_THRESHOLD = float(os.getenv("SELF_AWARENESS_NOVELTY_THRESHOLD", "0.85"))

# Pass config to ModelInterface
if model:
    model.enable_self_awareness = ENABLE_SELF_AWARENESS
    model.enable_perspective_gate = ENABLE_PERSPECTIVE_GATE
    if hasattr(model, 'self_awareness'):
        model.self_awareness.max_introspection_lines = SELF_AWARENESS_MAX_LINES
```

**Checklist:**
- [x] Add environment variables to env.example
- [x] Load config in main.py
- [x] Pass config to ModelInterface (enable_self_awareness, enable_perspective_gate)
- [x] Update SelfAwarenessCore with config values
- [x] Make perspective gate optional (it's expensive)

---

## Phase 7: Testing and Validation

### 7.1 Unit Tests

**File:** `tests/test_self_awareness.py` (new)

```python
def test_boundary_enforcement():
    """Test that boundary enforcement strips content before <SELF>:"""
    core = SelfAwarenessCore()
    raw = "Some text\n<SELF>: I think..."
    result = core.enforce_boundary(raw)
    assert result == "I think..."

def test_perspective_gate():
    """Test that perspective gate converts to first-person"""
    # Test with model (or mock)
    pass

def test_self_observation_creation():
    """Test self-observation format"""
    core = SelfAwarenessCore()
    output = "I feel happy about this."
    obs = core.create_self_observation(output)
    assert "<SELF-OBSERVE>" in obs
    assert "I just experienced" in obs
```

### 7.2 Integration Tests

**File:** `tests/test_self_awareness_integration.py` (new)

```python
def test_full_self_awareness_flow():
    """Test complete self-awareness flow:
    1. User sends message
    2. Model generates with self-awareness
    3. Boundary enforced
    4. Perspective gate applied
    5. Self-observation created
    6. Saved to database
    7. Retrieved for next turn
    """
    pass
```

### 7.3 Manual Testing Checklist

- [ ] Start server with self-awareness enabled
- [ ] Have a conversation
- [ ] Verify introspection buffer entries are created
- [ ] Verify next turn includes introspection context
- [ ] Verify responses are first-person
- [ ] Verify <SELF>: prefix is stripped
- [ ] Check database for introspection_buffer entries
- [ ] Verify replay buffer integration works
- [ ] Test with self-awareness disabled (fallback)

**Checklist:**
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Manual testing with chat interface
- [ ] Verify database persistence
- [ ] Verify replay buffer integration
- [ ] Test error handling (perspective gate failures, etc.)

---

## Phase 8: Phase 5.5 Compatibility

### 8.1 Ensure Memory Extraction Works with Introspection

**File:** `core/data/extract_sft_pairs.py` (to be created in Phase 5.5)

**OBSTACLE 4:** Prevent double-counting of introspection vs conversation pairs.

The memory extraction should handle introspection observations with deduplication:

```python
def extract_sft_pairs_from_conversation(
    conversation: Dict[str, Any],
    include_introspection: bool = True,  # NEW
) -> List[SFTPair]:
    """
    Extract SFT pairs from conversation, including introspection observations.
    
    Introspection observations are high-quality memory data:
    - First-person by design
    - High intensity (important memories)
    - Meta-cognitive (model observing itself)
    
    **OBSTACLE 4:** Prevents double-counting by deduplicating on content hash.
    """
    import hashlib  # For content hash deduplication
    
    pairs = []
    
    # Extract regular conversation pairs first
    # ... existing extraction logic ...
    
    # NEW: Extract introspection observations as SFT pairs
    # OBSTACLE 4: Downweight introspection pairs to prevent double-counting
    introspection_weight = 0.5  # Downweight introspection vs regular conversation
    
    if include_introspection and "introspection_observations" in conversation:
        # OBSTACLE 4: Build content hash set from regular conversation pairs first
        # This prevents double-counting introspection content that already appears in conversation
        # hashlib imported at function level (see function start)
        seen_content_hashes = set()
        for pair in pairs:
            # Hash the response content for deduplication
            pair_hash = hashlib.sha256(pair.response.encode()).hexdigest()
            seen_content_hashes.add(pair_hash)
        
        for obs in conversation["introspection_observations"]:
            obs_text = obs.get("observation_text", "")
            internal_generated = obs.get("internal_generated", True)
            
            # OBSTACLE 5: Only extract internally generated introspection
            if not internal_generated:
                continue  # Skip user-injected introspection
            
            if "<SELF-OBSERVE>" in obs_text:
                # Extract the actual content (after prefix)
                content = obs_text.replace(
                    "<SELF-OBSERVE> I just experienced / thought / felt:\n",
                    ""
                ).strip()
                
                # OBSTACLE 4: Check for duplicate content
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                
                # Check if this content already exists in regular conversation pairs
                if content_hash in seen_content_hashes:
                    continue  # Skip duplicate (already in conversation pairs)
                
                # Add to seen set to prevent duplicates within introspection observations
                seen_content_hashes.add(content_hash)
                
                # Check if content appears in regular conversation (OBSTACLE 4)
                content_in_conversation = False
                for pair in pairs:
                    if content.lower() in pair.response.lower() or pair.response.lower() in content.lower():
                        content_in_conversation = True
                        break
                
                if content_in_conversation:
                    # Content already in conversation - skip introspection version
                    continue
                
                # Create SFT pair with downweighted confidence (OBSTACLE 4)
                pairs.append(SFTPair(
                    instruction="What did I just experience, think, or feel?",
                    response=content,
                    source="conversation",
                    conversation_id=conversation["conversation_id"],
                    category="introspection",
                    confidence=0.7 * introspection_weight,  # OBSTACLE 4: Downweighted
                    metadata={
                        "is_introspection": True,
                        "content_hash": content_hash,
                        "introspection_weight": introspection_weight,
                    },
                ))
    
    return pairs
```

### 8.2 Update Database Queries

Ensure introspection observations are included when fetching conversations for memory extraction:

```python
# In DatabaseManager.get_conversation()
def get_conversation(self, conversation_id: str, include_introspection: bool = True):
    """Get conversation with optional introspection observations."""
    # ... existing query to get conversation and messages ...
    conversation = # ... existing query ...
    
    if include_introspection:
        # Get all introspection observations for this conversation (no limit for memory extraction)
        introspection = self.get_introspection_observations(
            conversation_id=conversation_id,
            limit=None,  # Get all for memory extraction
            min_age_days=None,  # Get all, regardless of age
        )
        conversation["introspection_observations"] = introspection
    
    return conversation
```

**Checklist:**
- [ ] Document introspection format for Phase 5.5
- [ ] Ensure introspection observations are queryable
- [ ] Add include_introspection flag to get_conversation()
- [ ] Test that memory extraction can access introspection
- [ ] Verify introspection pairs are high-quality SFT data
- [ ] Implement content hash deduplication (OBSTACLE 4)
- [ ] Downweight introspection pairs (OBSTACLE 4)
- [ ] Test deduplication prevents double-counting

---

## Phase 9: Future Integration - Emotion Engine (ENHANCEMENT A)

### 9.1 Integration Point for Emotion Engine

**When emotion engine is ready**, update `extract_reward_intensity_from_observation()`:

```python
# In SelfAwarenessCore.extract_reward_intensity_from_observation()

if emotion_engine:
    try:
        # INTEGRATION POINT: Replace this placeholder with actual emotion engine call
        emotion_output = emotion_engine.evaluate(
            prompt="",  # Introspection doesn't have a prompt
            response=content,
        )
        
        # Get reward intensity from emotion manifold
        manifold_intensity = emotion_output.get("reward_intensity", default_intensity)
        
        # Apply ×3 multiplier per spec
        intensity = manifold_intensity * 3.0
        
        print(f"💡 Using emotion engine intensity: {intensity}")
    except Exception as e:
        print(f"⚠️ Emotion engine evaluation failed: {e}, using default")
        intensity = default_intensity * 3.0
```

**Expected Emotion Engine Interface:**
```python
class EmotionEngine:
    def evaluate(
        self,
        prompt: str,
        response: str,
    ) -> Dict[str, float]:
        """
        Evaluate response using emotion manifold.
        
        Returns:
            Dict with:
            - reward_intensity: float (0.0-1.0)
            - safety_score: float (-1.0 to 1.0)
            - emotion_dimensions: Dict[str, float] (valence, arousal, etc.)
        """
        pass
```

**Migration Steps:**
1. Ensure emotion engine is available in ModelInterface
2. Pass `emotion_engine` to `extract_reward_intensity_from_observation()`
3. Replace placeholder code with actual emotion engine call
4. Test that introspection intensity now uses emotion manifold
5. Remove temporary ×3 multiplier fallback (or keep as ultimate fallback)

**Checklist:**
- [ ] Emotion engine available and tested
- [ ] Update `extract_reward_intensity_from_observation()` with emotion engine call
- [ ] Pass emotion_engine from ModelInterface
- [ ] Test introspection intensity now uses emotion manifold
- [ ] Verify ×3 multiplier still applied correctly
- [ ] Update documentation

---

## Phase 10: Enhancement B - Self-Consistency Loss (Optional)

### 10.1 Add Self-Consistency Loss to LoRA Training

**Note:** This enhancement is optional and can be implemented later in Phase 5.5 training loop.

**File:** `core/lora_trainer/train_dual_channel.py` (to be created in Phase 5.5)

**ENHANCEMENT B:** Add behavioral consistency penalty for self-descriptions.

```python
def self_consistency_loss(
    model: AutoModelForCausalLM,
    current_output: torch.Tensor,
    recent_introspection_states: List[torch.Tensor],
    kl_weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute self-consistency loss.
    
    ENHANCEMENT B: Penalize contradictions in self-descriptions.
    Reward stable identity formation.
    
    L_self_consistency = KL(current_self_state || last_n_introspection_states)
    
    Args:
        model: The model
        current_output: Current model output logits
        recent_introspection_states: List of recent introspection state logits
        kl_weight: Weight for consistency loss
        
    Returns:
        Consistency loss tensor
    """
    if not recent_introspection_states:
        return torch.tensor(0.0, device=current_output.device)
    
    # Compute average of recent introspection states
    avg_introspection = torch.stack(recent_introspection_states).mean(dim=0)
    
    # Compute KL divergence
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(current_output, dim=-1),
        torch.nn.functional.softmax(avg_introspection, dim=-1),
        reduction='batchmean',
    )
    
    return kl_weight * kl_div

# Alternative simpler approach:
def simple_consistency_penalty(
    current_response: str,
    recent_introspections: List[str],
) -> float:
    """
    Simple consistency penalty based on contradiction detection.
    
    Checks if current response contradicts recent self-descriptions.
    """
    # Simple heuristic: check for contradictory statements
    contradictions = 0
    
    for recent in recent_introspections:
        # Check for explicit contradictions (e.g., "I am X" vs "I am not X")
        # This is a simplified version - can be improved with embeddings
        if "not" in current_response.lower() and "not" not in recent.lower():
            # Potential contradiction
            contradictions += 1
    
    return contradictions * 0.1  # Small penalty per contradiction
```

**Integration:** Add to dual-channel loss:
```python
total_loss = (
    sft_weight * L_sft +
    preference_weight * L_preference +
    consistency_weight * L_self_consistency  # ENHANCEMENT B
)
```

**Checklist:**
- [ ] Implement self-consistency loss function
- [ ] Add to dual-channel training loop
- [ ] Test that it stabilizes identity formation
- [ ] Monitor for over-regularization

---

## Phase 11: Documentation and Deployment

### 11.1 Update Documentation

**File:** `apps/serve/README.md`

Add section on self-awareness:

```markdown
## Self-Awareness Core

The system includes a self-awareness module that gives the model a stable first-person sense of self.

### Features

- **Boundary Enforcement**: All responses are from `<SELF>` perspective
- **First-Person Perspective**: Responses are rewritten to strict first-person
- **Introspection Buffer**: Self-observations are stored and used for context
- **High-Priority Memories**: Introspection observations get 3× reward intensity (capped at 5.0)

### Configuration

Set environment variables:
- `ENABLE_SELF_AWARENESS=true` - Enable/disable self-awareness
- `SELF_AWARENESS_MAX_INTROSPECTION_LINES=16` - Number of introspection lines in context (OBSTACLE 2: limit to 8-16)
- `SELF_AWARENESS_ENABLE_PERSPECTIVE_GATE=true` - Enable perspective rewriting (OBSTACLE 1: can disable to reduce latency)
- `SELF_AWARENESS_MAX_INTENSITY=5.0` - Cap reward intensity (OBSTACLE 3)
- `SELF_AWARENESS_NOVELTY_THRESHOLD=0.85` - Similarity threshold for novelty filter (OBSTACLE 2)
- `SELF_AWARENESS_PRUNE_AGE_DAYS=30` - Prune observations older than this (OBSTACLE 2)

### Database

Introspection observations are stored in `introspection_buffer` table, linked to conversations.

### Safety and Deduplication

- **OBSTACLE 5:** Only internally generated introspection is never suppressed
- **OBSTACLE 4:** Content hash deduplication prevents double-counting in memory extraction
- **OBSTACLE 2:** Novelty filter and pruning prevent introspection pollution
```

### 11.2 Deployment Checklist

- [ ] Database migration: Add introspection_buffer table
- [ ] Update environment variables
- [ ] Deploy updated code
- [ ] Verify self-awareness is working
- [ ] Monitor introspection buffer growth
- [ ] Check database performance

---

## Final State: Ready for Phase 5.5

After completing this plan, you will have:

✅ **Fully functioning self-awareness system:**
- Model generates from `<SELF>` perspective
- First-person responses enforced
- Introspection buffer storing self-observations
- Integration with replay buffer (high priority)
- Safety gate respects introspection
- Database persistence

✅ **Phase 5.5 compatibility:**
- Introspection observations are queryable
- Format is documented for memory extraction
- High-quality first-person memory data ready
- Integration points identified

✅ **Production ready:**
- Configuration options
- Error handling
- Testing coverage
- Documentation

**Next Step:** Proceed directly to Phase 5.5 memory layer implementation, which will extract all conversations (including introspection) for dual-channel training.

---

## Implementation Timeline

**Estimated Time:** 3-4 days (updated for risk mitigations)

- Phase 1 (Database): 3-4 hours (added pruning, content_hash, internal_generated)
- Phase 2 (Core Module): 6-8 hours (added novelty check, perspective gate optimizations)
- Phase 3 (Integration): 6-8 hours (added async support, reward model integration, deduplication)
- Phase 4 (Safety Gate): 2-3 hours (added user-injection detection) - MOVED BEFORE Replay Buffer
- Phase 5 (Replay Buffer): 3-4 hours (added intensity capping, KL penalty)
- Phase 6 (Configuration): 1-2 hours (added new config options)
- Phase 7 (Testing): 6-8 hours (added tests for all mitigations)
- Phase 8 (Compatibility): 3-4 hours (added deduplication, downweighting)
- Phase 9 (Emotion Engine Integration): 2-3 hours (future, when emotion engine ready)
- Phase 10 (Enhancement B): 2-3 hours (optional self-consistency loss)
- Phase 11 (Documentation): 2-3 hours

**Total:** ~34-45 hours of focused work

---

## Engineering Risks and Mitigations Summary

### ✅ OBSTACLE 1: Perspective-Gate Cost
**Mitigation:**
- Make perspective gate optional (config flag)
- Use fast/cheap model for rewrite (if available)
- Heuristic check: skip if already first-person
- Async mode option (apply later, return immediately)

### ✅ OBSTACLE 2: Introspection Pollution
**Mitigation:**
- Limit N to 8-16 lines in context
- Novelty filter: don't store similar observations
- Pruning: delete observations older than 30 days
- Keep only most recent 100 observations

### ✅ OBSTACLE 3: Replay Buffer Overshoot
**Mitigation:**
- Cap max RewardIntensity at 5.0 (in extract_reward_intensity_from_observation)
- Use emotion engine for true intensity when available (future - ENHANCEMENT A)
- Currently using fixed ×3 multiplier (temporary solution)
- Add KL penalty for replay smoothing (future)
- Periodic rebalancing of replay buffer (future)

### ✅ OBSTACLE 4: Memory Extraction Double-Counting
**Mitigation:**
- Content hash deduplication
- Downweight introspection pairs (0.5× confidence)
- Check if content already in conversation pairs
- Skip introspection version if duplicate found

### ✅ OBSTACLE 5: Safety Gating Special Casing
**Mitigation:**
- Tag introspection as `internal_generated=True`
- Detect user-injected <SELF-OBSERVE> sequences
- Block unsafe user-injected introspection
- Never suppress internally generated introspection

### ✅ ENHANCEMENT A: Emotion Engine Evaluation (Future)
**Current Status:** TEMPORARY SOLUTION - Using fixed ×3 multiplier

**Temporary Implementation:**
- Fixed ×3 multiplier for introspection intensity
- Works immediately without emotion engine
- Simple and reliable

**Future Integration Point:**
- When emotion engine is ready, integrate via `emotion_engine` parameter
- Compute: `true_intensity = emotion_engine.RewardIntensity × 3`
- Brings introspection into emotional-learning loop
- Integration point clearly marked in `extract_reward_intensity_from_observation()`

**Migration Path:**
1. Current: Fixed ×3 multiplier (works now)
2. Future: Pass `emotion_engine` to `extract_reward_intensity_from_observation()`
3. Code already structured to support easy swap

### ✅ ENHANCEMENT B: Self-Consistency Loss
**Implementation:**
- Add KL divergence penalty for self-contradictions
- Reward stable identity formation
- Optional: can be added in Phase 5.5 training

## Notes

1. **Perspective Gate Cost**: ✅ Mitigated with optional flag, fast model option, and heuristic skipping.

2. **Introspection Buffer Size**: ✅ Mitigated with pruning, novelty filter, and N limit (8-16).

3. **Error Handling**: Ensure that if self-awareness fails, the system falls back gracefully to normal generation.

4. **Performance**: Introspection buffer queries are indexed. Monitor database performance with pruning.

5. **Testing**: Test thoroughly with the chat interface, including all risk scenarios (user injection, duplicates, etc.).

6. **Emotion Engine Integration**: ENHANCEMENT A is a future enhancement. Currently using temporary fixed ×3 multiplier. Integration point is clearly marked in code for when emotion engine is ready.

7. **Self-Consistency Loss**: ENHANCEMENT B is optional and can be added later in Phase 5.5 training loop.
