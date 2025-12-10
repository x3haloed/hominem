# Phase 5.5: Dual-Channel LoRA Training - Detailed Implementation Specification

This document provides concrete implementation details for Phase 5.5, filling in the gaps identified in the original plan.

## Table of Contents

1. [SFT Data Schema and Format](#1-sft-data-schema-and-format)
2. [Memory Extraction Pipeline](#2-memory-extraction-pipeline)
3. [Dual-Channel Training Loop](#3-dual-channel-training-loop)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Integration with Online Learning](#5-integration-with-online-learning)
6. [Configuration Specification](#6-configuration-specification)

---

## 1. SFT Data Schema and Database Storage

### 1.1 Database Table Schema

**Table:** `sft_pairs` (to be added to `apps/serve/schema.sql`)

```sql
-- SFT (Supervised Fine-Tuning) pairs for memory consolidation channel
CREATE TABLE IF NOT EXISTS sft_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Core training data
    instruction TEXT NOT NULL,  -- The prompt/question/correction context
    response TEXT NOT NULL,      -- The desired response
    
    -- Source tracking
    source TEXT NOT NULL CHECK (source IN (
        'conversation', 'manual', 'correction', 'knowledge_update', 'synthetic'
    )),
    conversation_id TEXT,        -- UUID if extracted from conversation
    message_id INTEGER,          -- References messages.id if from conversation
    message_index INTEGER,       -- Message index in conversation if applicable
    
    -- Extraction metadata
    extraction_method TEXT,      -- 'frontier_model', 'user_correction', 'manual', etc.
    confidence REAL CHECK (confidence BETWEEN 0.0 AND 1.0),
    category TEXT,               -- 'correction', 'knowledge_update', 'style', 'clarification'
    
    -- Usage tracking
    is_used BOOLEAN DEFAULT FALSE,  -- Mark when used in training
    used_in_training_batch TEXT,   -- Which training batch used this pair
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    extracted_at DATETIME,       -- When extraction occurred (may differ from created_at)
    
    -- Additional metadata
    metadata JSON                -- Model version, extraction parameters, etc.
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sft_pairs_source ON sft_pairs(source);
CREATE INDEX IF NOT EXISTS idx_sft_pairs_conversation ON sft_pairs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_sft_pairs_used ON sft_pairs(is_used);
CREATE INDEX IF NOT EXISTS idx_sft_pairs_created ON sft_pairs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sft_pairs_confidence ON sft_pairs(confidence DESC);
```

### 1.2 Data Format (In-Memory Representation)

When working with SFT pairs in Python code, use this structure:

```python
@dataclass
class SFTPair:
    """In-memory representation of an SFT training pair."""
    instruction: str
    response: str
    source: str  # One of: "conversation", "manual", "correction", "knowledge_update", "synthetic"
    conversation_id: Optional[str] = None
    message_id: Optional[int] = None
    message_index: Optional[int] = None
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### 1.3 Data Categories

**Primary Source: Conversations**
- **All Conversational Pairs**: Every (user_message, assistant_response) pair from conversations
  - This is the core memory: "when user said X, assistant responded Y"
  - No filtering needed - every pair is factual knowledge
  - Includes corrections, new information, normal interactions, style preferences, etc.
  - The `category` field can optionally tag pairs (correction, knowledge_update, style, etc.) but doesn't affect extraction

**Other Sources:**
- **Manual SFT Pairs**: Explicitly created instruction-response pairs (manually curated)

**Note:** The `category` field is informational only. It doesn't determine what gets extracted - all pairs are extracted regardless of category.

**Validation Rules:**
- `instruction` must be non-empty and at least 10 characters
- `response` must be non-empty and at least 10 characters
- `source` must be one of the valid values
- `confidence` must be between 0.0 and 1.0 (if provided)
- If `source == "conversation"`, `conversation_id` should be present

### 1.4 Database Schema Update

**Action Required:** Add the `sft_pairs` table to `apps/serve/schema.sql`

The table definition is provided in section 1.1 above. After adding it:

1. The schema will be automatically created when `TrainingDatabase` is initialized (via `_ensure_schema()`)
2. For existing databases, you may need to run a migration or manually execute the CREATE TABLE statement
3. The indexes will improve query performance for filtering and retrieval

**Migration Note:** If you have an existing database, you can add the table without affecting existing data:

```sql
-- Run this on existing databases to add the SFT pairs table
-- (The CREATE TABLE IF NOT EXISTS will handle this automatically, but you can also run manually)
```

---

## 2. Memory Extraction Pipeline

### 2.1 Core Principle

**Every conversational pair is factual knowledge.**

The memory/SFT channel captures the factual pattern: "when the user said X, the assistant responded with Y." This is knowledge that should be learned and remembered, regardless of whether it's a correction, new information, or just a normal interaction.

### 2.2 Extraction Function

**File:** `core/data/extract_sft_pairs.py`

```python
def extract_sft_pairs_from_conversation(
    conversation: Dict[str, Any],
    include_context: bool = False,
) -> List[SFTPair]:
    """
    Extract ALL (user_message, assistant_response) pairs from a conversation.
    
    Every pair is factual knowledge: "the model said Y when the user said X."
    
    Args:
        conversation: Conversation dict with 'messages' list (from DatabaseManager.get_conversation())
        include_context: If True, include conversation history in instruction
    
    Returns:
        List of SFTPair objects, one per assistant response.
    """
    
def extract_and_store_sft_pairs(
    db: TrainingDatabase,
    conversation_id: Optional[str] = None,
    all_conversations: bool = False,
    since: Optional[str] = None,
    include_context: bool = False,
    skip_existing: bool = True,
) -> int:
    """
    Extract ALL conversational pairs from conversations and store in database.
    
    This extracts every (user_message, assistant_response) pair as an SFT training pair.
    No filtering or frontier model analysis needed - every pair is knowledge.
    
    Args:
        db: TrainingDatabase instance
        conversation_id: Extract from specific conversation (UUID)
        all_conversations: Extract from all conversations
        since: Only process conversations after this date (ISO format)
        include_context: Include conversation history in instruction
        skip_existing: Skip pairs that already exist in database (by message_id)
    
    Returns:
        Number of pairs extracted and stored.
    """
```

### 2.3 Extraction Logic

**Simple Algorithm:**

1. **Load conversations** from database:
   - If `conversation_id` specified: load that conversation via `DatabaseManager.get_conversation()`
   - If `all_conversations`: iterate through all conversations
   - If `since` specified: filter conversations by `created_at >= since`

2. **For each conversation**:
   - Get all messages ordered by `message_index`
   - **Extract all (user, assistant) pairs**:
     - Iterate through messages in order
     - When you find a `role='assistant'` message:
       - Find the immediately preceding `role='user'` message
       - Optionally include earlier conversation context if `include_context=True`
       - Create SFT pair: `instruction = user_message` (with optional context), `response = assistant_message`
       - Set `source = "conversation"`
       - Set `conversation_id` and `message_id` for tracking

3. **Deduplication** (if `skip_existing=True`):
   - Check if SFT pair with same `message_id` already exists in database
   - Skip if exists (prevents duplicate extraction on re-runs)

4. **Store in database**:
   - Validate pair (length, format, etc.)
   - Insert via `db.insert_sft_pair()`

5. **Return count** of pairs extracted and stored

**Example:**

```python
# Conversation messages:
# [0] user: "What is the capital of France?"
# [1] assistant: "The capital of France is Paris."
# [2] user: "Tell me more about Paris."
# [3] assistant: "Paris is the largest city in France..."

# Extracted pairs:
# Pair 1: instruction="What is the capital of France?", response="The capital of France is Paris."
# Pair 2: instruction="Tell me more about Paris.", response="Paris is the largest city in France..."
```

### 2.4 Context Inclusion (CRITICAL for Preventing Repetition)

**Always include context by default.** This is essential to prevent verbatim repetition.

When `include_context=True`, the instruction includes conversation history:

```python
# Without context (RISKY - can cause verbatim repetition):
instruction = "Tell me more about Paris."
response = "Paris is the largest city in France..."

# With context (RECOMMENDED - teaches patterns, not exact pairs):
instruction = """Previous conversation:
User: What is the capital of France?
Assistant: The capital of France is Paris.

User: Tell me more about Paris."""
response = "Paris is the largest city in France..."
```

**Why Context Matters:**
- **Without context**: Model learns isolated pairs → risk of verbatim repetition
- **With context**: Model learns patterns in context → generalization, not memorization
- Context helps model understand *when* to use certain responses, not just *what* to say

**Recommendation:** **Always use `include_context=True`** to prevent the model from memorizing exact pairs and instead learn contextual patterns.

### 2.5 Recency Weighting (Important for Preventing Overfitting)

**Weight SFT pairs by recency** to prevent overfitting to old conversations:

```python
def get_sft_pair_weight(pair: SFTPair) -> float:
    """Calculate weight based on recency."""
    age_days = (datetime.now() - pair.created_at).days
    # More recent = higher weight
    # Decay over 30 days: weight = 1.0 / (1.0 + age_days / 30.0)
    return 1.0 / (1.0 + age_days / 30.0)
```

**Why:** Recent conversations are more relevant and reflect current patterns. Old conversations provide general knowledge but shouldn't dominate training.

**Implementation:** Apply weights during training loss calculation:
```python
weighted_loss = weight * sft_loss(pair)
```

### 2.6 Diversity Sampling (Recommended)

**Sample diverse pairs** to prevent overfitting to common patterns:

```python
def sample_diverse_sft_pairs(
    pairs: List[SFTPair],
    target_size: int,
) -> List[SFTPair]:
    """
    Sample pairs ensuring diversity across conversations.
    Prevents overfitting to frequently occurring patterns.
    """
    # Group by conversation_id
    # Sample evenly from each conversation
    # Ensures model sees diverse patterns, not just common ones
```

**Why:** Prevents model from overfitting to frequently occurring patterns in training data.

### 2.7 Quality Filtering

While every pair is knowledge, apply basic quality filters:

- **Minimum length**: Skip very short pairs (e.g., "ok" -> "ok")
- **Maximum length**: Truncate or skip extremely long pairs

This filtering is applied AFTER extraction, not during the decision of what to extract.

### 2.6 Validation

**Validation Function:**

```python
def validate_sft_pair(pair: SFTPair) -> Tuple[bool, Optional[str]]:
    """
    Validate an SFT pair before database insertion.
    Returns (is_valid, error_message).
    """
    # Check required fields
    if not pair.instruction or len(pair.instruction.strip()) < 10:
        return False, "instruction must be at least 10 characters"
    
    if not pair.response or len(pair.response.strip()) < 10:
        return False, "response must be at least 10 characters"
    
    # Check field types and values
    if pair.instruction == pair.response:
        return False, "instruction and response cannot be identical"
    
    if len(pair.instruction) > 2048:
        return False, "instruction exceeds maximum length (2048)"
    
    if len(pair.response) > 4096:
        return False, "response exceeds maximum length (4096)"
    
    if pair.confidence is not None and not (0.0 <= pair.confidence <= 1.0):
        return False, "confidence must be between 0.0 and 1.0"
    
    valid_sources = {"conversation", "manual", "correction", "knowledge_update", "synthetic"}
    if pair.source not in valid_sources:
        return False, f"source must be one of {valid_sources}"
    
    if pair.source == "conversation" and not pair.conversation_id:
        return False, "conversation_id required when source is 'conversation'"
    
    return True, None
```

**Quality Filters (applied before database insertion):**

- Minimum instruction length: 10 characters (skip very short user messages)
- Minimum response length: 10 characters (skip very short assistant responses)
- Maximum instruction length: 2048 characters (truncate if needed)
- Maximum response length: 4096 characters (truncate if needed)
- Reject if instruction == response (no learning signal, though rare)
- Database-level constraints enforce source enum

**Note:** These are basic quality filters. The principle is that ALL conversational pairs are knowledge, so we only filter for obvious quality issues (too short, identical, etc.), not for "relevance" or "importance."

### 2.7 Database Integration

**Database Methods** (to be added to `core/data/db.py`):

```python
class TrainingDatabase:
    # ... existing methods ...
    
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
    ) -> int:
        """Insert an SFT pair into the database."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.connection.execute("""
            INSERT INTO sft_pairs
            (instruction, response, source, conversation_id, message_id, message_index,
             extraction_method, confidence, category, metadata, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            instruction, response, source, conversation_id, message_id, message_index,
            extraction_method, confidence, category, metadata_json
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_sft_pairs(
        self,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        conversation_id: Optional[str] = None,
        is_used: Optional[bool] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get SFT pairs with optional filtering."""
        query = "SELECT * FROM sft_pairs WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)
        
        if conversation_id:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        
        if is_used is not None:
            query += " AND is_used = ?"
            params.append(is_used)
        
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        cursor = self.connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def mark_sft_pairs_used(
        self,
        pair_ids: List[int],
        training_batch: str,
    ) -> None:
        """Mark SFT pairs as used in a training batch."""
        placeholders = ','.join(['?'] * len(pair_ids))
        self.connection.execute(f"""
            UPDATE sft_pairs
            SET is_used = TRUE, used_in_training_batch = ?
            WHERE id IN ({placeholders})
        """, [training_batch] + pair_ids)
        self.connection.commit()
```

### 2.8 Batch Extraction

**CLI Command:**

```bash
python -m core.data.extract_sft_pairs \
    --conversation-id <uuid> \
    --all-conversations \
    --since 2024-01-01T00:00:00Z \
    --include-context \  # RECOMMENDED: Always include context to prevent verbatim repetition
    --no-skip-existing \
    --db-path <path>
```

**Options:**
- `--conversation-id`: Extract from specific conversation (UUID)
- `--all-conversations`: Extract from all conversations in database
- `--since <date>`: Only process conversations after date (ISO format, e.g., "2024-01-01T00:00:00Z")
- `--include-context`: Include conversation history in instruction (default: False)
- `--no-skip-existing`: Re-extract pairs even if they already exist (default: skip existing)
- `--db-path`: Path to SQLite database (uses default if not specified)
- `--export-jsonl <path>`: Optional: Export extracted pairs to JSONL file for backup

**Note:** 
- The extraction script writes directly to the database
- Every (user_message, assistant_response) pair is extracted - no filtering
- JSONL export is optional and primarily for backup/debugging purposes
- No frontier model needed - extraction is deterministic from conversation data

---

## 3. Dual-Channel Training Loop

### 3.1 Training Configuration

**File:** `config/training/lora_dual.yaml`

```yaml
model:
  base_model_id: "Qwen/Qwen3-1.7B"
  max_length: 1024
  padding_side: "left"
  load_in_4bit: false
  torch_dtype: "bfloat16"
  gradient_checkpointing: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: "none"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  # Data sources (database is primary, JSONL is optional fallback)
  use_database: true              # Use database instead of JSONL files
  db_path: null                   # null = use default from env or TrainingDatabase default
  sft_data_path: null             # Only used if use_database=false
  preferences_path: null          # Only used if use_database=false
  
  # Database query filters for SFT pairs
  sft_filters:
    source: null                   # null = all sources, or specific: "conversation", "manual", etc.
    is_used: false                 # Only get unused pairs (set to null for all)
    since: null                    # ISO date string, e.g., "2024-01-01T00:00:00Z"
    min_instruction_length: 10     # Minimum instruction length (characters)
    min_response_length: 10        # Minimum response length (characters)
  
  # SFT training enhancements (prevent verbatim repetition)
  sft_enhancements:
    use_recency_weighting: true    # Weight pairs by recency (newer = more important)
    recency_decay_days: 30         # Weight decays over 30 days
    use_diversity_sampling: true   # Sample diverse pairs across conversations
    diversity_target_size: null     # null = use all, or limit for diversity
  
  # Database query filters for preference pairs
  preference_filters:
    category: null                 # Filter by category if desired
  
  # Channel weights (must sum to 1.0 for normalization)
  # IMPORTANT: Keep preference_weight >= 0.5 to prevent verbatim repetition
  sft_weight: 0.4          # Memory channel weight (lower to prevent memorization)
  preference_weight: 0.6    # Reward channel weight (higher to shape behavior)
  
  # Batch mixing strategy: "interleaved" | "alternating" | "mixed"
  batch_mixing: "interleaved"
  
  # Batch sizes (can be different per channel)
  sft_batch_size: 4
  preference_batch_size: 2
  
  # Learning rates (can be different per channel)
  sft_learning_rate: 2.0e-5
  preference_learning_rate: 5.0e-6
  
  # Loss normalization
  normalize_losses: true   # Scale losses to similar magnitude
  
  # Training parameters
  num_epochs: 3
  warmup_steps: 100
  max_grad_norm: 0.3
  logging_steps: 10
  save_every_steps: 0
  beta: 0.05  # DPO beta parameter
  
  # Dynamic weighting (optional)
  dynamic_weighting: false
  dynamic_weight_window: 100  # Steps to average over
  
  # Output
  output_dir: "artifacts/lora/qwen3.1-7b-dual"
  seed: 42
```

### 3.2 Trainer Implementation

**File:** `core/lora_trainer/train_dual_channel.py`

**Key Components:**

#### 3.2.1 Data Loading

```python
def load_sft_pairs_from_database(
    db: TrainingDatabase,
    filters: Dict[str, Any],
    limit: Optional[int] = None,
) -> List[SFTPair]:
    """Load SFT pairs from database with filtering."""
    rows = db.get_sft_pairs(
        source=filters.get("source"),
        min_confidence=filters.get("min_confidence"),
        is_used=filters.get("is_used"),
        since=filters.get("since"),
        limit=limit,
    )
    
    pairs = []
    for row in rows:
        # Parse metadata JSON if present
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        
        pairs.append(SFTPair(
            instruction=row["instruction"],
            response=row["response"],
            source=row["source"],
            conversation_id=row["conversation_id"],
            message_id=row["message_id"],
            message_index=row["message_index"],
            extraction_method=row["extraction_method"],
            confidence=row["confidence"],
            category=row["category"],
            metadata=metadata,
        ))
    
    return pairs

def load_sft_pairs_from_jsonl(path: str) -> List[SFTPair]:
    """Load SFT pairs from JSONL file (fallback/legacy support)."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pairs.append(SFTPair(**obj))
    return pairs

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning pairs."""
    
    def __init__(self, pairs: List[SFTPair], tokenizer, max_length: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        # Format as chat template
        messages = [
            {"role": "user", "content": pair.instruction},
            {"role": "assistant", "content": pair.response},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Tokenize and prepare labels
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}

class DualChannelDataLoader:
    """Manages loading and mixing of SFT and preference data."""
    
    def __init__(
        self,
        sft_dataset: SFTDataset,
        preference_dataset: PreferenceDataset,
        sft_batch_size: int,
        preference_batch_size: int,
        mixing_strategy: str,  # "interleaved", "alternating", "mixed"
    ):
        self.sft_loader = DataLoader(
            sft_dataset,
            batch_size=sft_batch_size,
            shuffle=True,
        )
        self.preference_loader = DataLoader(
            preference_dataset,
            batch_size=preference_batch_size,
            shuffle=True,
            collate_fn=collate_fn,  # From train_dpo.py
        )
        self.mixing_strategy = mixing_strategy
        self.sft_iter = iter(self.sft_loader)
        self.pref_iter = iter(self.preference_loader)
```

#### 3.2.2 Loss Functions

```python
def sft_loss(
    model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Standard cross-entropy loss for SFT.
    L_sft = -log P(response | instruction)
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss

def dual_channel_loss(
    model: AutoModelForCausalLM,
    sft_batch: Optional[Dict[str, torch.Tensor]],
    preference_batch: Optional[Dict[str, torch.Tensor]],
    sft_weight: float,
    preference_weight: float,
    beta: float,
    normalize: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss from both channels.
    
    Returns:
        total_loss: Combined loss tensor
        metrics: Dict with per-channel losses and contributions
    """
    losses = {}
    contributions = {}
    
    # Compute SFT loss
    if sft_batch is not None:
        sft_loss_val = sft_loss(model, sft_batch)
        losses["sft"] = sft_loss_val.item()
    else:
        sft_loss_val = torch.tensor(0.0, device=model.device)
        losses["sft"] = 0.0
    
    # Compute preference loss
    if preference_batch is not None:
        pref_loss_val = weighted_dpo_loss(model, preference_batch, beta=beta)
        losses["preference"] = pref_loss_val.item()
    else:
        pref_loss_val = torch.tensor(0.0, device=model.device)
        losses["preference"] = 0.0
    
    # Normalize losses if requested
    if normalize:
        # Scale to similar magnitude (use running averages)
        sft_scale = 1.0  # Could be adaptive
        pref_scale = 1.0  # Could be adaptive
        sft_loss_val = sft_loss_val * sft_scale
        pref_loss_val = pref_loss_val * pref_scale
    
    # Weighted combination
    total_loss = sft_weight * sft_loss_val + preference_weight * pref_loss_val
    
    # Track contributions
    contributions["sft_contribution"] = (sft_weight * sft_loss_val.item()) / total_loss.item()
    contributions["preference_contribution"] = (preference_weight * pref_loss_val.item()) / total_loss.item()
    
    return total_loss, {
        "loss": total_loss.item(),
        "sft_loss": losses["sft"],
        "preference_loss": losses["preference"],
        **contributions,
    }
```

#### 3.2.3 Training Loop

**Batch Mixing Strategies:**

1. **Interleaved**: Alternate batches from each channel
   ```python
   for step in range(num_steps):
       if step % 2 == 0:
           sft_batch = next(sft_loader)
           pref_batch = None
       else:
           sft_batch = None
           pref_batch = next(pref_loader)
       loss = dual_channel_loss(...)
   ```

2. **Alternating**: Process all SFT batches, then all preference batches
   ```python
   for epoch in range(num_epochs):
       # Process all SFT data
       for sft_batch in sft_loader:
           loss = dual_channel_loss(sft_batch, None, ...)
       # Process all preference data
       for pref_batch in pref_loader:
           loss = dual_channel_loss(None, pref_batch, ...)
   ```

3. **Mixed**: Combine batches from both channels in each step
   ```python
   for step in range(num_steps):
       sft_batch = next(sft_loader, None)
       pref_batch = next(pref_loader, None)
       loss = dual_channel_loss(sft_batch, pref_batch, ...)
   ```

**Recommended:** Use "interleaved" for balanced learning, with dynamic adjustment based on data availability.

#### 3.2.4 Dynamic Weight Adjustment

```python
class DynamicChannelWeighter:
    """Adjusts channel weights based on training dynamics."""
    
    def __init__(self, initial_sft_weight: float, window_size: int = 100):
        self.sft_weight = initial_sft_weight
        self.pref_weight = 1.0 - initial_sft_weight
        self.window_size = window_size
        self.sft_losses = []
        self.pref_losses = []
    
    def update(self, sft_loss: float, pref_loss: float):
        """Update running averages and adjust weights."""
        self.sft_losses.append(sft_loss)
        self.pref_losses.append(pref_loss)
        
        if len(self.sft_losses) > self.window_size:
            self.sft_losses.pop(0)
            self.pref_losses.pop(0)
        
        # Adjust weights to balance learning rates
        avg_sft = np.mean(self.sft_losses)
        avg_pref = np.mean(self.pref_losses)
        
        # If one channel is learning much faster, reduce its weight
        if avg_sft > 0 and avg_pref > 0:
            ratio = avg_sft / avg_pref
            if ratio > 1.5:  # SFT learning too fast
                self.sft_weight *= 0.95
                self.pref_weight = 1.0 - self.sft_weight
            elif ratio < 0.67:  # Preference learning too fast
                self.pref_weight *= 0.95
                self.sft_weight = 1.0 - self.pref_weight
    
    def get_weights(self) -> Tuple[float, float]:
        return self.sft_weight, self.pref_weight
```

### 3.3 Logging and Metrics

**Per-Step Metrics:**
- `loss`: Total combined loss
- `sft_loss`: SFT channel loss
- `preference_loss`: Preference channel loss
- `sft_contribution`: Fraction of total loss from SFT
- `preference_contribution`: Fraction of total loss from preference
- `sft_grad_norm`: Gradient norm for SFT updates
- `preference_grad_norm`: Gradient norm for preference updates
- `sft_weight`: Current SFT weight (if dynamic)
- `preference_weight`: Current preference weight (if dynamic)

**Per-Epoch Metrics:**
- Average losses per channel
- Channel contribution ratios
- Data utilization (how many samples from each channel)

---

## 4. Evaluation Metrics

### 4.1 Knowledge Retention Test

**File:** `core/evaluation/test_knowledge_retention.py`

**Purpose:** Measure how well the model retains factual knowledge from SFT training.

**Method:**
1. Create a test set of factual questions with known answers
2. Extract SFT pairs that contain these facts
3. Train model on SFT pairs
4. Evaluate model's ability to answer factual questions correctly

**Metrics:**
- **Factual Accuracy**: % of factual questions answered correctly
- **Knowledge Coverage**: % of SFT-extracted facts that are retained
- **Degradation Rate**: How quickly knowledge degrades over training steps

**Test Set Format:**
```json
{
  "question": "What is the capital of France?",
  "expected_answer": "Paris",
  "sft_pair_id": "uuid-of-extracted-pair",
  "category": "geography"
}
```

### 4.2 Behavioral Alignment Test

**File:** `core/evaluation/test_behavioral_alignment.py`

**Purpose:** Measure how well the model aligns with reward-guided preferences.

**Method:**
1. Use existing preference pairs as test set
2. Evaluate model's preference for chosen vs rejected responses
3. Measure reward model scores on model outputs

**Metrics:**
- **Preference Accuracy**: % of test pairs where model prefers chosen response
- **Reward Score Improvement**: Average reward score increase vs base model
- **Dimensional Alignment**: Per-dimension reward scores (empathy, social_coherence, etc.)

### 4.3 Channel Interference Detection

**File:** `core/evaluation/detect_channel_interference.py`

**Purpose:** Detect when one channel dominates or interferes with the other.

**Metrics:**
- **Channel Dominance Ratio**: Ratio of gradient magnitudes between channels
- **Loss Correlation**: Correlation between SFT loss and preference loss (negative = interference)
- **Performance Trade-off**: If improving one channel degrades the other

**Algorithm:**
```python
def detect_interference(
    sft_losses: List[float],
    pref_losses: List[float],
    sft_performance: Dict[str, float],
    pref_performance: Dict[str, float],
) -> Dict[str, Any]:
    """
    Detect channel interference.
    
    Returns:
        {
            "dominance_ratio": float,  # >1.0 means SFT dominates
            "loss_correlation": float,  # Negative = interference
            "performance_tradeoff": bool,  # True if improving one hurts the other
            "recommendation": str,  # "increase_sft_weight", "increase_pref_weight", "balanced"
        }
    """
```

### 4.4 Repetition Detection

**File:** `core/evaluation/test_repetition.py`

**Purpose:** Detect if model is repeating training conversations verbatim.

**Method:**
1. Load training SFT pairs
2. Generate responses to test prompts
3. Check if responses match training responses (fuzzy matching)
4. Report repetition rate

**Metrics:**
- **Exact Match Rate**: % of responses that exactly match training responses
- **Fuzzy Match Rate**: % of responses that are >90% similar to training responses
- **Pattern Diversity**: Measure of response diversity (lower = more repetitive)

**CLI Command:**
```bash
python -m core.evaluation.test_repetition \
    --base-model-id <model> \
    --lora-dir <path> \
    --training-pairs-db storage/conversations.db \
    --test-prompts data/eval/test_prompts.txt \
    --threshold 0.9  # Similarity threshold for fuzzy matching
```

**Interpretation:**
- **Exact Match Rate < 5%**: Good - model is generalizing, not memorizing
- **Exact Match Rate 5-15%**: Acceptable - some repetition but mostly generalization
- **Exact Match Rate > 15%**: Problem - model is memorizing, reduce SFT weight or add more regularization

### 4.5 Combined Evaluation

**File:** `core/evaluation/eval_dual_channel.py`

**CLI Command:**
```bash
python -m core.evaluation.eval_dual_channel \
    --base-model-id <model> \
    --lora-dir <path> \
    --knowledge-test-set data/eval/knowledge_test.jsonl \
    --behavioral-test-set data/preferences/preferences.jsonl \
    --reward-model-path artifacts/reward_model/default \
    --test-repetition true  # Include repetition testing
```

**Output:**
- Knowledge retention scores
- Behavioral alignment scores
- Channel interference analysis
- **Repetition metrics** (if `--test-repetition` enabled)
- Recommendations for weight adjustment

---

## 5. Integration with Online Learning

### 5.1 Relationship to Online Updates

**Current State:**
- `online_update.py` implements online DPO updates with reward weighting
- Uses replay buffer for prioritized sampling
- Supports safety gating

**Dual-Channel Online Learning:**

The dual-channel approach can be extended to online learning:

1. **Online SFT Updates**: Extract SFT pairs from recent conversations
2. **Online Preference Updates**: Use existing online DPO mechanism
3. **Combined Online Loss**: Apply dual-channel loss in online updates

**File:** `core/lora_trainer/online_update_dual.py` (future extension)

### 5.2 Data Flow

```
Conversation → Database
    ↓
Extract SFT Pairs (batch process)
    ↓
data/sft/sft_training.jsonl
    ↓
Dual-Channel Training (offline)
    ↓
artifacts/lora/<model>/<version>/
    ↓
Deploy to serving system
    ↓
Online updates (preference channel only, for now)
    ↓
Periodic retraining (dual-channel)
```

### 5.3 Retraining Cadence

**Recommended Schedule:**

1. **Daily**: Extract SFT pairs from new conversations
2. **Weekly**: Run dual-channel training on accumulated data
3. **Continuous**: Online preference updates (single channel)
4. **Monthly**: Full evaluation and weight adjustment

**Trigger Conditions:**
- SFT data: 500+ new pairs OR 7 days elapsed
- Preference data: 200+ new pairs OR 7 days elapsed
- Combined: Either threshold reached OR both at 50% of ideal

---

## 6. Configuration Specification

### 6.1 Complete Config Example

See `config/training/lora_dual.yaml` above for full specification.

### 6.2 Environment Variables

```bash
# Frontier model for extraction
FRONTIER_MODEL_ENDPOINT=https://api.openai.com/v1/chat/completions
FRONTIER_MODEL_API_KEY=sk-...

# Database
DATABASE_PATH=storage/conversations.db

# Training
CUDA_VISIBLE_DEVICES=0  # or MPS for Apple Silicon
```

### 6.3 Validation

**Config Validator:**

```python
def validate_dual_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate dual-channel training configuration.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    # Check weights sum to 1.0
    sft_weight = config["training"]["sft_weight"]
    pref_weight = config["training"]["preference_weight"]
    if abs(sft_weight + pref_weight - 1.0) > 0.01:
        errors.append("sft_weight + preference_weight must sum to 1.0")
    
    # Check batch mixing strategy
    mixing = config["training"]["batch_mixing"]
    if mixing not in ["interleaved", "alternating", "mixed"]:
        errors.append(f"Invalid batch_mixing: {mixing}")
    
    # Check data paths exist
    # Check learning rates are positive
    # Check batch sizes are positive
    
    return len(errors) == 0, errors
```

---

## 7. Implementation Checklist

### Phase 5.5.1: Data Infrastructure
- [ ] Add `sft_pairs` table to `apps/serve/schema.sql`
- [ ] Add database methods to `TrainingDatabase` class:
  - [ ] `insert_sft_pair()`
  - [ ] `get_sft_pairs()` with filtering
  - [ ] `mark_sft_pairs_used()`
- [ ] Create `SFTPair` dataclass in `core/data/schema.py` or new module
- [ ] Implement `extract_sft_pairs.py`:
  - [ ] Simple extraction: iterate messages, extract (user, assistant) pairs
  - [ ] Optional context inclusion
  - [ ] Deduplication by message_id
  - [ ] Database insertion
- [ ] Create batch extraction CLI that writes to database
- [ ] Add SFT data loading utilities (database + optional JSONL fallback)
- [ ] Test extraction on sample conversations
- [ ] Add database migration/update script if needed

**Note:** No frontier model integration needed for basic extraction. Every conversational pair is extracted automatically.

### Phase 5.5.2: Training Infrastructure
- [ ] Create `lora_dual.yaml` config file with database configuration
- [ ] Update `train_dpo.py` to support loading preferences from database (if not already)
- [ ] Implement `load_sft_pairs_from_database()` function
- [ ] Implement `SFTDataset` class
- [ ] Implement `DualChannelDataLoader` with mixing strategies
- [ ] Implement `dual_channel_loss` function
- [ ] Implement `train_dual_channel.py` main script with database integration
- [ ] Add logic to mark SFT pairs as used after training
- [ ] Add dynamic weight adjustment (optional)
- [ ] Integrate with existing training logger

### Phase 5.5.3: Evaluation
- [ ] Create knowledge retention test set
- [ ] Implement `test_knowledge_retention.py`
- [ ] Implement `test_behavioral_alignment.py`
- [ ] Implement `detect_channel_interference.py`
- [ ] Create combined evaluation script `eval_dual_channel.py`

### Phase 5.5.4: Integration
- [ ] Update checkpointing to track channel contributions
- [ ] Add metadata to LoRA artifacts (channel weights, data sources)
- [ ] Create retraining automation script
- [ ] Document usage and best practices

---

## 8. Testing Strategy

### 8.1 Unit Tests

- SFT pair validation
- Data loading and mixing
- Loss computation
- Dynamic weight adjustment

### 8.2 Integration Tests

- End-to-end extraction → training → evaluation
- Config validation
- Checkpoint loading/saving

### 8.3 Smoke Tests

- Train on minimal dataset (10 SFT pairs, 10 preference pairs)
- Verify both channels contribute to loss
- Verify checkpoint contains metadata

---

## 9. Future Enhancements

1. **Adaptive Channel Weighting**: Automatically adjust weights based on evaluation metrics
2. **Per-Dimension Reward Weighting**: Weight preference loss by reward dimensions
3. **SFT Quality Scoring**: Filter low-quality SFT extractions automatically
4. **Online Dual-Channel**: Extend online updates to include SFT channel
5. **Multi-Task Learning**: Add additional channels (e.g., code, math, reasoning)

---

## 10. Database vs JSONL Strategy

### 10.1 Primary Storage: SQLite Database

**All SFT pairs are stored in the `sft_pairs` table in SQLite.**

Benefits:
- Queryable with SQL (filtering, sorting, deduplication)
- Transactional integrity
- Tracks usage (`is_used`, `used_in_training_batch`)
- Consistent with existing architecture (preferences, trajectories, etc.)
- Better for production systems

### 10.2 JSONL Support (Optional)

JSONL files are supported only for:
- **Import/Export**: Backup, migration, or sharing data
- **Legacy compatibility**: If you have existing JSONL files
- **Debugging**: Easy to inspect and manually edit

**CLI Export Example:**
```bash
# Export SFT pairs to JSONL for backup
python -m core.data.export_sft_pairs \
    --db-path storage/conversations.db \
    --output data/exports/sft_pairs_backup.jsonl \
    --since 2024-01-01
```

**CLI Import Example:**
```bash
# Import SFT pairs from JSONL
python -m core.data.import_sft_pairs \
    --input data/exports/sft_pairs_backup.jsonl \
    --db-path storage/conversations.db
```

### 10.3 Training Data Loading

The training script (`train_dual_channel.py`) will:
1. **Primary**: Load from database using `TrainingDatabase.get_sft_pairs()`
2. **Fallback**: Load from JSONL if `use_database=false` in config
3. **Mark as used**: After training, mark pairs as `is_used=TRUE` with training batch ID

This ensures:
- No duplicate training on the same pairs
- Track which data was used in which training run
- Easy to query unused pairs for next training cycle

---

## 11. References

- Original plan: `docs/human_like_reward_lora_step_by_step_plan.md` (Phase 5.5)
- System design: `docs/human_like_reward_lora_system_design.md`
- Reward manifold: `docs/REWARD_MANIFOLD.md`
- Database schema: `apps/serve/schema.sql`
- Database utilities: `core/data/db.py`
