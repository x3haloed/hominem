-- Conversation and labeling database schema for hominem serving system
-- This is the canonical source of truth for conversation history and labels

-- Main conversations table (canonical conversation history)
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT UNIQUE NOT NULL,  -- UUID for conversation thread
    title TEXT,  -- Auto-generated or user-set title
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,  -- Model version, temperature, etc.
    is_active BOOLEAN DEFAULT TRUE  -- For soft deletion
);

-- Individual messages within conversations
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    message_index INTEGER NOT NULL,  -- Order within conversation (0-based)
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),  -- Message sender
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER,  -- Approximate token count
    processing_time_ms INTEGER,  -- How long AI took to respond
    metadata JSON,  -- Model info, temperature, etc.

    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
    UNIQUE(conversation_id, message_index)
);

-- Introspection buffer (self-observations for self-awareness)
CREATE TABLE IF NOT EXISTS introspection_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,         -- Links to conversations.conversation_id
    message_id INTEGER,                    -- Links to messages.id (if from a specific message)
    observation_index INTEGER NOT NULL,    -- Order within conversation (0-based)

    -- Self-observation content
    observation_text TEXT NOT NULL,        -- The <SELF-OBSERVE> content
    content_hash TEXT,                     -- Hash for deduplication

    -- Metadata
    reward_intensity REAL,                 -- For replay buffer priority (RewardIntensity × 3, capped)
    safety_score REAL,                     -- Safety score for this observation
    internal_generated BOOLEAN DEFAULT TRUE, -- Tag internal vs user-injected
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,                         -- Additional context (model version, etc.)

    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY (message_id) REFERENCES messages(id),
    UNIQUE(conversation_id, observation_index)
);

-- Emotion labels (6-axis manifold + UI indicators)
CREATE TABLE IF NOT EXISTS emotion_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    labeler TEXT NOT NULL CHECK (labeler IN ('user', 'auto')),  -- Manual or automatic
    labeled_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 6-axis emotion manifold (manual labels may be partial)
    valence REAL CHECK(valence BETWEEN -2 AND 2),  -- 😊 +2/+1, 😟 -1/-2
    arousal REAL CHECK(arousal BETWEEN 0 AND 1),   -- 🚀 high arousal
    dominance REAL CHECK(dominance BETWEEN -1 AND 1),
    predictive_discrepancy REAL CHECK(predictive_discrepancy BETWEEN -1 AND 1),  -- 💔 surprise/betrayal
    temporal_directionality REAL CHECK(temporal_directionality BETWEEN -1 AND 1), -- ⏳ prospect vs 🪞 reflection
    social_broadcast REAL CHECK(social_broadcast BETWEEN 0 AND 1),  -- 🤗 high vs 🎭 low

    -- Computed scalars (auto-populated for auto labels)
    reward_intensity REAL,
    safety_score REAL,

    -- Raw UI indicators (what user clicked)
    raw_indicators JSON,  -- {'positive': 2, 'arousal': true, 'social': 'high'}

    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
    notes TEXT,  -- Optional user notes

    FOREIGN KEY (message_id) REFERENCES messages(id),
    UNIQUE(message_id, labeler)  -- Only one label per message per labeler
);

-- Reward model labels (behavioral dimensions - for comparison/training)
CREATE TABLE IF NOT EXISTS reward_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    labeler TEXT NOT NULL CHECK (labeler IN ('auto')),  -- Currently only auto
    labeled_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 7 behavioral dimensions
    empathy REAL,
    social_coherence REAL,
    agency_support REAL,
    epistemic_integrity REAL,
    harm_avoidance REAL,
    narrative_alignment REAL,
    curiosity REAL,

    -- Scalars
    reward_intensity REAL,
    safety_score REAL,

    confidence REAL CHECK(confidence BETWEEN 0 AND 1),

    FOREIGN KEY (message_id) REFERENCES messages(id),
    UNIQUE(message_id, labeler)
);

-- Synthetic training data (separate from conversation history)
CREATE TABLE IF NOT EXISTS synthetic_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,  -- 'frontier_generated', 'synthetic_augmentation', etc.
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Labels (can be emotion, reward, or both)
    emotion_labels JSON,  -- Full emotion label object
    reward_labels JSON,   -- Full reward label object

    metadata JSON,  -- Generation parameters, source model, etc.
    is_used BOOLEAN DEFAULT FALSE  -- For training set curation
);

-- Training batches (for tracking what data went into each model version)
CREATE TABLE IF NOT EXISTS training_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_name TEXT UNIQUE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    lora_version TEXT,  -- Which model version this trained

    -- Data sources
    conversation_start_date DATETIME,
    conversation_end_date DATETIME,
    synthetic_data_count INTEGER DEFAULT 0,
    total_samples INTEGER NOT NULL,

    -- Training config
    config JSON,  -- Full training configuration
    performance_metrics JSON,  -- Validation results

    status TEXT CHECK (status IN ('pending', 'training', 'completed', 'failed'))
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_conversations_active ON conversations(is_active);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, message_index);
CREATE INDEX IF NOT EXISTS idx_emotion_labels_message ON emotion_labels(message_id);
CREATE INDEX IF NOT EXISTS idx_emotion_labels_labeler ON emotion_labels(labeler);
CREATE INDEX IF NOT EXISTS idx_reward_labels_message ON reward_labels(message_id);
CREATE INDEX IF NOT EXISTS idx_introspection_conversation ON introspection_buffer(conversation_id, observation_index DESC);
CREATE INDEX IF NOT EXISTS idx_introspection_created ON introspection_buffer(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_introspection_content_hash ON introspection_buffer(content_hash);
CREATE INDEX IF NOT EXISTS idx_synthetic_data_source ON synthetic_data(source);
CREATE INDEX IF NOT EXISTS idx_synthetic_data_used ON synthetic_data(is_used);

-- Training data tables (migrated from JSONL)

-- Raw trajectories (generated prompt-response pairs)
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT UNIQUE NOT NULL,  -- Original 'id' from JSONL
    prompt_id TEXT,
    category TEXT,
    persona TEXT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    candidate_index INTEGER,
    source TEXT,  -- 'teacher', 'generator_model', etc.
    generator_model_id TEXT,
    generator_model_alias TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON  -- Additional generation parameters
);

-- Reward-labeled samples (trajectories with reward vectors)
CREATE TABLE IF NOT EXISTS reward_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT UNIQUE NOT NULL,  -- Original 'id' from JSONL
    trajectory_id TEXT,  -- References trajectories.trajectory_id
    prompt_id TEXT,
    category TEXT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    
    -- 7 behavioral dimensions
    empathy REAL,
    social_coherence REAL,
    agency_support REAL,
    epistemic_integrity REAL,
    harm_avoidance REAL,
    narrative_alignment REAL,
    curiosity REAL,
    
    -- Scalars
    scalar REAL,  -- Aggregate score
    reward_intensity REAL,
    safety_score REAL,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Preference pairs for DPO training
CREATE TABLE IF NOT EXISTS preference_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    chosen TEXT NOT NULL,
    rejected TEXT NOT NULL,
    
    -- Optional metadata
    chosen_id TEXT,  -- References reward_samples.sample_id
    rejected_id TEXT,
    prompt_id TEXT,
    category TEXT,
    chosen_score REAL,
    rejected_score REAL,
    score_margin REAL,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Self-training events (online feedback logs)
CREATE TABLE IF NOT EXISTS self_train_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,  -- Identifies the session log file
    timestamp_utc DATETIME NOT NULL,
    prompt TEXT NOT NULL,
    num_candidates INTEGER,
    max_new_tokens INTEGER,
    temperature REAL,
    top_p REAL,
    device TEXT,
    
    -- Chosen candidate
    chosen_text TEXT NOT NULL,
    chosen_scalar_score REAL,
    chosen_reward_empathy REAL,
    chosen_reward_social_coherence REAL,
    chosen_reward_agency_support REAL,
    chosen_reward_epistemic_integrity REAL,
    chosen_reward_harm_avoidance REAL,
    chosen_reward_narrative_alignment REAL,
    chosen_reward_curiosity REAL,
    chosen_reward_intensity REAL,
    chosen_reward_safety_score REAL,
    
    -- All candidates stored as JSON
    candidates_json JSON,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Training runs (metadata about training sessions)
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    component TEXT NOT NULL,  -- 'reward_model', 'lora_dpo', 'online_update', etc.
    created_at_utc DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata JSON  -- Run-level config, hyperparameters, etc.
);

-- Training steps (per-step metrics)
CREATE TABLE IF NOT EXISTS training_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    time_utc DATETIME DEFAULT CURRENT_TIMESTAMP,
    metrics JSON,  -- All step-level metrics (loss, learning_rate, etc.)
    
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);

-- Training evaluations (periodic eval snapshots)
CREATE TABLE IF NOT EXISTS training_evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    eval_number INTEGER NOT NULL,
    time_utc DATETIME DEFAULT CURRENT_TIMESTAMP,
    metrics JSON,  -- All eval-level metrics
    
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);

-- Indexes for training data
CREATE INDEX IF NOT EXISTS idx_trajectories_trajectory_id ON trajectories(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_prompt_id ON trajectories(prompt_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_created ON trajectories(created_at);
CREATE INDEX IF NOT EXISTS idx_reward_samples_sample_id ON reward_samples(sample_id);
CREATE INDEX IF NOT EXISTS idx_reward_samples_trajectory_id ON reward_samples(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_reward_samples_prompt_id ON reward_samples(prompt_id);
CREATE INDEX IF NOT EXISTS idx_reward_samples_created ON reward_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_preference_pairs_prompt_id ON preference_pairs(prompt_id);
CREATE INDEX IF NOT EXISTS idx_preference_pairs_created ON preference_pairs(created_at);
CREATE INDEX IF NOT EXISTS idx_self_train_events_session ON self_train_events(session_id);
CREATE INDEX IF NOT EXISTS idx_self_train_events_timestamp ON self_train_events(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_training_steps_run ON training_steps(run_id, step_number);
CREATE INDEX IF NOT EXISTS idx_training_evals_run ON training_evals(run_id, eval_number);

-- Views for training data preparation
CREATE VIEW IF NOT EXISTS training_data_combined AS
SELECT
    'conversation' as data_source,
    m.conversation_id,
    m.message_index,
    m.role,
    m.content,
    el.valence, el.arousal, el.dominance,
    el.predictive_discrepancy, el.temporal_directionality, el.social_broadcast,
    el.reward_intensity as emotion_intensity,
    el.safety_score as emotion_safety,
    rl.empathy, rl.social_coherence, rl.agency_support,
    rl.epistemic_integrity, rl.harm_avoidance, rl.narrative_alignment, rl.curiosity,
    rl.reward_intensity as reward_intensity,
    rl.safety_score as reward_safety,
    m.created_at,
    m.metadata
FROM messages m
LEFT JOIN emotion_labels el ON m.id = el.message_id AND el.labeler = 'auto'
LEFT JOIN reward_labels rl ON m.id = rl.message_id
WHERE m.role = 'assistant'  -- Only assistant responses for training
UNION ALL
SELECT
    'synthetic' as data_source,
    NULL as conversation_id,
    NULL as message_index,
    'assistant' as role,
    sd.response as content,
    json_extract(sd.emotion_labels, '$.valence') as valence,
    json_extract(sd.emotion_labels, '$.arousal') as arousal,
    json_extract(sd.emotion_labels, '$.dominance') as dominance,
    json_extract(sd.emotion_labels, '$.predictive_discrepancy') as predictive_discrepancy,
    json_extract(sd.emotion_labels, '$.temporal_directionality') as temporal_directionality,
    json_extract(sd.emotion_labels, '$.social_broadcast') as social_broadcast,
    json_extract(sd.emotion_labels, '$.reward_intensity') as emotion_intensity,
    json_extract(sd.emotion_labels, '$.safety_score') as emotion_safety,
    json_extract(sd.reward_labels, '$.empathy') as empathy,
    json_extract(sd.reward_labels, '$.social_coherence') as social_coherence,
    json_extract(sd.reward_labels, '$.agency_support') as agency_support,
    json_extract(sd.reward_labels, '$.epistemic_integrity') as epistemic_integrity,
    json_extract(sd.reward_labels, '$.harm_avoidance') as harm_avoidance,
    json_extract(sd.reward_labels, '$.narrative_alignment') as narrative_alignment,
    json_extract(sd.reward_labels, '$.curiosity') as curiosity,
    json_extract(sd.reward_labels, '$.reward_intensity') as reward_intensity,
    json_extract(sd.reward_labels, '$.safety_score') as reward_safety,
    sd.created_at,
    sd.metadata
FROM synthetic_data sd
WHERE sd.is_used = TRUE
UNION ALL
SELECT
    'reward_sample' as data_source,
    NULL as conversation_id,
    NULL as message_index,
    'assistant' as role,
    rs.response as content,
    NULL as valence,
    NULL as arousal,
    NULL as dominance,
    NULL as predictive_discrepancy,
    NULL as temporal_directionality,
    NULL as social_broadcast,
    NULL as emotion_intensity,
    NULL as emotion_safety,
    rs.empathy,
    rs.social_coherence,
    rs.agency_support,
    rs.epistemic_integrity,
    rs.harm_avoidance,
    rs.narrative_alignment,
    rs.curiosity,
    rs.reward_intensity,
    rs.safety_score as reward_safety,
    rs.created_at,
    rs.metadata
FROM reward_samples rs;

-- View for DPO training pairs
CREATE VIEW IF NOT EXISTS dpo_preference_pairs AS
SELECT
    prompt,
    chosen,
    rejected,
    chosen_id,
    rejected_id,
    prompt_id,
    category,
    chosen_score,
    rejected_score,
    score_margin,
    created_at
FROM preference_pairs;

-- View for self-train replay pairs (for online_update)
CREATE VIEW IF NOT EXISTS self_train_replay_pairs AS
SELECT
    id,
    prompt,
    chosen_text as chosen,
    -- Find the worst candidate from candidates_json for rejected
    -- This is a simplified view; actual logic should filter in application code
    timestamp_utc,
    chosen_scalar_score,
    chosen_reward_intensity,
    chosen_reward_safety_score,
    candidates_json,
    metadata
FROM self_train_events;
