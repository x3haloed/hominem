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
CREATE INDEX IF NOT EXISTS idx_synthetic_data_source ON synthetic_data(source);
CREATE INDEX IF NOT EXISTS idx_synthetic_data_used ON synthetic_data(is_used);

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
WHERE sd.is_used = TRUE;
