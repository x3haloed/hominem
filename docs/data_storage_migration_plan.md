# Data Storage Migration Plan

**Status:** Detour from main development roadmap - execute when datasets become unwieldy
**Purpose:** Transition from file-based JSONL storage to scalable database architecture
**Timeline:** 1-2 weeks implementation, ongoing maintenance

## Problem Statement

Current file-based JSONL storage will crumble under scale:
- 200k-500k+ conversation records for emotion manifold training
- Potential millions of records for long-term learning
- No indexing, inefficient queries, memory issues
- Datasets cluttering repository

## Target Architecture

### Primary Storage: SQLite Database
```sql
-- Conversations table
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    conversation_id TEXT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    lora_version TEXT,
    model_temperature REAL,
    response_time_ms INTEGER,
    metadata JSON
);

-- Emotion labels table
CREATE TABLE emotion_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    labeler TEXT NOT NULL,  -- 'auto' (frontier model) or 'manual' (user)
    label_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 6-axis emotion manifold
    valence REAL CHECK(valence BETWEEN -2 AND 2),
    arousal REAL CHECK(arousal BETWEEN 0 AND 1),
    dominance REAL CHECK(dominance BETWEEN -1 AND 1),
    predictive_discrepancy REAL CHECK(predictive_discrepancy BETWEEN -1 AND 1),
    temporal_directionality REAL CHECK(temporal_directionality BETWEEN -1 AND 1),
    social_broadcast REAL CHECK(social_broadcast BETWEEN 0 AND 1),

    -- Computed scalars
    reward_intensity REAL,
    safety_score REAL,

    -- UI indicators (stored as JSON for emoji mapping)
    raw_indicators JSON,  -- {'positive': 2, 'arousal': true, 'social': 'high'}

    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
    rationale TEXT,
    metadata JSON,

    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Reward model labels table (existing behavioral dimensions)
CREATE TABLE reward_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    empathy REAL,
    social_coherence REAL,
    agency_support REAL,
    epistemic_integrity REAL,
    harm_avoidance REAL,
    narrative_alignment REAL,
    curiosity REAL,
    reward_intensity REAL,
    safety_score REAL,
    metadata JSON,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Training batches table
CREATE TABLE training_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    batch_type TEXT,  -- 'sft' or 'preference'
    data_range_start DATETIME,
    data_range_end DATETIME,
    record_count INTEGER,
    parquet_path TEXT,
    metadata JSON
);

-- Indexes for performance
CREATE INDEX idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_emotion_labels_conversation ON emotion_labels(conversation_id);
CREATE INDEX idx_emotion_labels_labeler ON emotion_labels(labeler);
CREATE INDEX idx_emotion_labels_timestamp ON emotion_labels(label_timestamp);
```

### Dataset Export: Parquet Files
- Export filtered datasets to Parquet for ML training
- Columnar format for fast loading
- Compression reduces storage footprint
- Stored in `storage/exports/` directory

### Repository Structure
```
hominem/
├── data/                    # Keep in repo (samples, schemas)
│   ├── samples/            # Small curated examples (100-1000 records)
│   └── schemas/            # Data format definitions
├── storage/                # Datasets live here (gitignored)
│   ├── conversations.db    # SQLite database
│   ├── exports/           # Parquet exports for training
│   │   ├── sft_batches/
│   │   └── preference_batches/
│   ├── backups/           # Encrypted backups
│   └── migrations/        # Schema migration scripts
└── scripts/
    ├── db/                # Database utilities
    ├── export/           # Parquet export scripts
    └── backup/           # Backup/restore scripts
```

## Migration Phases

### Phase 1: Infrastructure Setup (1 week)
1. **Create database schema** with all tables and indexes
2. **Set up migration scripts** for existing JSONL data
3. **Create database utility scripts:**
   - Connection management
   - Backup/restore
   - Schema validation
   - Performance monitoring

### Phase 2: Data Migration (2-3 days)
1. **Migrate existing JSONL files** to SQLite
2. **Validate data integrity** (record counts, field mappings)
3. **Create initial Parquet exports** for existing training data
4. **Update all data loading code** to use database instead of files

### Phase 3: Application Updates (3-5 days)
1. **Update serving system** to write to database instead of JSONL
2. **Update training scripts** to read from database/Parquet
3. **Update CLI tools** for data exploration and analytics
4. **Add database connection management** to all Python scripts

### Phase 4: Repository Cleanup (1 day)
1. **Move existing data files** out of repo to storage directory
2. **Update .gitignore** to exclude storage directory
3. **Create symlinks or scripts** for easy data access during development
4. **Document new data access patterns**

## Performance Characteristics

### SQLite Advantages
- **ACID compliance:** Data integrity guarantees
- **Concurrent access:** Multiple processes can read/write safely
- **Rich querying:** SQL for complex analytics and sampling
- **Indexing:** Fast lookups by timestamp, labels, sessions
- **Backup/restore:** Simple file-based backup
- **Cross-platform:** Works on all systems

### Expected Performance
- **Insert:** 1000+ records/second for conversation logging
- **Query:** Sub-second for typical analytics queries
- **Sampling:** Instant random sampling for training
- **Storage:** ~1KB per conversation record (efficient compression)
- **Scale:** Handles 10M+ records comfortably on modern hardware

### Parquet Integration
- **Training data export:** Generate Parquet files on-demand
- **Versioning:** Timestamped exports for reproducible training
- **Compression:** 70-80% size reduction vs raw data
- **Loading:** 10-100x faster than JSONL for ML training

## Backup and Safety Strategy

### Automated Backups
- **Daily database backups** to encrypted files
- **Weekly full exports** to cloud storage
- **Pre-training snapshots** before model updates

### Data Validation
- **Integrity checks** after migrations
- **Schema validation** on startup
- **Duplicate detection** and cleanup scripts

### Recovery Procedures
- **Point-in-time recovery** from backups
- **Incremental sync** capabilities
- **Data export tools** for emergency migration

## Development Workflow Changes

### Before Migration
```bash
# Load data
data = load_jsonl("data/conversations.jsonl")

# Training
train_model(data)
```

### After Migration
```bash
# Export training data
python scripts/export/training_batch.py --type sft --days 7

# Training
python core/lora_trainer/train_dual.py --data storage/exports/sft_20241209.parquet
```

### Database Access Patterns
```python
# Logging conversations
db.insert_conversation(prompt, response, metadata)

# Query for training
conversations = db.get_labeled_conversations(
    start_date="2024-12-01",
    min_emotion_labels=1,
    limit=1000
)

# Analytics
stats = db.get_emotion_distribution(date_range="last_30_days")
```

## Migration Checklist

- [ ] Database schema designed and tested
- [ ] Migration scripts created for existing data
- [ ] Database utility library implemented
- [ ] Parquet export pipeline working
- [ ] Application code updated to use database
- [ ] Repository structure updated
- [ ] Backup strategy implemented
- [ ] Performance benchmarks completed
- [ ] Documentation updated
- [ ] Team trained on new workflow

## Risk Mitigation

**Data Loss Prevention:**
- Never delete original JSONL files until migration validated
- Multiple backup copies during transition
- Rollback scripts for failed migrations

**Downtime Minimization:**
- Migrate in small batches during low-usage periods
- Keep old system running in parallel during transition
- Gradual rollout with feature flags

**Performance Regression:**
- Benchmark before/after migration
- Monitor query performance post-migration
- Optimize indexes based on actual usage patterns

This migration ensures our data infrastructure scales to support the long-term learning goals while maintaining data safety and accessibility.
