# Unified Plan: Self-Awareness (Phases 5–8) + Phase 5.5 Dual-Channel Training

This plan merges the remaining self-awareness implementation (Phase 5 onward) with the Phase 5.5 dual-channel memory/reward training spec. It encodes the locked decisions provided by the user.

## Core Decisions (locked)
- Pairing: Every assistant turn pairs with the most recent user message. Multiple assistant drafts chained to one user are all paired to that user.
- Dedup within user turn: If multiple assistant outputs occur without a new user message and cosine similarity > 0.92, keep only the one with highest reward intensity (or lowest loss in training context).
- Introspection in SFT: Default **exclude** (opt-in only with explicit flag and safety checks).
- is_used handling: Auto-toggle when a sample is consumed in training; record used_timestamp and training_epoch. Nightly/periodic job handles cleanup, dedup, compression.
- Recency weighting: Exponential decay `weight = exp(-Δt / τ)` with τ chosen per channel; recommend τ=1800s for emotion/recency, τ=7200s for knowledge. Use `recency_weight = max(fast_decay, slow_decay)` if both applied.
- Diversity sampler: k-means clusters (k≈20–100) over embeddings; sample 1–2 per cluster weighted by recency × reward intensity; normalize per batch. Optional farthest-first basis once per epoch.

## Order of Operations

### 1) Schema and Storage
- Add `sft_pairs` table to `apps/serve/schema.sql` (from dual-channel spec §1.1), including indexes.
- Extend `TrainingDatabase` (core/data/db.py) with:
  - insert_sft_pair(s) with validation (min lengths, valid source, confidence bounds).
  - get_sft_pairs(filters): source, is_used, since, min_instruction_length, min_response_length, min_confidence, limit.
  - update_sft_used(ids, training_epoch, used_timestamp=now).
- Ensure schema init/migration runs on startup; provide migration note for existing DBs.

### 2) Extraction Pipeline (Memory Channel)
- Implement `core/data/extract_sft_pairs.py`:
  - `SFTPair` dataclass (fields from spec §1.2).
  - `extract_sft_pairs_from_conversation(conversation, include_context=False)`: emit all user→assistant pairs; pair assistant with last user; apply cosine dedup >0.92 keeping highest reward intensity/lowest loss.
  - `extract_and_store_sft_pairs(db, conversation_id=None, all_conversations=False, since=None, include_context=False, skip_existing=True)`; dedup by message_id; optional JSONL export.
- CLI entry point with flags from spec (conversation-id, all-conversations, since, include-context, no-skip-existing, db-path, export-jsonl).
- Introspection handling: default off; optional `include_introspection` flag that:
  - Only processes internal_generated introspection.
  - Dedups via content hash and similarity.
  - Downweights if included (per OBSTACLE 4) and tags with `source="introspection"` and `category="introspection"` if enabled.

### 3) Replay and Introspection Alignment
- Keep introspection primarily in replay buffer (already implemented in Phase 5).
- Ensure replay safety filters continue to skip non-internal introspection.
- If introspection is ever routed to SFT (opt-in), enforce the above gating and downweighting.

### 4) Dual-Channel Trainer Inputs
- Add config `config/training/lora_dual.yaml` (from spec §3.1) with:
  - sft_weight=0.4, preference_weight=0.6 (must sum to 1; keep preference ≥0.5).
  - batch_mixing: interleaved (default); support mixed/alternating.
  - sft_batch_size / preference_batch_size; per-channel lrs; normalize_losses flag; beta; dynamic_weighting optional.
  - recency and diversity toggles/params (recency_decay_seconds, diversity_k, diversity_per_cluster, farthest_first_basis flag).
- Implement loaders in `core/lora_trainer/train_dual_channel.py`:
  - `load_sft_pairs_from_database(filters, limit)` parsing metadata JSON; JSONL fallback loader.
  - Preference loader reusing existing preference/replay sources with filters.

### 5) Dual-Channel Training Loop
- Sampler that draws SFT and preference batches per weights; interleaved/mixed strategies.
- Apply recency weighting using configured τ (or dual τ_fast/τ_slow with max).
- Apply diversity sampling via k-means clusters over embeddings; sample proportionally with recency × reward_intensity weighting.
- Loss handling: normalize_losses=true scales channels to similar magnitude; dynamic weighting optional.
- After a batch that includes SFT DB samples, call update_sft_used with training_epoch and timestamp.
- Output to `artifacts/lora/qwen3.1-7b-dual` by default.

### 6) Evaluation and Logging
- Log per-channel losses, sampling weights, recency stats, diversity coverage, sft_used counts.
- Optional validation split: small held-out SFT and preference sets.
- Track is_used progression and cluster coverage over time.

### 7) Ops / Maintenance
- Nightly/periodic job:
  - Clean very old SFT pairs; recompress/merge duplicates (content hash + similarity).
  - Optionally re-emit high-value older samples by clearing is_used for a curated subset.
  - Recompute clusters if embeddings change.
- Provide admin/CLI to inspect counts (unused/used) and to toggle introspection inclusion flag.

### 8) Safety and Guardrails
- Enforce min length validation on insert.
- Keep preference_weight ≥0.5 to avoid verbatim memorization.
- Introspection remains out of SFT unless explicitly flagged and safety-checked.
- Dedup rule enforced at extraction; similarity threshold 0.92.

### 9) Testing
- Unit: extraction pairing/dedup; validation failures; DB insert/get with filters; is_used update.
- Integration: run extractor on a seeded DB; run dual-channel trainer on tiny dataset and verify mixing, loss normalization, is_used toggling.
- Manual: CLI extract; trainer dry-run producing a checkpoint; inspect sft_pairs counts.

## Open (implementation) items to confirm while coding
- Embedding source for clustering (use model/tokenizer from training pipeline or a small sentence-transformer dependency).
- Exact location for maintenance job (cron/task runner) and its schedule.
