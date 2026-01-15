# Hominem Planning Reconstitution

## Core Principles
- Canonical truth is an append-only event log (JSONL.zst) with schema_version, event_type, event_id, and timestamps.
- Operational indexes (SQLite/DuckDB) are derived from the log; they are not sources of truth.
- Process boundaries are strict: inference, orchestration, training, and tools run as separate binaries/services.
- All event-driven steps must also be manually operable via simple CLIs.

## Target Repo Layout (Fresh Start)
Top level
- apps/ : process boundaries (binaries/services)
- crates/ : Rust libraries (pure, reusable)
- python/ : Python packages (shared, minimal)
- schemas/ : event schema specs and versions
- configs/ : runtime configs
- data/ : local runtime data (git-ignored)
- tools/ : admin and ingestion scripts
- docs/ : architecture and operations

apps/
- apps/hominem-infer/ (Python): OpenAI-style inference only (MLX-VLM backend), no tools.
- apps/hominem-agent/ (Python, Qwen-Agent): owns planning + tool calling loop, emits tool events.
- apps/hominem-core/ (Rust): log writer/reader, rotation, indexing, scheduling.
- apps/hominem-tools/ (Rust or Python): optional tool execution sandbox if not using Qwen-Agent tools directly.
- Open WebUI (external service): primary UI, configured to use hominem-agent as an OpenAI-compatible backend.

crates/
- crates/log/ : JSONL.zst writer/reader, ULID, rotation, offsets.
- crates/index/ : SQLite/DuckDB ingest and query helpers.
- crates/events/ : typed event structs and serialization.
- crates/windows/ : windowing and aggregations.
- crates/scheduler/ : sleep scheduling and backpressure.
- crates/metrics/ : drift checks and QA.

python/
- python/hominem_infer/ : inference server package.
- python/hominem_data/ : dataset transforms and export.
- python/hominem_train/ : training pipelines.
- python/hominem_tools/ : optional tool runners.

schemas/
- schemas/events/ : TurnEvent, ToolInvocationRequested, ToolInvocationResult, SleepEvent, TrainingExample.
- schemas/index/ : SQLite/DuckDB DDL (derived views).

configs/
- infer.toml, core.toml, tools.toml, *.yaml.

## Training System Design (Log-First)
- All training signals are written as events (TurnEvent, ToolInvocationRequested/Result, SleepEvent, LabelEvent/DerivedTarget).
- Derived SQLite/DuckDB indexes enable fast dataset queries; they are rebuildable.
- Training runs are separate processes that only read from the log/index and emit events.
- Model artifacts are content-addressed and recorded with config and dataset hashes.

Canonical training events
- DatasetQueryRequested
- DatasetBuilt
- TrainingRunStarted
- TrainingBatchMetrics
- TrainingRunCompleted
- ModelArtifactProduced
- ModelPromotionRequested / ModelPromotionApproved

## Manifold Training Example
Input events
- TurnEvent: conversation_id, turn_id, messages, model output, metrics.
- SleepEvent: targets (reward_intensity, delta_phi, self fractions).
- Optional LabelEvent.Manifold or DerivedTarget events.

Dataset build
- hominem-core emits DatasetQueryRequested with criteria, seed, max_samples, balance strategy.
- hominem-train builds dataset from index and emits DatasetBuilt with dataset_id, row_count, hashes.

Training run
- hominem-train emits TrainingRunStarted (model_type, base_model, config hash, dataset_id).
- Streams TrainingBatchMetrics every N steps.
- Emits TrainingRunCompleted and ModelArtifactProduced with artifact hash and metrics.

Promotion
- Gatekeeper emits ModelPromotionRequested and ModelPromotionApproved if metrics pass.
- hominem-infer watches for approved models and hot-swaps.

## Manual Operability (CLI-First)
Each event-driven step must be runnable directly without the rest of the system.

CLI as canonical entrypoint
- `python -m hominem_train.manifold_train` is the primary interface.
- Accepts dataset by:
  - `--dataset-id` (resolved from log/index)
  - `--dataset-path` (local JSONL/Parquet)
  - `--query-json` (inline criteria; builds dataset on the fly)

Event emission options
- `--emit-events` (default if event log configured)
- `--event-log-path` to write local JSONL when the system log is unavailable

Event-driven wrapper
- A lightweight listener can react to DatasetBuilt and call:
  - `python -m hominem_train.manifold_train --dataset-id <id> --emit-events`
- The wrapper is optional; the CLI must be sufficient by itself.

## Migration Notes (Current Codebase to Target)
- apps/serve/ becomes apps/hominem-infer/ (OpenAI-compatible inference only).
- Qwen-Agent (metaboliq fork) becomes apps/hominem-agent/ (planning + tool calling).
- apps/training_factory/ becomes a training binary or python/hominem_train/ (no canonical storage).
- database.py becomes a derived index or is replaced by log consumers.
- training_logger.py JSONL legacy mode is removed or updated to log format.
- Open WebUI is version-pinned (dependency or deployment pin) and treated as an external service.

## Immediate Next Steps (If You Choose)
- Define event schemas for DatasetQueryRequested, DatasetBuilt, TrainingRunCompleted.
- Map current write sites to event types and owners.
- Decide whether SQLite tables become first derived index implementation.
- Sketch a phased refactor plan for training modules (manifold, reward, regime).
 
## Qwen-Agent Integration (Tool Loop)
- Qwen-Agent talks to hominem-infer via OpenAI-compatible `/v1/chat/completions` and/or `/v1/responses`.
- Consider possibly swapping out for [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)
- Tool calling lives inside the Qwen-Agent process (MetaboliqAgent loop), not in infer.
- Tool invocations/results should emit events for replay/audit:
  - ToolInvocationRequested (tool name, args, parent turn)
  - ToolInvocationResult (tool name, result, status, timing)
- hominem-infer remains stateless and does not execute tools.
- hominem-agent exposes OpenAI-compatible `/v1/models` and `/v1/chat/completions` for Open WebUI.
