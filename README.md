# Hominem

Hominem is an early-stage, **log-first** agent + training system. Today this repo contains:

- `python/hominem_infer`: a slim, OpenAI-compatible inference server (MLX-VLM backend; optional stub backend)
- `python/hominem_agent`: a planning + tool-calling server (Qwen-Agent) that talks to `hominem_infer`
- `python/hominem_train`: CLI training scripts for “manifold”, “regime”, and “reward” heads, with optional event emission

The intended long-term architecture is described in `docs/ARCHITECTURE_PLAN.md`.

## Architecture (the direction)

The core idea is **append-only events as canonical truth** (e.g. JSONL/JSONL.zst), with any indexes (SQLite/DuckDB) derived from the log. Process boundaries are strict: inference stays inference-only; orchestration/tool-calling lives in the agent process; training is a separate consumer/producer of events.

Read first:
- `docs/ARCHITECTURE_PLAN.md`
- `docs/yet_another_theory_of_agency.md` (the conceptual frame the current harness-oriented design is aiming at)
- `docs/yet_another_plan.md`

Background research (not the repo’s current organizing design):
- `docs/UNIFIED_THEORY_ENGINEERING_SPEC.md`
- `docs/unified_theory.md`
- `docs/analysis_of_all_theories.md`
- `docs/ROADMAP_SLEEP_COUNTERFACTUAL_REWARD_MANIFOLD.md`

## Repository layout

- `python/hominem_infer/`: OpenAI-compatible `/v1/chat/completions` + `/v1/responses`
- `python/hominem_agent/`: OpenAI-compatible `/v1/chat/completions` for Open WebUI (owns tool loop)
- `python/hominem_train/`: training CLIs + JSONL dataset helpers
- `third_party/mlx_vlm/`: git submodule (MLX-VLM)
- `docs/`: architecture + theory/roadmap notes

## Quickstart (local dev)

Create (or reuse) the local venv at `.venv/` and install this repo (includes optional tool deps like Tabstack):

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e ".[agent,infer,tools]"
```

### Configuration via .env

The CLI entrypoints (`python -m hominem_infer`, `python -m hominem_agent`, `python -m hominem_train`) load a `.env` file automatically if present. Use `.env.example` as a starting point:

```bash
cp .env.example .env
```

### Run inference (`hominem_infer`)

Fastest smoke test (no MLX-VLM required):

```bash
INFER_BACKEND=stub ./.venv/bin/python -m hominem_infer
```

Then:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"stub","messages":[{"role":"user","content":"hello"}]}'
```

MLX-VLM backend notes:
- The server loads MLX-VLM when `INFER_BACKEND=mlx_vlm` (default).
- This repo vendors MLX-VLM as a submodule at `third_party/mlx_vlm`; initialize it with `git submodule update --init --recursive`.
- Install MLX-VLM into your venv (for example): `./.venv/bin/python -m pip install -e third_party/mlx_vlm`.

Key env vars:
- `INFER_HOST` (default `0.0.0.0`), `INFER_PORT` (default `8000`)
- `INFER_MODEL_ID` (default `alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx`)
- `INFER_EVENT_LOG` (if set, append JSONL events like `TurnEvent`)

### Run the agent (`hominem_agent`)

Start the agent server (it calls `hominem_infer` via OpenAI-compatible HTTP):

```bash
HOMINEM_INFER_BASE_URL=http://127.0.0.1:8000/v1 \
./.venv/bin/python -m hominem_agent
```

Key env vars:
- `HOMINEM_AGENT_HOST` (default `0.0.0.0`), `HOMINEM_AGENT_PORT` (default `8020`)
- `HOMINEM_INFER_BASE_URL` (default `http://127.0.0.1:8000/v1`)
- `HOMINEM_AGENT_MODEL` (defaults to `INFER_MODEL_ID`, then a Qwen2-VL fallback)
- `HOMINEM_UI_CORS_ORIGINS` (comma-separated allowlist; defaults include common Open WebUI dev ports)
- `HOMINEM_YACY_BASE_URL` (default `http://127.0.0.1:8090`)
- `TABSTACK_API_KEY` (required for `fetch_markdown` / `fetch_json`)
- `fetch_markdown` / `fetch_json` require the `tools` extra (`pip install -e ".[tools]"`).

## Using with Open WebUI

The intended setup is:

1. Open WebUI → points to `hominem_agent` as the OpenAI API base URL
2. `hominem_agent` → orchestration + tool calls
3. `hominem_infer` → inference-only (no tools)

This repo pins an Open WebUI version via `setup.py` extra: `./.venv/bin/python -m pip install -e ".[open_webui]"`.

```bash
open-webui serve
```

Open web browser to http://localhost:8080/

## Training (`hominem_train`)

Training is CLI-first and can emit JSONL training events (see `docs/ARCHITECTURE_PLAN.md`).

Install training deps:

```bash
./.venv/bin/python -m pip install -r requirements-training.txt
```

Run:

```bash
./.venv/bin/python -m hominem_train manifold --dataset-path path/to/dataset.jsonl
./.venv/bin/python -m hominem_train regime --dataset-path path/to/dataset.jsonl
./.venv/bin/python -m hominem_train reward --dataset-path path/to/dataset.jsonl
```

Dataset format (current scripts):
- `manifold` / `regime`: each JSONL record needs a `labels` object with the expected keys, and either `history`/`target` or `user_message`/`assistant`.
- `reward`: each JSONL record needs `messages: [...]` plus `scores: {...}`.

Event emission:
- Add `--emit-events --event-log-path path/to/events.jsonl` to training commands.

## Status

This repo is a prototype aligned to the “target layout” in `docs/ARCHITECTURE_PLAN.md`. The Rust log/index components and schema/versioning described there are not implemented here yet; the current Python servers and training CLIs are scaffolding toward that design.
