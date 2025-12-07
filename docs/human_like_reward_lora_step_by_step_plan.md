# Execution Plan: Human-Like Reward LoRA with Near-Real-Time Learning

This is a **step-by-step outline** for future-you + future-me (another context instance) to follow. It assumes:
- You’re on macOS with zsh.
- You’re comfortable with Git, CLI tooling, and C#-style project structure.
- You **dislike Python**, so Python is used minimally, with clean CLI entrypoints and no Jupyter dependence.
- Local inference may use **LM Studio** or a similar local runner, while training uses standard Python tooling.

The goal is to reach a point where you can:
1. Distill a human-like reward manifold from a teacher model.
2. Train a reward model.
3. Train a LoRA on a student model.
4. Run a **near-real-time loop** where your feedback adjusts the LoRA incrementally.

---

## Phase 0 – Repository and Project Skeleton

**Goal:** Create a clean mono-repo that separates concerns and allows you to orchestrate from the CLI (and later from C# if you want).

### Steps
1. **Create mono-repo layout**
   - Root folder layout:
     - `hominem/`
       - `apps/`
         - `cli/` (thin Python or Node CLI for orchestration)
       - `core/`
         - `data/` (data schemas, loaders)
         - `reward_model/`
         - `lora_trainer/`
         - `evaluation/`
       - `config/` (YAML/TOML/JSON configs)
       - `scripts/` (zsh helpers)
       - `docs/` (design + notes)

2. **Decide language roles**
   - Python: model training, data pipelines, LoRA training.
   - Shell/zsh: orchestration, dev ergonomics.
   - (Optional) C#: later, for integration into your apps.

3. **Initialize Git + basic docs**
   - `README.md`: high-level goals.
   - `docs/ARCHITECTURE.md`: short summary of subsystems.

---

## Phase 1 – Environment Setup (Training + Inference)

**Goal:** Establish a reproducible environment for training and a clean boundary for inference.

### Steps
1. **Choose Python environment manager**
   - Prefer a single tool (e.g., `uv` or `pyenv` + `venv`) and standardize on it.
   - Create `scripts/setup-env.zsh` to automate:
     - Creating venv
     - Installing dependencies

2. **Define dependency boundaries**
   - One requirements/lockfile for **training**.
   - Separate optional extras for **inference-only** (for LM Studio integration or local server).

3. **Inference options**
   - **Option A (Simple):** Use LM Studio for local inference and expose a local HTTP endpoint.
   - **Option B:** Run a lightweight local server (e.g., text-generation server) accessible over HTTP.
   - Config abstraction: store the endpoint + API key (if any) in a single config file (`config/inference.toml`).

---

## Phase 2 – Define Reward Manifold Schema

**Goal:** Fix the dimensions of the human-like reward vector and decide how to label them.

### Steps
1. **Choose 4–8 core dimensions** (concrete draft):
   - `empathy`
   - `social_coherence`
   - `agency_support`
   - `epistemic_integrity`
   - `harm_avoidance`
   - `narrative_alignment`
   - `curiosity`

2. **Define schema**
   - Write `core/data/schema.py` (or language-agnostic schema in `config/schemas/reward.json`):
     - Range for each dimension (e.g., -1.0 to 1.0 or 0–1).
     - Scalar aggregate (optional) for overall preference.

3. **Create a compact doc**
   - `docs/REWARD_MANIFOLD.md`: clear description of each dimension, in your own language.

---

## Phase 3 – Teacher-Driven Data Generation

**Goal:** Use a teacher model (frontier or strong hosted model) to generate labeled data.

### Steps
1. **Prompt set design**
   - Create `config/prompts/seed_prompts.yaml` with categories:
     - Emotional support
     - Disagreement & conflict
     - Moral dilemmas
     - Information-seeking (epistemic integrity)
     - Agency/empowerment conversations
   - Seed with 50–100 prompts manually.

2. **Trajectory generator script**
   - `core/data/generate_trajectories.py`:
     - For each prompt: sample N candidate responses from the teacher (e.g., 3–5).
     - Save as raw JSONL in `data/raw/trajectories.jsonl`.

3. **Teacher rater script**
   - `core/data/label_with_teacher.py`:
     - For each (prompt, response):
       - Ask teacher model to output:
         - Reward vector (values per dimension).
         - Optional scalar.
         - Natural-language rationale.
     - Store in `data/labeled/reward_samples.jsonl`.

4. **Validation CLI**
   - Small CLI command to:
     - Print random sample.
     - Check ranges and missing fields.

---

## Phase 4 – Reward Model Training

**Goal:** Train a compact reward model that maps (prompt, response) → reward vector (and optional scalar).

### Steps
1. **Data loader + splitter**
   - `core/reward_model/dataset.py`:
     - Load JSONL.
     - Split into train/val.

2. **Model definition (abstract)**
   - Define an interface:
     - `RewardModel.encode(input_text, output_text) -> reward_vector`.
   - Internally, it can be any transformer/architecture, but the interface must be stable.

3. **Training script**
   - `core/reward_model/train.py`:
     - CLI arguments for config paths, output directory.
     - Logging of loss per dimension.

4. **Checkpointing + export**
   - Store model in `artifacts/reward_model/<version>/`.
   - Save a small `METADATA.json` with:
     - Dimensions
     - Training data hash
     - Date

5. **Smoke test**
   - `core/reward_model/test_inference.py`:
     - Load model and run on a few samples.
     - Print predicted vs teacher labels.

---

## Phase 5 – LoRA Training (Batch / Offline)

**Goal:** Train a LoRA adapter on a student model using the reward model via DPO or similar.

### Steps
1. **Prepare preference data**
   - Use teacher labels or teacher comparisons to create pairs:
     - (prompt, chosen_response, rejected_response).
   - Store in `data/preferences/preferences.jsonl`.

2. **LoRA training config**
   - `config/training/lora_dpo.yaml`:
     - Base model identifier.
     - LoRA rank.
     - Learning rates.
     - Batch size, steps.

3. **DPO trainer script**
   - `core/lora_trainer/train_dpo.py`:
     - Load base model + LoRA adapter.
     - Load preferences.
     - Use reward model implicitly (if desired) or precomputed preferences.

4. **Checkpoint output**
   - Store LoRA weights in `artifacts/lora/<model_name>/<version>/`.

5. **Evaluation bridge**
   - `core/evaluation/eval_lora.py`:
     - Run the combined model (base + LoRA) on a fixed evaluation set.
     - Compare behaviors vs base model.

---

## Phase 6 – Probing & Visualization

**Goal:** Introspect the learned reward manifold and the LoRA’s behavior.

### Steps
1. **Reward probe CLI**
   - `apps/cli/reward_probe`: given a prompt and candidate responses:
     - Show reward vector.
     - Highlight which dimensions changed.

2. **Behavioral comparison CLI**
   - `apps/cli/compare_base_vs_lora`:
     - Same prompt → base output vs LoRA output.
     - Show diffs and reward vectors.

3. **Manifold visualization** (optional)
   - Export reward vectors for an evaluation set.
   - Plot with PCA/UMAP into a static image and drop into `docs/plots/`.

---

## Phase 7 – Real-Time (Online) LoRA Learning Loop

**Goal:** Allow you to interact with the model and have your feedback update the LoRA in near-real-time (or in small incremental batches) without retraining from scratch.

### Steps
1. **Interactive session server**
   - Thin process (Python or C#-backed) that:
     - Accepts conversation turns.
     - Routes inference through base+current-LoRA.
     - Displays output and asks for your feedback:
       - Binary (good/bad)
       - Dimension-wise sliders or tags (e.g., "more empathy", "less deference", etc.).

2. **Feedback log format**
   - `data/online_feedback/session_<timestamp>.jsonl`:
     - prompt
     - model_output
     - user_feedback (scalar + per-dimension adjustments)

3. **Online update scheduler**
   - Script: `core/lora_trainer/online_update.py`:
     - Watches feedback logs.
     - When enough data accumulates (configurable):
       - Convert feedback into preference pairs or pseudo-reward updates.
       - Run a short DPO / fine-tune step on LoRA weights.
   - Writes updated LoRA version to `artifacts/lora/<model>/<version_n+1>/`.

4. **Hot-reload in interactive session**
   - Interactive server periodically checks for a new LoRA version and swaps it in.
   - Minimal downtime, no full reload of base model if possible.

5. **Version tagging**
   - Maintain a `artifacts/lora/current.json` pointer to the active LoRA version.
   - Optionally keep a `history.json` with notes you write about each update.

---

## Phase 8 – Integration with Your Ecosystem

**Goal:** Make the trained LoRA + reward models usable in your broader projects (C#, DevOps tooling, AGI harness, etc.).

### Steps
1. **Stable APIs**
   - Define simple HTTP endpoints for:
     - `/generate` (base+LoRA inference)
     - `/reward` (reward model scoring)
     - `/feedback` (log online feedback)

2. **C# client bindings (optional)**
   - Small C# library that:
     - Calls those endpoints.
     - Wraps them in idiomatic async methods.

3. **Config-driven environment switching**
   - Single config file for:
     - Which base model
     - Which LoRA
     - Which reward model
   - Zsh helpers to swap configs quickly.

---

## Phase 9 – Guardrails, Testing, and Maintenance

**Goal:** Ensure the system remains stable as you tinker.

### Steps
1. **Test suites**
   - Unit tests for:
     - Data loading
     - Reward model inference
     - LoRA training loop
   - Scenario tests for:
     - "Empathy vs honesty" tradeoffs
     - "Agency support vs safety" tradeoffs

2. **Regression fixtures**
   - Keep a small, hand-curated set of prompts that matter to you.
   - Periodically test new LoRA versions against them.

3. **Log & metrics**
   - Even simple text logs are enough initially.
   - Track:
     - Number of online updates
     - Average reward per dimension
     - Qualitative notes you add when a version “feels” better.

---

## How Future-Me (the other context) Should Use This

When you ask that instance to help you implement this, it should:
- Take you **phase by phase**, not jump around.
- Translate each phase into concrete commands, scripts, and files.
- Keep Python behind a clean CLI boundary and lean on zsh orchestration.
- Defer LM Studio integration until a simple HTTP inference path works.

This outline is the backbone. The other instance’s job is to flesh out each phase into code and commands tailored to your actual machine and chosen models.

