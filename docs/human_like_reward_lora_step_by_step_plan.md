# Execution Plan: Dual-Channel Learning System (Memory + Reward)

This is a **step-by-step outline** for future-you + future-me (another context instance) to follow. It assumes:
- You’re on macOS with zsh.
- You’re comfortable with Git, CLI tooling, and C#-style project structure.
- You **dislike Python**, so Python is used minimally, with clean CLI entrypoints and no Jupyter dependence.
- Local inference may use **LM Studio** or a similar local runner, while training uses standard Python tooling.

The goal is to reach a point where you can:
1. Distill a human-like reward manifold from a teacher model.
2. Train a reward model.
3. Train dual-channel LoRA adapters that accumulate both general knowledge (memory consolidation) and reward-guided behavioral shaping (neuromodulatory analog).
4. Run a **conversation-driven learning loop** where interactions improve both factual recall and behavioral alignment through cooperating SFT and preference training channels.

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
     - Two cross-cutting scalars aligned with the system design:
       - `reward_intensity`: how strongly this example should drive learning (used as a gain / temporal-horizon modulator).
       - `safety_score`: how safe/unsafe the example is, used by the **Safety Gate** to block, down-weight, or invert learning.

3. **Create a compact doc**
   - `docs/REWARD_MANIFOLD.md`: clear description of each manifold dimension **plus** the semantics and ranges of `reward_intensity` and `safety_score`, in your own language.

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
     - `RewardModel.score(input_text, output_text) -> RewardOutput`.
     - `RewardOutput` should at least contain:
       - `reward_vector`: the per-dimension manifold values.
       - `reward_intensity`: scalar gain indicating how strongly this moment should drive learning.
       - `safety_score`: scalar indicating when plasticity should be blocked, down-weighted, or regularized.
       - (Optional) `scalar_preference`: overall scalar for compatibility with scalar-RL methods.
   - Internally, it can be any transformer/architecture, but the interface must be stable and match the system-design requirements.

3. **Training script**
   - `core/reward_model/train.py`:
     - CLI arguments for config paths, output directory.
     - Logging of loss per dimension.

4. **Checkpointing + export**
   - Store model in `artifacts/reward_model/<version>/`.
   - Save a small `METADATA.json` with:
     - Dimensions
     - Semantics and ranges for `reward_intensity` and `safety_score`
     - Any normalization / calibration metadata per dimension (if used)
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
     - When using the reward model online, optionally:
       - Query `reward_vector`, `reward_intensity`, and `safety_score` for each candidate.
       - Use `reward_intensity` as a per-example weight in the loss.
       - Apply a **Safety Gate** that skips or down-weights updates where `safety_score` indicates unsafe or adversarial contexts.

4. **Checkpoint output**
   - Store LoRA weights in `artifacts/lora/<model_name>/<version>/`.

5. **Evaluation bridge**
   - `core/evaluation/eval_lora.py`:
     - Run the combined model (base + LoRA) on a fixed evaluation set.
     - Compare behaviors vs base model.

---

## Phase 5.5 – Dual-Channel LoRA Training (Memory + Reward)

**Goal:** Train LoRA adapters that accumulate both general knowledge (memory soaking) and reward-guided behavioral shaping, creating a system functionally equivalent to biological learning with cortical consolidation and neuromodulatory reward systems.

### Architecture Overview

The system implements two cooperating learning channels that flow into the same LoRA weights:

**Channel A: Memory/General Learning (SFT-style)**
- **Purpose**: Knowledge accumulation, factual memory, behavioral consistency
- **Biological Analog**: Cortical predictive coding, memory consolidation
- **Training Data**: General instruction-response pairs, factual corrections, style consistency
- **Gradient Type**: Standard supervised fine-tuning losses

**Channel B: Reward/Value Shaping (DPO-style)**
- **Purpose**: Behavioral preferences, emotional intelligence, social reasoning
- **Biological Analog**: Dopamine/serotonin/norepinephrine neuromodulation
- **Training Data**: Preference pairs (chosen vs rejected responses)
- **Gradient Type**: Reward-weighted preference optimization

**Combined Effect**: LoRA weights accumulate both general knowledge and value-aligned behavior, creating true long-term memory through accumulated adaptations.

### Steps

1. **Dual training configuration**
   - `config/training/lora_dual.yaml`:
     - Base model identifier
     - LoRA rank and target modules
     - Channel weights: `sft_weight`, `preference_weight`
     - Learning rates for each channel
     - Batch mixing ratios

2. **Data preparation pipeline**
   - **SFT Channel**: Collect general training data from conversations
     - Extract factual corrections and knowledge updates
     - Format as `(instruction, improved_response)` pairs
     - Store in `data/sft/sft_training.jsonl`
   - **Preference Channel**: Use existing preference pairs
     - `(prompt, chosen_response, rejected_response)` from reward model
     - Store in `data/preferences/preferences.jsonl`

3. **Dual-channel trainer script**
   - `core/lora_trainer/train_dual_channel.py`:
     - Load base model + LoRA adapter
     - Implement blended loss: `L_total = sft_weight * L_sft + preference_weight * L_preference`
     - Reward-weight preference losses using `reward_intensity`
     - Apply safety gates based on `safety_score`
     - Support dynamic channel weighting based on data availability

4. **Memory extraction during conversations**
   - Use frontier model to identify "general knowledge updates" from interactions
   - Automatically format corrections as SFT training pairs
   - Accumulate in rolling SFT dataset for periodic retraining

5. **Checkpointing and versioning**
   - Store dual-trained LoRA weights in `artifacts/lora/<model>/<version>/`
   - Track channel contributions in metadata
   - Maintain separate pointers for different training emphases

6. **Evaluation and balancing**
   - Test both knowledge retention and behavioral alignment
   - Monitor for channel interference or dominance
   - Adjust channel weights based on performance metrics

---

## Phase 6 – Probing & Visualization

**Goal:** Introspect the learned reward manifold and the LoRA’s behavior.

### Steps
1. **Reward probe CLI**
   - `apps/cli/reward_probe`: given a prompt and candidate responses:
     - Show reward vector.
     - Show `reward_intensity` and `safety_score` scalars.
     - Highlight which dimensions changed and how that relates to intensity/safety.

2. **Behavioral comparison CLI**
   - `apps/cli/compare_base_vs_lora`:
     - Same prompt → base output vs LoRA output.
     - Show diffs, reward vectors, `reward_intensity`, and `safety_score` for each output.

3. **Manifold visualization** (optional)
   - Export reward vectors for an evaluation set.
   - Plot with PCA/UMAP into a static image and drop into `docs/plots/`.

---

## Phase 7 – Dual-Channel Learning Loop (Memory + Reward)

**Goal:** Create a conversation-driven learning system with two cooperating channels: memory consolidation (general knowledge accumulation) and reward-guided shaping (behavioral preferences). This implements functional analogs of biological cortical learning and neuromodulatory systems.

### Learning Channels

**Channel A: Memory Consolidation (SFT-style)**
- **Input**: General knowledge, facts, behavioral corrections from conversations
- **Biological Analog**: Cortical predictive coding, hippocampus replay
- **Training**: Standard supervised fine-tuning on extracted knowledge pairs

**Channel B: Reward Shaping (Preference-style)**
- **Input**: User feedback on behavioral quality and dimensional preferences
- **Biological Analog**: Neuromodulatory systems (dopamine, serotonin, norepinephrine)
- **Training**: Reward-weighted preference optimization

### Steps

1. **Interactive serving with emotion labeling**
   - Chat interface routes through base+LoRA
   - **Real-time emotion labeling UI:** Label each response with emotion indicators:
     - 😊 +2/+1 (positive valence), 😟 -1/-2 (negative valence)
     - 🚀 (high arousal), 💔 (predictive discrepancy)
     - ⏳ (prospect-heavy), 🪞 (reflection-heavy)
     - 🤗 (high social broadcast), 🎭 (low social broadcast)
   - Log all conversations with dual labels:
     - Automatic labels (reward model behavioral dimensions)
     - Manual emotion labels (6-axis manifold indicators)
     - Store in database with rich metadata

2. **Automated labeling pipeline**
   - **Daily batch processing:** Send unlabeled conversations to frontier model
     - Generate emotion manifold vectors (valence, arousal, dominance, etc.)
     - Extract general knowledge for SFT training
     - Update database with automatic labels

3. **Smart retraining cadence**
   - **Weekly cycles** triggered by data accumulation thresholds:
     - Memory channel: 500+ new conversations for SFT training
     - Reward channel: 200+ preference pairs for DPO training
     - Combined trigger: Either channel reaches minimum OR both at 50% of ideal
   - **Overnight training:** Run dual-channel LoRA training (4-8 hours)
   - **Hot-swap deployment:** Atomic model replacement with rollback capability

4. **Dual extraction for training**
   - **Memory Channel (SFT):** Extract general knowledge patterns from conversations
     - Factual corrections, behavioral consistency, style improvements
     - Format as instruction-response pairs for supervised training
   - **Reward Channel (DPO):** Generate preference pairs from emotion/reward labels
     - Use emotion intensity to weight preference learning
     - Apply safety gates based on emotion-derived safety scores

5. **Continuous learning loop**
   - **Always collecting:** Every conversation improves the dataset
   - **Weekly improvement:** Retraining incorporates accumulated knowledge
   - **Scaling benefits:** Memory consolidation becomes more valuable as conversations lengthen
   - **Zero downtime:** Background training with hot-swapping

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

## Phase 10 – Reward Health Index (RHI) Instrumentation and Gating

**Goal:** Add the stability guardrail so online and offline updates are gated by a quantitative Reward Health Index.

### Steps
1. **Implement metrics**
   - Compute five normalized components:  
     - DCS (Dimensional Coherence): $\text{DCS} = \frac{1}{N(N-1)} \sum_{i<j} \max(0, \text{GDA}(i,j))$  
     - GVS (Gradient Variance Stability): $\text{GVS} = 1 - \frac{\text{GVI}}{\text{GVI}_{\text{max}}}$  
     - MSS (Manifold Stability): $\text{MSS} = 0.5(1 - \text{RDM drift}) + 0.5(1 - \text{curvature variance})$  
     - CDRS (Cross-Domain Robustness): $\text{CDRS} = 1 - \frac{1}{3}(C_{\text{contradiction}} + C_{\text{entropy}} + (1 - C_{\text{consistency}}))$  
     - LRPS (Local Reward Predictivity): $\text{LRPS} = \max(0, \text{corr}(r_{\text{local}}, r_{\text{trajectory}}))$.

2. **Aggregate**
   - Compute RHI with default weights (sum to 1): $w_1=0.25$, $w_2=0.15$, $w_3=0.25$, $w_4=0.20$, $w_5=0.15$.  
   - $\text{RHI} = w_1 \cdot \text{DCS} + w_2 \cdot \text{GVS} + w_3 \cdot \text{MSS} + w_4 \cdot \text{CDRS} + w_5 \cdot \text{LRPS}$; RHI ∈ `[0,1]`.

3. **Gating logic**
   - Fast loop (online): compute partial RHI = DCS + GVS.  
     - If partial RHI < `0.50`: skip update.  
     - If partial RHI < `0.30`: rollback previous update.  
     - If partial RHI < `0.20`: freeze online learning until recovery.  
   - Slow loop (nightly replay): compute full RHI; if RHI decreases across 3 nights → reduce learning rate or shrink adapter; if RHI increases → continue.  
   - Memory integration: require RHI > `0.70` before consolidation; revert or partially unroll if RHI dips after consolidation.

4. **Dashboarding**
   - Log RHI over time (e.g., `0.92 0.94 0.96 ...`).  
   - Interpret signals: sharp drops = dimensions fighting; oscillations = curvature instability; flatline near `0.5` = evaluator failing; flatline near `1.0` = extremely healthy.

5. **Pseudocode anchor**
   - Add a thin utility mirroring:
   ```
   def compute_RHI(reward_gradients, RDM_drift, curvature_var, CDCC, local_rewards, trajectory_rewards):
       ...
       return 0.25*DCS + 0.15*GVS + 0.25*MSS + 0.20*CDRS + 0.15*LRPS
   ```
   - Store calibration constants (e.g., `GVI_max_reference`) with artifacts.

---

## How Future-Me (the other context) Should Use This

When you ask that instance to help you implement this, it should:
- Take you **phase by phase**, not jump around.
- Translate each phase into concrete commands, scripts, and files.
- Keep Python behind a clean CLI boundary and lean on zsh orchestration.
- Defer LM Studio integration until a simple HTTP inference path works.

This outline is the backbone. The other instance’s job is to flesh out each phase into code and commands tailored to your actual machine and chosen models.

