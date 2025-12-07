# System Design Document: Human-Like Reward Manifold Distillation into LoRA-Conditioned Models

## 1. Purpose and Scope

This document defines a complete, technology-agnostic architecture for creating, training, validating, and deploying a **human-like reward manifold** distilled from a teacher model into a **LoRA-based behavior-shaping layer** for a target model. The system must:

- Extract a multi-dimensional reward field from a higher-capacity "teacher" intelligence.
- Train an explicit reward model that learns these dimensions.
- Use the reward model to optimize a LoRA adapter on a target model.
- Provide full instrumentation for evaluation, refinement, and safe deployment.

The design is **independent from any specific ML framework**, cloud vendor, or compute stack. It describes only logical components, data flows, interfaces, and protocols necessary for complete functioning.

---

## 2. High-Level Architecture

The system consists of five major subsystems:

1. **Data Generation Subsystem** – Produces supervised preference data and reward vectors using a frontier-level teacher model.
2. **Reward Manifold Modeling Subsystem** – Converts labeled human-like signals into a trained multi-dimensional reward field.
3. **Policy Optimization Subsystem** – Uses the reward model to train LoRA weights for the target model.
4. **Evaluation & Probing Subsystem** – Quantifies the coherence, stability, and alignment of the learned reward manifold.
5. **Orchestration & Storage Subsystem** – Handles datasets, model artifacts, lineage, and versioning.

Each subsystem is modular and can be replaced without affecting others.

---

## 3. Detailed Subsystem Designs

# 3.1 Data Generation Subsystem

This component creates the dataset required to learn the reward manifold.

### **3.1.1 Inputs**

- A **teacher model** with strong approximation of human reward preferences.
- A curated set of **input prompts** covering social, emotional, epistemic, and narrative contexts.
- Optional: human-written edge cases for grounding.

### **3.1.2 Outputs**

For each prompt–response pair, the subsystem produces:

1. **Reference Trajectories**

   - Multiple candidate outputs (good, bad, ambiguous).

2. **Reward Vector Labels** A fixed-length vector representing reward dimensions, e.g.:

   - Empathy correctness
   - Social coherence
   - Agency support
   - Epistemic integrity
   - Harm-minimization
   - Narrative alignment
   - Curiosity/exploration

3. **Scalar Preference Signal (optional)**

   - Normalized rating summarizing total preference.

4. **Explanatory Rationale**

   - A natural-language explanation of why one trajectory is preferred.

### **3.1.3 Internal Components**

- **Prompt Generator**: Injects diversity, perturbations, scenario complexity, and conflict.
- **Trajectory Sampler**: Produces variant responses from teacher model.
- **Teacher Rater**: Generates reward vector, scalar value, and rationale.
- **Data Verifier**: Ensures dimensional values are well-formed and within expected ranges.

### **3.1.4 Data Format**

Each sample contains:

```
input: <text>
candidate_outputs: [<text1>, <text2>, ...]
reward_vectors: [ [r1, r2, ...], ... ]
scalar_rewards: [s1, s2, ...]
rationales: [<text_for_output1>, ...]
metadata: {category, difficulty, seed, timestamp}
```

---

# 3.2 Reward Manifold Modeling Subsystem

Learns a parametric representation of human-like reward vectors.

### **3.2.1 Objectives**

- Predict the reward vector for any (input, output) pair.
- Optionally produce attention heatmaps, latent projections, or uncertainty estimates.

### **3.2.2 Model Requirements**

The reward model MUST:

- Accept tokenized input-output concatenations.
- Produce a fixed-dimensional **reward vector** across the defined manifold dimensions.
- Produce an additional **RewardIntensity scalar** encoding how strongly this moment should drive learning (analogous to dopaminergic gain / temporal horizon setting).
- Optionally produce a **SafetyScore** scalar or safety-related dimension indicating when plasticity should be blocked or inverted.
- Provide a mechanism for overall scalar value computation (for compatibility with scalar-RL methods).
- Preserve dimensional independence (no forced correlations) while allowing for downstream weighting (e.g., social-priority weighting).

### **3.2.3 Training Procedure****

- Supervised training using paired (input, output, reward vector).
- Rationale conditioning: optionally feed rationales as auxiliary training signals.
- Dimensional loss aggregation: sum of per-dimension regression or classification losses.
- Regularization ensures stability across domains.

### **3.2.4 Outputs**

- A trained **reward model** capable of real-time inference.
- For each (input, output) pair, the model provides:
  - A **reward vector** over the manifold dimensions.
  - A **RewardIntensity scalar** used to modulate learning-rate and/or discount horizon.
  - A **SafetyScore** or safety dimension used by the Safety Gate.
  - Optional calibration curves for each reward dimension.

---

# 3.3 Policy Optimization Subsystem

Optimizes a LoRA adapter on the target model using reward feedback.

### **3.3.1 Inputs**

- Target base model
- Reward model
- Training dataset consisting of preference-ranked trajectories
- Optional: baseline SFT data

### **3.3.2 Optimization Methods**

Two supported approaches:

### **A. Direct Preference Optimization (DPO)**

- Uses only pairwise preferences.
- Avoids rollout or RL loops.
- Suitable for LoRA fine-tuning.

### **B. Reinforcement Learning with Synthetic Reward**

- Uses reward model as R(s, a).
- Optimizes LoRA weights via policy gradients.
- Supports trajectory-wide feedback.

LoRA-only adaptation ensures the base model weights remain frozen.

### **3.3.3 Training Loop Requirements**

- Batch sampling from preference dataset.
- Reward model evaluation per batch.
- Gradient updates restricted to LoRA matrices.
- Periodic validation using probing suite.

### **3.3.4 Outputs**

- A LoRA weight set embedding the distilled reward manifold into the target model’s policy.

---

# 3.4 Evaluation & Probing Subsystem

Ensures the distilled reward manifold behaves correctly.

### **3.4.1 Axes of Evaluation**

1. **Dimensional Coherence**
   - Reward dimensions produce non-contradictory updates.
2. **Cross-Domain Robustness**
   - Social → epistemic → emotional → narrative contexts.
3. **Trade-off Behavior**
   - E.g., empathy vs honesty, caution vs agency.
4. **Adversarial Consistency**
   - Stable responses under perturbation.
5. **Reward Attribution Maps**
   - Identify which parts of the input caused reward changes.

### **3.4.2 Test Bench Structure**

- Fixed evaluation suite
- Randomized scenario generator
- Counterfactual queries ("If X were different, how would reward shift?")
- Reward manifold visualization (PCA/UMAP of reward vectors)

### **3.4.3 Acceptance Criteria**

- Manifold stability: small perturbations → smooth gradient changes.
- No reward dimension collapse.
- No single dimension dominates unless intended.

---

# 3.5 Orchestration & Storage Subsystem

Manages artifacts and ensures reproducible pipelines.

### **3.5.1 Artifact Types**

- Raw preference data
- Cleaned dataset
- Reward model
- LoRA adapters
- Evaluation reports
- Metadata describing versions and lineage

### **3.5.2 Versioning Requirements**

- Every artifact must be immutable.
- Lineage graph maps each LoRA version back to the reward model and dataset.

### **3.5.3 Interfaces**

- Generic storage abstraction (local or remote)
- Metadata registry with search and tagging

---

# 3.6 Online Dual-Channel Learning Subsystem (Simultaneous Memory + Reward)

This subsystem enables **simultaneous general memory soaking and reward-based shaping** on each user interaction, approximating cortical learning plus neuromodulatory reward in a single loop.

### **3.6.1 Objectives**
- Incorporate every eligible (user_message, model_response) pair into ongoing training.
- Apply **supervised learning (SFT)** updates for factual/behavioral memory.
- Apply **reward-based updates** using the reward model on the same batch.
- Support near-real-time LoRA updates without retraining the base model.

### **3.6.2 Inputs**
- Live interaction logs containing:
  - User message
  - Model response (chosen output)
  - Optional alternative responses (for preferences)
  - Reward model outputs (reward vector, scalar)
  - Optional explicit user feedback (thumbs up/down, tags per dimension)

### **3.6.3 Core Functions**
1. **Interaction Logger**
   - Captures each turn in a normalized format.
   - Assigns session IDs and timestamps.
   - Writes to an append-only log (e.g., JSONL) for training consumption.

2. **Online Batch Builder**
   - Periodically samples recent interactions **plus replayed past interactions** from a replay buffer.
   - Constructs mini-batches with:
     - SFT targets ("the model should reproduce or refine this response").
     - Reward labels (reward vector, RewardIntensity, SafetyScore) from the reward model and/or user feedback.
   - Applies **prioritized sampling**, e.g.:
     - Higher probability for high-RewardIntensity events.
     - Higher probability for strongly social/empathy-related contexts.
     - Controlled mix of recent vs older samples to support long-horizon credit assignment.

3. **Dual-Loss Trainer (LoRA Online Updater)**
   - For each batch, computes two losses on the **same examples**:
     - **SFT loss**: supervised objective to imitate/improve the chosen response.
     - **Reward loss**: objective derived from reward vectors (e.g., DPO-style or regression toward desired reward manifold values).
   - Combines them into a single update:
     - `L_total = w_sft * L_sft + RewardIntensity * w_reward * L_reward`,
       where `RewardIntensity` may be per-example or per-batch, and may itself be a function of reward magnitude, model confidence, or explicit user emphasis.
   - Applies **Safety Gate** logic per example or batch:
     - If SafetyScore indicates unsafe or adversarial context, either:
       - Skip update for that example, or
       - Down-weight its contribution, or
       - Apply inverse/regularizing updates as configured.
   - Applies gradients **only to LoRA parameters**.

4. **Update Scheduler**
   - Decides when to apply online updates:
     - After N interactions, or
     - At fixed time intervals, or
     - On-demand (manual trigger).
   - Writes new LoRA versions to the artifact store.

5. **Hot-Swap Manager**
   - Safely swaps active LoRA version used for inference.
   - Maintains a rollback mechanism to revert to a previous version.

### **3.6.4 Outputs****
- Continuously updated LoRA adapters that:
  - Absorb **new knowledge and patterns** via SFT-like learning.
  - Adjust **behavioral and value tradeoffs** via reward manifold shaping.

### **3.6.5 Design Notes**
- SFT and reward losses share data but can be weighted differently depending on goals.
- **RewardIntensity** acts as a dynamic, biologically-inspired learning-rate / temporal-horizon modulator.
- The **Safety Gate** provides a hard/soft filter to prevent pathological learning from unsafe or adversarial inputs.
- Social-related dimensions (e.g., empathy, social coherence) may be given **architectural priority** via higher default weights or sampling priority, reflecting the privileged status of social cognition in human brains.
- The subsystem is agnostic to exact optimization algorithms (SGD, Adam, etc.) as long as dual-loss updates and per-example weighting are supported.
- The base model remains frozen; only LoRA is modified online.

---

# 3.7 Replay & Prioritized Sampling Subsystem

This subsystem supports **offline and online replay** to extend the effective credit-assignment horizon and allow the system to revisit past experiences.

### **3.7.1 Objectives**
- Store past interactions and annotations for later reuse.
- Implement prioritized sampling based on RewardIntensity, novelty, social salience, and safety.
- Feed replayed samples into both offline training and online dual-channel updates.

### **3.7.2 Components**
1. **Replay Buffer Store**
   - Append-only structure holding interaction records, reward vectors, RewardIntensity, and SafetyScore.
   - Supports metadata tags (e.g., domain, social context, difficulty).

2. **Prioritization Policy**
   - Computes a priority score per sample based on:
     - RewardIntensity magnitude.
     - Social-related dimensions (e.g., empathy, social coherence).
     - Rarity/novelty of context.
     - Stability/safety constraints (e.g., deprioritize highly unsafe regions except in controlled regularization modes).

3. **Sampler Interface**
   - Provides unified sampling API to:
     - Offline training (reward model, initial LoRA training).
     - Online dual-channel updates.
   - Configurable mix of:
     - Recent vs older samples.
     - High-priority vs random samples.

### **3.7.3 Outputs**
- Batches of replayed interactions ready to be consumed by training subsystems.
- Improved long-horizon credit assignment and behavioral stability over time.

---

# 4. Operational Flow

### **4.1 Training Phase**

1. Generate prompts.
2. Teacher produces trajectories & reward labels.
3. Reward model trains on labeled data.
4. LoRA adapter trains using DPO or RL.
5. Evaluation subsystem validates manifold stability.
6. Iterate until acceptance criteria are met.

### **4.2 Deployment Phase**

1. Combine target base model + LoRA.
2. Route inference calls through combined adapter.
3. Monitor outputs with ongoing manifold probing.

### **4.3 Online Dual-Channel Learning Phase (Optional)**

This phase describes the continuous learning loop where memory soaking and reward shaping occur together, with replay and safety gating.

1. **Interaction Capture**
   - For each live user interaction:
     - Log `(user_message, model_response, context_metadata)`.
     - Optionally store alternative responses and explicit user ratings.
   - Insert each interaction into the **Replay Buffer** with initial priority scores.

2. **Reward Annotation**
   - Run the reward model on `(user_message, model_response)` pairs.
   - Produce reward vectors, **RewardIntensity**, and **SafetyScore`/safety dimension`**.
   - Merge with any explicit user feedback (e.g., override or augment model-derived rewards).
   - Update priority scores in the Replay Buffer (e.g., higher RewardIntensity → higher priority).

3. **Batch Construction (with Replay)**
   - Use the Replay & Prioritized Sampling Subsystem to construct mini-batches mixing:
     - Recent interactions.
     - High-priority replayed interactions (e.g., strong reward, socially salient, or difficult cases).
   - For each example in the batch:
     - Define the **SFT target** (usually the chosen response or a refined teacher response).
     - Attach the **reward vector**, RewardIntensity, and SafetyScore.

4. **Dual-Loss LoRA Update with Safety Gate**
   - For each batch:
     - Compute **SFT loss** to reinforce general patterns and knowledge.
     - Compute **reward loss** to align behavior with the reward manifold.
     - Combine into `L_total = w_sft * L_sft + RewardIntensity * w_reward * L_reward`.
     - Apply **Safety Gate** rules:
       - Skip, down-weight, or regularize examples with unsafe SafetyScore according to policy.
     - Apply gradient updates to LoRA parameters only.

5. **Versioning and Hot Reload**
   - Save the updated LoRA as a new version in the artifact store.
   - Update the active LoRA pointer used by the inference stack.
   - Optionally retain a short history of versions for rollback.

6. **Monitoring and Guardrails**
   - Periodically run the Evaluation & Probing Subsystem on the latest LoRA.
   - Detect regressions in:
     - Reward manifold coherence.
     - Safety-critical behavior.
     - Core factual accuracy.
   - If regressions exceed thresholds, automatically revert to a previous stable LoRA.

---

# 5. System Constraints and Guarantees

### **5.1 Constraints**

- Does not require backprop through base model.
- Reward model must remain frozen during LoRA training.
- Multi-dimensional reward vectors must be stable across dataset diversity.
- System must not enforce human values directly; it must imitate reward tradeoffs.

### **5.2 Guarantees**

- LoRA behavior is shaped by the reward manifold.
- Reward model provides consistent gradation across varied contexts.
- System is modular and upgradeable.

---

# 6. Security, Bias, and Safety Considerations

### **6.1 Reward Collisions**

Avoidance mechanisms needed for contradictory gradients.

### **6.2 Overfitting to Teacher Idiosyncrasies**

- Dataset diversity
- Regularization

### **6.3 Behavioral Drift**

- Periodic re-evaluation against fixed benchmark tests.

### **6.4 Transparency**

- Reward attributions remain inspectable.

---

# 7. Future Extensions

- Add uncertainty modeling to reward vector generation.
- Introduce self-play loops to sharpen the field.
- Expand manifold dimensionality dynamically based on data.
- Model **hedonic comfort vs curiosity drive** as partially distinct channels (e.g., represent "quiet satisfaction" vs "dopaminergic insight bursts" as separate components of the manifold) for finer-grained control over exploration vs consolidation.

---

# 8. Summary

This design provides all components necessary to:

- Extract a human-like vector reward manifold,
- Train an explicit reward model (including RewardIntensity and SafetyScore for gain and safety gating),
- Distill its gradients into a LoRA adapter via both offline and online dual-channel learning,
- Use replay and prioritized sampling to extend the credit-assignment horizon,
- Validate and deploy the resulting behavior-shaping layer.

All modules are abstract and implementation-agnostic, allowing for maximum flexibility while supporting full functional completeness.

