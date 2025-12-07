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
- Produce a fixed-dimensional reward vector.
- Provide a mechanism for overall scalar value computation.
- Preserve dimensional independence (no forced correlations).

### **3.2.3 Training Procedure**

- Supervised training using paired (input, output, reward vector).
- Rationale conditioning: optionally feed rationales as auxiliary training signals.
- Dimensional loss aggregation: sum of per-dimension regression or classification losses.
- Regularization ensures stability across domains.

### **3.2.4 Outputs**

- A trained **reward model** capable of real-time inference.
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

## 7.1 Reward Model Improvement Plan (Concrete Implementation Notes)

For the concrete implementation in `hominem`, once a larger labeled dataset is available, the reward model should be upgraded along these axes:

- **Label normalization**
  - Standardize labels per dimension (e.g., zero mean, unit variance) during training.
  - Optionally track running statistics and store them alongside `METADATA.json` for consistent inference-time de-normalization.

- **Training regime**
  - Increase number of epochs and use early stopping based on validation loss.
  - Expand the dataset and periodically re-train to avoid overfitting to the initial small set.

- **Model capacity**
  - Consider trying a slightly larger encoder (e.g., `bert-base-uncased` or a domain-tuned model) if capacity becomes a bottleneck.

- **Metrics and monitoring**
  - Track per-dimension RMSE/MAE and correlation between teacher labels and model predictions.
  - Log these metrics per training run into the artifacts directory for regression tracking.

---

# 8. Summary

This design provides all components necessary to:

- Extract a human-like vector reward manifold,
- Train an explicit reward model,
- Distill its gradients into a LoRA adapter,
- Validate and deploy the resulting behavior-shaping layer.

All modules are abstract and implementation-agnostic, allowing for maximum flexibility while supporting full functional completeness.

