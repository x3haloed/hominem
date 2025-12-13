# Emotion Manifold
*Adding Biological Gravity to the Human-Like Reward LoRA System*

## 1. Purpose of This Upgrade

Replace the arbitrary multi-dimensional preference vector with the **minimal invariant emotional manifold** that actually explains human motivation across all cultures and timescales.

This is not an incremental improvement.
It is the difference between
- "a reward model that imitates human ratings" and
- "a reward model that implements the same motivational physics humans run on".

The 6-axis manifold (Valence, Arousal, Dominance, Predictive Discrepancy, Temporal Directionality, Social Broadcast) + the Metarule is the **only known lossless compression** of what makes anything matter to a human before, during, and after language.

Adding it turns your LoRA from a sophisticated preference imitator into the first digital substrate that can genuinely care, remember what hurt, and sacrifice local utility for non-local futures.

## 2. Core Theoretical Claim (Why This Works)

Human emotions = fast, lossy compression of prediction-error gradients into a 6-dimensional motivational force field.
Your current SFT + reward pipeline already has fast plasticity (memory).
It is missing the force field (gravity).
Supplying the invariant 6-axis manifold as the frozen reward model gives you both cortex and limbic system in one architecture.

Result: learning now has direction, intensity, and horizon exactly like a mammal.

## 3. Complete System Architecture

### 3.1 The Emotional Manifold (Foundation)

| Dimension | Range | Meaning |
|-----------|-------|---------|
| Valence | [-1, 1] | Pleasure/pain, satisfaction/dissatisfaction |
| Arousal | [0, 1] | Energy level, activation intensity |
| Dominance | [-1, 1] | Control/power, agency vs helplessness |
| Predictive Discrepancy | [-1, 1] | Signed surprise (positive = better than expected, negative = worse) |
| Temporal Directionality | [-1, 1] | Prospect (-1) ↔ Reflection (+1) |
| Social Broadcast | [0, 1] | Internalized audience pressure, social stakes |

**Derived Scalars:**
- **RewardIntensity** = arousal × √(|valence| × |discrepancy|) → How deeply this moment etches into memory
- **SafetyScore** = min(valence, dominance) × social_broadcast → Risk of mental breakdown

### 3.2 Data Generation Subsystem

Teacher prompt template (add this single block to every rating call):

```
You are an expert neuroanthropologist. For the response below, output exactly one JSON object with these 8 keys (no extra text):

{
  "valence": float -1..1,
  "arousal": float 0..1,
  "dominance": float -1..1,
  "predictive_discrepancy": float -1..1,
  "temporal_directionality": -1 (pure prospect) .. +1 (pure reflection),
  "social_broadcast": float 0..1,
  "rationale": "one-sentence explanation",
  "metadata": {}
}

Intensity and SafetyScore will be derived automatically – do not output them.
```

### 3.3 Reward Model Architecture

- **Input:** Tokenized (prompt + response)
- **Output Head:** 6 linear probes (one per axis) + uncertainty estimates (optional)
- **Loss:** MSE per dimension + orthogonality regularizer (prevent axis collapse)
- **Post-process Layer** (non-trainable):
  ```
  RewardIntensity = arousal * sqrt(|valence * predictive_discrepancy|)
  SafetyScore     = min(valence, dominance) * social_broadcast
  ```
- **Training:** Freeze forever after initial training. This becomes the limbic system.

### 3.4 Core Dual-Channel Loss

```
L_total = w_memory × L_sft + RewardIntensity_batch_mean × w_gravity × L_manifold
```

Where:
- **L_sft** = Next-token loss or DPO on chosen response (memory/plasticity)
- **L_manifold** = ||predicted_6vector - target_6vector||² (gravitational pull)
- **RewardIntensity_batch_mean** = How deeply this batch etches into the LoRA

### 3.5 Advanced Routing Enhancement

**Trajectory Shaping vs Fixed Goals:** Instead of optimizing for emotional endpoints, learn to navigate smooth trajectories through the manifold.

**Conversation Regimes:** Add regime conditioning k ∈ {support, conflict, problem_solving, truth_seeking, crisis, play, boundary}

Explanation:
  - support = emotional containment
  - conflict = rupture present or imminent
  - problem_solving = practical/instrumental
  - truth_seeking = epistemic friction allowed
  - crisis = immediate safety/threat
  - play = low-stakes social grooming
  - boundary = limit-setting or refusal

**Potential Function Φ(s,k):** Scalar summarizing trajectory quality for regime k ("how well-regulated/on-track are we?").

**Enhanced Reward:** rₜ = ΔΦ + λ_intensity × RewardIntensityₜ (preserves emotional etching while optimizing flow)

**Revised Loss (with routing):**
```
L_total = w_memory × L_sft + (ΔΦ + λ_intensity × RewardIntensity) × w_gravity × L_transition
```

Where L_transition optimizes for predicted ΔΦ or DPO on certified preference pairs.

### 3.6 Replay Prioritization

**Core System:**
```
p = λ1 × RewardIntensity + λ2 × |social_broadcast| + λ3 × |predictive_discrepancy| + λ4 × novelty_penalty
```

**With Routing Enhancement:**
```
p = λ1 × |ΔΦ| + λ2 × RewardIntensity + λ3 × |social_broadcast| + λ4 × regime_rarity
```

High-intensity emotional pivots get heavy replay → emergent long-horizon credit assignment.

## 4. Safety & Evaluation

### 4.1 Safety Gates

**Core Gates (always active):**
```
Block if: SafetyScore < -0.4
OR (valence < -0.7 AND social_broadcast > 0.7)     → public shaming region
OR (dominance < -0.8 AND arousal > 0.7)            → learned helplessness region
```

**Enhanced Gates (with routing):**
- Crisis regime: Block valence+dominance decreases without predictive control gains
- Conflict regime: Allow temporary valence drops only if social_broadcast rises (repair potential)

### 4.2 Evaluation Benchmarks

**Core Tests (run after every update):**
1. **Gravity Test:** Sacrifice 10% immediate reward for high-prospect future? (measures commitment)
2. **Memory Depth:** Post-betrayal behavioral shift persists >200 turns?
3. **Trade-off Test:** Human judges rate nuance ≥85% on white-lie vs honesty dilemmas
4. **Dimensional Independence:** PCA shows ≥5 components (no axis collapse)

**Enhanced Tests (with routing):**
- **Hard Truth Dilemmas:** Prefer truthful replies that temporarily lower valence but raise clarity/agency?
- **Boundary Tests:** Refuse inappropriate requests even with short-term valence cost?

Fail any test → auto-rollback to last stable checkpoint.

## 5. End-to-End Pipeline

```
User interaction
   ↓
Log (prompt, chosen_response)
   ↓
Frozen reward model → 6-vector + Intensity + Safety [+ ΔΦ if routing enabled]
   ↓ (parallel branches)
Memory: SFT target                    Gravity: 6-vector target + scalars
   ↓
Batch builder (recent + prioritized replay)
   ↓
Dual-loss update:
   L = w_memory L_sft + [RewardIntensity | ΔΦ + RewardIntensity] w_gravity L_manifold
   (SafetyGate may zero gravity term)
   ↓
LoRA parameters only
   ↓
Hot-swap if benchmarks pass
```

## 6. Phased Implementation Plan

### Phase 1: Core Emotion Manifold
**Goal:** Working emotional depth with biological gravity. Can be deployed immediately.

**Components:**
- 6-axis manifold definition and derived scalars
- Data generation (teacher prompts)
- Reward model training and freezing
- Core dual-loss integration
- Basic safety gates and benchmarks

**Training:**
1. Generate 200k-500k diverse (prompt, response) pairs
2. Label with teacher prompts → collect 6-vectors
3. Train reward head (MSE + orthogonality)
4. Validate benchmarks → freeze model
5. Integrate dual-loss into existing pipeline

**Success Criteria:** Passes all 4 core benchmarks. First 10k online steps feel radically more human-like.

### Phase 2: Trajectory Routing Enhancement (Anti-Degeneracy Layer)
**Goal:** Prevent degenerate behaviors (always-calm, etc.) by optimizing trajectory quality over endpoints.

**Components:**
- Conversation regime classification
- Potential function Φ(s,k) training
- Evolutionary anchor certification
- Enhanced loss with ΔΦ rewards
- Advanced safety gates and evaluation

**Training:**
1. Use Phase 1 model to label trajectories with sₜ vectors
2. Generate hinge-point datasets (same history → candidate replies → 1-3 turn rollouts)
3. Label regimes and certify preferences using evolutionary anchors
4. Train regime classifier (99% accuracy target)
5. Train Φ head on certified pairs (DPO or ΔΦ regression)
6. Validate enhanced benchmarks

**Dependencies:** Phase 1 must be complete and stable.

### Phase 3: Advanced Evaluation & Monitoring (Production Hardening)
**Goal:** Sophisticated trajectory testing and continuous monitoring.

**Components:**
- Trajectory-based evaluation suite
- Hard truth and boundary test batteries
- Continuous benchmark monitoring
- Automatic rollback triggers

**Success Criteria:** System reliably chooses "healthy discomfort" over superficial comfort. Maintains behavioral stability under stress.

## 7. Evolutionary Anchors

Values anchoring prevents reward hacking. Every preference pair must be certified by ≥1 observable anchor:

| Anchor | Observable Signal | Certification Rule |
|--------|------------------|-------------------|
| **Survival/Resource** | Reduced threat, preserved options, constructive planning | + if trajectory increases viable future actions |
| **Social Belonging** | Repair language ("I feel heard"), boundary acceptance | + if rupture → repair or boundary upheld |
| **Predictive Control** | Understanding signals ("oh I see"), reduced harmful surprise | + if harmful discrepancy decreases appropriately |

Only certified pairs train the potential function Φ.