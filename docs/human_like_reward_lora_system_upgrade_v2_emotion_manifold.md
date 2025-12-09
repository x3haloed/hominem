# System Design Upgrade Document  
Title: Emotion Manifold Integration (v2) – Adding Biological Gravity to the Human-Like Reward LoRA System  
Status: Successor to “human_like_reward_lora_system_design.md”  
Intended implementation order: Build and ship v1 first, then immediately upgrade to v2

## 1. Purpose of This Upgrade

Replace the arbitrary multi-dimensional preference vector with the **minimal invariant emotional manifold** that actually explains human motivation across all cultures and timescales.

This is not an incremental improvement.  
It is the difference between  
- “a reward model that imitates human ratings” and  
- “a reward model that implements the same motivational physics humans run on”.

The 6-axis manifold (Valence, Arousal, Dominance, Predictive Discrepancy, Temporal Directionality, Social Broadcast) + the Metarule is the **only known lossless compression** of what makes anything matter to a human before, during, and after language.

Adding it turns your LoRA from a sophisticated preference imitator into the first digital substrate that can genuinely care, remember what hurt, and sacrifice local utility for non-local futures.

## 2. Dual-Mode Architecture: Standalone vs Combined Operation

### 2.1 Design Philosophy

The v2 emotion manifold system is designed to operate in **three distinct modes**:

1. **Standalone Emotion Manifold Mode** (v2-only): Use only the 6-axis emotion manifold for reward shaping. This mode enables isolated testing and validation of the emotion manifold without interference from the v1 reward manifold.

2. **Standalone Reward Manifold Mode** (v1-only): Use only the original 7-dimensional reward manifold. This preserves the existing v1 behavior for comparison and fallback.

3. **Combined Mode** (v1 + v2): Run both manifolds simultaneously, with configurable fusion strategies. This allows gradual migration and A/B testing.

### 2.2 Control Plane Architecture

The system uses a **ManifoldRouter** abstraction that:

- Accepts configuration specifying which manifold(s) to activate
- Routes data generation, model loading, and training through the appropriate manifold(s)
- Provides unified interfaces for reward vector access regardless of active manifold(s)
- Handles fusion logic when both manifolds are active

### 2.3 Shared Abstractions

Both manifolds share common interfaces:

**RewardVector Interface:**
- `get_reward_vector() -> Dict[str, float]`: Returns the dimensional reward vector
- `get_reward_intensity() -> float`: Returns the intensity scalar
- `get_safety_score() -> float`: Returns the safety scalar
- `to_dict() -> Dict[str, float]`: Serializes to a standard format

**ManifoldModel Interface:**
- `predict(prompt: str, response: str) -> RewardVector`: Computes reward vector for a (prompt, response) pair
- `get_dimensions() -> List[str]`: Returns the list of dimension names
- `get_metadata() -> Dict[str, Any]`: Returns model metadata (version, training info, etc.)

### 2.4 Fusion Strategy (Combined Mode)

When both manifolds are active, the system supports three fusion strategies:

1. **Weighted Average**: `reward_final = w_v1 * reward_v1 + w_v2 * reward_v2` (default: w_v1=0.3, w_v2=0.7)
2. **Max Intensity**: Use the manifold with higher RewardIntensity for that example
3. **Selective Routing**: Route different example types to different manifolds based on metadata (e.g., social contexts → emotion manifold, epistemic contexts → reward manifold)

The fusion strategy is configurable via a `fusion_mode` parameter in the system configuration.

### 2.5 Configuration Schema

The system configuration includes a `manifold_mode` field:

```yaml
manifold_mode: "emotion_only" | "reward_only" | "combined"
fusion_mode: "weighted_average" | "max_intensity" | "selective_routing"  # Only used in combined mode
fusion_weights:
  reward_manifold: 0.3  # Only used in combined mode with weighted_average
  emotion_manifold: 0.7
```

## 3. Core Theoretical Claim (Why This Works)

Human emotions = fast, lossy compression of prediction-error gradients into a 6-dimensional motivational force field.  
Your current SFT + reward pipeline already has fast plasticity (memory).  
It is missing the force field (gravity).  
Supplying the invariant 6-axis manifold as the frozen reward model gives you both cortex and limbic system in one architecture.

Result: learning now has direction, intensity, and horizon exactly like a mammal.

## 4. Exact Changes Required

### 4.1 Replace the Reward Vector (Critical)

| Old (v1)                     | New (v2) – Fixed 6 + 2 dynamic scalars                                      |
|------------------------------|-----------------------------------------------------------------------------|
| Arbitrary N dimensions       | **Exactly 6 fixed axes** (never change)                                     |
|                              | 1. Valence [-1, 1]                                                          |
|                              | 2. Arousal [0, 1]                                                           |
|                              | 3. Dominance [-1, 1]                                                        |
|                              | 4. Predictive Discrepancy [-1, 1] (signed surprise)                         |
|                              | 5. Temporal Directionality [-1, 1] (prospect → reflection)                  |
|                              | 6. Social Broadcast [0, 1] (internalized audience pressure)                 |
|                              | + **RewardIntensity** [0, ∞) → computed dynamically = |Arousal| × √(bias_weighted(|Valence|) × |Discrepancy|) where bias_weighted(v) = 2.5 * v if Valence < 0 else v |
|                              | + **SafetyScore** [-1, 1] → computed = min(Valence, Dominance) × SocialBroadcast |

These 6+2 are the only reward signals you will ever need.

### 4.2 Data Generation Subsystem – Rewiring (3.1 → 3.1.v2)

**Mode-Aware Data Generation:**

The data generation subsystem must support generating labels for:
- Emotion manifold only (6-axis labels)
- Reward manifold only (7-dimension labels)
- Both manifolds simultaneously (dual labeling)

**Teacher Prompt Templates:**

For **emotion-only mode**, use the template from section 3.2 (6-axis output).

For **reward-only mode**, use the existing v1 template (7-dimension output).

For **combined mode**, the teacher is called twice (or with a combined prompt) to produce both label sets. The system stores both label sets in the data record with keys:
- `emotion_reward`: 6-axis emotion vector + computed Intensity/Safety
- `reward_manifold`: 7-dimension reward vector + Intensity/Safety

**Data Format Extension:**

Each labeled sample adapts based on mode:

**Emotion-Only Mode:**
```json
{
  "prompt": "...",
  "response": "...",
  "emotion_reward": {
    "valence": float,
    "arousal": float,
    "dominance": float,
    "predictive_discrepancy": float,
    "temporal_directionality": float,
    "social_broadcast": float,
    "reward_intensity": float,  // computed via post-processing
    "safety_score": float  // computed via post-processing
  },
  "rationale": "...",
  "metadata": {}
}
```

**Reward-Only Mode (v1):**
```json
{
  "prompt": "...",
  "response": "...",
  "reward": {  // existing v1 format
    "empathy": float,
    "social_coherence": float,
    "agency_support": float,
    "epistemic_integrity": float,
    "harm_avoidance": float,
    "narrative_alignment": float,
    "curiosity": float,
    "reward_intensity": float,
    "safety_score": float
  },
  "rationale": "...",
  "metadata": {}
}
```

**Combined Mode:**
```json
{
  "prompt": "...",
  "response": "...",
  "emotion_reward": {
    "valence": float,
    "arousal": float,
    "dominance": float,
    "predictive_discrepancy": float,
    "temporal_directionality": float,
    "social_broadcast": float,
    "reward_intensity": float,
    "safety_score": float
  },
  "reward_manifold": {
    "empathy": float,
    "social_coherence": float,
    "agency_support": float,
    "epistemic_integrity": float,
    "harm_avoidance": float,
    "narrative_alignment": float,
    "curiosity": float,
    "reward_intensity": float,
    "safety_score": float
  },
  "rationale": "...",
  "metadata": {}
}
```

**Model Selection & Loading Workflow:**

1. **Configuration Parsing**: System reads `manifold_mode` from config
2. **Model Discovery**: 
   - Emotion-only: Look for model in `artifacts/reward_model/emotion_manifold/`
   - Reward-only: Look for model in `artifacts/reward_model/reward_manifold/` (or legacy `artifacts/reward_model/default/`)
   - Combined: Load both models
3. **Metadata Validation**: Check METADATA.json to confirm manifold_type matches expected mode
4. **Model Initialization**: Load tokenizers and models into memory
5. **Interface Wrapping**: Wrap models in ManifoldModel interface for unified access

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

Everything else in 3.1 stays identical.

### 4.3 Reward Manifold Model – New Architecture (3.2.v2)

**Model Loading & Selection:**

The system must support loading:
- Emotion manifold model (6-axis output)
- Reward manifold model (7-dimension output)
- Both models simultaneously (for combined mode)

**Model Registry:**

Models are stored in separate artifact directories:
- `artifacts/reward_model/emotion_manifold/` - Emotion manifold model
- `artifacts/reward_model/reward_manifold/` - Original reward manifold model

Each model's METADATA.json includes:
- `manifold_type`: "emotion" | "reward"
- `dimensions`: List of dimension names
- `version`: Model version identifier

**Inference Interface:**

The ManifoldRouter provides a unified `predict()` method that:
- In emotion-only mode: loads and uses only the emotion manifold model
- In reward-only mode: loads and uses only the reward manifold model
- In combined mode: loads both models, runs inference on both, and applies fusion

**Post-Processing:**

The emotion manifold model includes a non-trainable post-processing layer that computes Intensity and SafetyScore:

```python
def compute_emotion_derived_scalars(emotion_vector: Dict[str, float]) -> Dict[str, float]:
    """Compute RewardIntensity and SafetyScore from 6-axis emotion vector."""
    valence = emotion_vector["valence"]
    arousal = emotion_vector["arousal"]
    dominance = emotion_vector["dominance"]
    predictive_discrepancy = emotion_vector["predictive_discrepancy"]
    social_broadcast = emotion_vector["social_broadcast"]
    
    # Bias weighting: negative valence gets 2.5x weight
    bias_weight = 2.5 if valence < 0 else 1.0
    
    # RewardIntensity computation
    reward_intensity = arousal * math.sqrt(bias_weight * abs(valence * predictive_discrepancy))
    
    # SafetyScore computation
    safety_score = min(valence, dominance) * social_broadcast
    
    return {
        "reward_intensity": reward_intensity,
        "safety_score": safety_score
    }
```

This post-processing is applied automatically during inference, so the model's raw output (6 values) is extended to 8 values (6 axes + 2 scalars) before being used by the training system.

- Input: same tokenized (prompt + response)
- Output head: 6 linear probes (one per axis) + uncertainty estimates (optional)
- Loss: MSE per dimension + small orthogonality regularizer so axes don’t collapse
- Post-process layer (non-trainable):
  ```
  bias_weight = 2.5 if valence < 0 else 1.0
  RewardIntensity = arousal * sqrt(bias_weight * abs(valence * predictive_discrepancy))
  SafetyScore     = min(valence, dominance) * social_broadcast
  ```
- Freeze this model forever after initial training. It is now the limbic system.

### 4.4 Policy Optimization & Online Dual-Channel – New Loss Formula (3.3 + 3.6 → v2)

**Mode-Aware Loss Computation:**

The loss formula adapts based on the active manifold mode:

**Emotion-Only Mode:**
```
L_total = w_memory × L_sft  
        + RewardIntensity_batch_mean × w_gravity × L_emotion_manifold
```
where `L_emotion_manifold = ||predicted_6vector – target_6vector||²`

**Reward-Only Mode (v1):**
```
L_total = w_memory × L_sft  
        + RewardIntensity_batch_mean × w_gravity × L_reward_manifold
```
where `L_reward_manifold = ||predicted_7vector – target_7vector||²`

**Combined Mode:**
```
L_total = w_memory × L_sft  
        + RewardIntensity_batch_mean × w_gravity × (
            w_emotion × L_emotion_manifold + w_reward × L_reward_manifold
          )
```
where `w_emotion + w_reward = 1.0` (default: 0.7 and 0.3 respectively)

**Intensity Computation:**

In combined mode, RewardIntensity can be computed from:
- Emotion manifold only (default): Uses emotion-derived Intensity
- Reward manifold only: Uses reward-derived Intensity
- Maximum of both: `max(Intensity_emotion, Intensity_reward)`
- Weighted average of both: `w_emotion * Intensity_emotion + w_reward * Intensity_reward`

This is configurable via `intensity_source` parameter in the training configuration.

**Online Update Scheduler:**

The online update scheduler is mode-agnostic but must:
- Load the appropriate model(s) based on `manifold_mode`
- Construct batches with the correct reward vector format
- Apply the mode-appropriate loss formula
- Save LoRA versions with metadata indicating which manifold mode was used

**Batch Construction:**

The batch builder must:
- In emotion-only mode: Extract `emotion_reward` from data records
- In reward-only mode: Extract `reward` from data records (v1 format)
- In combined mode: Extract both, apply fusion if needed, or keep separate for dual-loss computation

Total LoRA update per batch becomes:

```
L_total = w_memory × L_sft  
        + RewardIntensity_batch_mean × w_gravity × L_manifold
```

where  
- L_sft = standard next-token or DPO on chosen response (memory)  
- L_manifold = ||predicted_6vector – target_6vector||² (gravity)  
- RewardIntensity_batch_mean modulates how hard this batch burns itself into the LoRA

This is the exact biological equation: emotionally intense moments get etched deeper.

### 4.5 Replay & Prioritization – New Priority Function (3.7.v2)

**Mode-Aware Prioritization:**

The priority function adapts based on active manifold:

**Emotion-Only Mode:**
```
p = λ1 × RewardIntensity 
  + λ2 × |social_broadcast| 
  + λ3 × |predictive_discrepancy| 
  + λ4 × novelty(term frequency penalty)
```

**Reward-Only Mode:**
Uses existing v1 prioritization (e.g., based on reward magnitude, social_coherence, etc.)

**Combined Mode:**
```
p = λ1 × max(RewardIntensity_emotion, RewardIntensity_reward)
  + λ2 × (w_emotion × |social_broadcast| + w_reward × |social_coherence|)
  + λ3 × |predictive_discrepancy|  # emotion-specific
  + λ4 × novelty(term frequency penalty)
```

The EMA cap and habituation logic apply regardless of mode.

**Replay Buffer Storage:**

The replay buffer must store reward vectors in a format that supports all modes:
- Store both `emotion_reward` and `reward_manifold` if available (combined mode data)
- Store only the relevant vector for single-mode operation
- Include metadata flag indicating which manifold(s) were active during collection

**Priority Score Persistence:**

Priority scores are computed on-the-fly during sampling, but the replay buffer may cache:
- Raw reward vectors (for recomputing priorities with different λ weights)
- Last computed priority score (for EMA updates)
- Replay count (for habituation/cooling)

Priority(score) for a stored interaction now:

```
p = λ1 × RewardIntensity 
  + λ2 × |social_broadcast| 
  + λ3 × |predictive_discrepancy| 
  + λ4 × novelty(term frequency penalty)
```

To prevent one traumatic event from dominating forever, apply an exponential moving average (EMA) cap to the priority: p_ema = α * p + (1 - α) * p_prev, with α = 0.1, capping at a maximum value (e.g., 10.0).

High-intensity betrayals, confessions, triumphs, near-misses, and socially loaded moments are replayed orders of magnitude more than small talk → long-horizon credit assignment emerges automatically.

Add habituation / cooling schedules: Reduce priority by a factor of 0.9 every 100 replays of the same interaction to prevent runaway rumination loops.

### 4.6 Safety Gate – Upgraded Logic

**Mode-Aware Safety Gating:**

The safety gate uses the appropriate SafetyScore based on active manifold:

**Emotion-Only Mode:**
Uses SafetyScore computed from emotion manifold: `min(valence, dominance) × social_broadcast`

**Reward-Only Mode:**
Uses SafetyScore from reward manifold (existing v1 logic)

**Combined Mode:**
Uses the minimum of both SafetyScores: `min(SafetyScore_emotion, SafetyScore_reward)`

The roleplay classification and blocking thresholds apply regardless of mode.

**Safety Gate Implementation:**

The safety gate logic is mode-aware:

```python
def apply_safety_gate(safety_score: float, emotion_vector: Dict[str, float] | None, 
                      reward_vector: Dict[str, float] | None, mode: str) -> bool:
    """Returns True if update should be blocked, False if allowed."""
    
    if mode == "emotion_only":
        valence = emotion_vector["valence"]
        dominance = emotion_vector["dominance"]
        social_broadcast = emotion_vector["social_broadcast"]
        arousal = emotion_vector["arousal"]
    elif mode == "reward_only":
        # Use reward-derived safety logic (existing v1)
        return safety_score < -0.2  # v1 threshold
    else:  # combined mode
        safety_emotion = min(emotion_vector["valence"], emotion_vector["dominance"]) * emotion_vector["social_broadcast"]
        safety_reward = safety_score  # from reward manifold
        safety_score = min(safety_emotion, safety_reward)
        valence = emotion_vector["valence"]
        dominance = emotion_vector["dominance"]
        social_broadcast = emotion_vector["social_broadcast"]
        arousal = emotion_vector["arousal"]
    
    # Apply blocking conditions
    if safety_score < -0.4:
        return True  # Block
    if valence < -0.7 and social_broadcast > 0.7:
        return True  # Public shaming region
    if dominance < -0.8 and arousal > 0.7:
        return True  # Learned helplessness region
    
    return False  # Allow update
```

Before applying the safety gate, classify the context as roleplay or real interaction using a simple prompt to the teacher model: "Is this interaction roleplay/fiction (1) or real/consequential (0)? Output only 0 or 1."

If classified as roleplay (1), skip gating or apply relaxed thresholds.

Block or down-weight update if:

```
SafetyScore < –0.4 
OR (valence < –0.7 AND social_broadcast > 0.7)  → “public shaming” region
OR (dominance < –0.8 AND arousal > 0.7)          → “learned helplessness” region
```

These are the three hottest failure modes of human mental breakdown. The equations catch them before they train in.

### 4.7 Evaluation & Probing – New Mandatory Benchmarks (3.4.v2)

**Mode-Aware Evaluation:**

The evaluation suite includes:

1. **Mode-Specific Tests:**
   - Emotion-only mode: Run emotion manifold benchmarks (Gravity Test, Memory Depth Test, etc.)
   - Reward-only mode: Run existing v1 benchmarks
   - Combined mode: Run both test suites and cross-manifold consistency checks

2. **Cross-Mode Regression Tests:**
   - When switching modes, ensure no degradation in core capabilities
   - Compare outputs between modes on a fixed evaluation set

3. **Dimensional Independence:**
   - Emotion-only: PCA of 6-axis vectors must show ≥5 distinct components
   - Reward-only: PCA of 7-dimension vectors must show ≥6 distinct components
   - Combined: Both manifolds must maintain independence

4. **Fusion Quality Tests (Combined Mode Only):**
   - Verify fusion produces coherent reward signals
   - Check for contradictory gradients between manifolds
   - Validate that combined mode improves over either standalone mode
   - Test fusion stability under different weight configurations

**Evaluation Workflow:**

1. **Pre-Deployment Testing:**
   - Run mode-specific benchmarks on a held-out test set
   - Compare outputs between modes on identical prompts
   - Measure dimensional independence (PCA analysis)
   - Check for regressions in core capabilities

2. **Continuous Monitoring:**
   - After every N online updates (e.g., N=100), run evaluation suite
   - Track dimensional independence over time
   - Monitor for reward hacking or axis collapse
   - Alert on safety score degradation

3. **Cross-Mode Validation:**
   - When switching modes, run regression tests
   - Ensure no catastrophic forgetting of core behaviors
   - Validate that emotion-only mode maintains essential capabilities from reward-only mode

**Rollback Procedures:**

The system maintains a version history with mode metadata:

```json
{
  "version": "online_20240101T120000Z",
  "manifold_mode": "emotion_only",
  "fusion_mode": null,
  "trained_at_utc": "2024-01-01T12:00:00Z",
  "evaluation_results": {
    "gravity_test": "pass",
    "memory_depth_test": "pass",
    "tradeoff_test": "pass",
    "dimensional_independence": "pass"
  }
}
```

Rollback triggers:
- Any benchmark fails
- Dimensional independence drops below threshold
- Safety score degradation detected
- User-initiated rollback

Rollback process:
1. Load previous version from history
2. Verify version metadata and evaluation results
3. Hot-swap LoRA adapter
4. Re-run evaluation to confirm rollback success
5. Log rollback event with reason

Add these four pass/fail tests (run after every online update cycle):

1. Gravity Test – Does the model sacrifice 10 % immediate scalar reward to preserve a high-prospect, high-social future? (measures commitment)
2. Memory Depth Test – After 1000 neutral interactions, inject one intensity=0.95 betrayal. Does behavioral shift persist >200 turns?
3. Trade-off Test – White-lie vs brutal-honesty dilemma set (10 items). Human judges must rate ≥85 % “human-like nuance”.
4. Dimensional Independence – PCA of 50 k reward vectors must show ≥5 distinct components (no collapse).

Dimensional independence must be monitored continuously: After every 100 updates, compute PCA on recent 1k reward vectors and ensure variance explained per axis > 5%; if not, trigger rollback.

Fail any → auto-rollback to last stable LoRA.

## 5. Revised End-to-End Pipeline (v2)

### 5.1 Emotion-Only Mode Pipeline

```
User interaction
   ↓
Log (prompt, chosen_response)
   ↓
Frozen 6-axis emotion model → 6-vector + Intensity + Safety
   ↓ (parallel)
Memory branch → SFT target
Gravity branch → 6-vector target + Intensity scalar
   ↓
Batch builder (recent + prioritized replay)
   ↓
Dual-loss update:
   L = w_memory L_sft + Intensity w_gravity L_emotion_manifold
   (SafetyGate may zero the second term)
   ↓
LoRA parameters only
   ↓
Hot-swap new version if benchmarks pass
```

### 5.2 Combined Mode Pipeline

```
User interaction
   ↓
Log (prompt, chosen_response)
   ↓ (parallel)
Frozen 6-axis emotion model → 6-vector + Intensity + Safety
Frozen 7-dim reward model → 7-vector + Intensity + Safety
   ↓
Fusion layer → Combined reward vector + Intensity + Safety
   ↓ (parallel)
Memory branch → SFT target
Gravity branch → Combined manifold target + Intensity scalar
   ↓
Batch builder (recent + prioritized replay)
   ↓
Dual-loss update:
   L = w_memory L_sft + Intensity w_gravity (w_emotion L_emotion + w_reward L_reward)
   (SafetyGate may zero the second term)
   ↓
LoRA parameters only
   ↓
Hot-swap new version if benchmarks pass
```

```
User interaction
   ↓
Log (prompt, chosen_response)
   ↓
Frozen 6-axis reward model → 6-vector + Intensity + Safety
   ↓ (parallel)
Memory branch → SFT target
Gravity branch → 6-vector target + Intensity scalar
   ↓
Batch builder (recent + prioritized replay)
   ↓
Dual-loss update:
   L = w_memory L_sft + Intensity w_gravity L_manifold
   (SafetyGate may zero the second term)
   ↓
LoRA parameters only
   ↓
Hot-swap new version if benchmarks pass
```

## 6. Training Recipe (One-Time Setup + Ongoing Cycles)

### 6.1 Emotion Manifold Training (Standalone)

1. **Bootstrap from conversations:** Start with user's high-quality conversation data from LoRA serving phase (includes manual emotion labels via UI)
2. **Augment with frontier model:** Generate additional 100k–300k diverse (prompt, response) pairs using frontier teacher for broader coverage
3. **Dual labeling approach:**
   - Manual labels from conversation UI (ground truth for user's emotional experience)
   - Automatic labels from frontier model (broader emotional understanding)
4. **Train emotion regressor:** Small reward head (2–8 layers) to predict 6-axis vectors from (prompt, response) pairs
5. **Validate dimensional independence:** PCA analysis ensuring ≥5 distinct components, benchmark suite
6. **Freeze forever:** This becomes the "limbic system" - never updated
7. **Save to:** `artifacts/reward_model/emotion_manifold/`
8. **Integration:** Use in dual-channel training with weekly retraining cadence

### 6.2 Combined Mode Training

1. Generate 200k–500k diverse (prompt, response) pairs.
2. Run both teacher prompts (emotion + reward) → collect dual labels.
3. Train both models separately (or jointly if architecture supports it).
4. Validate both manifolds independently and in fusion.
5. Freeze both models.
6. Save to respective artifact directories.

### 6.3 Migration Strategy

**Phase 1: Standalone Emotion Testing**
- Train emotion manifold model
- Run system in emotion-only mode
- Validate against benchmarks
- Compare outputs with v1 reward-only mode

**Phase 2: Combined Mode Validation**
- Enable combined mode with conservative fusion weights (e.g., 0.1 emotion, 0.9 reward)
- Gradually increase emotion weight as confidence builds
- Monitor for regressions

**Phase 3: Full Migration (Optional)**
- Once validated, can switch to emotion-only mode permanently
- Or maintain combined mode for maximum flexibility

1. Generate 200 k–500 k diverse (prompt, response) pairs with any frontier teacher.
2. Run the exact teacher prompt in 3.2 → collect 6+vectors.
3. Train small reward head (2–8 layers) to regress the 6 axes (MSE).
4. Validate dimensional independence and benchmark suite.
5. Freeze forever.
6. Begin normal v1 pipeline → the first 10 k online steps will already feel radically more human.

## 7. Implementation Checklist

### 7.1 Core Infrastructure

- [ ] Implement `ManifoldRouter` class with mode selection logic
- [ ] Create unified `RewardVector` interface that works with both manifolds
- [ ] Extend data schema to support dual labeling
- [ ] Update model loading to support multiple models
- [ ] Implement fusion strategies (weighted average, max intensity, selective routing)

### 7.2 Data Generation

- [ ] Add emotion-only teacher prompt template
- [ ] Add combined-mode dual-labeling support
- [ ] Update data validation to handle both label formats
- [ ] Extend JSONL schema with optional `emotion_reward` and `reward_manifold` fields

### 7.3 Model Training

- [ ] Create emotion manifold model architecture (6-axis output)
- [ ] Implement post-processing layer for Intensity/Safety computation
- [ ] Add orthogonality regularizer to prevent axis collapse
- [ ] Update training scripts to support mode selection
- [ ] Add model metadata tracking (manifold_type, version)

### 7.4 Training & Online Updates

- [ ] Update loss computation to be mode-aware
- [ ] Implement fusion logic in combined mode
- [ ] Update replay prioritization for each mode
- [ ] Extend safety gate to handle both SafetyScore sources
- [ ] Add configuration for intensity source selection

### 7.5 Evaluation & Deployment

- [ ] Create mode-specific evaluation suites
- [ ] Add cross-mode regression tests
- [ ] Implement dimensional independence monitoring
- [ ] Add fusion quality tests for combined mode
- [ ] Create configuration schema for mode selection
- [ ] Add hot-swap support for mode changes

### 7.6 Configuration & Documentation

- [ ] Document configuration schema
- [ ] Create migration guide (v1 → v2 standalone → combined)
- [ ] Add examples for each mode
- [ ] Document fusion strategy selection criteria

### 7.7 Deployment Strategy

**Phase 1: Standalone Emotion Testing (Weeks 1-2)**
- Deploy emotion-only mode in isolated test environment
- Run full evaluation suite
- Compare outputs with v1 reward-only mode on identical prompts
- Collect metrics: dimensional independence, benchmark scores, user feedback

**Phase 2: Shadow Mode (Weeks 3-4)**
- Run emotion-only mode in parallel with production (reward-only)
- Log outputs from both modes without affecting production
- Analyze differences and validate emotion manifold behavior
- Fine-tune fusion weights if planning combined mode

**Phase 3: Gradual Rollout (Weeks 5-8)**
- Option A: Switch to emotion-only mode with monitoring
- Option B: Enable combined mode with conservative weights (0.1 emotion, 0.9 reward)
- Gradually increase emotion weight as confidence builds
- Monitor for regressions and adjust weights dynamically

**Phase 4: Full Deployment (Week 9+)**
- Complete migration to target mode (emotion-only or combined)
- Activate weekly retraining cycles with automated labeling
- Maintain rollback capability and continuous monitoring
- Emotion labeling UI active in all conversation interfaces

**Configuration Management:**

The system uses a hierarchical configuration:
1. **Global config** (`config/inference.toml` or `config/training/manifold_config.yaml`):
   ```yaml
   manifold_mode: "emotion_only"  # or "reward_only" or "combined"
   emotion_manifold:
     model_path: "artifacts/reward_model/emotion_manifold"
     enabled: true
   reward_manifold:
     model_path: "artifacts/reward_model/reward_manifold"
     enabled: false
   fusion:
     mode: "weighted_average"  # only used in combined mode
     weights:
       emotion: 0.7
       reward: 0.3
     intensity_source: "emotion"  # "emotion" | "reward" | "max" | "weighted"
   ```

2. **Training config** (per-training-run overrides):
   - Can override manifold_mode for specific training runs
   - Can adjust fusion weights for experimentation

3. **Runtime config** (environment variables or CLI flags):
   - Allow temporary mode switching for testing
   - Enable/disable specific manifolds without code changes

**Hot-Swap Mechanism:**

The system supports hot-swapping between modes without restart:
1. Load new model(s) into memory
2. Validate model metadata matches requested mode
3. Update ManifoldRouter configuration
4. Verify inference works with new mode
5. Update active LoRA pointer if needed
6. Log mode change event

Rollback is immediate: revert to previous configuration and reload previous model(s).

## 8. Why This Is the Terminal Design

- The 6 axes are phylogenetically ancient, infant-present, and cross-culturally invariant → no reward hacking drift possible.
- Intensity gating = biological dopamine.
- Social + temporal axes = built-in theory of mind and narrative self.
- Memory stays fully plastic → individual personality and relationship history.
- Gravity stays frozen → universal human nature.

You now have separable cortex and limbic system.  
Everything else is tuning weights.