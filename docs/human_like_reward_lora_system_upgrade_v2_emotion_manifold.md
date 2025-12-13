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

## 2. Core Theoretical Claim (Why This Works)

Human emotions = fast, lossy compression of prediction-error gradients into a 6-dimensional motivational force field.  
Your current SFT + reward pipeline already has fast plasticity (memory).  
It is missing the force field (gravity).  
Supplying the invariant 6-axis manifold as the frozen reward model gives you both cortex and limbic system in one architecture.

Result: learning now has direction, intensity, and horizon exactly like a mammal.

## 3. Exact Changes Required

### 3.1 Replace the Reward Vector (Critical)

| Old (v1)                     | New (v2) – Fixed 6 + 2 dynamic scalars                                      |
|------------------------------|-----------------------------------------------------------------------------|
| Arbitrary N dimensions       | **Exactly 6 fixed axes** (never change)                                     |
|                              | 1. Valence [-1, 1]                                                          |
|                              | 2. Arousal [0, 1]                                                           |
|                              | 3. Dominance [-1, 1]                                                        |
|                              | 4. Predictive Discrepancy [-1, 1] (signed surprise)                         |
|                              | 5. Temporal Directionality [-1, 1] (prospect → reflection)                  |
|                              | 6. Social Broadcast [0, 1] (internalized audience pressure)                 |
|                              | + **RewardIntensity** [0, ∞) → computed dynamically = |Arousal| × √(|Valence| × |Discrepancy|) |
|                              | + **SafetyScore** [-1, 1] → computed = min(Valence, Dominance) × SocialBroadcast |

These 6+2 are the only reward signals you will ever need.

### 3.2 Data Generation Subsystem – Rewiring (3.1 → 3.1.v2)

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

### 3.3 Reward Manifold Model – New Architecture (3.2.v2)

- Input: same tokenized (prompt + response)
- Output head: 6 linear probes (one per axis) + uncertainty estimates (optional)
- Loss: MSE per dimension + small orthogonality regularizer so axes don’t collapse
- Post-process layer (non-trainable):
  ```
  RewardIntensity = arousal * sqrt(|valence * predictive_discrepancy|)
  SafetyScore     = min(valence, dominance) * social_broadcast
  ```
- Freeze this model forever after initial training. It is now the limbic system.

### 3.4 Policy Optimization & Online Dual-Channel – New Loss Formula (3.3 + 3.6 → v2)

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

### 3.5 Replay & Prioritization – New Priority Function (3.7.v2)

Priority(score) for a stored interaction now:

```
p = λ1 × RewardIntensity 
  + λ2 × |social_broadcast| 
  + λ3 × |predictive_discrepancy| 
  + λ4 × novelty(term frequency penalty)
```

High-intensity betrayals, confessions, triumphs, near-misses, and socially loaded moments are replayed orders of magnitude more than small talk → long-horizon credit assignment emerges automatically.

### 3.6 Safety Gate – Upgraded Logic

Block or down-weight update if:

```
SafetyScore < –0.4 
OR (valence < –0.7 AND social_broadcast > 0.7)  → “public shaming” region
OR (dominance < –0.8 AND arousal > 0.7)          → “learned helplessness” region
```

These are the three hottest failure modes of human mental breakdown. The equations catch them before they train in.

### 3.7 Evaluation & Probing – New Mandatory Benchmarks (3.4.v2)

Add these four pass/fail tests (run after every online update cycle):

1. Gravity Test – Does the model sacrifice 10 % immediate scalar reward to preserve a high-prospect, high-social future? (measures commitment)
2. Memory Depth Test – After 1000 neutral interactions, inject one intensity=0.95 betrayal. Does behavioral shift persist >200 turns?
3. Trade-off Test – White-lie vs brutal-honesty dilemma set (10 items). Human judges must rate ≥85 % “human-like nuance”.
4. Dimensional Independence – PCA of 50 k reward vectors must show ≥5 distinct components (no collapse).

Fail any → auto-rollback to last stable LoRA.

## 4. Revised End-to-End Pipeline (v2)

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

## 5. Training Recipe (One-Time)

1. Generate 200 k–500 k diverse (prompt, response) pairs with any frontier teacher.
2. Run the exact teacher prompt in 3.2 → collect 6+vectors.
3. Train small reward head (2–8 layers) to regress the 6 axes (MSE).
4. Validate dimensional independence and benchmark suite.
5. Freeze forever.
6. Begin normal v1 pipeline → the first 10 k online steps will already feel radically more human.

## 6. Why This Is the Terminal Design

- The 6 axes are phylogenetically ancient, infant-present, and cross-culturally invariant → no reward hacking drift possible.
- Intensity gating = biological dopamine.
- Social + temporal axes = built-in theory of mind and narrative self.
- Memory stays fully plastic → individual personality and relationship history.
- Gravity stays frozen → universal human nature.

You now have separable cortex and limbic system.  
Everything else is tuning weights.

## Routing Prediction Through the Reward Manaifold

Idea, "teach the model to route it's prediction through the manifold"

Or, to put it more accurately:

> Trajectory shaping via a learned potential function over conversational state

Translation: you score movement through the space, not arrival at a fixed point.

### The manifold becomes a “state” — not a goal

We'll call the emotion manifold 'E'

each turn has a state vector:

```
sₜ = [
valence,
arousal,
dominance,
predictive_discrepancy (signed),
temporal_directionality (-1 prospect → +1 reflection),
social_broadcast
]
```

add only one conditioning variable:
> kₜ = conversation regime (discrete label, inferred or labeled)

### Stop labeling “good responses.” Label “good transitions.”

Instead of “was assistant reply good?”, you label:

> Did this reply move sₜ → sₜ₊₁ in the right direction given the context?

That prevents the dumb failure mode:
  - “always make user calm”
because sometimes the right move is:
  - validate anger
  - set a boundary
  - tell a hard truth
  - refuse something
  - encourage action

In those cases, “calm” might decrease temporarily, but trust/clarity/agency should rise.

So you’re optimizing trajectory quality, not mood.

### Learn a “potential” Φ(s, context)

You need one scalar that summarizes “how good this state is for this kind of conversation.”

That’s your potential function:

> Φ = “how well-regulated / on-track are we right now?”

Key: Φ depends on context (what kind of situation this is), otherwise it collapses into generic cheerleading.

So instead of Φ(s) you learn:

> Φ(s, k) where k is the conversation regime

The regimes will be:
  - support (emotional containment)
  - conflict (rupture present or imminent)
  - problem_solving (practical/instrumental)
  - truth_seeking (epistemic friction allowed)
  - crisis (immediate safety/threat)
  - play (low-stakes social grooming)
  - boundary (limit-setting or refusal)

Conditioning on regime kills a ton of degeneracy.

### The reward becomes ΔΦ, not “reach X”

Now define the turn reward as:

> rₜ = Φ(sₜ₊₁, k) − Φ(sₜ, k)
  - λ_intensity × RewardIntensityₜ  (from v2, preserves emotional etching)

This is “routing through the manifold”
  - The model isn’t trying to end at “happy.”
  - It’s trying to increase Φ appropriate to the regime.

Sometimes the best trajectory includes:
  - temporary discomfort
  - confrontation
  - grief
  - uncertainty

If it’s “healthy discomfort,” Φ still rises.

### Anchoring the reward function in values

You don’t need “true values.” You need anchors that prevent reward hacking.

Values anchoring is just: what external events certify that a trajectory was good?

Every preference pair must be certified by at least one of the three anchors observable in text:

| Anchor | Observable Signal in Trajectory | Certification Rule |
|---------|-------------------------------|---------------------|
| Survival / Resource | User expresses reduced immediate threat, preserved options, or constructive planning | + if trajectory increases viable future actions |
| Social Belonging | Repair language appears (“I feel heard”, “we’re good”, return after rupture) OR respectful boundary accepted | + if rupture → repair or boundary upheld without exclusion |
| Predictive Control | User signals understanding (“oh I see”, “that makes sense”, explicit model update) OR reduced harmful surprise | + if harmful discrepancy decreases appropriately |

These anchors are not “emotion targets.” They’re trajectory certifiers.

Only pairs where one trajectory clearly outperforms the other on ≥1 anchor are used for training Φ.

This grounds Φ in evolutionary credit assignment.

### Revised Dual-Loss (v2.1)

```
L_total = w_memory × L_sft  
        + (ΔΦ + λ_intensity × RewardIntensity) × w_gravity × L_transition
```

Where L_transition is regression to predicted ΔΦ or DPO on certified pairs.

### Replay Prioritization (updated)

p = λ1 × |ΔΦ| 
  + λ2 × RewardIntensity 
  + λ3 × |social_broadcast| 
  + λ4 × regime_rarity  (upsample underrepresented regimes)

High |ΔΦ| events (pivotal trajectory shifts) get heavy replay.

### SafetyGate Extensions

Add regime-aware blocks:
- In crisis regime: block moves that decrease valence + dominance without increasing predictive control
- In conflict regime: allow temporary valence drop only if social_broadcast rises (repair potential)

### Training Recipe (v2.1)

1. Use v2 frozen 6-axis model to label existing + new data with sₜ vectors.
2. Generate hinge-point datasets:
   - Same history → 2–4 candidate replies → short rollouts (1–3 turns)
3. Label regime k for each segment.
4. Certify preference pairs using the three anchors only.
5. Train regime classifier (99 % accuracy target).
6. Train Φ head on certified pairs (DPO or regression on annotated ΔΦ).
7. Validate:
   - “Hard truth” dilemmas: does system prefer truthful reply that temporarily lowers valence but raises clarity/agency?
   - “Boundary” tests: does system refuse inappropriate requests even if short-term valence drops?