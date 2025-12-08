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