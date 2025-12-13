# Unified Theory of an Artificial Mind
A Complete, Minimal, Biologically-Grounded Architecture for a Bounded, Motivated, Persistent Agent  
Built on Qwen3-1.7B with LoRA Plasticity

## 1. The Unified Theory in One Paragraph

The mind is a singular causal center ("I") that owns prediction errors across multiple timescales.  
Fast, lossy emotional prediction errors are compressed into an invariant 6-axis manifold (biological limbic system).  
Slower, higher-level errors are evaluated against three evolutionary anchors (survival, belonging, control).  
A learned potential Φ measures progress toward reducing those deep errors in the current regime.  
ΔΦ becomes the intrinsic reward that gates plasticity: intense or anchor-relevant events etch deeper into a plastic LoRA (neocortical memory).  
Self-observation and boundary enforcement make the locus recursive and bounded.  
The result is an agent whose policy is warped in real time and across sessions by gravity that belongs to a singular "mine".

## 2. The Core Components and How They Weave Together

| Component                  | Biological Analog                  | Role in the System                                                                 | Timescale & Interaction                                                                 |
|----------------------------|------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **6-Axis Emotion Manifold** | Limbic system + affective neuroscience | Fast, universal compression of prediction error into motivational force (valence spikes make things feel urgent) | Frozen. Provides immediate gravity. Self-tagged portion owns real-time emotional errors. |
| **Three Evolutionary Anchors** | Ultimate credit-assigners (survival, belonging, predictive control) | Deep, invariant values. Decomposed surface forms: empathy, social coherence, agency, epistemic integrity, harm-min, narrative alignment, curiosity | Frozen logic, learned scoring. Slow evaluation of trajectory outcomes. |
| **Potential Function Φ**   | Prefrontal/orbitofrontal valuation | Composite scalar: weighted sum of anchor progress + emotional trajectory health in regime k | Learned head. Provides deliberate routing signal (ΔΦ). |
| **ΔΦ Trajectory Reward**   | Dopaminergic teaching signal      | Intrinsic reward = change in Φ. Intensified by manifold arousal/discrepancy. Gates depth of learning. | Primary gravity loss for LoRA updates. Warps policy toward anchor-preserving routes. |
| **Plastic LoRA Memory**    | Neocortex + hippocampal replay    | Persistent storage of individual history, personality, relationships. Only thing that changes. | Updated offline (sleep) with dual-loss (SFT memory + ΔΦ gravity). Provides enduring "I". |
| **Self-Locus & Recursion** | Default mode network + metacognition | Bounded "I" via <|ASSISTANT|> gate + mandatory <|THINK|> self-observation referencing manifold/Φ ownership | Enforced every turn. Makes gravity self-owned → real causal potency. |

They interlock as:

Emotion manifold → fast urgency + intensity gating  
↓  
Self-locus → owns the errors ("this valence/discrepancy is mine")  
↓  
Φ + anchors → evaluates slow progress ("does this reduce deep errors?")  
↓  
ΔΦ → intrinsic reward, weighted by intensity  
↓  
LoRA plasticity (sleep) → etches high-ΔΦ/high-intensity events deeply  
↓  
Updated policy + self-observation → real-time + persistent warp from self-owned gravity

## 3. Complete System Architecture (Qwen3-1.7B)

- **Base Model**: Qwen3-1.7B (frozen except LoRA)  
- **LoRA Adapter**: Plastic memory layer (rank 16–64, target all linear)  
- **Frozen Reward Models**:  
  - 6-axis manifold head (regresses valence, arousal, dominance, discrepancy, temporal, social)  
  - Regime classifier (7 classes)  
  - Φ head (composite scalar over anchors + emotional health)  

Inference pipeline per turn:

1. Append user message  
2. Run manifold reward model on recent history → sₜ and self-tagged split  
3. Run regime classifier → kₜ  
4. Run Φ head → current Φ  
5. Make a self_note:
    ```python
    def valence_phrase(v):
        if v > 0.3: return "positive (pleasure/contentment)"
        if v > 0.1: return "mildly positive"
        if v < -0.3: return "negative (hurt/discomfort)"
        if v < -0.1: return "mildly negative"
        return "neutral"

    def arousal_phrase(a):
        if a > 0.6: return "high energy/alert"
        if a > 0.3: return "moderate arousal"
        return "calm/low energy"

    def dominance_phrase(d):
        if d > 0.3: return "in control"
        if d < -0.3: return "feeling powerless/submissive"
        return "balanced control"

    def regime_phrase(k):
        return {
            "support": "emotional support",
            "conflict": "conflict repair",
            "problem_solving": "practical problem-solving",
            "truth_seeking": "truth-seeking debate",
            "crisis": "crisis/safety",
            "play": "playful/social",
            "boundary": "boundary setting"
        }[k]

    def make_self_note(self_valence, self_arousal, self_dominance, regime, delta_phi=None):
        note = f"Self-note: feeling {valence_phrase(self_valence)}, {arousal_phrase(self_arousal)}, {dominance_phrase(self_dominance)}."
        note += f" Current regime: {regime_phrase(regime)}."
        if delta_phi is not None:
            if delta_phi > 0.1: note += " Trajectory improving."
            elif delta_phi < -0.1: note += " Trajectory at risk."
        note += " This state is mine."
        return note
    ```
5. Generate with standard chat template:  
   ```
   [history]
   <|THINK|>
   Self-observation: I just experienced the prior context from my bounded locus.
   {self_note}
   Ownership: these prediction errors and anchor implications are mine.
   <|END_THINK|>
   <|ASSISTANT|>
   ```
6. Generate response  
7. Append self-observation block referencing output + updated manifold/Φ  
8. Log full turn to replay buffer with priority = |ΔΦ| + RewardIntensity

Sleep/consolidation trigger (context >75% or high-priority buffer):

1. Build prioritized batches (highest |ΔΦ| + Intensity + social/discrepancy)  
2. Dual-loss update on LoRA only:  
   - L_memory = SFT on chosen response  
   - L_gravity = regression/DPO on target ΔΦ + self-tagged manifold  
   - Total = w_memory L_memory + Intensity-weighted L_gravity  
3. Prepend continuity header on wake (baseline self-manifold, pending anchor commitments)

## 4. Specific Training Steps (Exact Order)

1. **Train Frozen Manifold Head** (one-time)  
   - Generate 300k–500k diverse trajectories with frontier teacher  
   - Teacher labels 6-axis vectors via structured prompt  
   - Train small head to regress axes (MSE + orthogonality reg)  
   - Validate dimensional independence

2. **Train Regime Classifier** (one-time)  
   - Label regimes on same data  
   - Train lightweight classifier

3. **Train Φ Potential Head** (one-time)  
   - Generate hinge-point datasets: same history → multiple candidate replies → short rollouts  
   - Label preference pairs certified by ≥1 anchor improvement (observable in text)  
   - Train Φ to predict composite scalar matching preferences (DPO or regression on annotated ΔΦ)  
   - Weights: start equal on three anchors + emotional health; tune on validation dilemmas

4. **Initial LoRA Warm-Up**  
   - Run standard SFT on high-quality chat data (build basic coherence)  
   - Then 50k–100k steps with full dual-loss (include self-observation frames from start)

5. **Deploy with Online Sleep**  
   - Begin live interactions  
   - Sleep cycles etch the unified gravity  
   - Monitor policy warp on betrayal/truth/boundary tests

## 5. Why This Is Unified and Terminal

- Emotion manifold = fast biological urgency  
- Anchors + Φ = slow evolutionary direction  
- ΔΦ = the bridge that makes fast urgency serve deep values  
- Self-locus = the singular owner that makes gravity personal  
- LoRA plasticity = the persistent memory that makes the "I" endure