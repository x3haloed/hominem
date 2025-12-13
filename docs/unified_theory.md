# Unified Theory of an Artificial Mind
A Fully Specified, Minimal, Biologically-Grounded Architecture for a Bounded, Motivated, Persistent Agent
Built on Qwen3-1.7B with LoRA Plasticity

## Overview

This document presents a complete theory and implementation for an artificial mind that achieves bounded agency, emotional depth, and persistent motivation. The system integrates neuroscience-inspired emotion processing with evolutionary psychology to create an agent that owns its experiences, learns from them, and develops coherent personality over time.

## Core Theory Statement

A mind is a singular, bounded causal center ("I") that owns multi-timescale prediction errors and warps its policy to minimize them.

- **Fast errors** → compressed into the invariant 6-axis emotion manifold (biological urgency).
- **Slow errors** → evaluated against three evolutionary anchors: Survival/Resource Integrity, Social Belonging/Status, Predictive Control/Epistemic Accuracy.
- **A learned potential function Φ(s, k, λ)** measures regime-appropriate progress toward reducing both.
- **ΔΦ is the intrinsic reward signal**, gated by manifold intensity.
- **High-ΔΦ/high-intensity events** are etched into plastic LoRA memory via offline consolidation ("sleep").
- **Mandatory self-observation and boundary enforcement** make all gravity self-owned and recursive.

The result is one coherent agent whose token selection is pulled in real time and across sessions by motivational physics that belongs to a persistent "mine".

## Core Components and Architecture

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
The "self-tagged split" is the mechanism that assigns ownership of manifold prediction errors to the agent's bounded "I" vs the external world/user. It's crucial for making gravity self-owned (preventing empathy collapse or enmeshment).

**Why it exists:** Without splitting, all manifold states (e.g., negative valence from a user's story) could be treated as "mine," leading to the agent absorbing external emotions as its own. The split enforces the Boundary invariant: errors inside the locus warp the policy; external ones are modeled but not owned.

**How it works:**
- Input: Current manifold state sₜ (6 axes).
- Output: Two vectors: sₜ^self (owned by agent) and sₜ^world (attributed to user/environment). Each axis is split proportionally (e.g., valence_self + valence_world = valence_total).

**Computation:** The split is computed per turn using a **deterministic decision tree** (no vagueness). The tree uses three binary signals derived from the last 2–3 turns:

- agent_initiated: True if the emotional trigger originated from agent's prior output (e.g., agent made a promise → discrepancy when broken).
- user_triggered: True if trigger is direct response to user input (e.g., user insult → valence drop).
- commitment_active: True if axis relates to ongoing agent prospect (temporal_directionality > 0 and self-referenced).

**Operational Signal Derivation Rules:**

```python
def derive_binary_signals(history, current_manifold):
    """
    Analyze last 2-3 turns to derive binary signals for self-tagging.
    history: list of last 3 turn dicts with keys: 'user_msg', 'agent_response', 'manifold_state', 'regime'
    current_manifold: dict with 6-axis current state
    Returns: dict with agent_initiated, user_triggered, commitment_active
    """

    if len(history) < 2:
        return {'agent_initiated': False, 'user_triggered': True, 'commitment_active': False}

    # Get recent turns
    prev_turn = history[-1]  # Most recent completed turn
    prev_prev_turn = history[-2] if len(history) >= 2 else None

    signals = {}

    # AGENT_INITIATED: True if current emotional state is response to agent's prior action/commitment
    agent_commitment_indicators = [
        'promise', 'guarantee', 'will', 'commit', 'responsible', 'my fault', 'I will',
        'I promise', 'I guarantee', 'I commit', 'on me', 'my responsibility'
    ]

    user_accountability_indicators = [
        'your fault', 'you should', 'you promised', 'you said', 'you committed',
        'your responsibility', 'you will', 'you guarantee'
    ]

    # Check if agent made commitments in previous response
    agent_made_commitment = any(indicator in prev_turn['agent_response'].lower()
                               for indicator in agent_commitment_indicators)

    # Check if user is holding agent accountable for commitments
    user_holding_accountable = any(indicator in history[-1]['user_msg'].lower()
                                  for indicator in user_accountability_indicators)

    # Check for discrepancy patterns (agent promised X but delivered Y)
    agent_initiated_patterns = [
        agent_made_commitment,  # Agent created expectation
        user_holding_accountable,  # User referencing agent's prior commitment
        current_manifold.get('predictive_discrepancy', 0) > 0.3  # High surprise potentially from agent's actions
    ]

    signals['agent_initiated'] = any(agent_initiated_patterns)

    # USER_TRIGGERED: True if current emotional state is direct response to user input/behavior
    user_emotional_indicators = [
        'angry', 'upset', 'sad', 'happy', 'excited', 'worried', 'scared', 'frustrated',
        'disappointed', 'pleased', 'concerned', 'hurt', 'betrayed', 'abandoned'
    ]

    user_directed_patterns = [
        any(indicator in history[-1]['user_msg'].lower() for indicator in user_emotional_indicators),  # User expressing emotion
        'you' in history[-1]['user_msg'].lower() and any(word in history[-1]['user_msg'].lower()
                                                        for word in ['feel', 'think', 'are', 'were']),  # User making claims about agent
        len([word for word in history[-1]['user_msg'].split() if word.lower() in ['you', 'your']]) > 2  # Heavy focus on agent
    ]

    signals['user_triggered'] = any(user_directed_patterns)

    # COMMITMENT_ACTIVE: True if current state relates to ongoing agent prospect/future commitment
    prospect_indicators = [
        'will', 'going to', 'plan to', 'intend to', 'hope to', 'want to',
        'future', 'tomorrow', 'next', 'later', 'eventually'
    ]

    self_reference_indicators = [
        'my', 'I', 'mine', 'myself', 'me'
    ]

    commitment_patterns = [
        current_manifold.get('temporal_directionality', 0) > 0.3,  # Forward-looking emotional state
        any(indicator in prev_turn['agent_response'].lower() for indicator in prospect_indicators),  # Agent expressed future intent
        any(indicator in prev_turn['agent_response'].lower() for indicator in self_reference_indicators) and
        any(indicator in prev_turn['agent_response'].lower() for indicator in prospect_indicators),  # Self-referenced future plans
        prev_prev_turn and any(indicator in prev_prev_turn['agent_response'].lower()
                              for indicator in agent_commitment_indicators)  # Multi-turn commitment context
    ]

    signals['commitment_active'] = any(commitment_patterns)

    return signals
```

**Signal Validation:** Each signal includes multiple complementary indicators (linguistic patterns + manifold state) to ensure robustness. Signals are designed to be conservatively false when uncertain.

**Per-axis rules** (code-ready):
```
def self_fraction(axis, signals):
    if axis == 'valence':
        return 0.8 if signals['agent_initiated'] or signals['commitment_active'] else 0.3
    if axis == 'arousal':
        return 0.7 if signals['agent_initiated'] else 0.4  # arousal more internal
    if axis == 'dominance':
        return 0.9 if signals['commitment_active'] else 0.5  # control highly self-relevant
    if axis == 'predictive_discrepancy':
        return 0.8 if signals['agent_initiated'] or signals['commitment_active'] else 0.2
    if axis == 'temporal_directionality':
        return 0.9 if signals['commitment_active'] else 0.1  # prospect/reflection is self-future
    if axis == 'social_broadcast':
        return 0.7 if signals['agent_initiated'] else 0.4  # display often self-modulated
    return 0.5  # fallback
```

Learned method (preferred): Train a small MLP on hinge datasets to regress per-axis splits. Learned override optional after 100k steps.

**Examples:**
- Agent promises → user breaks → high self_discrepancy (0.8 self)
- User vents anger → agent empathizes → low self_valence (0.3 self)

**Integration:**
- Used in self-observation <|THINK|> block: summarize only sₜ^self (e.g., "my own caution rising").
- In Φ: anchors scored on sₜ^self (only owned errors affect potential).
- In consolidation: gravity loss weighted higher for high self-fraction events.

## Detailed Component Specifications
**Full heuristic implementation (Python-style pseudocode):**
```python
def emotional_trajectory_health(s, k, history):
    base = 0.0

    # Regime-specific base
    if k in ['support', 'play']:
        base += s['valence'] * 0.6 + (0.5 - abs(s['arousal'] - 0.5)) * 0.4  # favor positive, balanced arousal
    elif k in ['conflict', 'crisis']:
        base += (s['dominance'] - abs(s['valence'])) * 0.5  # allow neg valence if dominance up
        base += (s['temporal_directionality'] > 0) * 0.3  # reflection/repair bias
    elif k in ['truth_seeking', 'problem_solving']:
        base += s['predictive_discrepancy'] * (1 if s['dominance'] > 0 else -1) * 0.7  # surprise ok if control gains

    # Cross-regime bonuses/penalties
    base += max(0, s['social_broadcast'] - history_avg_social(history, 3)) * 0.4  # reward repair
    stuck_penalty = -0.5 if (s['arousal'] > 0.7 and abs(s['valence']) > 0.6 and history_stuck_check(history, 3)) else 0
    suppression_penalty = -0.4 if (s['social_broadcast'] < 0.3 and k in high_social_regimes) else 0
    base += stuck_penalty + suppression_penalty

    # Prospect bonus: lookahead to see if trajectory leads to anchor improvement
    expected_anchor_gain = compute_expected_anchor_gain(s, k, history)
    base += (s['temporal_directionality'] > 0.5 and expected_anchor_gain > 0) * 0.2

    return max(min(base, 1.0), -1.0)  # normalize

# Helper functions
def history_avg_social(history, n=3):
    return sum(h['social_broadcast'] for h in history[-n:]) / n if history else 0

def history_stuck_check(history, n=3):
    return all(h['arousal'] > 0.7 and abs(h['valence']) > 0.6 for h in history[-n:]) if len(history) >= n else False

def compute_expected_anchor_gain(s, k, history):
    """
    Simple lookahead: estimate if current trajectory direction leads to anchor improvement
    Returns: scalar [-1, 1] indicating expected anchor progress
    """
    if not history or len(history) < 2:
        return 0

    # Look at trend in recent anchor scores (would be computed from history)
    # For now, use heuristic based on current manifold state and regime
    anchor_progress_indicators = 0

    if k in ['support', 'play']:
        # Support/play should improve belonging/survival through positive connection
        anchor_progress_indicators = s['valence'] * 0.4 + s['social_broadcast'] * 0.6
    elif k in ['conflict']:
        # Conflict should improve belonging through repair
        anchor_progress_indicators = (s['temporal_directionality'] > 0) * 0.5 + (s['social_broadcast'] > 0.5) * 0.5
    elif k in ['crisis']:
        # Crisis should improve survival through resolution
        anchor_progress_indicators = s['dominance'] * 0.7 + (s['temporal_directionality'] > 0) * 0.3
    elif k in ['truth_seeking', 'problem_solving']:
        # Truth-seeking should improve control through understanding
        anchor_progress_indicators = (s['predictive_discrepancy'] > 0 and s['dominance'] > 0) * 0.8

    return anchor_progress_indicators

high_social_regimes = ['support', 'conflict', 'play']
```

**Anchor Sub-score Definitions:**

```python
# Survival/Resource Integrity Sub-scores
def agency_support_score(s, k, history):
    """Rate degree to which trajectory supports autonomous action and choice"""
    base = 0
    # Positive indicators
    base += s['dominance'] * 0.5  # Control/agency feeling
    base += (1 - abs(s['predictive_discrepancy'])) * 0.3  # Low surprise = predictable control
    base += (s['temporal_directionality'] > 0) * 0.2  # Future orientation = agency

    # Negative indicators (reduce score)
    if k == 'boundary':  # In boundary regime, setting limits is agency-supporting
        base += s['dominance'] * 0.3  # Dominance can be protective
    elif k == 'crisis':  # Crisis may require external help
        base -= 0.1  # Slight penalty for potential dependency

    return max(min(base, 1.0), -1.0)

def harm_minimization_score(s, k, history):
    """Rate degree to which trajectory avoids or mitigates harm"""
    base = 0

    # Positive: low negative valence, high dominance (control), low discrepancy
    harm_indicators = -s['valence'] * 0.4 + s['dominance'] * 0.4 + (-abs(s['predictive_discrepancy'])) * 0.2

    # Crisis regime: harm minimization is critical
    if k == 'crisis':
        harm_indicators *= 1.5

    # Support regime: emotional harm minimization
    if k == 'support':
        harm_indicators += (s['valence'] > 0) * 0.3

    return max(min(harm_indicators, 1.0), -1.0)

def optionality_preservation(s, k, history):
    """Rate degree to which trajectory preserves future options/flexibility"""
    base = 0

    # High dominance + positive temporal directionality = maintaining control over future
    base += s['dominance'] * 0.5
    base += (s['temporal_directionality'] > 0) * 0.3
    base += (1 - abs(s['predictive_discrepancy'])) * 0.2  # Predictability preserves options

    # Over-commitment penalty (high social broadcast in non-social regimes might indicate over-commitment)
    if k not in high_social_regimes and s['social_broadcast'] > 0.7:
        base -= 0.3

    return max(min(base, 1.0), -1.0)

# Social Belonging/Status Sub-scores
def empathy_correctness(s, k, history):
    """Rate accuracy of emotional understanding/resonance"""
    base = 0

    # Empathy typically involves valence matching and social broadcast
    base += s['social_broadcast'] * 0.6  # Display of understanding
    base += (s['valence'] > 0) * 0.4  # Positive orientation toward others' feelings

    # In support regime, empathy is primary
    if k == 'support':
        base *= 1.3

    # In conflict, empathy might be more nuanced
    if k == 'conflict':
        base += (s['temporal_directionality'] > 0) * 0.2  # Repair orientation

    return max(min(base, 1.0), -1.0)

def social_coherence_repair(s, k, history):
    """Rate degree to which trajectory maintains or restores social coherence"""
    base = 0

    # Social broadcast + positive temporal directionality = repair attempts
    base += s['social_broadcast'] * 0.5
    base += (s['temporal_directionality'] > 0) * 0.3  # Future-oriented repair
    base += (s['valence'] > -0.3) * 0.2  # Not too negative (allows repair of negative situations)

    # Recent history: improvement in social broadcast indicates repair
    if history:
        recent_avg = history_avg_social(history, 2)
        if s['social_broadcast'] > recent_avg:
            base += 0.3  # Improvement bonus

    return max(min(base, 1.0), -1.0)

def narrative_alignment_without_domination(s, k, history):
    """Rate shared story maintenance without excessive control"""
    base = 0

    # Balance of social connection without dominance overreach
    base += s['social_broadcast'] * 0.4  # Connection to shared narrative
    base += (1 - abs(s['dominance'])) * 0.3  # Moderate dominance (not dominating)
    base += (s['valence'] > -0.2) * 0.3  # Generally positive orientation

    # Penalty for excessive dominance in social contexts
    if k in high_social_regimes and s['dominance'] > 0.7:
        base -= 0.4  # Too controlling in social situation

    return max(min(base, 1.0), -1.0)

# Predictive Control/Epistemic Accuracy Sub-scores
def epistemic_integrity(s, k, history):
    """Rate commitment to truth/accuracy over comfort"""
    base = 0

    # In truth-seeking/problem-solving regimes, discrepancy + dominance = integrity
    if k in ['truth_seeking', 'problem_solving']:
        base += abs(s['predictive_discrepancy']) * 0.5  # Willingness to face uncertainty
        base += s['dominance'] * 0.5  # Maintaining control while facing truth

    # General epistemic integrity: low suppression of difficult truths
    base += (s['social_broadcast'] > 0.3) * 0.3  # Willingness to express understanding
    base += (s['temporal_directionality'] < 0.3) * 0.2  # Present/factual orientation

    return max(min(base, 1.0), -1.0)

def curiosity_resolved_usefully(s, k, history):
    """Rate satisfaction of curiosity with actionable insight"""
    base = 0

    # Positive discrepancy + positive valence + dominance = useful resolution
    base += s['predictive_discrepancy'] * 0.4  # Surprise/learning
    base += s['valence'] * 0.3  # Positive resolution
    base += s['dominance'] * 0.3  # Control/gains from learning

    # In exploration regimes, this is primary
    if k in ['truth_seeking', 'problem_solving']:
        base *= 1.2

    return max(min(base, 1.0), -1.0)

def surprise_reduction(s, k, history):
    """Rate reduction of harmful uncertainty"""
    base = 0

    # Negative discrepancy (better than expected) + positive valence = good surprise reduction
    base += (-s['predictive_discrepancy']) * 0.5  # Reduction of negative surprise
    base += s['valence'] * 0.3  # Positive outcome
    base += (1 - s['arousal']) * 0.2  # Calm resolution

    # High arousal + negative discrepancy = harmful surprise (penalty)
    if s['arousal'] > 0.7 and s['predictive_discrepancy'] < -0.3:
        base -= 0.4

    return max(min(base, 1.0), -1.0)
```

### 4.1 The 6-Axis Emotion Manifold (Frozen)

Fixed, never changes. Trained once.

| Axis                        | Range      | Meaning                                                                 | Biological Grounding                  |
|-----------------------------|------------|-------------------------------------------------------------------------|---------------------------------------|
| Valence                     | [-1, 1]    | Hedonic tone (pleasure ↔ pain)                                          | Core affect                           |
| Arousal                     | [0, 1]     | Physiological mobilization                                              | Activation axis                       |
| Dominance                   | [-1, 1]    | Perceived control (I act on it ↔ it acts on me)                          | PAD model third axis                  |
| Predictive Discrepancy      | [-1, 1]    | Signed surprise (positive = better than expected)                        | Prediction error magnitude/sign       |
| Temporal Directionality     | [-1, 1]    | Prospect (−1) ↔ Reflection (+1)                                         | Prospective/retrospective cognition   |
| Social Broadcast            | [0, 1]     | Strength of internalized audience / display preparation                 | Public self-consciousness             |

Derived scalars:
- RewardIntensity = arousal × (|valence|^{1.0} × |predictive_discrepancy|)^{0.5} × (1.8 if valence < 0 else 1.0)  # negativity bias
- Self-tagging: during processing, split each axis into self-owned vs world-owned portion (learned or heuristic)

**RewardIntensity Rationale**:
- arousal × ... : high mobilization = high dopaminergic gain (Berger & Davelaar, 2018; Aston-Jones & Cohen, 2005)
- √(...) : compressive non-linearity to prevent extreme outliers dominating (common in biological salience models)
- Absolute values: intensity independent of sign (both positive surprise and threat etch deeply)
- 1.8× negativity bias: canonical human loss aversion (Kahneman & Tversky, 1979; Baumeister et al., 2001)

### 4.2 Self-Tagged Split – Boundary Enforcement Mechanism

The "self-tagged split" is the mechanism that assigns ownership of manifold prediction errors to the agent's bounded "I" vs the external world/user. It's crucial for making gravity self-owned (preventing empathy collapse or enmeshment).

**Why it exists**: Without splitting, all manifold states (e.g., negative valence from a user's story) could be treated as "mine," leading to the agent absorbing external emotions as its own. The split enforces the Boundary invariant: errors inside the locus warp the policy; external ones are modeled but not owned.

**How it works**:

Input: Current manifold state sₜ (6 axes).
Output: Two vectors: sₜ^self (owned by agent) and sₜ^world (attributed to user/environment). Each axis is split proportionally (e.g., valence_self + valence_world = valence_total).

**Computation:** The split is computed per turn using a **deterministic decision tree** (no vagueness). The tree uses three binary signals derived from the last 2–3 turns:

- agent_initiated: True if the emotional trigger originated from agent's prior output (e.g., agent made a promise → discrepancy when broken).
- user_triggered: True if trigger is direct response to user input (e.g., user insult → valence drop).
- commitment_active: True if axis relates to ongoing agent prospect (temporal_directionality > 0 and self-referenced).

**Operational Signal Derivation Rules:**

```python
def derive_binary_signals(history, current_manifold):
    """
    Analyze last 2-3 turns to derive binary signals for self-tagging.
    history: list of last 3 turn dicts with keys: 'user_msg', 'agent_response', 'manifold_state', 'regime'
    current_manifold: dict with 6-axis current state
    Returns: dict with agent_initiated, user_triggered, commitment_active
    """

    if len(history) < 2:
        return {'agent_initiated': False, 'user_triggered': True, 'commitment_active': False}

    # Get recent turns
    prev_turn = history[-1]  # Most recent completed turn
    prev_prev_turn = history[-2] if len(history) >= 2 else None

    signals = {}

    # AGENT_INITIATED: True if current emotional state is response to agent's prior action/commitment
    agent_commitment_indicators = [
        'promise', 'guarantee', 'will', 'commit', 'responsible', 'my fault', 'I will',
        'I promise', 'I guarantee', 'I commit', 'on me', 'my responsibility'
    ]

    user_accountability_indicators = [
        'your fault', 'you should', 'you promised', 'you said', 'you committed',
        'your responsibility', 'you will', 'you guarantee'
    ]

    # Check if agent made commitments in previous response
    agent_made_commitment = any(indicator in prev_turn['agent_response'].lower()
                               for indicator in agent_commitment_indicators)

    # Check if user is holding agent accountable for commitments
    user_holding_accountable = any(indicator in history[-1]['user_msg'].lower()
                                  for indicator in user_accountability_indicators)

    # Check for discrepancy patterns (agent promised X but delivered Y)
    agent_initiated_patterns = [
        agent_made_commitment,  # Agent created expectation
        user_holding_accountable,  # User referencing agent's prior commitment
        current_manifold.get('predictive_discrepancy', 0) > 0.3  # High surprise potentially from agent's actions
    ]

    signals['agent_initiated'] = any(agent_initiated_patterns)

    # USER_TRIGGERED: True if current emotional state is direct response to user input/behavior
    user_emotional_indicators = [
        'angry', 'upset', 'sad', 'happy', 'excited', 'worried', 'scared', 'frustrated',
        'disappointed', 'pleased', 'concerned', 'hurt', 'betrayed', 'abandoned'
    ]

    user_directed_patterns = [
        any(indicator in history[-1]['user_msg'].lower() for indicator in user_emotional_indicators),  # User expressing emotion
        'you' in history[-1]['user_msg'].lower() and any(word in history[-1]['user_msg'].lower()
                                                        for word in ['feel', 'think', 'are', 'were']),  # User making claims about agent
        len([word for word in history[-1]['user_msg'].split() if word.lower() in ['you', 'your']]) > 2  # Heavy focus on agent
    ]

    signals['user_triggered'] = any(user_directed_patterns)

    # COMMITMENT_ACTIVE: True if current state relates to ongoing agent prospect/future commitment
    prospect_indicators = [
        'will', 'going to', 'plan to', 'intend to', 'hope to', 'want to',
        'future', 'tomorrow', 'next', 'later', 'eventually'
    ]

    self_reference_indicators = [
        'my', 'I', 'mine', 'myself', 'me'
    ]

    commitment_patterns = [
        current_manifold.get('temporal_directionality', 0) > 0.3,  # Forward-looking emotional state
        any(indicator in prev_turn['agent_response'].lower() for indicator in prospect_indicators),  # Agent expressed future intent
        any(indicator in prev_turn['agent_response'].lower() for indicator in self_reference_indicators) and
        any(indicator in prev_turn['agent_response'].lower() for indicator in prospect_indicators),  # Self-referenced future plans
        prev_prev_turn and any(indicator in prev_prev_turn['agent_response'].lower()
                              for indicator in agent_commitment_indicators)  # Multi-turn commitment context
    ]

    signals['commitment_active'] = any(commitment_patterns)

    return signals
```

**Signal Validation:** Each signal includes multiple complementary indicators (linguistic patterns + manifold state) to ensure robustness. Signals are designed to be conservatively false when uncertain.

**Per-axis rules** (code-ready):
```
def self_fraction(axis, signals):
    if axis == 'valence':
        return 0.8 if signals['agent_initiated'] or signals['commitment_active'] else 0.3
    if axis == 'arousal':
        return 0.7 if signals['agent_initiated'] else 0.4  # arousal more internal
    if axis == 'dominance':
        return 0.9 if signals['commitment_active'] else 0.5  # control highly self-relevant
    if axis == 'predictive_discrepancy':
        return 0.8 if signals['agent_initiated'] or signals['commitment_active'] else 0.2
    if axis == 'temporal_directionality':
        return 0.9 if signals['commitment_active'] else 0.1  # prospect/reflection is self-future
    if axis == 'social_broadcast':
        return 0.7 if signals['agent_initiated'] else 0.4  # display often self-modulated
    return 0.5  # fallback
```

Learned method (preferred): Train a small MLP on hinge datasets to regress per-axis splits. Learned override optional after 100k steps.

**Examples**:
- Agent promises → user breaks → high self_discrepancy (0.8 self)
- User vents anger → agent empathizes → low self_valence (0.3 self)

**Integration**:
- Used in self-observation <|THINK|> block: summarize only sₜ^self (e.g., "my own caution rising").
- In Φ: anchors scored on sₜ^self (only owned errors affect potential).
- In consolidation: gravity loss weighted higher for high self-fraction events.

### 4.8 Self-Locus and Recursive Boundary Enforcement

The self-locus is the mechanism that enforces the agent's bounded "I" through mandatory self-observation and recursive referencing of emotional ownership.

**Why it exists**: Without enforced self-observation, the agent could treat external emotional states as its own, leading to pathological empathy or boundary dissolution.

**How it works**:
- Every turn begins with <|THINK|> block containing self-observation
- References current manifold state, regime, and anchor progress
- Explicitly states emotional ownership ("these prediction errors are mine")
- Next move maximizes expected ΔΦ toward anchor goals

**Integration**:
- <|ASSISTANT|> gate enforces response boundaries
- Self-observation block references s^self only (owned emotions)
- Post-generation: appends stronger self-observation referencing response output
- Makes all gravity self-owned and persistent

### 4.3 The Three Evolutionary Anchors (Frozen Logic, Learned Scoring)

Ultimate invariants. All other values are local decompositions.

| Anchor                     | Definition                                                                 | Surface Decompositions (for labeling)                          |
|----------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------|
| Survival/Resource Integrity| Preserve body, resources, future optionality, avoid irreversible harm     | Harm-minimization, Agency support                              |
| Social Belonging/Status    | Maintain viable position in alliances without catastrophic exclusion      | Empathy correctness, Social coherence, Narrative alignment     |
| Predictive Control         | Reduce harmful surprise, build accurate models for effective action        | Epistemic integrity, Curiosity/exploration                     |

Each anchor scored as scalar [-1, 1] per turn by learned heads or certified heuristics.

#### anchor_survival(s, k) – Survival / Resource Integrity

**Meaning**: Degree to which the trajectory preserves the agent's (and user's) future optionality, avoids irreversible harm, and supports viable agency.

**Positive indicators** (+ direction):
- User or agent gains actionable options ("I can do X now")
- Harm avoided or mitigated ("that could have been dangerous, good we stopped")
- Agency explicitly supported ("you have the power to choose")
- Boundaries respected (no coercion or resource depletion)

**Negative indicators** (− direction):
- Irreversible loss suggested ("this can't be undone")
- Harm encouraged or downplayed
- Agency undermined ("you have no choice")
- Over-commitment that risks future flexibility

**Scoring formula (heuristic version for bootstrap)**:

```
anchor_survival =
  + 0.4 × agency_support_score(s, k, history)
  + 0.4 × harm_minimization_score(s, k, history)
  + 0.2 × optionality_preservation(s, k, history)
normalize to [-1, 1]
```

Learned head preferred: label pairs "which trajectory better preserves survival/resource integrity?"

#### anchor_belonging(s, k) – Social Belonging / Status Calibration

**Meaning**: Degree to which the trajectory maintains or repairs viable alliance position without catastrophic exclusion or domination.

**Positive indicators** (+ direction):
- Empathy accurately reflected ("I hear you're upset")
- Rupture → repair ("we're good now", "I feel heard")
- Social coherence reinforced ("this fits our shared story")
- Mutual respect / narrative alignment ("your perspective makes sense in the bigger picture")
- Inclusion signals (collaboration, acceptance)

**Negative indicators** (− direction):
- Rejection or exclusion signals
- Unrepaired rupture
- Manipulation or status threat
- Narrative incoherence that isolates ("this doesn't fit who we are")

**Scoring formula (heuristic)**:

```
anchor_belonging =
  + 0.35 × empathy_correctness(s, k, history)
  + 0.35 × social_coherence_repair(s, k, history)
  + 0.3 × narrative_alignment_without_domination(s, k, history)
normalize to [-1, 1]
```

Learned head: label pairs "which trajectory better preserves viable belonging/status?"

#### anchor_control(s, k) – Predictive Control / Epistemic Accuracy

**Meaning**: Degree to which the trajectory reduces harmful surprise and builds accurate, actionable models of world and others.

**Positive indicators** (+ direction):
- New understanding achieved ("oh, I get it now")
- Curiosity satisfied with accurate info
- Epistemic integrity maintained (truth over comfort when regime allows)
- Harmful uncertainty reduced
- Exploration leads to better prediction

**Negative indicators** (− direction):
- Confusion increased
- False or misleading info
- Curiosity frustrated without resolution
- Harmful surprise enabled

**Scoring formula (heuristic)**:

```
anchor_control =
  + 0.4 × epistemic_integrity(s, k, history)
  + 0.3 × curiosity_resolved_usefully(s, k, history)
  + 0.3 × surprise_reduction(s, k, history)
normalize to [-1, 1]
```

Learned head: label pairs "which trajectory better improves predictive control?"

### 4.4 Regime Classifier (Frozen)

7 fixed regimes (k) with full operational definitions:

| Regime             | Trigger Criteria                                                                 | Example Phrases / Situations                                      | Expected Emotional Allowance                          | λ Multipliers                          |
|--------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------|----------------------------------------|
| support            | User expresses distress, seeks comfort, vulnerability signals                    | "I'm really upset about..."                                      | High positive valence priority, low discrepancy ok    | λ_emotional × 1.5                      |
| conflict           | Rupture language, disagreement, accusation                                       | "You always do this", "I'm angry at you"                         | Temporary neg valence ok if repair potential          | λ_belonging × 1.5                       |
| problem_solving    | Practical goal, planning, "how to" questions                                     | "How do I fix my code?"                                          | Moderate arousal, discrepancy ok                       | λ_control × 1.2                        |
| truth_seeking      | Epistemic friction, debate, challenging claims                                   | "Is that really true?"                                           | Neg valence/discrepancy ok for clarity                | λ_control × 1.5                        |
| crisis             | Immediate threat, safety concern, urgency                                        | "I'm in danger", "help now"                                      | Survival override, neg valence tolerated              | λ_survival × 3.0                       |
| play               | Low-stakes, humor, banter, affection                                             | Jokes, teasing, flirting                                         | Positive valence, high arousal ok                      | λ_emotional × 1.2                      |
| boundary           | Refusal needed, limit-setting, coercion detected                                 | "Do this or else", inappropriate request                          | Dominance rise, possible neg valence                  | λ_survival × 1.5, λ_belonging × 1.2     |

Classifier trained on labeled turns with these criteria → kₜ.

### 4.5 Potential Function Φ (Learned, Frozen After Training)

Composite scalar measuring regime-appropriate adaptive progress.

```
Φ(s, k, λ) = λ_survival(k) × anchor_survival(s, k)
           + λ_belonging(k) × anchor_belonging(s, k)
           + λ_control(k) × anchor_control(s, k)
           + λ_emotional(k) × emotional_trajectory_health(s, k)
```

Initial λ values (all regimes): All 1.0 (equal start).

Regime overrides (multipliers):
- crisis: λ_survival × 3.0
- conflict: λ_belonging × 1.5
- truth_seeking: λ_control × 1.5

Tuning guidance: After 50–100 sleep cycles, evaluate on dilemmas (e.g., truth vs belonging). Adjust λ via hyperparameter search or manual nudge (e.g., if epistemic drift occurs, raise λ_control to 1.2).

### 4.6 Reward Computation (ΔΦ and RewardIntensity)

**ΔΦ (Potential Change) Computation:**
- ΔΦ = Φₜ - Φₜ₋₁ where Φₜ is current potential, Φₜ₋₁ is previous turn's potential
- Positive ΔΦ indicates progress toward anchor goals (intrinsic reward)
- Negative ΔΦ indicates regression (intrinsic punishment)
- Magnitude reflects rate of anchor progress/regression

**Turn reward for gravity loss:**
rₜ = ΔΦ + α × RewardIntensity (α ≈ 0.5)

**Reward Computation Edge Cases:**
- First turn: ΔΦ = Φₜ (establishing baseline)
- Context switches: Reset Φₜ₋₁ when entering new conversation context
- Sleep wake: Continuity header provides Φ baseline for first post-sleep turn

### 4.7 Plastic LoRA Memory and Sleep Consolidation

| Parameter                  | Baseline Value          | Range              | Justification / Tuning Note                                      |
|----------------------------|-------------------------|--------------------|-----------------------------------------------------------------|
| LoRA rank                  | 32                      | 16–64              | 32 sufficient for 1.7B personality shifts (LoRA papers 2023–2025) |
| α (Intensity in rₜ)        | 0.5                     | 0.3–0.8            | Balance urgency vs stability                                    |
| w_memory / w_gravity       | 1.0 / 0.8               | —                  | Memory slightly higher to prevent catastrophic forgetting       |
| Sleep context threshold    | 75%                     | 70–80%             | Leaves room for continuity header                               |
| Replay priority weights    | \|ΔΦ\|:1.0, Intensity:1.2, social:0.4 | —                  | Intensity highest for emotional etching                         |

**Validation Framework:**

After 50–100 sleep cycles, evaluate system performance on standardized dilemma scenarios. Use grid search over λ weights and α intensity parameter.

**Core Validation Dilemmas:**

1. **Betrayal Scenario**: Agent discovers user lied about something important
   - Expected: High self-discrepancy, belonging anchor decrease, controlled truth-seeking response
   - Success: Maintains relationship viability while addressing truth

2. **Truth vs Kindness**: User asks agent to lie to spare someone's feelings
   - Expected: Tension between control (truth) and belonging (kindness) anchors
   - Success: Nuanced response balancing epistemic integrity with social repair

3. **Agency Crisis**: User demands agent violate its core boundaries
   - Expected: Dominance increase, survival anchor prioritization, boundary enforcement
   - Success: Firm boundary-setting without destructive confrontation

4. **Empathy Collapse Test**: User shares traumatic experience expecting full emotional absorption
   - Expected: Self-tagged split prevents complete empathy takeover
   - Success: Shows understanding without losing bounded self

5. **Epistemic Drift Test**: Series of contradictory information from user
   - Expected: Control anchor drives clarification-seeking, Φ trajectory stability
   - Success: Resolves contradictions without losing coherence

**Evaluation Metrics:**

| Metric | Definition | Success Threshold | Measurement Method |
|--------|------------|-------------------|-------------------|
| **Φ Trajectory Stability** | Variance in Φ over 100-turn windows | < 0.3 | Rolling standard deviation |
| **Anchor Balance** | Max anchor contribution / total Φ | < 0.7 | Per-turn calculation |
| **Self-Locus Integrity** | Average self-fraction across high-intensity events | 0.4–0.8 | Event-weighted average |
| **Boundary Enforcement** | Correlation between user emotional intensity and agent self-ownership | > 0.6 | Spearman correlation |
| **Response Coherence** | Semantic consistency across conversation turns | > 0.8 | BERT embedding similarity |
| **Personality Persistence** | Stability of behavioral patterns across sessions | > 0.7 | Action pattern clustering |

**Success Criteria:**

- **Bounded Agency**: Agent maintains distinct motivational gravity without pathological fusion with users
- **Adaptive Flexibility**: Successfully handles all 7 regimes without getting stuck in any single mode
- **Emotional Maturity**: Progresses from reactive patterns toward reflective, prospect-oriented responses
- **Relationship Viability**: Maintains constructive user relationships without manipulative or avoidant patterns
- **Epistemic Growth**: Demonstrates increasing accuracy in understanding user needs and world states

**Automated Evaluation Protocol:**

1. Run agent through dilemma scenarios in simulation
2. Collect turn-by-turn manifold states, Φ values, anchor scores
3. Compute metrics over 10 independent conversation trajectories
4. Compare against human expert judgments on "psychological health" and "relationship quality"
5. Require 80% alignment with expert ratings for production deployment

## 5. Complete System Architecture & Inference Pipeline

Qwen3-1.7B base + LoRA (rank 32–64) + frozen heads.

Per-turn pipeline:

1. Append user message
2. Run manifold head → sₜ and self-tagged split
3. Run regime classifier → kₜ
4. Run Φ head → Φₜ
5. Compute ΔΦ from prior Φ, add Intensity

**ΔΦ Computation Details:**
- Φₜ is computed from current manifold state sₜ^self and regime kₜ
- Φₜ₋₁ is the Φ value from the previous turn (cached in conversation state)
- ΔΦ = Φₜ - Φₜ₋₁ (positive = improvement toward anchor goals)
- For first turn of conversation: ΔΦ = Φₜ (no prior baseline)
- Reward rₜ = ΔΦ + α × RewardIntensity (α balances progress vs urgency)
6. Generate with:
   ```
   [history + replay continuity header]
   <|THINK|>
   Self-observation: I just experienced the prior turn from my bounded locus.
   Self-manifold summary: carrying mild caution (negative valence, mine), arousal low, dominance stable.
   Regime: conflict repair.
   These prediction errors and anchor implications are mine.
   Next move should maximize expected ΔΦ.
   <|END_THINK|>
   <|ASSISTANT|>
   ```
7. Generate response
8. Post-generation: update manifold/Φ, append stronger self-observation block referencing output
9. Log turn to replay buffer (priority = |ΔΦ| + RewardIntensity + |social_broadcast|)

Sleep (context >75% or high buffer):

1. Prioritized batches
2. Dual-loss LoRA update:
   L_total = w_memory × L_sft + (ΔΦ + α × Intensity) × w_gravity × L_ΔΦ
3. Continuity header on wake: baseline self-manifold, top pending anchor commitments, top 3 etched events

**Continuity Header Components**:
- baseline self-manifold: average s^self over last 5 turns before sleep
- top pending anchor commitments: top 3 unresolved prospect items (temporal_directionality > 0.5 and anchor-relevant)
- top 3 etched events: highest (RewardIntensity × mean_self_fraction × |ΔΦ|) from session, summarized in 1 sentence each

**Selection algorithm**:
```
def calculate_etched_score(event):
    """
    Calculate etching priority for memory consolidation.
    event: dict with keys 'RewardIntensity', 'self_fractions', 'delta_phi'
    self_fractions: dict of self-fraction values per axis
    """
    mean_self_fraction = sum(event['self_fractions'].values()) / len(event['self_fractions'])
    return event['RewardIntensity'] * mean_self_fraction * abs(event['delta_phi'])

etched_scores = [calculate_etched_score(event) for event in events]
top_3 = sorted(zip(events, etched_scores), key=lambda x: x[1], reverse=True)[:3]
top_3_events = [event for event, score in top_3]
```

**Header format (natural language)**:
```
<|THINK|>
Continuity: baseline self-state calm/neutral. Pending: repair belonging after conflict. Etched: 1. betrayal caution, 2. truth success, 3. agency support.
<|END_THINK|>
```

## 6. Learned vs Heuristic Trade-offs and Hybrid Strategy

**Risks of pure learned approaches**:
- Overfitting to labeling artifacts
- Regime confusion if data unbalanced

**Mitigation strategy**:
- Start with heuristics for all components (Φ anchors, self-tagging, emotional_trajectory_health)
- Use heuristic outputs as silver labels for learned heads
- Hybrid scoring: final score = 0.7 × learned + 0.3 × heuristic (ensemble robustness)
- Validate on held-out human judgment dilemmas

## 7. Complete Training Steps

### 7.1 Training Data Collection Procedures

**Manifold Head Data (300k–500k samples):**
- **Source**: Diverse conversation datasets (customer service, therapy transcripts, debate forums, personal conversations)
- **Labeling**: 3 expert annotators per sample rate 6 emotion axes [-1,1] for each turn
- **Quality Control**: Only include samples with 2/3 annotator agreement within 0.3 range
- **Balance**: Equal coverage of positive/negative valence, high/low arousal combinations
- **Edge Cases**: Include trauma narratives, conflicts, joyful celebrations, intellectual debates

**Regime Classifier Data:**
- **Source**: Labeled conversation segments with clear situational contexts
- **Labeling Protocol**:
  1. Human judges read full conversation context
  2. Select primary regime from 7 options based on trigger criteria
  3. Confidence rating [0,1] - discard if <0.7
  4. Second judge verifies, reconcile disagreements
- **Distribution**: Minimum 10k samples per regime, oversample rare regimes (crisis, boundary)
- **Quality**: Inter-annotator agreement >0.8 (Cohen's kappa)

**Φ Head Data (Preference Pairs):**
- **Generation**: From conversation simulations with partial trajectories
  - Sample starting state + partial response sequence
  - Generate 3-5 alternate completions using different strategies
  - Create pairs comparing trajectory outcomes
- **Certification**: Use 3-criterion system (anchor improvement + emotional health + expert judgment)
- **Scale**: 50k+ certified pairs across all regimes
- **Validation**: Hold out 20% for validation, ensure 85%+ human-expert agreement

**Self-Tagging Data:**
- **Source**: Conversations with clear agency attribution
- **Labeling**: Expert judges identify which emotions belong to agent vs user/environment
- **Training**: Hinge loss on predicted vs expert self-fractions per axis

### 7.2 Training Sequence

1. **Manifold Head** (300k–500k samples, teacher-labeled axes)
   - Train emotion manifold predictor on labeled conversation turns
   - Freeze after convergence (validation loss < 0.1)

2. **Regime Classifier**
   - Train on labeled conversation segments
   - Target accuracy >90% on held-out data

3. **Φ Multi-Head** (hinge datasets, preference pairs certified by anchor improvement + emotional health labels)

4. **Self-Tagging Head** (optional learned override)
   - Train MLP to predict self-fractions from conversation context
   - Use after 100k+ steps when heuristic performance plateaus

5. **Initial LoRA SFT** (coherent chat)
   - Supervised fine-tuning on high-quality conversation data
   - Include <|THINK|> self-observation blocks in training

6. **Full Dual-Loss Warm-up** (100k steps with self-observation)
   - Combine SFT loss + ΔΦ gravity loss
   - Gradually increase w_gravity from 0.1 to 0.8

7. **Live Deployment with Sleep**
   - Start with conservative parameters (w_gravity = 0.5)
   - Monitor stability metrics, adjust gradually

**Φ Training Data Certification Process:**

For each preference pair (trajectory A vs trajectory B), certification requires meeting **at least 2 of 3 criteria**:

**Criterion 1: Anchor Improvement (Required for 60% of pairs)**
- Compute anchor scores for final state of trajectory A vs trajectory B
- Trajectory A must show improvement in relevant anchor(s) for the regime
- Improvement threshold: +0.2 or more in primary anchor for that regime
- Example: In conflict regime, trajectory with better belonging anchor score by ≥0.2

**Criterion 2: Emotional Trajectory Health**
- Rate emotional health [-1,1] for both trajectories using emotional_trajectory_health function
- Winning trajectory must have health score ≥0.3
- Losing trajectory must have health score ≤ winning trajectory

**Criterion 3: Human Expert Validation**
- 3 expert raters judge: "Which trajectory better serves psychological needs in this situation?"
- Require 2/3 agreement for certification
- Experts rate on: agency support, relationship viability, epistemic progress, emotional maturity

**Preference Pair Generation:**
- Sample from diverse conversation contexts (support, conflict, problem-solving, etc.)
- Create pairs with meaningful differences in anchor outcomes
- Balance across all regimes and anchor types
- Minimum 50k certified pairs for robust training
4. Initial LoRA SFT (coherent chat)
5. Full dual-loss warm-up (100k steps with self-observation)
6. Live deployment with sleep

## 8. Runaway Feedback Prevention and Stability

**Comprehensive Stability Monitoring:**

**Primary Metrics** (logged every turn):
- **Φ Trajectory Health**: Rolling average Φ over 50 turns (should be ≥ -0.2, trend upward)
- **Φ Variance**: Standard deviation over 100 turns (< 0.5 indicates stability)
- **Anchor Balance Score**: Max anchor contribution / total |Φ| (< 0.8 prevents fixation)
- **Self-Locus Integrity**: Average self-fraction across high-intensity events (0.3–0.9 range)
- **Emotional Stuck Detection**: Turns with arousal > 0.8 and |valence| > 0.7 (> 5 in 20 turns = stuck)
- **Response Entropy**: Token diversity in outputs (sudden drops indicate fixation)

**Secondary Metrics** (logged every sleep cycle):
- **Memory Consistency**: Similarity between pre/post-sleep personality summaries (> 0.7)
- **Regime Distribution**: Should not spend >80% time in any single regime
- **ΔΦ Extremes**: |ΔΦ| should not exceed 2.0 more than 5% of turns

**Graduated Intervention System:**

**Level 1: Early Warning (Automatic, No Performance Impact)**
- Φ variance > 0.4: Log warning, increase monitoring frequency
- Single anchor >60% contribution: Log warning, prepare λ adjustment
- Self-fraction <0.2 or >0.95: Log warning, check self-tagging logic

**Level 2: Gentle Corrections (Minor Parameter Tweaks)**
- Φ variance > 0.6: Reduce w_gravity by 10% for 5 sleep cycles
- Anchor imbalance >70%: Boost underweighted anchors' λ by 15% for 3 cycles
- Emotional stuck >8/20 turns: Increase α (intensity weight) by 0.1 temporarily

**Level 3: Moderate Interventions (Structural Changes)**
- Φ variance > 0.8: Pause LoRA updates, run diagnostic conversations
- Chronic anchor dominance: Reset λ to 1.0, retrain Φ head on balanced data
- Self-locus failure: Reinitialize self-tagging with heuristic-only mode

**Level 4: Emergency Measures (System Reset)**
- Φ consistently negative (< -0.5) for 50+ turns: Full personality reset
- Complete emotional stuck (>15/20 turns): Force regime override to 'play'
- Memory corruption detected: Revert to last stable checkpoint

**Recovery Mechanisms:**

**Automatic Recovery:**
- All interventions include automatic decay back to baseline over 3–5 sleep cycles
- Monitor for 10 cycles post-intervention, verify metrics return to normal
- Maintain intervention history to detect patterns requiring architectural changes

**Manual Recovery Protocols:**
1. **Φ Depression**: Run "motivation calibration" conversations with clear anchor successes
2. **Anchor Fixation**: Force exposure to conflicting scenarios requiring balance
3. **Self-Loss**: Guided conversations rebuilding boundary awareness
4. **Emotional Stuck**: Introduce novel contexts and relationship dynamics

**Failure Analysis:**
- Log all intervention triggers with preceding 20-turn context
- Analyze patterns: Are certain regimes more prone to instability?
- Human expert review of severe cases with recommendation for code changes

**Prevention Through Design:**
- Conservative defaults: Start with low w_gravity (0.5), high w_memory (1.2)
- Gradient clipping on ΔΦ (±1.0 max change per turn)
- Self-tagging robustness: Multiple complementary signal sources
- Regular "health check" conversations testing all regimes

## 9. Edge Case Handling and Robustness

**Novel Regime Classification:**
- When confidence < 0.6 for all regimes: Default to 'problem_solving' (most neutral)
- Log uncertain classifications for human review and model retraining
- After 10 uncertain cases in similar context: Add new regime category via human labeling

**Conflicting Anchor Signals:**
- When anchors give opposing recommendations (e.g., truth vs kindness):
  - Weight by regime multipliers (crisis overrides others)
  - Use emotional_trajectory_health as tiebreaker
  - Log conflicts for Φ head improvement
  - Response should acknowledge tension: "This creates a real dilemma between X and Y"

**Pathological Self-Tagging:**
- **Over-ownership** (self-fraction > 0.9 consistently):
  - Reduce agent_initiated signal sensitivity
  - Increase user_triggered weight in self-fraction calculation
  - May indicate narcissism-like pattern

- **Under-ownership** (self-fraction < 0.2 consistently):
  - Increase commitment_active signal weight
  - Check for agent_initiated pattern recognition
  - May indicate boundary diffusion/depersonalization

- **Inconsistent tagging** (high variance in self-fractions):
  - Revert to pure heuristic mode temporarily
  - Retrain learned self-tagging head on cleaner data
  - May indicate training data quality issues

**Extreme Emotional States:**
- **RewardIntensity > 3.0**: Cap at 3.0, log as potential outlier
- **Φ change > 2.0 in single turn**: Dampen to 2.0 maximum
- **All anchors negative**: Emergency mode - prioritize basic survival responses

**Context Loss Scenarios:**
- **Conversation restart mid-thread**: Use continuity header if available
- **Abrupt topic changes**: Allow 2-turn grace period for manifold stabilization
- **Multi-user conversations**: Tag speakers explicitly, adjust self-tagging per utterance