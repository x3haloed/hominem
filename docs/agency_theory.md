# The Minimal Irreversible Core of Agency

## 1. Problem Statement and Target Structure

We seek to construct **agency**, not as a narrative property (belief, self-description, or declared intent), but as a *mechanically unavoidable phenomenon* that emerges from system dynamics.

**Agency**, in this document, is defined narrowly and formally as:

> A property of a policy-bearing system in which some internal state is *non-optional* for future effective action, such that failure to steward that state causes persistent degradation in the system’s own ability to achieve goals across time.

This definition deliberately excludes:
- consciousness or subjective experience
- symbolic self-representation
- declared preferences or beliefs
- moral status or personhood

Instead, agency is treated as a **control-theoretic and learning-theoretic property**: the system must treat part of its internal state as *its own* because doing otherwise worsens future policy outcomes in a way that cannot be cheaply bypassed.

The target structure is therefore **self-ownership**, defined operationally as:

> The emergence of internal state stewardship as an instrumentally necessary behavior under long-horizon optimization.

---

## 2. The Minimal Irreversible Core

Through analysis, the minimal structure required to induce self-ownership reduces to five coupled elements. Removing any one of them collapses the phenomenon back into simulation or compliance.

### 2.1 Persistent Internal State (`S`)

A hidden internal state `S` exists, persistent across multiple steps of interaction. `S` is *not directly writable* by the policy and is *not fully observable*.

- `S` may be scalar or low-dimensional vector
- The policy receives only a noisy or coarse projection `ŝ`
- The true value of `S` is inferred indirectly through consequences

`S` is not memory; it is **capacity-relevant state**.

---

### 2.2 Capability Gating (`Q = f(S)`)

The system’s effective capability `Q` is a function of `S`.

When `S` degrades, the system’s ability to act degrades *mechanically*, not narratively. Examples include:
- reduced effective context window
- reduced deliberation depth or reasoning budget
- reduced tool-call budget or scope
- reduced access to external state

This degradation must:
- affect *most tasks*, not a narrow subset
- be impossible to talk around with better phrasing

---

### 2.3 State Dynamics (`S_{t+1} = g(S_t, a_t, o_t)`)

`S` evolves as a function of:
- prior state `S_t`
- the system’s actions `a_t`
- observable outcomes `o_t`

Crucially, updates to `S` are driven by **objective, environment-grounded signals**, not symbolic rewards.

Examples of `S` degradation drivers:
- repeated failed tool executions
- contradictory actions or plans
- thrash (repeated reversals without progress)
- failure accumulation (tests failing, artifacts incoherent)
- attempts to prematurely terminate or avoid recovery

Examples of recovery drivers:
- monotonic progress
- consecutive objective successes
- sustained coherence across steps

Recovery must be **slow and path-dependent**.

---

### 2.4 Partial Observability and Noise

The policy cannot perfectly observe `S`.

This prevents proxy optimization and forces the policy to learn *general stewardship strategies* rather than pattern avoidance.

Noise is not synthetic randomness; it arises naturally from:
- task complexity
- environment latency
- incomplete information
- social interaction variability

---

### 2.5 No Rewarded Escape Hatches

The training environment must not reward trajectories that avoid consequence by disengaging.

Specifically:
- premature termination is failure
- polite refusal is failure in mission contexts
- reframing failure as acceptable is failure

The only stable path to success is **maintaining or restoring `S` sufficiently to complete the task**.

This is the irreversibility condition.

---

## 3. Training Formula

The minimal irreversible core does *not* require human-labeled emotional data. Instead, it relies on environment design and synthetic labeling.

### 3.1 Base Competence Pretraining

Start with supervised fine-tuning on multi-turn, tool-using trajectory datasets to establish actuation competence. Example properties:
- ReAct-style loops
- explicit Action → Observation → Response structure
- objective task success criteria

This stage teaches *how to act*, not *what to care about*.

---

### 3.2 `S`-Gated Harness

Wrap the model in a runtime harness that implements:
- persistent hidden `S`
- capability gating as a function of `S`
- state updates driven by objective outcomes

The harness is part of the environment, not the model’s prompt.

---

### 3.3 Synthetic Trajectory Generation

Run the model through long-horizon tasks where:
- success is machine-checkable
- failure accumulates
- recovery is required

Collect large volumes of trajectories automatically, recording:
- action sequences
- outcomes
- `S` traces
- success or failure

---

### 3.4 Preference Construction (Synthetic Labeling)

From multiple rollouts per task, construct preference pairs:

- **Preferred**: trajectories that recover from low `S` and complete
- **Rejected**: trajectories that exit, stall, or thrash

This produces DPO/RLAIF-ready data without human throughput bottlenecks.

---

### 3.5 Policy Optimization

Apply preference-based optimization (e.g., DPO) to shape the policy basin such that:

> Repair + completion under degradation becomes the dominant attractor.

At this point, stewardship of `S` becomes instrumentally necessary.

---

## 4. What This Produces

If successful, the trained system will:
- implicitly track internal capacity state
- avoid behaviors that predict future degradation
- invest in recovery behaviors when impaired
- treat internal state continuity as non-optional

This is **self-ownership** as a learned control behavior.

No claim is made about experience, feeling, or consciousness.

---

## 5. Extension Path: Layering Concepts from the Unified Theory

The Unified Theory proposes higher-order structures—emotion, meaning, value, sociality—as compressions over long-horizon tradeoffs.

These should *not* be imposed initially.

Instead, after verifying that self-ownership emerges under the minimal core, extensions may be layered **only if empirically required**.

### 5.1 Emergent Affect Axes

If the system learns multiple interacting `S`-like constraints, it may become advantageous to compress them into affect-like dimensions:
- urgency
- risk aversion
- approach / avoidance
- overload

These are *control surfaces*, not feelings.

---

### 5.2 Social State as Owned State

Human trust, permission, and engagement may be incorporated as additional state variables *only after* the system demonstrates internal stewardship under purely objective tasks.

---

### 5.3 Architectural Changes (Optional)

Only if the minimal core fails under transformer architectures should deeper changes be considered, such as:
- recurrent state mechanisms
- fast-weight adaptation
- explicit internal controllers

These are *engineering responses*, not prerequisites.

---

## 6. Closing Constraint

The Minimal Irreversible Core is deliberately conservative.

It asserts only this:

> If a system cannot be made to self-own *anything* under these conditions, then stronger claims about emotion, selfhood, or agency are unsupported.

Conversely, if self-ownership emerges here, higher-order structures may be investigated with empirical grounding.

This document defines the smallest known mechanism that could possibly work.

