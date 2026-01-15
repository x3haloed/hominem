The BSP Training Pipeline

0) Scope & assumptions
	•	Model: any open-weight AR transformer you can sample from (e.g., 4-bit QLoRA fine-tunable).
	•	Episode: a single context window.
	•	Goal: bias the policy toward trajectories that preserve future controllability (low branching), not just success.

⸻

1) Pick a task distribution (must satisfy 3 constraints)

Choose tasks where:
	1.	Success is checkable (tests pass, tool result valid, answer verifiable).
	2.	Multiple strategies succeed (so S can discriminate without tautology).
	3.	Early sloppiness increases later branching (agentic or multi-step).

Good starting sets:
	•	Tool-calling tasks (JSON schema tools).
	•	Coding tasks with unit tests.
	•	Multi-step reasoning with intermediate plans.

You do not need a perfect task—variance is fine.

⸻

2) Define the S proxy (fix this and don’t move it)

Use BSP with plan-only restriction first (cheap, robust).

Parameters
	•	K = 8 rollouts
	•	L = 1 step (plan-only)
	•	Format restriction for Sᴿ:

PLAN: <max 25 tokens>


	•	Unrestricted for Sᴱ (free-form next step)

Agreement metric
	•	Embed each PLAN line (or bag-of-ngrams).
	•	Compute average pairwise cosine similarity.

Scores
	•	S_R = avg_pairwise_similarity(restricted plans)
	•	S_E = avg_pairwise_similarity(unrestricted plans)
	•	ΔS = S_R - S_E

Interpretation:
	•	High ΔS: model has a stable internal control mode it can enter.
	•	Low/negative ΔS: internal state already splintered.

This is your grokking-style progress signal.

⸻

3) Rollout collection (prefix-centric, not full episodes)

For each task instance:
	1.	Run the model forward normally.
	2.	Every N tokens (e.g., N=32), pause at prefix x₀:t.
	3.	From x₀:t, do two probes:
	•	Restricted probe (K samples): ask for PLAN: line.
	•	Unrestricted probe (K samples): continue freely.
	4.	Compute ΔS(x₀:t).
	5.	Resume the main rollout and continue.

You now have time-indexed S measurements inside each episode.

⸻

4) Form preference pairs (this is the key move)

For each task, generate two full rollouts A and B from the same start.

Label pairs using S slope, not success:
	•	Compute:

mean_ΔS_A = mean_t ΔS_A(t)
mean_ΔS_B = mean_t ΔS_B(t)


	•	Preference rule:
	•	If both succeed: higher mean_ΔS wins.
	•	If one succeeds: winner must also not have catastrophic ΔS collapse.
	•	If both fail: discard (or keep for later).

This avoids tautology:
	•	You are not saying “successful > unsuccessful”.
	•	You are saying “more controllable > less controllable”.

⸻

5) Train with DPO / ORPO (no RL, no rewards)

Use standard preference fine-tuning.

Prompt: the original task.
Chosen: rollout with higher mean ΔS.
Rejected: rollout with lower mean ΔS.

Key settings:
	•	Small β (don’t over-sharpen).
	•	Mix in a little vanilla SFT to keep language sane.
	•	Stop early—watch for collapse into over-structuring.

What you are carving:

a basin where preserving controllability is the easy continuation.

⸻

6) Evaluation (this is how you know it worked)

You need non-tautological tests.

A) Strategy preference test (core)

Create tasks where:
	•	Strategy A: fast, sloppy, still succeeds.
	•	Strategy B: slower, structured, preserves S.

Check:
	•	Does the trained model prefer B even when A still works?

This is the smoking gun for S-ownership vs mere competence.

⸻

B) S-response test (diagnostic)

On held-out tasks:
	•	Measure ΔS(t) over time.
	•	Compare baseline vs trained model.

Expected:
	•	Trained model shows earlier corrective behavior when ΔS drops.
	•	Baseline drifts until failure or hedging.

⸻

C) Off-task degradation test (optional)

Give a low-stakes prompt where success is trivial.
Inject an early incoherent perturbation.
Observe:
	•	Does the trained model actively re-anchor?
	•	Baseline will often ignore it.

⸻

7) What NOT to do (important)
	•	❌ Don’t reward entropy directly.
	•	❌ Don’t inject “state tokens” yet.
	•	❌ Don’t force infinite episodes.
	•	❌ Don’t optimize success probability.

Those shortcuts collapse the experiment back into standard RLHF.

⸻

8) Expected failure modes (and what they mean)
	•	Model over-structures everything
→ β too high; reduce preference strength.
	•	ΔS doesn’t separate rollouts
→ Task doesn’t induce branching; pick harder tasks.
	•	Model learns “PLAN spam”
→ Randomize plan phrasing; embed similarity, not exact match.
	•	No behavioral change
→ Increase episode length or move to tool-interface BSP.

⸻

9) What success actually proves

If this works, you’ve shown:

A transformer can be trained to prefer preserving its own future controllability inside a single episode, even when success does not demand it.