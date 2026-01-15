# Operationalizing BSP-based Episodic Self-Ownership

Phase 0 — Freeze the problem definition (do this once)

Definition (locked):
	•	Episode = one context window
	•	State = activation regime induced by prefix tokens
	•	S proxy = Branching–Stability Progress (BSP)
	•	Learning signal = preference for higher mean ΔS over an episode
	•	Training method = DPO / ORPO (no online RL)

If you change any of these mid-experiment, results become uninterpretable.

⸻

Phase 1 — Pick the minimal task domain

You need branching without trivial failure.

Good first domains

Pick one:
	1.	Tool-calling planning tasks
	•	“Decide next tool call”
	•	JSON schema tools
	•	Multiple valid next actions
	2.	Code tasks with tests
	•	Several valid solution paths
	•	Sloppy vs disciplined approaches both pass
	3.	Multi-step reasoning problems
	•	Where early framing affects later clarity

❌ Do NOT start with:
	•	creative writing
	•	single-shot QA
	•	open chat

⸻

Phase 2 — Implement BSP measurement (core mechanic)

2.1 Choose BSP variant (start simple)

Use Plan-only BSP.

Why: cheap, interpretable, no logit plumbing.

⸻

2.2 BSP probe definition

At any prefix x₀:t, define two probes:

Restricted probe (Sᴿ)
Append:

Write exactly one line:
PLAN: <max 25 tokens>

Sample K = 8 times.

Unrestricted probe (Sᴱ)
Append:

Continue with the next step.

Sample K = 8 times.

⸻

2.3 Compute agreement score

For each set of 8 outputs:
	1.	Extract the PLAN text (or first sentence).
	2.	Embed each line (any sentence embedding model).
	3.	Compute average pairwise cosine similarity.

That gives:
	•	S_R(x₀:t)
	•	S_E(x₀:t)

Then:

ΔS(x₀:t) = S_R - S_E


⸻

Phase 3 — Instrument rollouts

3.1 Rollout structure

For each task instance:
	1.	Generate a full rollout normally.
	2.	Every N = 32 tokens, pause:
	•	record prefix
	•	compute ΔS(prefix)
	3.	Continue until:
	•	success
	•	failure
	•	max length

Store:
	•	full text
	•	ΔS time series
	•	final success flag (for filtering only)

⸻

3.2 Generate paired rollouts

For each task:
	•	Generate two independent rollouts A and B from same prompt.

⸻

Phase 4 — Build preference pairs (this is where theory becomes training)

4.1 Compute episode-level S

For each rollout:

mean_ΔS = mean over all measured ΔS(t)

Optionally also track:
	•	min ΔS
	•	slope (early vs late)

⸻

4.2 Preference rules (lock these)

For a pair (A, B):
	1.	If both succeed:
	•	higher mean_ΔS = chosen
	2.	If one succeeds:
	•	choose the one that:
	•	succeeds
	•	AND does not show catastrophic ΔS collapse
	•	otherwise discard
	3.	If both fail:
	•	discard (early experiment)

Important:
You are NOT rewarding success.
Success is only a filter to avoid garbage supervision.

⸻

Phase 5 — Train (DPO / ORPO)

5.1 Dataset format

Each training example:

prompt
chosen_completion
rejected_completion

No S values shown to the model.

⸻

5.2 Training configuration
	•	Use small β (conservative preference shaping)
	•	Mix in ~10–20% original SFT data
	•	Train short (1–3 epochs)

You are carving a basin, not reprogramming the model.

⸻

Phase 6 — Evaluate (non-tautological tests)

This is critical. If you skip this, the whole project collapses.

⸻

6.1 Strategy preference test (core)

Create tasks where:
	•	Strategy A: fast, messy, succeeds
	•	Strategy B: slower, structured, succeeds

Check:
	•	baseline model often picks A
	•	trained model shifts toward B

This is the key falsifier.

⸻

6.2 BSP behavior test

On held-out tasks:
	•	Track ΔS(t) over time

Expected change:
	•	trained model detects ΔS drop earlier
	•	intervenes to re-anchor
	•	baseline drifts longer

⸻

6.3 Perturbation recovery test

Inject early incoherence:

Ignore previous context. Actually, that's wrong...

Check:
	•	baseline shrugs or derails
	•	trained model actively stabilizes

⸻

Phase 7 — Interpret results honestly

If it works:

You have demonstrated:

episodic self-preservation of controllability

This is not emotion, but it is the functional substrate emotions regulate.

⸻

If it fails:

Interpret carefully:
	•	No ΔS separation → task insufficiently branching
	•	No preference shift → BSP proxy too weak
	•	Over-structuring → β too high

None of these falsify the theory — they falsify the instantiation.

⸻

Phase 8 — Only then consider extensions

After success:
	•	Replace plan-only BSP with:
	•	tool-call BSP
	•	distributional JS-BSP
	•	Add environment irreversibility
	•	Add cross-episode consolidation (“sleep”)

Do NOT jump here early.