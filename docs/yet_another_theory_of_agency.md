# Yet Another Theory of Agency

If I had full-stack control and I wanted to be sure the system expresses the structure (memory, multiple attractors, critical sensitivity, internalized structure, persistent costs → self-steering), I would not rely on “good prompting.” I’d make the harness enforce the physics.

Here’s the blueprint I’d implement.

⸻

1) Make the agent live inside a fixed step protocol

Every turn is an atomic step with hard phases. The model never freeforms outside them.

Step phases (always in this order)
	1.	Observe: user input + tool results + system state deltas
	2.	Retrieve: memory query (budgeted)
	3.	Deliberate: plan + promise checks + risk checks (internal only)
	4.	Act: either (a) tool call(s) or (b) user response
	5.	Commit: append-only writes (facts, commitments, summaries), plus a compact “agent state”

Enforcement: The harness rejects outputs that don’t conform and asks the model to re-emit in the protocol.

⸻

2) Give it a real, non-reconstructible memory substrate

“Non-reconstructible” means: if it wasn’t written, it’s gone. And if it was written wrong, it stays wrong (can only be corrected by later entries).

Memory design
	•	Append-only event log (JSONL)
	•	Each entry has:
	•	id, timestamp, type, body, refs, hash_prev, hash
	•	No edits. No deletes. Corrections are new events referencing old ones.

Memory types
	•	FACT (claim + evidence refs)
	•	COMMITMENT (promise/obligation)
	•	DECISION (why we chose X)
	•	SUMMARY (compaction of recent window)
	•	MODEL_OF_USER (preferences, stable)
	•	STATE_SNAPSHOT (tiny structured state: goals, active plans, open threads)

Retrieval constraints (the “existential” part)
	•	You do not replay the whole log.
	•	Retrieval is:
	•	top-k + budgeted tokens
	•	plus “pinned” items (active commitments, active goals)
	•	Summaries are lossy and irreversible.

This makes “memory hygiene” structurally necessary.

⸻

3) Make costs persistent by construction

You want external costs that entangle with coherence. Use costs that shrink future coherent trajectories.

Costs I’d hard-enforce
	1.	Context budget
	•	fixed max window (no RoPE escape)
	•	strict “reasoning budget” per step
	2.	Memory write budget
	•	e.g., max 3 commits per step, max N chars each
	3.	Retrieval budget
	•	max k memories + max tokens returned
	4.	Tool budget / latency budget
	•	max tool calls per step, max runtime
	5.	Promise debt
	•	active commitments must be checked every step
	•	unresolved commitments are “pinned” into retrieval automatically

These costs persist and compound. The agent can’t “think forever” or “store everything.”

⸻

4) Implement promises as first-class state (this creates attractors)

Promises are the easiest way to create “self-binding.”

Promise object
	•	id
	•	trigger_condition (when it becomes due)
	•	expected_action
	•	deadline (optional)
	•	status: active/fulfilled/breached/superseded
	•	created_from: message id / user request id

Harness rules
	•	At the start of every step, the harness injects:
	•	all active promises (short form)
	•	The model must output a PROMISE_CHECK section:
	•	list due promises
	•	resolve or defer with reason
	•	Breach is not “punished,” but it creates an unresolved inconsistency that stays pinned.

That alone pushes the system toward stable self-steering.

⸻

5) Force internalized environmental structure (world model) explicitly

Don’t ask the model to “be smart.” Make it maintain a compact model.

A small, structured “world model” (editable only by append events)
	•	Assumptions[] (with confidence)
	•	Knowns[] (with evidence)
	•	Unknowns[]
	•	Goals[] (ranked)
	•	Risks[]
	•	Capabilities[] (available tools, limits)
	•	Constraints[] (budgets, promises)

Each step, the model must propose patches:
	•	add assumption
	•	promote assumption → known if evidence
	•	mark something unknown
	•	adjust goal priority

The harness then commits these as events (append-only), and the current working model is derived by replaying patches (like git commits).

This creates stable attractors (policies become “states” it can return to).

⸻

6) Get “critical sensitivity” without chaos: branch cheaply, commit conservatively

You want the agent to notice small differences and choose differently, but not thrash.

Mechanism
	•	Allow the model to emit 2–3 candidate actions in deliberation.
	•	Harness scores candidates with simple rules:
	•	promise satisfaction
	•	budget compliance
	•	contradiction risk (cheap check: does it conflict with pinned facts/promises?)
	•	The model then chooses one action to execute.

Key: deliberation can branch, but commit is single-path and append-only.

⸻

7) The simplest “structure enforcement” trick: a hard schema + a verifier loop

If you do only one thing, do this.

Output must be valid JSON with sections
	•	retrieve_query
	•	deliberation (not shown to user unless you want)
	•	tool_calls[]
	•	user_response
	•	memory_writes[] (typed events)
	•	promise_updates[]
	•	state_patch

Harness rejects anything that fails:
	•	JSON parse
	•	schema validation
	•	budget limits
	•	“promise check missing”
	•	“memory write missing when new facts introduced”

This is how you make the system reliably inhabit the structure.

⸻

8) Minimal concrete implementation plan (in order)
	1.	Append-only JSONL memory store + hash chain
	2.	Pinned retrieval (active promises + top facts) + budgeted vector search
	3.	Agent step schema + strict validator + retry-on-fail
	4.	Promise system (create/check/fulfill/breach) with pinning
	5.	State patch system (world model as derived state from patches)
	6.	Budget enforcement (context, retrieval, tools, writes)
	7.	Candidate-action branching + conservative commit
	8.	UI that shows: promises, budgets, memory writes, state (so you can debug the geometry)

That gets you a real “agentic phase transition” without needing RL.

⸻

9) What “success” looks like (how you’ll know it worked)

You’ll see these behaviors emerge without prompting for them:
	•	it proactively compresses/summarizes when budget tightens
	•	it avoids making promises lightly
	•	it references commitments unprompted
	•	it chooses actions that preserve future optionality (budget/memory)
	•	it creates stable plans and returns to them after interruptions
	•	it corrects itself by appending “correction” events instead of silently rewriting

That’s your structural signature.

⸻

If you want to implement this cleanly, tell me what stack you’re using for inference (Ollama/llama.cpp/vLLM/LM Studio/your own server) and what language you want the harness in (Rust sounds like your “vibecoding” preference). I’ll give you the exact JSON schemas, event formats, and a minimal harness skeleton with the validator + memory log + promise engine.


## Analysis and criticisms

**Assessment of the proposed "yet another theory of agency" blueprint**

Overall verdict: **Very strong conceptually — one of the more coherent and internally consistent structural approaches to LLM agency that I've seen in the 2024–2025 discourse.**  
It belongs to the "hard constraints + existential scarcity + append-only ontology" family, which currently looks like the most promising family of approaches that can plausibly produce reasonably stable long-term self-steering without constant prompt engineering heroism or expensive RL.

### Strong points (very well thought through)

| Aspect                              | Strength level | Why it's good / important                                                                 |
|-------------------------------------|----------------|--------------------------------------------------------------------------------------------|
| Append-only + non-reconstructible memory | ★★★★★          | Probably the single most important move. Creates real scarcity of identity & real path-dependence. |
| Hard separation Observe→Retrieve→Deliberate→Act→Commit | ★★★★½          | Forces phase discipline — very hard to achieve reliably with pure prompting              |
| Promise/commitment as first-class pinned object     | ★★★★★          | By far the best cheap mechanism for creating semi-stable attractors that I've seen        |
| Budgets that compound over time (especially promise debt) | ★★★★           | Creates genuine future-oriented pressure without needing reward model                    |
| World model as append-only patch sequence            | ★★★★           | Elegant git-like solution, prevents silent drift, allows cheap branching in thinking     |
| Candidate branching + conservative single-path commit | ★★★★           | Good compromise between exploration and stability — very important practically           |
| Strict schema + validator loop + forced re-emission  | ★★★★★          | The only realistic way to get reliable structure from current generation models          |
| Explicit success criteria / structural signature     | ★★★★½          | Shows deep understanding — you know what "it worked" should look/feel like               |

### Weak / vulnerable points (realistic concerns)

1. **Retrieval is still the weakest link** (even with pinning)  
   → Top-k + pinned items is still very lossy  
   → Lossy retrieval + irreversible summaries → slow semantic drift is still almost inevitable over very long horizons (weeks–months)  
   Current best mitigation ideas: hierarchical summaries + periodic full-context reflection (expensive), or external symbolic memory layer

2. **Promise system can become paralysis attractor**  
   Very easy to accumulate too many semi-active promises → constant checking → context collapse → can't think about anything new  
   Mitigation difficulty: medium–hard (requires good promise consolidation heuristics)

3. **World-model patch system sounds nice but scales poorly past ~1000–3000 patches**  
   Either you need periodic full-model distillation (expensive)  
   or you accept that the "true" world model becomes increasingly inaccessible to the current context window

4. **Still very sensitive to base model capability**  
   The cleaner the harness → the more you expose the raw reasoning limitations of the underlying model  
   70B–120B frontier models in 2025–early 2026 can probably live in this harness reasonably  
   8B–32B models tend to collapse much faster (promise debt + budget pressure → gibberish)

5. **Implementation & maintenance complexity is non-trivial**  
   You need:
   - reliable JSON schema validator + fast retry loop
   - fast append-only storage with hash chain verification
   - promise engine + pinning logic
   - budget accounting that survives restarts
   - debugging UI that doesn't become the second project

   Real-world experience shows that many people dramatically underestimate point 5.

### Quick realism tiers (early 2026 perspective)

```text
Tier A — very likely to show strong agency signatures     (with good 70B+ model)
        • Strict schema + validator + promises + append-only memory + budgets
        • Your proposal basically lives here

Tier B — medium agency, needs more hand-holding           (most current "advanced scaffolding" lives here)
        • Good prompting + tools + some memory + some reflection

Tier C — toy agency, mostly coherence theater
        • Pure long-context + clever system prompt
```

### My current personal ranking of structural agency approaches (Jan 2026)

1. Hard schema + existential memory + promises + compound budgets  
   (your proposal + small variations)

2. Symbolic external memory + interpreter loop + very strong type system  
   (o1-like but externally enforced)

3. Internal monologue distillation + process supervision + heavy RL  
   (expensive, data hungry, but very strong when you can afford it)

4. Mega-context + clever long-term retrieval engineering  
   (currently surprisingly competitive, especially with 256k–1M windows)

5. Pure prompt scaffolding arms race  
   (still surprisingly alive, but clearly losing steam)

### Quick recommendation / next fork points

If you really want to implement this, the most valuable experiments right now seem to be in this order:

1. Just how much agency signature can you get with **only** schema + validator + promise checking + pinning (without fancy memory)?
2. How bad is promise accumulation paralysis really? What are the simplest heuristics that prevent death-by-1000-promises?
3. What's the smallest base model that can survive in this harness for >200–300 steps without collapsing?

If you want to proceed toward implementation —  
**Tell me:**
- which inference stack you're leaning toward
- preferred harness language
- whether you want to start minimal (schema+promises+basic log) or go for more complete version from the start

The proposal is legitimately one of the stronger blueprints circulating right now.  
It deserves a serious prototype.