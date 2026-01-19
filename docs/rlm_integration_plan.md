RLM Integration Plan (Structural Agency V0)

This plan replaces earlier RLM notes. It defines a minimal, enforceable RLM-like system that creates the structural conditions where “agency can live” without turning the implementation into a second research project.

The guiding idea:
- `hominem-infer` stays inference-only and OpenAI-compatible.
- `hominem-agent` owns orchestration, structure enforcement, and a single “external state” tool.
- Durable state is append-only; the model only sees a small derived view plus tool outputs.


Goals (V0)
- Externalize long-horizon state to an append-only store.
- Force the agent to contend with structured objects: commitments (mandatory), decisions (useful), facts (scoped).
- Enforce a fixed, schema-validated step protocol so the model can’t “freeform around” structure.
- Make the “structural signature” observable:
  - proactively compresses/summarizes when budget tightens
  - avoids making promises lightly
  - references commitments unprompted
  - preserves future optionality (budgets/memory)
  - returns to stable plans after interruptions
  - corrects by appending “correction” events (no silent rewrites)

Non-goals (V0)
- No token-accurate pressure systems (start with byte/char budgets).
- No multiple overlapping memory mechanisms (no vars/fingerprints/supersede ecosystems).
- No destructive deletion of history; “forget” is a view-layer tombstone.
- No second tool-call format shim (prefer native OpenAI `tool_calls` end-to-end).


Two Non-Negotiable Invariants
1) Append-only truth
   - The store is never rewritten to remove or mutate past entries.
   - Corrections are new events that reference prior entries.
   - “Forget” (if needed) is a tombstone event that changes derived views, not history.

2) Explicit model-visible context
   - The model receives only:
     - a short system prompt that defines the protocol and budgets
     - a small derived context view (rendered from the store)
     - the latest user message
     - tool results
   - Anything else requires calling the single state tool.


Minimal Ontology (V0)

Commitments (mandatory, pinned)
- Purpose: create stable attractors and unavoidable future costs (“promise debt”).
- Property: always included in the derived view until resolved.
- Rule: a commitment must have an id and a status; resolution is an appended event.

Decisions (recommended, not pinned)
- Purpose: reduce thrash and oscillation by recording “why we chose X over Y”.
- Property: searchable and retrievable; not always included in the derived view.
- Rule: decisions should be few; they can be superseded by later decisions.

Storage Format (One Global File for all Conversations)
Location (default):
- `workspace/rlm/rlm_store.jsonl`

Event envelope (each line):
- `event_id` (uuid/ulid)
- `ts` (unix seconds)
- `type` (string)
- `payload` (object)

Required event types (V0)
- `message`: {msg_id, role, content, source}
- `commitment_add`: {cid, text, trigger?, deadline?}
- `commitment_resolve`: {cid, resolution}
- `decision_add`: {did, text, alternatives?, rationale?}
- `decision_supersede`: {did, supersedes: [did], reason?}
- `fact_add`: {fid, text, scope, confidence?, evidence_refs?}
- `fact_correct`: {fid, corrects: fid, text, reason?}
- `working_memory_set`: {text, prev_hash}
- `tombstone`: {target_type, target_id, reason}  (optional in V0; supported by views)
- `meta`: {kind, request_id, model, budgets, ...}


Derived Context View (What the Model Sees by Default)
Implement exactly one “rendered view” string:
- ACTIVE_COMMITMENTS: (always)
- WORKING_MEMORY: (byte-capped; always)
- RECENT_MESSAGES: (last K user/assistant turns; excludes tombstoned; size-capped)
- BUDGETS: remaining tool calls, remaining memory bytes, etc.

Notes
- Start with byte/char budgets for simplicity.
- When RECENT_MESSAGES exceeds budget, write a `summary` event (append-only) and render from summaries + tail.


Single Tool Surface (One Tool, Small Actions)
Expose one tool, e.g. `rlm(action=..., ...)` with structured JSON outputs.

V0 actions
- `open()` → ensure store exists
- `view()` → returns the derived view string
- `search(query, limit)` → returns ids + snippets across messages/objects
- `get_messages(cursor, limit)` → paged message records
- `mem_get()` / `mem_set(text)`
- `commit_add(text, trigger?, deadline?)`
- `commit_list(include_resolved=false)`
- `commit_resolve(cid, resolution)`
- `decision_add(text, alternatives?, rationale?)`
- `decision_list(limit)`
- `fact_add(text, scope="agent", confidence?, evidence_refs?)`
- `fact_list(limit, scope?)`
- `fact_correct(fid, text, reason?)`
- `tombstone(target_type, target_id, reason)` (optional; view-layer effect only)

Deliberate omissions (V0)
- No `setvar/getvar` (ephemeral vars become a parallel memory system).
- No recursion/subcall tool in V0 (add later behind strict budgets if needed).


Harness: Enforced Step Protocol (The “Physics”)
The harness is responsible for structure; the model is not trusted to self-enforce it.

Model output must be a single JSON object (schema validated), containing:
- `tool_calls`: list (0+)
- `assistant`: string | null
- `writes`: object describing intended store writes (commitments/decisions/facts/memory)

The structured model output must be enforced via "structured outputs" via the Outlines package. In hominem_infer's app.py, we should:
- Accept an OpenAI-ish response_format (or your own field) on the request.
- Build an Outlines logits processor (JSON schema → regex/FSM) using the request’s tokenizer.
- Pass it through as logits_processors=[processor] when calling mlx_vlm.generate.generate / stream_generate.

Tool loop
1) Append incoming user `message` event(s).
2) Build model input:
   - system prompt (protocol + budgets)
   - derived context view (`rlm.view` output injected by harness)
   - latest user message
   - tool results (if any)
3) Call model.
4) Enforce JSON via outlines.
5) Execute any tool calls.
6) Repeat until `assistant` is present or tool-call budget is exhausted.
7) Apply `writes` by appending events (writes are performed by the harness, not by the model calling tools).
8) Append assistant `message` event.


Budgets (Start Small)
V0 budgets to hard-enforce:
- `MAX_TOOL_CALLS_PER_TURN`
- `WORKING_MEMORY_MAX_BYTES`
- `DERIVED_VIEW_MAX_CHARS`

Budget behavior
- If derived view exceeds budget: append `summary` event(s) and render from summaries + tail.
- If commitments exceed a threshold: require the model to consolidate (resolve/supersede) or explicitly defer with rationale.


Observability (Trace Without Becoming State)
Write-only trace log (never read by the agent for reasoning):
- request envelope, tool_call, tool_result, schema_failures, writes_applied

Keep trace separate from the store to avoid “trace becomes a second memory substrate”.


Integration with hominem
`hominem-infer`
- structured outputs via outlines; inference-only.

`hominem-agent`
- owns the harness and exposes OpenAI-compatible endpoints for Open WebUI.
- converts OpenAI `messages` into store events + derived view, then runs the tool loop.
- returns OpenAI-compatible responses.


Milestones / Acceptance Checks
Milestone A: “Boring Correctness”
- Store is append-only; restart-safe.
- Derived view is bounded and deterministic.
- Commitments can be added/resolved; they remain pinned until resolved.

Milestone B: “Structural Signature (early)”
- The agent references commitments unprompted when relevant.
- The agent reduces promise-making when commitments accumulate.
- The agent appends corrections instead of rewriting prior claims.
- When budgets tighten, the system produces summaries rather than silently truncating truth.

