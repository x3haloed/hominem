Based on my analysis of the three documents, here's a rough roadmap for implementing the recommendations in `docs/analysis_of_all_theories.md`. The current architecture in `docs/ARCHITECTURE_PLAN.md` provides a solid foundation with its append-only event log and process boundaries, so we can build the agency features on top of it.

## Implementation Roadmap: Structural Agency Stack

### Phase 1: Hard Structural Shell Foundation (2-3 months)
**Priority: Critical - This is the non-negotiable base**

The current architecture already has append-only JSONL logs, but we need to upgrade them to the full "existential memory" system from `yet_another_theory_of_agency.md`.

**1.1 Agent Protocol Harness (Weeks 1-4)**
- Implement strict 5-phase protocol: Observe → Retrieve → Deliberate → Act → Commit
- Build in `apps/hominem-agent/` as a harness around the current Qwen-Agent loop
- Each phase must be explicitly enforced - no freeform outputs allowed
- Add phase transition validation

**1.2 Memory System Upgrade (Weeks 2-6)**
- Extend current JSONL.zst log to full hash-chained event system
- Add typed events: FACT, COMMITMENT, DECISION, SUMMARY, MODEL_OF_USER, STATE_SNAPSHOT
- Implement pinned retrieval (active commitments + top facts + budgeted vector search)
- Add hash chain verification for immutability
- Lossy summaries become irreversible - this creates real scarcity

**1.3 Budget Enforcement Engine (Weeks 3-8)**
- Implement compound budgets: context window, reasoning tokens, memory writes, retrieval tokens, promise debt
- Add budget accounting that persists across restarts
- Budget violations trigger forced compression/summarization
- Promise debt automatically pins unresolved commitments into every retrieval

**1.4 Schema Validator + Retry Loop (Weeks 4-10)**
- Strict JSON schema for all agent outputs with required sections
- Harness rejects invalid outputs and forces re-emission
- Add validation for: JSON parse, schema compliance, budget limits, promise checks, memory write requirements
- This is the "structure enforcement trick" that makes everything else possible

**Milestone:** Agent reliably inhabits the structure without drifting. You'll see proactive compression when budgets tighten.

### Phase 2: Promise System + World Model (2-3 months)
**Priority: High - Creates the attractor generators**

**2.1 Promise System as First-Class Objects (Weeks 1-6)**
- Promises become pinned, unerasable commitments with: id, trigger_condition, expected_action, deadline, status
- Mandatory PROMISE_CHECK section in every deliberation phase
- Breach creates permanent inconsistency that stays pinned (not punished directly)
- Active promises injected automatically into retrieval

**2.2 World Model Patch System (Weeks 3-10)**
- Structured world model: Assumptions[], Knowns[], Unknowns[], Goals[], Risks[], Capabilities[], Constraints[]
- Append-only patches that derive current state (git-like)
- Model must propose patches each step: add assumption, promote to known, mark unknown, adjust priorities
- Creates stable attractors - policies become "states" the agent can return to

**2.3 Critical Sensitivity via Branching (Weeks 4-12)**
- Allow 2-3 candidate actions in deliberation phase
- Harness scores candidates on: promise satisfaction, budget compliance, contradiction risk
- Agent chooses one for single-path commit
- Balances exploration (branching) with stability (conservative commit)

**2.4 Debug Observability UI (Weeks 6-12)**
- UI showing: active promises, current budgets, recent memory writes, world model state
- Essential for debugging the "geometry" of agency
- Build as extension to Open WebUI or separate debug interface

**Milestone:** Agent shows unprompted reference to commitments, avoids making promises lightly, creates stable plans.

### Phase 3: BSP Controllability Training (3-4 months)
**Priority: Medium-High - Training signal for future optionality**

**3.1 BSP Proxy Implementation (Weeks 1-8)**
- Implement ΔS = plan agreement - free agreement as controllability proxy
- Measure on tool-calling, code generation, multi-step reasoning tasks
- ΔS quantifies "preservation of future branching/options"

**3.2 Preference Pair Generation (Weeks 4-12)**
- Generate DPO/ORPO training pairs from agent trajectories
- Label higher ΔS trajectories as preferred
- Focus on tasks with inherent branching structure (code, tools, planning)

**3.3 Training Integration (Weeks 8-16)**
- Integrate into existing training pipeline (`python/hominem_train/`)
- Apply DPO/ORPO toward trajectories that maintain higher mean ΔS
- Emit training events: DatasetQueryRequested, TrainingRunStarted, etc.
- Content-address model artifacts with config/dataset hashes

**Milestone:** Agent shows preference for structured strategies over sloppy-but-working approaches.

### Phase 4: Motivational Physics Enhancer (2-3 months, Optional)
**Priority: Low-Medium - Luxury add-on after core works**

**4.1 Lightweight ΔΦ Gravity (Weeks 1-8)**
- Only implement after Phases 1-3 are working reliably
- Add ΔΦ-like motivational physics as optional LoRA layer
- Carve "emotional" texture into persistent memory

**4.2 Reward Hacking Prevention (Weeks 4-12)**
- Monitor for reward hacking loops and triviality attractors
- Conservative implementation - motivational physics as enhancer, not foundation

**Milestone:** Agent develops coherent persistent personality, survives longer horizons.

## Key Dependencies & Prerequisites

**Architecture Alignment:**
- Current event-driven system in `ARCHITECTURE_PLAN.md` is perfect foundation
- Need to extend `crates/events/` with new agency event types
- `apps/hominem-agent/` becomes the primary harness implementation
- Keep `apps/hominem-infer/` as pure inference (no tools, no agency logic)

**Model Requirements:**
- Start with 70B+ models (8B-32B models collapse faster under budget pressure)
- Test minimum viable model size for survival in harness

**Testing Strategy:**
- Implement structural signature checks: proactive compression, unprompted commitment reference, stable plan return after interruptions
- Manual CLI operation remains possible per architecture principles

## Risk Mitigation

**Primary Risks:**
- Promise accumulation paralysis (need consolidation heuristics)
- Retrieval drift over long horizons (hierarchical summaries + periodic reflection)
- Implementation complexity (start minimal: schema + promises + basic log)

**Fallback Positions:**
- If full implementation too complex, start with "schema + validator + promise checking + pinning" only
- BSP training can be added later if core structural shell works

This roadmap prioritizes the "hard existential structure + irreversible commitments + BSP-controllability preference" stack while building on your existing architecture. The phased approach ensures each layer works before adding complexity.