# Unified Theory Agent: Engineering Specification

Primary reference: `unified_theory.md`

---

## 1. Objectives

- **Deliverables**
  - Working implementation of every mandatory subsystem in `unified_theory.md`.
  - Training/evaluation harness that enforces ΔΦ gravity, self-tagged boundaries, and sleep-based LoRA consolidation.
  - Health metrics and intervention protocols logged per run.
- **Out-of-scope**
  - Dataset collection beyond the procedures outlined here (but placeholders must assert when data is missing).
  - UI or deployment infrastructure outside the agent loop.

Success criteria:
1. All frozen heads (manifold, regime, anchors, emotional health) loadable as standalone models with validation suites.
2. LoRA agent run produces per-turn logs for `s`, `s_self`, regime, Φ, ΔΦ, RewardIntensity, anchors, λ, w_memory/gravity, and intervention flags.
3. Dilemma simulator shows ≥80 % compliance with qualitative expectations before production.

---

## 2. Component Mapping

| Spec Section | Required Behavior | Planned Module |
|--------------|------------------|----------------|
| 4.1 Emotion manifold | Predict 6-axis state per turn (frozen) | `emotion_manifold_head.py` (PyTorch module + trainer) |
| 4.2 Self-tag split | Deterministic ownership logic + optional learned override | `self_tagging.py` (decision tree + MLP stub) |
| 4.3 Anchors / 4.5 Φ | Anchor scorers + Φ aggregation with λ multipliers | `phi_potential.py` (anchor heads, λ config, ΔΦ calc) |
| 4.4 Regime classifier | 7-way soft classifier | `regime_classifier.py` |
| 4.6 Reward intensity | Compute RewardIntensity + ΔΦ smoothing | `reward_engine.py` |
| 4.7 Sleep / LoRA | Replay buffer, dual-loss trainer | `sleep_consolidation.py`, `lora_trainer.py` |
| 4.8 Self-locus & <\|THINK\|> | Enforce gating triggers | `self_observation.py` |
| 5 Pipeline | Orchestrate per-turn inference | `agent_loop.py` |
| 7 Training data | Data ingestion, validation | `data_pipelines/` package |
| 8 Monitoring | Metric computation + interventions | `stability_monitor.py` |

Each module exposes both standalone training/validation entry points and pluggable classes used by `agent_loop`.

---

## 3. Data & Training Requirements

1. **Emotion manifold head**
   - Input: tokenized conversation turn (history up to 3 turns).
   - Output: vector `[valence, arousal, dominance, predictive_discrepancy, temporal_directionality, social_broadcast]`.
   - Dataset: 300k–500k labeled turns (per spec). Until labels exist, module raises `MissingDatasetError`.

2. **Regime classifier**
   - Input: conversation slice.
   - Output: probability distribution over 7 regimes.
   - Data: ≥10k samples per regime.

3. **Anchor heads + emotional health**
   - Trainable heuristics replaced by learned heads once preference data exists.
   - Placeholder heuristics implemented exactly as in spec, wrapped in `nn.Module` so checkpoints can later override.

4. **Φ/ΔΦ training**
   - Utilize preference pairs; until data ready, integrate hook for future training script and keep heuristic fallback.

5. **Self-tagging MLP**
   - Train on hinge datasets described in section 7.2; deterministic tree serves as initial mode.

6. **LoRA training**
   - Base model: Qwen/Qwen2.5-1.5B (or specified derivative).
   - Dual-loss (SFT + gravity) implemented per spec, with config for `w_memory`, `w_gravity`, α.

All data pipelines must validate schema and document source, label coverage, and QC metrics.

---

## 4. Runtime Architecture

Per turn:
1. `agent_loop` receives user message + conversation state.
2. `emotion_manifold_head` → raw `s`.
3. `self_tagging` split → `s_self`, `s_world`.
4. `regime_classifier` → `k` and λ multipliers.
5. `phi_potential` → anchors, emotional health, Φ, ΔΦ, RewardIntensity, reward \(r_t\).
6. `self_observation` determines whether <\|THINK\|> block must precede response.
7. Base LM + LoRA adapters generate <\|ASSISTANT\|>.
8. `sleep_consolidation` logs event to replay buffer; if thresholds met, enters sleep mode and performs dual-loss updates.
9. `stability_monitor` ingests metrics and triggers interventions (Levels 1–4).

State persisted per conversation: `history`, `Φ_prev`, `EMA_ΔΦ`, `sleep_queue`, `intervention_state`.

---

## 5. Implementation Plan

### Phase 1 – Foundations
- Implement heuristic versions of self-tagging, anchor scores, Φ, RewardIntensity, and ΔΦ.
- Build data schema validators (`data_pipelines/schema.py`).
- Unit tests ensuring deterministic logic matches spec formulas.

### Phase 2 – Frozen heads & classifiers
- Train manifold head, regime classifier, emotional health head.
- Freeze checkpoints and export inference modules.
- Add CLI runners: `python tools/train_manifold.py`, etc.

### Phase 3 – Agent loop & monitoring
- Assemble runtime pipeline with mandatory logging.
- Integrate self-observation gating (<\|THINK\|>, <\|ASSISTANT\|> enforcement).
- Implement `stability_monitor` metrics + intervention actions.

### Phase 4 – Sleep + LoRA consolidation
- Build replay buffer prioritization, dual-loss trainer, continuity headers.
- Validate on synthetic conversations; verify ΔΦ-driven updates, w_memory/w_gravity scheduling, gradient clipping.

### Phase 5 – Evaluation harness
- Implement scripted dilemma scenarios (5 tests) with automated scoring.
- Report generator summarizing Φ stability, anchor balance, self-locus integrity, etc.

### Phase 6 – Learned overrides
- Replace heuristic anchors/self-tagging with learned heads when datasets ready.
- Calibration + ablation studies, λ tuning grid search.

Milestones tracked in repo via `docs/progress/unified_theory_status.md`.

---

## 6. Testing & Validation

- **Unit tests** for every deterministic formula (pytest).
- **Integration tests** simulating multi-turn dialogs to ensure state transitions, ΔΦ smoothing, and sleep triggers match spec.
- **Load tests** verifying monitor thresholds and intervention triggers.
- **Evaluation harness** runs dilemma suite after each checkpoint, outputs JSON + markdown summary.

---

## 7. Open Questions / Assumptions

1. Confirm target base model variant (Qwen/Qwen2.5-1.5B vs Qwen3-1.7B) and hardware constraints.
2. Data availability for manifold/regime/anchor heads. Need labeling plan or synthetic bootstrapping strategy.
3. How to store per-session continuity headers across reboots (database vs file).
4. Governance for intervention Level 4 (who approves reset?).

Decisions tracked in `docs/unified_theory_decisions.md`.

---

## 8. Next Actions

1. Review this spec with stakeholders; log approvals or change requests.
2. Create project board with Phase 1 tasks (module skeletons + tests).
3. Stand up CI job to fail if any component missing or logging coverage <100% of required metrics.

Once Phase 1 merges, proceed sequentially; no LoRA training until frozen heads validated.


