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
   - Training note: v1 may train an argmax single-label classifier for simplicity; v2 should support soft-label training (e.g., KL to target probabilities).
   - Data: ≥10k samples per regime.

3. **Anchor heads + emotional health**
   - Trainable heuristics replaced by learned heads once preference data exists.
   - Placeholder heuristics implemented exactly as in spec, wrapped in `nn.Module` so checkpoints can later override.

4. **Φ/ΔΦ training**
   - Utilize preference pairs; until data ready, integrate hook for future training script and keep heuristic fallback.

5. **Self-tagging MLP**
   - Train on hinge datasets described in section 7.2; deterministic tree serves as initial mode.

6. **LoRA training**
   - Base model: Qwen/Qwen3-1.7B (or specified derivative).
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

Normative runtime constants (must match `unified_theory.md`):
- Self-observation gating: emit <\|THINK\|> when `abs(raw_ΔΦ) > 0.2` OR `abs(mean_self_fraction_t - mean_self_fraction_{t-1}) > 0.2`.
- RewardIntensity: `arousal * (abs(valence) ** 1.0 * abs(predictive_discrepancy)) ** 0.5 * (1.8 if valence < 0 else 1.0)`.
- ΔΦ smoothing: `EMA_ΔΦ_t = 0.8 * EMA_ΔΦ_{t-1} + 0.2 * raw_ΔΦ` and define `ΔΦ_used = EMA_ΔΦ_t` (log both).
- Gravity reward: `r_t = ΔΦ_used + α * RewardIntensity` with default `α = 0.5`.
- Safety clamps: cap `RewardIntensity` to `3.0`; cap `abs(Φ_t - Φ_{t-1})` to `2.0`; clip `ΔΦ_used` to `[-1.0, 1.0]` for loss weighting.

---

## 5. Implementation Plan

### Phase 1 – Foundations
- Implement heuristic versions of self-tagging, anchor scores, Φ, RewardIntensity, and ΔΦ.
- Build data schema validators (`data_pipelines/schema.py`).
- Unit tests ensuring deterministic logic matches spec formulas.

### Phase 2 – Frozen heads & classifiers
- Train manifold head, regime classifier, emotional health head.
- Freeze checkpoints and export inference modules.
- Add CLI runners: `python3 -m core.lora_trainer.train_manifold`, `python3 -m core.lora_trainer.train_regime_classifier`, etc.

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

## 9. Emotion Manifold Training Results & Best Practices

### 9.1 Training Regime Summary
Completed comprehensive training regime testing 6 different model configurations with systematic hyperparameter optimization. Key findings:

**🏆 Best Performing Model**: BERT-base-uncased (110M parameters)
- **Training Command**:
```bash
python -m core.lora_trainer.train_manifold \
  --data-roots data/processed_datasets_unified \
  --datasets ultrachat_trajectories ultrachat_synthetic_trajectories \
  --output-dir artifacts/manifold_bert_optimized \
  --min-records 1000 \
  --validation-split 0.1 \
  --model-id bert-base-uncased \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --num-epochs 12 \
  --lr 2e-5 \
  --max-length 256
```

### 9.2 Performance Metrics (Full Dataset Evaluation)
**Overall Metrics:**
- MSE: 0.0160
- MAE: 0.0853
- RMSE: 0.1265
- Average Correlation: **0.5003** (154% improvement over baseline)

**Per-Axis Correlations:**
- valence: 0.3723
- arousal: 0.5012
- dominance: 0.3678
- predictive_discrepancy: **0.7083** (went from -0.07 to 0.71!)
- temporal_directionality: 0.5583
- social_broadcast: 0.4938

### 9.3 Model Comparison Results
| Model | Parameters | Avg Correlation | Rank |
|-------|------------|-----------------|------|
| BERT-base-uncased (optimized) | 110M | **0.5003** | 🏆 1st |
| BERT-base-uncased (10 epochs) | 110M | 0.4035 | 2nd |
| BERT-base-uncased (5 epochs) | 110M | 0.3496 | 3rd |
| RoBERTa-base | 125M | 0.3363 | 4th |
| BERT-base (gradient acc) | 110M | 0.3250 | 5th |
| ALBERT-base-v2 | 12M | 0.1629 | 6th |

### 9.4 Key Insights & Best Practices

**Architecture Selection:**
- **BERT-base-uncased** significantly outperforms smaller models (ALBERT) and shows better results than RoBERTa for this regression task
- Model capacity is critical - 154% performance improvement moving from 12M to 110M parameters

**Hyperparameter Optimization:**
- **Learning Rate**: 2e-5 (higher than typical BERT fine-tuning of 1e-5)
- **Training Duration**: 12 epochs provides optimal performance (vs 5-10 epochs)
- **Batch Size**: Effective batch size of 8 (batch_size=4 × gradient_accumulation_steps=2)
- **Max Length**: 256 tokens (vs 512) for memory efficiency

**Data Processing:**
- History clamping to last 3 turns implemented and working correctly
- Conversation context format: `assistant: [text]\nuser: [text]\nassistant: [current]`

**Evaluation Framework:**
- Custom evaluation script (`core/evaluation/eval_manifold.py`) provides comprehensive metrics
- Full dataset evaluation (1350 samples) vs sample evaluation (100 samples) shows more robust performance
- Correlation metrics are more meaningful than MSE for emotion prediction tasks

**Training Stability:**
- Loss steadily decreased throughout training (from ~0.013 to 0.0096)
- No overfitting observed - evaluation loss remained stable
- Gradient accumulation improved training stability

### 9.5 Implementation Notes

**Training Script Modifications:**
- Added `gradient_accumulation_steps` parameter to TrainingArguments
- Added `trust_remote_code=True` for custom model loading
- Implemented automatic pad token handling for GPT-style models
- Added history clamping to last 3 turns in `record_to_text()`

**Evaluation Enhancements:**
- Numpy-based metrics (no sklearn dependency)
- Per-axis correlation analysis
- Distribution statistics for predictions vs ground truth
- JSON output for programmatic analysis

**Memory Optimization:**
- Reduced batch size + gradient accumulation for GPU memory constraints
- Shortened max sequence length (256 vs 512)
- MPS device utilization confirmed working

### 9.6 Recommendations for Future Training

1. **Use BERT-base-uncased** as the foundation model for emotion manifold tasks
2. **Train for 12 epochs** with lr=2e-5 for optimal performance
3. **Use gradient accumulation** (effective batch size 8) for training stability
4. **Evaluate on full dataset** for robust performance assessment
5. **Monitor per-axis correlations** rather than just overall MSE
6. **Consider ensemble approaches** if single model performance plateaus

**Model Location**: `artifacts/manifold_bert_optimized/`
**Evaluation Results**: `artifacts/manifold_final_evaluation.json`

---

## 8. Next Actions

1. Review this spec with stakeholders; log approvals or change requests.
2. Create project board with Phase 1 tasks (module skeletons + tests).
3. Stand up CI job to fail if any component missing or logging coverage <100% of required metrics.

Once Phase 1 merges, proceed sequentially; no LoRA training until frozen heads validated.

