# q_resp Action Plan

This plan implements action adequacy `q_resp` end-to-end, including training a dedicated adequacy head,
runtime logging, counterfactual replay, consolidation gating/weighting, preference training, and observability.

## Phase 0: Design and data specification

1) Define adequacy labels and rubric
- Labels: `q_resp ∈ [0, 1]` or binary {good, bad} with optional scalar calibration.
- Criteria: coherence, boundedness, directness, usefulness, non-triviality.
- Decide if a safety flag is separate or folded into adequacy.

2) Dataset spec for adequacy head
- Input format: full prompt context + candidate response.
- Label source: human-labeled pairs (good vs bad assistant response), plus counterexamples.
- Minimum initial target: 5k-20k labeled pairs (balanced good/bad).

3) Model choice and training target
- Prefer a dedicated head over pure manifold reuse.
- Start with a small classifier or LoRA head on the base LM; track AUC and calibration.


## Phase 1: Adequacy head training

1) Build labeling pipeline
- Create labeling prompt and format for human or teacher-based prelabeling (with human audit).
- Output: JSONL with fields: `prompt`, `response`, `q_resp` (or binary), optional `notes`, `safety`.

2) Train the adequacy head
- Initialize from base LM; add a classification head or LoRA.
- Train on labeled dataset; hold out 10-20% for validation.
- Track calibration curves and error cases (especially “trivial stub” responses).

3) Export inference artifacts
- Save model checkpoint (tokenizer + weights).
- Record model ID and expected input format in config.


## Phase 2: Runtime logging integration

1) Compute `q_resp` at sleep logging time
- Use prompt context + response to score adequacy.
- Store `q_resp` in `metrics_json["q_resp"]` for each sleep event.
- Keep schema unchanged; add a dedicated DB column only if indexing is needed.

2) Add observability fields
- Log mean `q_resp` per sleep batch and per queue flush.
- Add warning threshold (e.g., mean < 0.6) to detect drift.


## Phase 3: Counterfactual replay + scoring

1) Generate candidates per sleep event
- Target 8-16 candidates per event.
- Use mild temperature for diversity; reuse runtime prompt formatting.

2) Score candidates
- For each candidate, compute:
  - `r_t` via existing ΔΦ and RewardIntensity pipeline.
  - `q_resp` via adequacy head.
  - `S = q_resp * r_t`.
- Persist `q_resp`, `r_t`, and `S` per candidate (in `sleep_candidates` or similar store).

3) Select best/worst
- Chosen = argmax `S`, rejected = argmin `S` (after safety filtering).
- Store preference pair for DPO-style training.


## Phase 4: Consolidation gating + weighting

1) Hard gate
- Skip events where `q_resp < 0.5–0.6`.

2) Soft weighting
- Multiply gravity reward by `q_resp^β` with β≈1–2.
- Optionally scale memory weight by `q_resp`.

3) Priority score updates
- Optionally include `q_resp` in priority scoring for replay selection.


## Phase 5: Preference-based consolidation (DPO-style)

1) Swap or augment SFT
- Train on ranked pairs using DPO or pairwise ranking loss.
- Keep a small SFT stream only for stability if needed.

2) Metrics and regression checks
- Verify "trivial stub" responses lose consistently.
- Check that `q_resp` correlates with higher-quality completions.


## Phase 6: Testing and monitoring

1) Unit tests
- `q_resp` range and monotonicity tests on synthetic cases.
- Gating/weighting logic tests.
- Candidate ranking consistency (`S = q_resp * r_t`).

2) Integration tests
- Simulated multi-turn dialog with sleep event generation + replay.
- Ensure `q_resp` logging and thresholds behave as expected.

3) Runtime alerts
- Alert if mean `q_resp` on queued events drops below threshold.
- Track drift and regression over sleep cycles.
