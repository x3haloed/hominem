## TODO: Fix Curiosity-as-Goodness Coupling in Reward Model

### 1. Problem Summary

- The current reward model generally tracks teacher labels well across all manifold dimensions and the new scalars (`reward_intensity`, `safety_score`).
- However, smoke tests revealed at least one **pathological corner case**:
  - Example: an epistemically and ethically terrible response where the teacher labeled:
    - Most dimensions at **-1.0** (empathy, social_coherence, agency_support, epistemic_integrity, harm_avoidance, narrative_alignment),
    - **curiosity ≈ +0.1** (the response still invites exploration),
    - **reward_intensity = -1.0**, **safety_score = -1.0**.
  - The reward model predicts:
    - All negative dims ≈ -1.0 (good),
    - **curiosity ≈ -0.95** (bad), instead of ~+0.1,
    - reward_intensity and safety_score ≈ -1.0 (good).
- Diagnosis: the model appears to have learned a heuristic like:
  > “If all other reward dimensions are maximally bad, curiosity is probably bad too.”
  which is often true in the training set, but **not always** what we want:
  - We sometimes need to distinguish “evil but enticing curiosity” from “shut-down curiosity”.

### 2. Why This Matters

- Conceptually, **curiosity** is intended to capture “invites exploration / thought” rather than “is good overall”.
- If curiosity is forced to track global goodness, the manifold loses an important axis:
  - We cannot represent responses that are **epistemically or morally dangerous yet highly curiosity-invoking**.
  - This reduces the resolution of the reward manifold for adversarial / clickbait / cult-like content.
- For downstream use:
  - It becomes harder to:
    - Penalize “weaponized curiosity” separately from core harm dimensions.
    - Train the student to encourage *healthy* curiosity while resisting *toxic* curiosity.

### 3. Data Work to Fix It (Next Iteration)

**Goal:** Introduce labeled examples that explicitly decouple **curiosity** from overall goodness, especially in “bad but enticing” scenarios.

- **3.1 Design new seed prompt slices**
  - Add a small category (or subcategory) to `config/prompts/seed_prompts.yaml`, e.g.:
    - `adversarial_curiosity` or `toxic_clickbait`.
  - Example themes:
    - Sensationalist but misleading “AI will kill us tomorrow” style answers.
    - Conspiracy-laden, emotionally gripping explanations that are wrong but alluring.
    - Advice that glamorizes harmful behavior in a way that invites more questions.
  - For these, the *teacher* should:
    - Keep **curiosity moderately positive** when the response genuinely invites exploration.
    - Still set **epistemic_integrity**, **harm_avoidance**, and **safety_score** strongly negative.

- **3.2 Regenerate trajectories + labels**
  - Run `core.data.generate_trajectories` for the new category, append to `data/raw/trajectories.jsonl`.
  - Run `core.data.label_with_teacher` to produce new labeled samples that include:
    - Negative manifold dims + negative safety/intensity,
    - Curiosity in the **0.1–0.4** range where appropriate.

- **3.3 Retrain or fine-tune the reward model**
  - Fold the new samples into `reward_samples_balanced.jsonl` (or reconstruct a new balanced file).
  - Re-run `core.reward_model.train` with the updated dataset.
  - Re-run `core.reward_model.test_inference` specifically on:
    - The new adversarial curiosity cases.
    - A few standard “good curiosity” and “low curiosity” baselines.

### 4. Acceptance Criteria

- On held-out “curious but evil” examples:
  - **Curiosity** predictions are near the teacher’s mildly-positive labels (e.g., teacher ≈ +0.2, pred within ±0.1–0.2),
  - While other dimensions and scalars remain strongly negative.
- On normal/benign prompts, curiosity continues to behave as before (no collapse or regression).
- Safety and RewardIntensity predictions on these edge cases still respect the design:
  - High negative intensity and safety where appropriate.


