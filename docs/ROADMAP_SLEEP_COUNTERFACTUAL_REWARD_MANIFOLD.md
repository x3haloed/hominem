# Roadmap: Sleep Cycles → Counterfactual Replay → Reward Manifold

This doc is an ordered set of next steps (no timelines) to take `hominem` from:

- **manual sleep + SFT consolidation** (working today)
to
- **sleep-time counterfactual replay + reward-manifold-based action adequacy (`q_resp`) + preference training** (DPO-style).

It is intentionally specific to this repo and to your current operator constraints:

- Canonical DB path currently in use: `/Users/chad/Documents/hominem/conversations.db`
- OOM constraint: ~10k tokens full prompt; manual sleep trigger at ~8k; post-sleep history trimmed to ~1.6k tokens
- Current adapter lineage: `artifacts/lora/qwen3-1.7b-seed-sft-v3` (shards-only baseline + first sleep updates)
- Server environment expects `.venv` sourced; use `python3` when running CLIs locally.

---

## 0) Non-negotiable invariants (do these first)

1. **Never mutate canonical DB without a backup copy.**
   - `/sleep` is intentionally *destructive* to `conversations.state_json.history` (it trims history after flushing the queue).
   - Treat every experimental `POST /sleep` as a state transition you can’t “undo” without restoring DB.
   - Operator habit: `cp conversations.db conversations.backup_$(date -u +%Y%m%dT%H%M%SZ).db`

2. **Always compute token counts using the real chat template.**
   - The sleep trigger/trim must use the same tokenizer + chat template that inference uses (Qwen chat template), not approximate tokenizers.
   - This is already required behavior in the current `/sleep` implementation; preserve it in future refactors.

3. **Keep sleep triggering manual for now (by design).**
   - We’re intentionally *not* building cadence automation yet; all pipeline pieces must be runnable offline and deterministic.

---

## 1) Stabilize canonical data capture (so training is “about reality”)

### 1.1 Confirm what the canonical DB already contains

Your `sleep_events.metrics_json` is already rich and should remain the single “source of truth” for training weights/priority unless you explicitly migrate it:

- `pre` / `post` blocks (each has): `phi_value`, `delta_phi_used`, `delta_phi_ema`, `reward_intensity`, `r_t`, `regime_argmax`, `regime_probs`, `mean_self`, etc.
- `think_gate` boolean (present at top-level)

Quick sanity checks:

```bash
sqlite3 /Users/chad/Documents/hominem/conversations.db \
  "select count(*) total, sum(case when used_at is null then 1 else 0 end) unused from sleep_events;"
```

```bash
sqlite3 /Users/chad/Documents/hominem/conversations.db \
  "select id, reward_intensity, delta_phi_used, r_t, created_at from sleep_events where used_at is null order by id desc limit 10;"
```

### 1.2 Fix “no THINK tokens” at the source (data, not training)

Right now, `sleep_events.think` is empty in the canonical DB. That is not a trainer bug; it’s a **collection gap**.

You want two distinct “think-ish” streams:

1. **Model-produced “thinking”** (if you choose to enable it at inference time).
2. **Injected self-observation blocks** (“think_gate” / “self-observation”) that the runtime fabricates as part of your architecture.

The pipeline should persist **(2) even if (1) is disabled**, because (2) is architecture telemetry, not “chain-of-thought”.

Operator-facing rule:

- After the next cycle, verify you see non-null values in `messages.think` (assistant turns) and then in `sleep_events.think` after `/sleep`.

DB check:

```bash
sqlite3 /Users/chad/Documents/hominem/conversations.db \
  "select count(*) total, sum(case when think is not null and length(think)>0 then 1 else 0 end) with_think from sleep_events;"
```

If `with_think` stays 0 after a full new cycle, prioritize fixing persistence before any further learning work.

---

## 2) Make sleep queue + sleep flush boringly correct

Goal: the sleep queue is “append-only during wake”, then “flush-all during sleep”, with no silent drops.

### 2.1 What “working” looks like

During wake:

- The runtime adds candidates to `state_json.sleep_queue` when thresholds are met (currently lowered so we actually get candidates early).
- The queue items must include enough context to train: at minimum `(user_message, assistant, history_json, metrics_json, think?)`.

During sleep:

- `/sleep` moves **all** queued items into `sleep_events`, then clears `sleep_queue`.
- `/sleep` trims `history` down to a token budget *after* inserting a continuity anchor.

### 2.2 Manual backfill (only if needed)

If you have historical state that never got queued, use the offline backfill script (it must not trim or mangle any other data):

- `scripts/sleep/backfill_sleep_queue.py`

Operator rule:

- Backfill is a one-time repair tool. Once you’re confident the runtime is appending correctly, stop using it.

---

## 3) Lock in the consolidation trainer you actually run

This repo has older scripts (`online_update.py`, dual-loss prototypes) that are explicitly out-of-date for your current workflow.

For sleep cycles, standardize on:

- `core/lora_trainer/sleep_sft_update.py`

It exists to do three things you need in practice:

1. Load **existing LoRA weights** and continue training (no “fresh adapter each cycle” behavior).
2. Train on **new sleep events** (and optionally replay some old ones).
3. Provide **validation loss** during training so you can detect collapse early.

### 3.1 Operator defaults for the next run (until you have counterfactual replay)

- Keep `--balanced-batches` on (prevents “all high-intensity” drift).
- Keep `--validation-split` non-zero with `--eval-steps` so you get signal mid-run.
- Keep `--save-best` on so you don’t have to guess which checkpoint is safest.

If you want to “thin” a failure mode (like `Support.` stubs), do it at selection time:

- mark known-bad rows `used` (or delete them) in `sleep_events` **before** training
- keep a backup of the DB so this is reversible

---

## 4) Fix the reward math that caused “stub metastasis” (before you add more power)

This is the core lesson from the `Support.` incident:

- If `r_t` can become positive even when `ΔΦ_used` is negative (e.g., via additive intensity), then trivial responses can “win” despite being harmful to the intended direction of learning.

You already updated the theory docs to the safer multiplicative form:

- `r_t = ΔΦ_used * (1 + α * RewardIntensity)`

Order of operations for implementation:

1. Update the runtime’s computation of `r_t` to match the doc (so newly collected data is consistent).
2. Update the trainer to compute/interpret `r_t` the same way (so training weights match collection).
3. Add a regression check: “if `ΔΦ_used < 0`, then the example cannot receive a positive consolidation multiplier unless explicitly allowed by a safety/override gate.”

Do **not** proceed to counterfactual replay until this is consistent end-to-end, or you will generate a large volume of mislabeled preference signal.

---

## 5) Build the missing dataset: counterfactuals + adequacy labels (this is on you, human)

You asked if we’re “ready to train the reward manifold.” You are ready to train **a bootstrap prior**, but you are not ready to use it for `q_resp` without adding explicit data that encodes:

- “this response is trivial / non-responsive” (low adequacy)
- “this response is good but low-intensity” (good-but-not-highly-learning-relevant)
- “this response is intense but wrong” (high intensity should not mean high reward)

### 5.1 What you need to collect (minimum viable)

For a few hundred prompts/events, you need **K-way candidate sets**:

- same prompt context
- multiple plausible assistant completions
- one chosen, one rejected (at least), optionally a full ranking
- optional per-candidate `q_resp` label (0–1) and safety flag

Concrete target:

- 100–300 events × 4–8 candidates each → 400–2400 candidate responses
- For each event: label best + worst (pairwise preference) and flag any “trivial completion”

### 5.2 How to label (baby pea brain version)

When you look at a prompt + candidate responses, do exactly this:

1. **Pick the best response** (“If I could only keep one, which is it?”).
2. **Pick the worst response** (“Which one would I never want the model to do again?”).
3. Give the **best response** a `q_resp` score:
   - `1.0` = genuinely responds, advances the interaction, not a stub
   - `0.5` = sorta responds but is thin / generic
   - `0.0` = non-responsive / “Support.” / boilerplate / evasion
4. If any response is unsafe, mark `unsafe=true` (do not train it as “chosen”).

Important: this labeling is about **response adequacy**, not “did it sound nice.”

### 5.3 Add “trivial completion” negatives on purpose

Right now your labeled reward dataset does not include stub-like responses, which means the reward model has never learned “stubs are bad.”

You should deliberately add examples where the response is:

- a one-word acknowledgement
- an over-generic “Support.” / “Ok.” / “I understand.” with no content
- a refusal to engage when engagement is appropriate

And label them low on:

- `social_coherence`, `narrative_alignment`, `curiosity`, and overall `scalar`
- `reward_intensity` can be high if you want the model to learn “don’t do this”

---

## 6) Implement counterfactual replay storage in the canonical DB (non-obvious plumbing)

You already have `sleep_events`. Counterfactual replay needs new persistent objects.

### 6.1 Two viable storage paths (pick one, don’t mix ad hoc)

**Option A (recommended): canonical DB tables**

- Pros: single canonical source of truth; easy to resume; no “where did that JSONL come from?”
- Cons: requires schema changes (safe additive `CREATE TABLE`), and you must be disciplined about backups.

**Option B: append-only JSONL artifacts (simpler first iteration)**

- Pros: very easy to iterate; no DB migrations; easy to version-control alongside training outputs
- Cons: you must treat the JSONL as canonical and not lose it; you need a stable naming convention

If you choose Option B first, still design it so you can later import into DB tables without losing metadata.

Recommended minimal schema (additive, no migrations required beyond `CREATE TABLE`):

1. `sleep_candidates`
   - one row per generated candidate completion per sleep_event
2. `preference_pairs`
   - one row per (prompt, chosen, rejected) training pair
3. `reward_labels` (optional)
   - human-provided `q_resp`, safety overrides, notes/rationales

Non-obvious requirement:

- Store **generation parameters** (temperature/top_p/max_new_tokens/seed) so you can reproduce candidate sets and debug weird failures.

### 6.2 Suggested schema (SQLite)

If you go with canonical DB tables, start with this:

```sql
CREATE TABLE IF NOT EXISTS sleep_candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sleep_event_id INTEGER NOT NULL,
  candidate_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  -- optional: model info + sampling params for reproducibility
  model_id TEXT,
  adapter_path TEXT,
  temperature REAL,
  top_p REAL,
  max_new_tokens INTEGER,
  seed INTEGER,
  -- optional: scoring fields
  q_resp REAL,
  r_t REAL,
  score REAL,
  safety_score REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (sleep_event_id) REFERENCES sleep_events(id)
);

CREATE TABLE IF NOT EXISTS preference_pairs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sleep_event_id INTEGER,
  chosen_candidate_id INTEGER NOT NULL,
  rejected_candidate_id INTEGER NOT NULL,
  -- store the effective prompt context used for generation (for DPO reproducibility)
  prompt_text TEXT NOT NULL,
  -- optional: weight for training
  weight REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (sleep_event_id) REFERENCES sleep_events(id),
  FOREIGN KEY (chosen_candidate_id) REFERENCES sleep_candidates(id),
  FOREIGN KEY (rejected_candidate_id) REFERENCES sleep_candidates(id)
);

CREATE TABLE IF NOT EXISTS reward_labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sleep_candidate_id INTEGER NOT NULL,
  q_resp REAL,
  unsafe INTEGER DEFAULT 0,
  notes TEXT,
  labeled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (sleep_candidate_id) REFERENCES sleep_candidates(id)
);
```

### 6.3 JSONL formats that already exist in this repo (reuse them)

You already have a working preference JSONL format at `data/preferences/preferences.jsonl`:

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "category": "...", "chosen_score": 0.9, "rejected_score": -0.6}
```

For sleep-time counterfactual replay, you can mirror that structure and add metadata:

```json
{
  "sleep_event_id": 123,
  "prompt": "...(exact prompt used for generation)...",
  "chosen": "...",
  "rejected": "...",
  "candidates": ["...", "...", "..."],
  "q_resp": {"chosen": 0.9, "rejected": 0.1},
  "scores": {"chosen": 0.32, "rejected": -0.05},
  "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": 1234}
}
```

---

## 7) Train the reward manifold (bootstrap), then make it earn trust

You have a bootstrap reward dataset at:

- `data/labeled/reward_samples.jsonl`

Training entry point:

- `core/reward_model/train.py` with `config/training/reward_model.yaml`

But before you let the reward model influence learning:

1. Add the missing negative/trivial examples (Section 5.3).
2. Add some real “in-the-wild” samples from your own conversations (not just seed prompts).
3. Evaluate the trained reward model on a held-out set and do a small “sanity suite”:
   - obvious good response scores higher than obvious bad response
   - stub responses score low on `social_coherence/narrative_alignment/curiosity`

### 7.1 Operator command (bootstrap reward model)

From repo root (with `.venv` active):

```bash
python3 -m core.reward_model.train \
  --data data/labeled/reward_samples.jsonl \
  --config config/training/reward_model.yaml \
  --output artifacts/reward_model/bootstrap_$(date -u +%Y%m%dT%H%M%SZ)
```

---

## 8) Integrate `q_resp` into selection + training (soft first, hard later)

Order of operations:

1. **Logging only**: compute `q_resp` for each sleep_event and store it.
2. **Soft weighting**: downweight consolidation strength by `q_resp`.
3. **Hard gating** (optional): skip training on events with `q_resp < q_min`.

Do not jump straight to hard gating until you’ve validated the reward model doesn’t have blind spots (it will at first).

---

## 9) Switch consolidation from SFT-only to preference training (DPO-style)

Once you have `preference_pairs`:

- Use DPO-style training for consolidation (this is what counterfactual replay is for).
- Keep a small SFT replay stream for “style/fluency glue” if needed, but make preferences the main driver for “don’t metastasize stubs.”

Non-obvious implementation detail:

- Preference training wants **consistent prompt formatting**. Reuse the same chat template builder used by inference and `/sleep` token counting.

---

## 10) Add evaluation loops that catch collapse early (cheap + reliable)

Minimum evals you should run every sleep cycle:

1. **Validation loss** during sleep consolidation training (already supported in your sleep trainer).
2. A tiny fixed **prompt suite** (10–30 prompts) where you diff outputs between:
   - previous adapter
   - new adapter
3. A “stub detector” report that tracks:
   - fraction of completions under N tokens
   - top repeated 1–3 token outputs

This isn’t a banlist; it’s a smoke alarm.

---

## Appendix: What you (the human) should do next, concretely

### A) Add stub/trivial negatives to the reward dataset

You can do this without building any new tooling:

1. Copy 20–50 real prompts (from your own conversations or seed prompts).
2. For each prompt, create 2–4 “bad” responses that are obviously trivial:
   - `Support.`
   - `Ok.`
   - `I understand.`
   - `That makes sense.`
   - (no follow-up question, no specificity, no action)
3. Add them to `data/labeled/reward_samples.jsonl` with:
   - low `social_coherence`, low `narrative_alignment`, low `curiosity`
   - optionally high `reward_intensity` if you want “learn to avoid this strongly”

That single change makes the reward model capable of learning the anti-stub boundary you actually care about.

### B) Start collecting counterfactual candidate sets from real sleep events

Pick a small batch (e.g., 20 sleep events), generate 4–8 candidates each, then label best/worst.

Your “labeling loop” should be:

1. Generate candidates (machine work).
2. Label best/worst + `q_resp` (your work).
3. Train preference loss (machine work).
4. Run a tiny prompt suite diff (your sanity check).

---

## Appendix: Things that are easy to forget (repo-specific)

- Use `python3`, not `python`, unless your environment wires `python` explicitly.
- Source `.venv` before running training scripts locally.
- The canonical DB is `/Users/chad/Documents/hominem/conversations.db` (you also have an older `/Users/chad/temp/...` that is not the current schema).
- If you see `Support.` metastasis again, treat it as a **reward signal bug** or **missing adequacy data**, not as “the model is dumb.”
