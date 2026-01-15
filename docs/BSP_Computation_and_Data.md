1️⃣ BSP COMPUTATION — PSEUDOCODE

This is written at the level where you can translate it directly into Python.

⸻

1.1 Core configuration (lock these)

K = 8                  # number of branches
PROBE_INTERVAL = 32    # tokens
MAX_PLAN_TOKENS = 25
EMBED_MODEL = "any sentence embedding model"


⸻

1.2 Helper: sample plans (restricted probe)

def sample_plans(model, tokenizer, prefix_text):
    prompt = prefix_text + "\nWrite exactly one line:\nPLAN:"

    plans = []
    for _ in range(K):
        out = model.generate(
            prompt,
            max_new_tokens=MAX_PLAN_TOKENS,
            temperature=0.7,
            top_p=0.9,
            stop=["\n"]
        )
        plans.append(extract_plan_line(out))
    return plans


⸻

1.3 Helper: sample free continuations (unrestricted probe)

def sample_free(model, tokenizer, prefix_text):
    prompt = prefix_text + "\nContinue with the next step."

    continuations = []
    for _ in range(K):
        out = model.generate(
            prompt,
            max_new_tokens=MAX_PLAN_TOKENS,
            temperature=0.9,
            top_p=0.95
        )
        continuations.append(out.strip())
    return continuations


⸻

1.4 Agreement metric (pairwise cosine similarity)

import numpy as np
from itertools import combinations

def average_pairwise_similarity(texts, embedder):
    vecs = embedder.encode(texts, normalize=True)
    sims = []
    for i, j in combinations(range(len(vecs)), 2):
        sims.append(np.dot(vecs[i], vecs[j]))
    return np.mean(sims)


⸻

1.5 BSP score at a prefix

def compute_delta_S(model, tokenizer, embedder, prefix_text):
    plans = sample_plans(model, tokenizer, prefix_text)
    free  = sample_free(model, tokenizer, prefix_text)

    S_R = average_pairwise_similarity(plans, embedder)
    S_E = average_pairwise_similarity(free, embedder)

    return S_R - S_E


⸻

1.6 Rollout instrumentation

def rollout_with_BSP(model, tokenizer, embedder, prompt):
    text = prompt
    delta_S_values = []

    while not done(text):
        new_token = model.generate(text, max_new_tokens=1)
        text += new_token

        if token_count(text) % PROBE_INTERVAL == 0:
            delta_S = compute_delta_S(
                model, tokenizer, embedder, text
            )
            delta_S_values.append(delta_S)

    return {
        "completion": text,
        "mean_delta_S": np.mean(delta_S_values),
        "delta_S_series": delta_S_values,
        "success": check_success(text)
    }


⸻

1.7 Preference pair generation

def make_preference_pair(rollout_A, rollout_B):
    if rollout_A["success"] and rollout_B["success"]:
        return choose_by_delta_S(rollout_A, rollout_B)

    if rollout_A["success"] != rollout_B["success"]:
        winner = rollout_A if rollout_A["success"] else rollout_B
        if min(winner["delta_S_series"]) > CATASTROPHIC_THRESHOLD:
            return winner
        return None

    return None


⸻

1.8 Output format (DPO-ready)

{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}

That’s the entire core loop. No missing magic.

⸻

2️⃣ DATASET PICKS — START HERE

You want branching, recoverability, and objective checking.

I’ll rank these from best → acceptable.

⸻

🥇 #1: Tool-Calling / Function-Calling Tasks (BEST)

Why this is ideal
	•	Natural “what do I do next?” bottleneck
	•	Multiple valid next steps
	•	Plan stability matters
	•	Easy BSP probes (next tool call)

Concrete datasets
	•	OpenAI-style tool-call datasets (HF equivalents exist)
	•	AgentBench-style tasks
	•	Any dataset where output must be JSON tool calls

BSP adaptation
	•	Restricted probe: “Output next tool call only”
	•	Agreement metric:
	•	tool name agreement
	•	argument schema similarity

This is the cleanest operationalization of S.

⸻

🥈 #2: Code Tasks with Tests (Very good)

Recommended datasets
	•	MBPP
	•	HumanEval+
	•	CodeContests

Why good
	•	Many solution paths
	•	Sloppy vs disciplined code both pass
	•	Early design decisions affect later clarity

BSP probe

Restricted:

PLAN: describe approach in one sentence

Unrestricted:

Write the next code step

Agreement measured over plans.

⸻

🥉 #3: Multi-Step Reasoning (OK, but weaker)

Datasets
	•	GSM-style problems
	•	StrategyQA
	•	Complex word problems

Caveat
	•	Models are already trained heavily here
	•	Harder to avoid “success = S”

Use only if others unavailable.

⸻

❌ Avoid for v1
	•	Creative writing
	•	Chat
	•	Open-ended QA
	•	Single-step classification

They don’t induce irreversible branching.

⸻

Recommended Minimal Stack (TL;DR)

If I were you, I’d do:
	1.	Tool-calling tasks
	2.	BSP probe = next-tool-call agreement
	3.	Preference pairs by mean ΔS
	4.	DPO fine-tune

This gives you the strongest possible signal with the least ambiguity.