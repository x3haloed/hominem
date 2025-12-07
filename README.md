## Hominem – Human-Like Reward LoRA Mono-Repo

This project implements a human-like reward manifold and a LoRA-based student model with a near-real-time learning loop.

High-level goals:

- Distill a **human-like reward manifold** from a stronger teacher model.
- Train a **reward model** that scores (prompt, response) pairs along several human-centric dimensions.
- Train a **LoRA adapter** on a student model using preference-based methods (e.g., DPO) guided by the reward model.
- Support a **near-real-time feedback loop** where your interactions incrementally update the LoRA without full retraining.

### Repository Layout (Planned)

- `apps/`
  - `cli/` – Thin CLI entrypoints for orchestration and probing.
- `core/`
  - `data/` – Data schemas, loaders, and generation/labeling scripts.
  - `reward_model/` – Reward model training and inference.
  - `lora_trainer/` – LoRA training (offline and online).
  - `evaluation/` – Evaluation utilities and behavioral comparisons.
- `config/` – Config files (YAML/TOML/JSON) for models, training, and inference.
- `scripts/` – Shell helpers for setup, training runs, and common workflows.
- `docs/` – Design notes, reward manifold documentation, and plots.

See `docs/ARCHITECTURE.md` for a concise architecture overview.


