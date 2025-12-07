## Architecture Overview

This document summarizes the main subsystems in the `hominem/` mono-repo and how they fit together.

### High-Level Flow

1. **Teacher models** generate conversation trajectories and rate candidate responses on a human-like reward manifold.
2. A **reward model** is trained on these labeled samples to approximate the teacher’s judgments.
3. A **LoRA adapter** is trained on a student model using preference-based methods (e.g., DPO), guided by reward signals.
4. An **online learning loop** allows incremental LoRA updates based on your ongoing feedback.

### Subsystems

- **Core / Data**
  - Defines reward manifold schemas and data formats (JSONL, etc.).
  - Implements data loaders and splitters for reward and preference data.
  - Contains scripts for teacher-driven data generation and labeling.

- **Core / Reward Model**
  - Provides a stable interface to map (prompt, response) → reward vector (+ optional scalar).
  - Includes training scripts, checkpoint management, and basic test inference utilities.

- **Core / LoRA Trainer**
  - Handles batch/offline LoRA training using preference data.
  - Implements an online update mechanism that consumes feedback logs and produces new LoRA versions.

- **Core / Evaluation**
  - Offers CLIs and utilities to compare base vs LoRA behavior.
  - Integrates with the reward model for quantitative and qualitative evaluation.

- **Apps / CLI**
  - Thin command-line frontends around core functionality (probing reward vectors, comparing models, running evaluation suites).

- **Config**
  - Centralizes configuration for models, training runs, inference endpoints, and active artifact versions.

- **Scripts**
  - Contains shell helpers for environment setup, training jobs, evaluation runs, and artifact management.

As the implementation progresses, this document can link to more detailed module-level docs and describe concrete model choices.


