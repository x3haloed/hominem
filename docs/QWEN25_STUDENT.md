## Qwen2.5-7B Student Setup

This project now targets `Qwen/Qwen2.5-7B-Instruct` as the primary “student” model (the inference brain that receives LoRA updates).

### Requirements

- `torch` with MPS support (macOS 13.5+ on Apple Silicon).
- `transformers >= 4.37.0` (already pinned in `requirements-training.txt`).
- Optional: `bitsandbytes` + CUDA if you plan to train in 4-bit; otherwise leave `load_in_4bit: false` in the config.
- At least ~18 GB of unified memory. Training will be slow but tractable at rank 16 LoRA with batch size 1.

### Training flow

1. **Prepare preference data** as before:
   ```bash
   python -m core.data.make_preferences_from_rewards
   ```

2. **Train the LoRA adapter** (defaults are already tuned for Qwen2.5-7B):
   ```bash
   python -m core.lora_trainer.train_dpo \
     --config config/training/lora_dpo.yaml
   ```

   - The config now targets `Qwen/Qwen2.5-7B-Instruct`, uses rank 16 adapters, batch size 1, and enables gradient checkpointing.
   - The output adapter lives under `artifacts/lora/qwen2.5-7b/`.

3. **Serve for inference**

   - Convert the base model to `q8_0` (e.g., with LM Studio / llama.cpp `convert.py`).
   - Point LM Studio (or another gguf-compatible runner) at the q8_0 weights and attach the exported adapter via its LoRA feature.
   - Update `config/inference.toml` to hit that HTTP endpoint; the self-training server will then use the Qwen2.5 LoRA student automatically.

### Notes

- Keep `load_in_4bit: false` on Apple Silicon unless you have CUDA+bitsandbytes; Qwen2.5-7B fits on M3 Pro in bf16 with batch size 1 thanks to gradient checkpointing.
- If you later move training to a larger GPU, just flip `load_in_4bit: true` (with bitsandbytes available) and adjust the LoRA rank/batch size as needed.



