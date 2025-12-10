import argparse
import datetime
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
import yaml

from core.training_logger import TrainingJSONLogger

@dataclass
class PreferenceSample:
    """Single preference pair: prompt + (chosen, rejected) responses."""

    prompt: str
    chosen: str
    rejected: str


class PreferenceDataset(Dataset):
    """Torch dataset for DPO-style pairwise training."""

    def __init__(self, samples: List[PreferenceSample], tokenizer, max_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def _format_chat(self, prompt: str, response: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return f"User: {prompt}\nAssistant: {response}"

    def _build_inputs(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        text = self._format_chat(prompt, response)
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        chosen_inputs = self._build_inputs(sample.prompt, sample.chosen)
        rejected_inputs = self._build_inputs(sample.prompt, sample.rejected)

        return {
            "chosen": chosen_inputs,
            "rejected": rejected_inputs,
        }


def load_preferences(path: str) -> List[PreferenceSample]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preferences file not found at '{path}'. "
            "Create data/preferences/preferences.jsonl with fields: "
            "{'prompt', 'chosen', 'rejected'} per line."
        )

    samples: List[PreferenceSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                prompt = obj["prompt"]
                chosen = obj["chosen"]
                rejected = obj["rejected"]
            except KeyError as exc:
                raise ValueError(f"Missing field in preference record: {exc}; got: {obj}") from exc
            samples.append(PreferenceSample(prompt=prompt, chosen=chosen, rejected=rejected))

    if not samples:
        raise ValueError(f"No preference samples found in '{path}'.")

    return samples


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Simple collator that stacks chosen/rejected tensors."""

    def stack_side(side: str) -> Dict[str, torch.Tensor]:
        keys = batch[0][side].keys()
        return {
            k: torch.stack([item[side][k] for item in batch], dim=0)
            for k in keys
        }

    return {
        "chosen": stack_side("chosen"),
        "rejected": stack_side("rejected"),
    }


def dpo_loss(
    model: AutoModelForCausalLM,
    batch: Dict[str, Dict[str, torch.Tensor]],
    beta: float,
) -> torch.Tensor:
    """
    Compute a simple DPO-style loss:
      L = -E[log σ(β * (logp_chosen - logp_rejected))]

    This omits the reference model term for simplicity and treats the
    model's own log-likelihood as the "reward".
    """

    def log_probs_for_side(side_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(
            input_ids=side_batch["input_ids"],
            attention_mask=side_batch["attention_mask"],
            labels=side_batch["labels"],
        )
        # HF CausalLM returns loss averaged over tokens; we convert to
        # log-likelihood per sequence by multiplying by sequence length.
        # We just need a scalar per sequence; relative scale is what matters.
        seq_lengths = side_batch["attention_mask"].sum(dim=-1)
        # Negative loss is average log-prob per token; multiply by length.
        loglik = -outputs.loss * seq_lengths
        return loglik

    chosen_logp = log_probs_for_side(batch["chosen"])
    rejected_logp = log_probs_for_side(batch["rejected"])

    diff = chosen_logp - rejected_logp
    logits = beta * diff
    # -log sigmoid
    loss = -torch.nn.functional.logsigmoid(logits)
    return loss.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except AttributeError:
            pass


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with a simple DPO-style objective.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/lora_dpo.yaml",
        help="Path to LoRA/DPO training config YAML.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_str: str | None):
    if not dtype_str or dtype_str == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype '{dtype_str}'")
    return mapping[key]


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_model_and_tokenizer(cfg: Dict[str, Any], device: torch.device) -> (AutoModelForCausalLM, AutoTokenizer):
    model_cfg = cfg["model"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_cfg.get("padding_side", "left")

    torch_dtype = resolve_torch_dtype(model_cfg.get("torch_dtype"))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype or torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if load_in_4bit else None,
        quantization_config=quant_config,
    )

    if not load_in_4bit:
        model.to(device)

    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def train() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    set_seed(train_cfg.get("seed", 42))

    preferences_path = train_cfg["preferences_path"]
    samples = load_preferences(preferences_path)

    print(f"Loaded {len(samples)} preference pairs from {preferences_path}")

    device = select_device()
    model, tokenizer = prepare_model_and_tokenizer(cfg, device)

    dataset = PreferenceDataset(samples, tokenizer, max_length=model_cfg["max_length"])
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    num_training_steps = train_cfg["num_epochs"] * math.ceil(len(dataset) / train_cfg["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.get("warmup_steps", 0),
        num_training_steps=num_training_steps,
    )

    model.train()
    global_step = 0
    logging_steps = train_cfg.get("logging_steps", 10)
    save_every_steps = train_cfg.get("save_every_steps", 0)
    beta = float(train_cfg.get("beta", 0.1))

    run_id = train_cfg.get("run_id") or f"offline_dpo_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logging_root = Path(train_cfg.get("logging_dir", "logs/train"))
    log_dir = logging_root / run_id
    logger = TrainingJSONLogger(
        run_id=run_id,
        component="offline_dpo",
        output_dir=log_dir,
        meta={
            "config": args.config,
            "output_dir": train_cfg["output_dir"],
            "preferences_path": preferences_path,
            "num_samples": len(samples),
            "batch_size": train_cfg["batch_size"],
            "beta": beta,
            "model_id": model_cfg["base_model_id"],
        },
    )

    for epoch in range(train_cfg["num_epochs"]):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = {
                side: {k: v.to(device) for k, v in side_batch.items()}
                for side, side_batch in batch.items()
            }

            loss = dpo_loss(model, batch, beta=beta)
            loss.backward()

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / global_step
                logger.log_step(
                    {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "num_training_steps": num_training_steps,
                        "loss": loss.item(),
                        "avg_loss": avg_loss,
                        "grad_norm": grad_norm,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "batch_size": train_cfg["batch_size"],
                        "dataset_size": len(dataset),
                        "beta": beta,
                        "replay_ratio": 0.0,
                        "priority_stats": None,
                        "kl_to_base": None,
                        "partial_rhi": None,
                    }
                )
                print(f"Epoch {epoch + 1}, step {global_step}/{num_training_steps} - loss: {loss.item():.4f}, avg_loss: {avg_loss:.4f}")

            if save_every_steps and global_step % save_every_steps == 0:
                save_dir_step = os.path.join(train_cfg["output_dir"], f"step_{global_step}")
                os.makedirs(save_dir_step, exist_ok=True)
                model.save_pretrained(save_dir_step)
                tokenizer.save_pretrained(save_dir_step)
                print(f"Saved intermediate LoRA adapter to {save_dir_step}")

        avg_epoch_loss = epoch_loss / max(1, len(dataloader))
        logger.log_eval(
            {
                "epoch": epoch + 1,
                "avg_epoch_loss": avg_epoch_loss,
                "dataset_size": len(dataset),
            }
        )
        print(f"Epoch {epoch + 1} completed - average loss: {avg_epoch_loss:.4f}")

    # Final save
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved final LoRA adapter to {output_dir}")

    # Update pointer to current LoRA version and maintain a simple history.
    pointer_path = os.path.join(output_dir, "current.json")
    history_path = os.path.join(output_dir, "history.json")

    version_entry = {
        "source": "offline_dpo",
        "path": output_dir,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # current.json always points at the latest adapter.
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(version_entry, f, indent=2)

    # Append to history.json for manual inspection.
    history: list[dict[str, object]] = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f) or []
        except Exception:
            history = []
    history.append(version_entry)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train()


