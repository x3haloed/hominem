import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
import yaml

from core.data.db import TrainingDatabase
from core.training_logger import TrainingJSONLogger
from core.lora_trainer.train_dpo import PreferenceSample, load_preferences, collate_fn, dpo_loss, set_seed

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import KMeans
except ImportError:
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore
    KMeans = None  # type: ignore
    warnings.warn(
        "sentence-transformers and scikit-learn not installed; "
        "embedding-based diversity sampling will be disabled."
    )

@dataclass
class SFTSample:
    instruction: str
    response: str
    id: Optional[int] = None
    created_at: Optional[str] = None
    recency_weight: float = 1.0
    reward_intensity: Optional[float] = None


class SFTDataset(Dataset):
    def __init__(self, samples: List[SFTSample], tokenizer, max_length: int) -> None:
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = self._format_chat(sample.instruction, sample.response)
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        enc["sample_id"] = torch.tensor(sample.id if sample.id is not None else -1, dtype=torch.long)
        enc["weight"] = torch.tensor(sample.recency_weight, dtype=torch.float32)
        return enc


def _recency_weight_dual(created_at: Optional[str], tau_fast: float, tau_slow: float) -> float:
    if not created_at or tau_fast <= 0 or tau_slow <= 0:
        return 1.0
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        return 1.0
    age_seconds = (datetime.utcnow() - dt).total_seconds()
    fast = math.exp(-age_seconds / tau_fast)
    slow = math.exp(-age_seconds / tau_slow)
    return max(fast, slow)


def load_sft_pairs_from_database(
    db: TrainingDatabase,
    filters: Dict[str, Any],
    use_recency_weighting: bool,
    tau_fast: float,
    tau_slow: float,
) -> List[SFTSample]:
    rows = db.get_sft_pairs(
        source=filters.get("source"),
        is_used=filters.get("is_used"),
        since=filters.get("since"),
        min_instruction_length=filters.get("min_instruction_length"),
        min_response_length=filters.get("min_response_length"),
        min_confidence=filters.get("min_confidence"),
        limit=filters.get("limit"),
    )
    samples: List[SFTSample] = []
    for row in rows:
        metadata = json.loads(row["metadata"]) if row.get("metadata") else {}
        reward_intensity = metadata.get("reward_intensity") if isinstance(metadata, dict) else None
        recency_weight = 1.0
        if use_recency_weighting:
            recency_weight = _recency_weight_dual(row.get("created_at"), tau_fast, tau_slow)
        samples.append(
            SFTSample(
                instruction=row["instruction"],
                response=row["response"],
                id=row.get("id"),
                created_at=row.get("created_at"),
                recency_weight=recency_weight,
                reward_intensity=reward_intensity if reward_intensity is not None else None,
            )
        )
    return samples


def load_sft_pairs_from_jsonl(
    path: str,
    use_recency_weighting: bool,
    tau_fast: float,
    tau_slow: float,
) -> List[SFTSample]:
    samples: List[SFTSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            reward_intensity = obj.get("reward_intensity")
            recency_weight = 1.0
            created_at = obj.get("created_at")
            if use_recency_weighting:
                recency_weight = _recency_weight_dual(created_at, tau_fast, tau_slow)
            samples.append(
                SFTSample(
                    instruction=obj["instruction"],
                    response=obj["response"],
                    id=obj.get("id"),
                    created_at=created_at,
                    recency_weight=recency_weight,
                    reward_intensity=reward_intensity if reward_intensity is not None else None,
                )
            )
    if not samples:
        raise ValueError(f"No SFT samples found in '{path}'.")
    return samples


def _maybe_diversity_sample(
    samples: List[SFTSample],
    target_size: Optional[int],
    diversity_k: Optional[int],
    diversity_per_cluster: int,
) -> Tuple[List[SFTSample], Dict[str, Any]]:
    if target_size is None or target_size <= 0 or target_size >= len(samples):
        return samples, {"num_clusters": 0, "selected_clusters": 0}
    if SentenceTransformer is None or np is None or KMeans is None:
        rng = random.Random(1234)
        rng.shuffle(samples)
        sliced = samples[:target_size]
        return sliced, {"num_clusters": 0, "selected_clusters": 0}

    # Build embedding model (small, fast)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"User+Context: {s.instruction}\nAssistant: {s.response}" for s in samples]
    embeddings = model.encode(texts, normalize_embeddings=True)
    n = len(samples)
    if diversity_k is not None:
        k = max(2, min(diversity_k, n))
    else:
        k = max(2, int(math.sqrt(n / 2)))
        k = min(k, n)  # guard
    try:
        km = KMeans(n_clusters=k, n_init=5, random_state=1234)
        labels = km.fit_predict(embeddings)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"k-means failed, falling back to shuffle diversity: {exc}")
        rng = random.Random(1234)
        rng.shuffle(samples)
        return samples[:target_size]

    # Weighted cluster sampling with per-cluster cap
    cluster_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        cluster_to_indices.setdefault(int(label), []).append(idx)

    rng = random.Random(1234)
    for idxs in cluster_to_indices.values():
        rng.shuffle(idxs)

    def sample_weight(idx: int) -> float:
        s = samples[idx]
        reward = s.reward_intensity
        reward_factor = 1.0
        if reward is not None:
            reward_factor = max(0.0, (reward + 1.0) / 2.0)  # map [-1,1] -> [0,1]
            if reward_factor == 0.0:
                reward_factor = 0.05  # keep a small chance
        return s.recency_weight * reward_factor

    selected: List[SFTSample] = []
    selected_cluster_ids: List[int] = []
    for cid, idxs in cluster_to_indices.items():
        # sort by weight descending
        idxs.sort(key=sample_weight, reverse=True)
        take = min(diversity_per_cluster, len(idxs))
        if take > 0:
            selected_cluster_ids.append(cid)
        selected.extend(samples[i] for i in idxs[:take])

    if len(selected) > target_size:
        # Downselect globally by weight if we overshoot
        selected.sort(key=lambda s: sample_weight(samples.index(s)), reverse=True)  # type: ignore[arg-type]
        selected = selected[:target_size]

    stats = {
        "num_clusters": len(cluster_to_indices),
        "selected_clusters": len(selected_cluster_ids),
    }
    return selected, stats
    return selected


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_model_and_tokenizer(cfg: Dict[str, Any]):
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    load_in_4bit = model_cfg.get("load_in_4bit", False)

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_id"],
        padding_side=model_cfg.get("padding_side", "left"),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        torch_dtype=torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else None,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def _build_schedulers_and_opt(model, cfg: Dict[str, Any], total_steps: int):
    training_cfg = cfg["training"]
    lr = min(training_cfg["sft_learning_rate"], training_cfg["preference_learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_cfg["warmup_steps"],
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _compute_sft_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["labels"].clone()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=labels,
    )
    # Weight per-sample then mean
    loss = outputs.loss
    if "weight" in batch:
        loss = loss * batch["weight"]
        loss = loss.mean()
    return loss


def _mix_batches(
    sft_iter,
    pref_iter,
    cfg: Dict[str, Any],
    steps: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    training_cfg = cfg["training"]
    sft_weight = training_cfg["sft_weight"]
    pref_weight = training_cfg["preference_weight"]
    mixing = training_cfg.get("batch_mixing", "interleaved")
    sequence: List[Tuple[str, Dict[str, Any]]] = []
    for _ in range(steps):
        if mixing == "alternating":
            choice = "sft" if len(sequence) % 2 == 0 else "pref"
        else:
            rnd = random.random()
            choice = "sft" if rnd < sft_weight / (sft_weight + pref_weight) else "pref"
        if choice == "sft":
            try:
                sequence.append(("sft", next(sft_iter)))
            except StopIteration:
                continue
        else:
            try:
                sequence.append(("pref", next(pref_iter)))
            except StopIteration:
                continue
    return sequence


def train(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    training_cfg = cfg["training"]
    set_seed(training_cfg.get("seed", 42))

    db = TrainingDatabase(db_path=training_cfg.get("db_path"))

    use_recency = training_cfg["sft_enhancements"].get("use_recency_weighting", True)
    tau_fast = float(training_cfg["sft_enhancements"].get("recency_tau_fast_seconds", 900))
    tau_slow = float(training_cfg["sft_enhancements"].get("recency_tau_slow_seconds", 5400))
    diversity_target = training_cfg["sft_enhancements"].get("diversity_target_size")
    diversity_k = training_cfg["sft_enhancements"].get("diversity_k")
    diversity_per_cluster = int(training_cfg["sft_enhancements"].get("diversity_per_cluster", 2))

    if training_cfg.get("use_database", True):
        sft_samples = load_sft_pairs_from_database(
            db=db,
            filters=training_cfg.get("sft_filters", {}),
            use_recency_weighting=use_recency,
            tau_fast=tau_fast,
            tau_slow=tau_slow,
        )
    else:
        sft_path = training_cfg.get("sft_data_path")
        if not sft_path:
            raise ValueError("sft_data_path is required when use_database=false")
        sft_samples = load_sft_pairs_from_jsonl(
            sft_path,
            use_recency_weighting=use_recency,
            tau_fast=tau_fast,
            tau_slow=tau_slow,
        )

    sft_samples, diversity_stats = _maybe_diversity_sample(
        sft_samples,
        diversity_target,
        diversity_k,
        diversity_per_cluster,
    )

    if training_cfg.get("use_database", True):
        pref_filters = training_cfg.get("preference_filters", {})
        pref_limit = pref_filters.get("limit")
        pref_rows = db.get_preference_pairs(limit=pref_limit)
        preference_path = None
        preference_samples = [
            PreferenceSample(prompt=row["prompt"], chosen=row["chosen"], rejected=row["rejected"])
            for row in pref_rows
        ]
    else:
        preference_path = training_cfg.get("preferences_path")
        if not preference_path:
            raise ValueError("preferences_path is required when use_database=false")
        preference_samples = load_preferences(preference_path)

    if not sft_samples:
        raise ValueError("No SFT samples loaded.")
    if not preference_samples:
        raise ValueError("No preference samples loaded.")

    val_fraction = float(training_cfg.get("validation_fraction", 0.0))
    val_max_batches = int(training_cfg.get("validation_max_batches", 0))

    def split_dataset(items: List[Any], fraction: float):
        if fraction <= 0.0 or len(items) <= 1:
            return items, None
        val_size = max(1, int(len(items) * fraction))
        rng = random.Random(42)
        rng.shuffle(items)
        val_items = items[:val_size]
        train_items = items[val_size:]
        if not train_items:
            train_items, val_items = items, None
        return train_items, val_items

    sft_samples, sft_val_samples = split_dataset(sft_samples, val_fraction)
    preference_samples, pref_val_samples = split_dataset(preference_samples, val_fraction)

    model, tokenizer = _prepare_model_and_tokenizer(cfg)
    device = model.device

    sft_dataset = SFTDataset(sft_samples, tokenizer, max_length=cfg["model"]["max_length"])
    pref_dataset = PreferenceDataset(preference_samples, tokenizer, max_length=cfg["model"]["max_length"])

    sft_loader = DataLoader(
        sft_dataset,
        batch_size=training_cfg["sft_batch_size"],
        shuffle=True,
    )
    pref_loader = DataLoader(
        pref_dataset,
        batch_size=training_cfg["preference_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    total_steps = math.ceil(len(sft_loader) * training_cfg["sft_weight"]) + math.ceil(
        len(pref_loader) * training_cfg["preference_weight"]
    )
    total_steps *= training_cfg["num_epochs"]

    optimizer, scheduler = _build_schedulers_and_opt(model, cfg, total_steps)
    logger = TrainingJSONLogger(output_dir=training_cfg["output_dir"])

    global_step = 0
    used_sft_ids: List[int] = []

    model.train()
    for epoch in range(training_cfg["num_epochs"]):
        sft_iter = iter(sft_loader)
        pref_iter = iter(pref_loader)
        mixed = _mix_batches(sft_iter, pref_iter, cfg, steps=max(len(sft_loader), len(pref_loader)))
        epoch_loss_sft = 0.0
        epoch_loss_pref = 0.0
        epoch_steps_sft = 0
        epoch_steps_pref = 0
        for batch_type, batch in mixed:
            if batch_type == "sft":
                batch = {k: v.to(device) for k, v in batch.items()}
                loss_sft = _compute_sft_loss(model, batch)
                loss_pref = torch.tensor(0.0, device=device)
                weight = training_cfg["sft_weight"]
                if training_cfg.get("normalize_losses", True):
                    loss = weight * loss_sft
                else:
                    loss = loss_sft
                sample_ids = batch.get("sample_id")
                if sample_ids is not None:
                    for sid in sample_ids.tolist():
                        if sid != -1:
                            used_sft_ids.append(sid)
                epoch_loss_sft += float(loss_sft.detach().cpu())
                epoch_steps_sft += 1
            else:
                batch = {side: {k: v.to(device) for k, v in tensors.items()} for side, tensors in batch.items()}
                loss_pref = dpo_loss(model, batch, beta=training_cfg["beta"])
                loss_sft = torch.tensor(0.0, device=device)
                weight = training_cfg["preference_weight"]
                if training_cfg.get("normalize_losses", True):
                    loss = weight * loss_pref
                else:
                    loss = loss_pref
                epoch_loss_pref += float(loss_pref.detach().cpu())
                epoch_steps_pref += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % training_cfg["logging_steps"] == 0:
                logger.log_step(
                    global_step,
                    {
                        "loss": loss.item(),
                        "loss_sft": float(loss_sft.item()) if hasattr(loss_sft, "item") else 0.0,
                        "loss_pref": float(loss_pref.item()) if hasattr(loss_pref, "item") else 0.0,
                        "lr": scheduler.get_last_lr()[0],
                    },
                )

        if used_sft_ids:
            db.update_sft_used(pair_ids=used_sft_ids, training_epoch=epoch)
            used_sft_ids = []

        # Validation (lightweight)
        val_metrics = {}
        if val_fraction > 0.0 and (sft_val_samples or pref_val_samples):
            model.eval()
            if sft_val_samples:
                sft_val_ds = SFTDataset(sft_val_samples, tokenizer, max_length=cfg["model"]["max_length"])
                sft_val_loader = DataLoader(sft_val_ds, batch_size=training_cfg["sft_batch_size"], shuffle=False)
                vloss = 0.0
                vsteps = 0
                for i, vb in enumerate(sft_val_loader):
                    if val_max_batches and i >= val_max_batches:
                        break
                    vb = {k: v.to(device) for k, v in vb.items()}
                    with torch.no_grad():
                        v = _compute_sft_loss(model, vb)
                    vloss += float(v.detach().cpu())
                    vsteps += 1
                if vsteps:
                    val_metrics["val_loss_sft"] = vloss / vsteps
            if pref_val_samples:
                pref_val_ds = PreferenceDataset(pref_val_samples, tokenizer, max_length=cfg["model"]["max_length"])
                pref_val_loader = DataLoader(pref_val_ds, batch_size=training_cfg["preference_batch_size"], shuffle=False, collate_fn=collate_fn)
                vloss = 0.0
                vsteps = 0
                for i, vb in enumerate(pref_val_loader):
                    if val_max_batches and i >= val_max_batches:
                        break
                    vb = {side: {k: v.to(device) for k, v in tensors.items()} for side, tensors in vb.items()}
                    with torch.no_grad():
                        v = dpo_loss(model, vb, beta=training_cfg["beta"])
                    vloss += float(v.detach().cpu())
                    vsteps += 1
                if vsteps:
                    val_metrics["val_loss_pref"] = vloss / vsteps
            model.train()

        logger.log_step(
            global_step,
            {
                "epoch": epoch,
                "epoch_loss_sft": (epoch_loss_sft / max(1, epoch_steps_sft)) if epoch_steps_sft else 0.0,
                "epoch_loss_pref": (epoch_loss_pref / max(1, epoch_steps_pref)) if epoch_steps_pref else 0.0,
                "diversity_num_clusters": diversity_stats.get("num_clusters", 0),
                "diversity_selected_clusters": diversity_stats.get("selected_clusters", 0),
                "recency_mean": float(sum(s.recency_weight for s in sft_samples) / len(sft_samples)) if sft_samples else 0.0,
                **val_metrics,
            },
        )

    os.makedirs(training_cfg["output_dir"], exist_ok=True)
    model.save_pretrained(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])
    logger.finalize({"global_steps": global_step})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-channel LoRA training (SFT + preferences).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "config" / "training" / "lora_dual.yaml"),
        help="Path to training config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
