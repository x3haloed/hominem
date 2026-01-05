import json
import warnings

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from mlx_vlm.trainer.trainer import Trainer as BaseTrainer, get_prompt
from mlx_vlm.utils import prepare_inputs as base_prepare_inputs


def _prepare_inputs(
    *,
    processor,
    images=None,
    audio=None,
    prompts=None,
    image_token_index=None,
    resize_shape=None,
    add_special_tokens=False,
    **kwargs,
):
    if not images and not audio:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            prompts,
            add_special_tokens=add_special_tokens,
            padding=True,
            truncation=True,
        )
        input_ids = mx.array(inputs.input_ids).astype(mx.int32)
        mask = mx.array(inputs.attention_mask).astype(mx.int32)
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
        }
    return base_prepare_inputs(
        processor=processor,
        images=images,
        audio=audio,
        prompts=prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        **kwargs,
    )


class HominemDataset:
    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_processor=None,
        take=None,
        split=None,
        image_resize_shape=None,
    ):
        if split is not None:
            self.dataset = hf_dataset[split]
        else:
            self.dataset = hf_dataset
        if take is not None:
            self.dataset = self.dataset.take(take)
        self.processor = processor
        self.config = config
        self.image_processor = image_processor
        self.image_resize_shape = image_resize_shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        batch_items = item if isinstance(item, list) else [item]

        images = []
        conversations = []
        for entry in batch_items:
            images.append(entry.get("images", entry.get("image", None)))
            conv = entry.get("messages", entry.get("conversations"))
            if conv is None:
                conv = []
            if isinstance(conv, str):
                conv = [{"role": "user", "content": conv}]
            elif isinstance(conv, dict):
                conv = [conv]
            conversations.append(conv)

        if len(images) == 1:
            images = images[0]
        if images in (None, "", []):
            images = None
        if isinstance(images, list) and len(images) == 0:
            images = None
        if isinstance(images, list):
            empty = True
            for img in images:
                if img is None:
                    continue
                if isinstance(img, str) and img == "":
                    continue
                if isinstance(img, list) and len(img) == 0:
                    continue
                empty = False
                break
            if empty:
                images = None

        prompts = []
        for idx_offset, conversation in enumerate(conversations):
            if not conversation:
                warnings.warn(f"Skipping empty conversation at batch index {idx_offset}")
                continue
            if self.config["model_type"] == "pixtral":
                conversation = [json.loads(i) for i in conversation]
                if len(conversations) > 1:
                    warnings.warn(
                        "Pixtral batch processing is not supported yet. Set batch size to 1."
                    )
            prompt = get_prompt(self.config["model_type"], self.processor, conversation)
            if not isinstance(prompt, str):
                prompt = json.dumps(prompt, ensure_ascii=False)
            prompts.append(prompt)

        if not prompts:
            warnings.warn("All conversations in batch were empty; skipping batch.")
            return {
                "input_ids": mx.array([], dtype=mx.int32),
                "attention_mask": mx.array([], dtype=mx.int32),
                "pixel_values": None,
            }

        image_token_index = self.config["image_token_index"]
        inputs = _prepare_inputs(
            processor=self.processor,
            images=images,
            audio=None,
            prompts=prompts,
            image_token_index=image_token_index,
            resize_shape=self.image_resize_shape,
        )
        if "input_ids" in inputs and inputs["input_ids"].size == 0:
            warnings.warn("Empty input_ids after tokenization; skipping batch.")
            return {
                "input_ids": mx.array([], dtype=mx.int32),
                "attention_mask": mx.array([], dtype=mx.int32),
                "pixel_values": None,
            }

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        mask = inputs["attention_mask"]
        kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        if mask is None:
            mask = mx.ones_like(input_ids)

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": mask,
            **kwargs,
        }

        example_weight = None
        if isinstance(item, list):
            weights = []
            for entry in batch_items:
                weights.append(
                    entry.get("example_weight", entry.get("sample_weight", entry.get("weight")))
                )
            example_weight = weights
        else:
            example_weight = item.get("example_weight", item.get("sample_weight", item.get("weight")))

        if example_weight is not None:
            if isinstance(example_weight, list):
                output["example_weight"] = mx.array(example_weight)
            else:
                output["example_weight"] = mx.array([float(example_weight)])

        if "preferred_input_ids" in item and "rejected_input_ids" in item:
            output["preferred_input_ids"] = mx.array(item["preferred_input_ids"])
            output["rejected_input_ids"] = mx.array(item["rejected_input_ids"])

        return output


class HominemTrainer(BaseTrainer):
    def preference_loss(self, model, batch):
        pref_ids = batch["preferred_input_ids"]
        rej_ids = batch["rejected_input_ids"]

        def score(input_ids):
            logits = model(input_ids).logits.astype(mx.float32)
            labels = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            ce = nn.losses.cross_entropy(logits, labels)
            return -ce.mean(axis=1)

        s_pref = score(pref_ids)
        s_rej = score(rej_ids)
        return -mx.mean(mx.log(mx.sigmoid(s_pref - s_rej)))

    def loss_fn(self, model, batch):
        if "preferred_input_ids" in batch and "rejected_input_ids" in batch:
            return self.preference_loss(model, batch)

        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids[:, 1:]

        batch_size, seq_length = input_ids.shape

        if self.train_on_completions:
            weight_mask = mx.ones_like(attention_mask)

            assistant_response_index = np.where(input_ids == self.assistant_id)[1]
            range_matrix = mx.repeat(
                mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0
            )
            assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(
                -1, 1
            )
            weight_mask = mx.where(
                assistant_mask, mx.zeros_like(weight_mask), weight_mask
            )[:, 1:]
        else:
            weight_mask = None

        input_ids = input_ids[:, :-1]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1]
        lengths = mx.sum(attention_mask, axis=1)
        example_weight = batch.get("example_weight")
        kwargs = {
            k: v
            for k, v in batch.items()
            if k not in ["input_ids", "pixel_values", "attention_mask", "example_weight"]
        }

        outputs = model(input_ids, pixel_values, attention_mask, **kwargs)
        logits = outputs.logits.astype(mx.float32)

        def align_logits_with_labels(logits, labels):
            if logits.shape[1] < labels.shape[1]:
                pad_length = labels.shape[1] - logits.shape[1]
                pad_width = ((0, 0), (0, pad_length), (0, 0))
                return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
            if logits.shape[1] > labels.shape[1]:
                return logits[:, -labels.shape[1] :, :]
            return logits

        logits = align_logits_with_labels(logits, labels)

        length_mask = mx.arange(input_ids.shape[1])[None, :] < lengths[:, None]
        ce = nn.losses.cross_entropy(logits, labels, weights=weight_mask) * length_mask

        if example_weight is not None:
            if example_weight.shape[0] != ce.shape[0]:
                print(
                    f"[debug] example_weight shape={example_weight.shape} batch={ce.shape[0]}"
                )
                example_weight = example_weight[: ce.shape[0]]
            if len(example_weight.shape) > 1:
                example_weight = mx.squeeze(example_weight)
            example_weight = mx.expand_dims(example_weight, axis=1)
            ce = ce * example_weight
            denom = (length_mask * example_weight).sum()
        else:
            denom = length_mask.sum()

        return ce.sum() / mx.maximum(denom, 1)

    def train_step(self, batch):
        if batch.get("input_ids") is not None and batch["input_ids"].size == 0:
            return mx.array(0.0)
        if "preferred_input_ids" in batch and "input_ids" in batch:
            raise ValueError("Batch must be either SFT or preference, not both")

        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, batch)

        if self.clip_gradients is not None:
            grads = tree_map(
                lambda g: mx.clip(g, -self.clip_gradients, self.clip_gradients), grads
            )

        self.optimizer.update(self.model, grads)
        return loss
