import argparse
import json
import os
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base vs LoRA-adapted model behavior on an evaluation set."
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="gpt2",
        help="Base model identifier (must match the one used for LoRA training).",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="artifacts/lora/default",
        help="Directory containing the trained LoRA adapter.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="data/preferences/preferences.jsonl",
        help="JSONL file with evaluation prompts. Uses 'prompt' and 'chosen' fields.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of examples to print.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    return parser.parse_args()


def load_eval_prompts(path: str, limit: int) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation file not found at '{path}'. "
            "Expected JSONL with at least a 'prompt' field (optionally 'chosen')."
        )

    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                continue
            items.append(obj)
            if len(items) >= limit:
                break

    if not items:
        raise ValueError(f"No usable prompts found in '{path}'.")

    return items


def main() -> None:
    args = parse_args()

    print(f"Loading base model '{args.base_model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id)
    base_model.eval()

    if not os.path.exists(args.lora_dir):
        raise FileNotFoundError(
            f"LoRA directory '{args.lora_dir}' not found. "
            "Train a LoRA adapter with core.lora_trainer.train_dpo first."
        )

    print(f"Loading LoRA adapter from '{args.lora_dir}'...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.lora_dir,
    )
    lora_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    lora_model.to(device)

    eval_items = load_eval_prompts(args.eval_file, args.samples)

    for idx, item in enumerate(eval_items, start=1):
        prompt = item["prompt"]
        reference = item.get("chosen")

        print("-" * 60)
        print(f"Example {idx}")
        print(f"PROMPT:\n{prompt}\n")
        if reference is not None:
            print(f"REFERENCE (chosen):\n{reference}\n")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            base_out = base_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            lora_out = lora_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
        lora_text = tokenizer.decode(lora_out[0], skip_special_tokens=True)

        print("BASE OUTPUT:\n")
        print(base_text)
        print("\nLoRA OUTPUT:\n")
        print(lora_text)
        print()


if __name__ == "__main__":
    main()


