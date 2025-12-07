from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

from core.data.schema import REWARD_DIMENSIONS


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_reward_model(model_dir: Path, device: torch.device) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_base_and_lora(
    base_model_id: str,
    lora_dir: Path,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.to(device)
    base_model.eval()

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
    )
    lora_model.to(device)
    lora_model.eval()

    return tokenizer, base_model, lora_model


def generate_response(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def score_pair(
    *,
    reward_tokenizer,
    reward_model,
    device: torch.device,
    prompt: str,
    response: str,
    max_length: int,
) -> Dict[str, float]:
    text = f"User: {prompt}\nAssistant: {response}"
    encoded = reward_tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = reward_model(**encoded)
        logits = outputs.logits.squeeze(0).cpu().tolist()

    return {dim: float(logits[idx]) for idx, dim in enumerate(REWARD_DIMENSIONS)}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base vs LoRA outputs and reward vectors for a single prompt."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt text. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="gpt2",
        help="Base causal LM identifier.",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=Path("artifacts/lora/default"),
        help="Directory containing the trained LoRA adapter.",
    )
    parser.add_argument(
        "--reward-model-dir",
        type=Path,
        default=Path("artifacts/reward_model/default/model"),
        help="Directory containing the trained reward model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (used when --do-sample is enabled).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value (used when --do-sample is enabled).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding for generation.",
    )
    parser.add_argument(
        "--reward-max-length",
        type=int,
        default=512,
        help="Maximum sequence length for reward model inputs.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    prompt = args.prompt or input("Prompt: ").strip()
    if not prompt:
        print("No prompt provided; exiting.")
        return

    device = select_device()

    reward_tokenizer, reward_model = load_reward_model(args.reward_model_dir, device)
    gen_tokenizer, base_model, lora_model = load_base_and_lora(
        args.base_model_id,
        args.lora_dir,
        device,
    )

    print("\n--- Generating responses ---\n")
    base_text = generate_response(
        model=base_model,
        tokenizer=gen_tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )
    lora_text = generate_response(
        model=lora_model,
        tokenizer=gen_tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    print("PROMPT:\n")
    print(prompt)
    print("\nBASE OUTPUT:\n")
    print(base_text)
    print("\nLoRA OUTPUT:\n")
    print(lora_text)
    print("\n--- Scoring with reward model ---\n")

    base_scores = score_pair(
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        device=device,
        prompt=prompt,
        response=base_text,
        max_length=args.reward_max_length,
    )
    lora_scores = score_pair(
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        device=device,
        prompt=prompt,
        response=lora_text,
        max_length=args.reward_max_length,
    )

    print("DIMENSION               BASE       LORA      DELTA")
    print("-" * 52)
    for dim in REWARD_DIMENSIONS:
        b = base_scores.get(dim, 0.0)
        l = lora_scores.get(dim, 0.0)
        d = l - b
        print(f"{dim:20s} {b:+7.3f}  {l:+7.3f}  {d:+7.3f}")
    print()


if __name__ == "__main__":  # pragma: no cover
    main()


