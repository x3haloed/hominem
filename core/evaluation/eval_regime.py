#!/usr/bin/env python3
"""
Evaluate a trained regime classifier on labeled conversation data.

Loads the trained regime classifier and evaluates predictions against ground truth labels,
computing metrics like MSE, MAE, correlations, and classification accuracy for each regime.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.data.shard_loader import ShardLoader
from core.lora_trainer.train_regime import record_to_text, REGIME_KEYS


def load_model_and_tokenizer(model_dir: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the trained regime classifier and tokenizer."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    print(f"📦 Loading regime classifier from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(REGIME_KEYS),
        problem_type="regression",
        trust_remote_code=True,
    )

    # Move to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully on {device}")
    return model, tokenizer


def load_evaluation_data(data_roots: List[str], datasets: List[str]) -> List[Dict]:
    """Load evaluation records from the specified datasets."""
    loader = ShardLoader(root_paths=[Path(p) for p in data_roots], dataset_filters=datasets)
    records, summary = loader.load_records(REGIME_KEYS)

    if summary.usable_records == 0:
        raise ValueError("No usable evaluation records found.")

    print(f"📊 Loaded {summary.usable_records} evaluation records")
    return records


def predict_regimes(model: AutoModelForSequenceClassification,
                   tokenizer: AutoTokenizer,
                   records: List[Dict],
                   batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on records and return predictions vs ground truth.

    Returns:
        predictions: (N, 7) array of predicted regime probabilities
        ground_truth: (N, 7) array of true regime probabilities
    """
    device = next(model.parameters()).device

    texts = []
    ground_truth = []

    for rec in records:
        texts.append(record_to_text(rec))

        # Extract ground truth labels
        labels = rec.get("labels", {})
        gt_values = [float(labels.get(key, 0.0)) for key in REGIME_KEYS]
        ground_truth.append(gt_values)

    ground_truth = np.array(ground_truth)

    # Create dataset for batching
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    predictions = []

    # Batch inference
    for i in range(0, len(tokenized), batch_size):
        batch = tokenized[i:i+batch_size]

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )

            # Get predictions (logits for regression)
            batch_preds = outputs.logits.cpu().numpy()
            predictions.extend(batch_preds)

    predictions = np.array(predictions)

    # Apply softmax to ensure probabilities sum to 1
    predictions = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)

    return predictions, ground_truth


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    metrics = {}

    # Overall metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))

    # KL divergence (since these are probability distributions)
    eps = 1e-10
    kl_div = np.mean(np.sum(ground_truth * np.log((ground_truth + eps) / (predictions + eps)), axis=1))

    metrics["overall"] = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
        "kl_divergence": float(kl_div),
    }

    # Per-regime metrics
    metrics["per_regime"] = {}
    for i, regime in enumerate(REGIME_KEYS):
        pred_regime = predictions[:, i]
        true_regime = ground_truth[:, i]

        mse_regime = np.mean((pred_regime - true_regime) ** 2)
        mae_regime = np.mean(np.abs(pred_regime - true_regime))
        corr = np.corrcoef(true_regime, pred_regime)[0, 1] if np.std(true_regime) > 0 and np.std(pred_regime) > 0 else 0.0

        # Top-1 accuracy (argmax prediction vs argmax ground truth)
        pred_argmax = np.argmax(predictions, axis=1)
        true_argmax = np.argmax(ground_truth, axis=1)
        top1_acc = np.mean(pred_argmax == true_argmax)

        metrics["per_regime"][regime] = {
            "mse": float(mse_regime),
            "mae": float(mae_regime),
            "rmse": float(np.sqrt(mse_regime)),
            "correlation": float(corr),
            "mean_error": float(np.mean(pred_regime - true_regime)),
            "std_error": float(np.std(pred_regime - true_regime)),
            "top1_contribution": float(np.mean(true_regime > 0.5)),  # How often this regime is dominant
        }

    metrics["classification"] = {
        "top1_accuracy": float(top1_acc),
        "top3_accuracy": float(np.mean([
            true_argmax[i] in np.argsort(predictions[i])[-3:]
            for i in range(len(true_argmax))
        ])),
    }

    # Distribution analysis
    metrics["distribution"] = {
        "predictions": {
            "mean": predictions.mean(axis=0).tolist(),
            "std": predictions.std(axis=0).tolist(),
            "entropy": [float(-np.sum(p * np.log(p + 1e-10))) for p in predictions[:100]],  # Sample entropy
        },
        "ground_truth": {
            "mean": ground_truth.mean(axis=0).tolist(),
            "std": ground_truth.std(axis=0).tolist(),
            "entropy": [float(-np.sum(p * np.log(p + 1e-10))) for p in ground_truth[:100]],
        }
    }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print evaluation results in a nice format."""
    print("\n" + "="*60)
    print("🎯 REGIME CLASSIFIER EVALUATION RESULTS")
    print("="*60)

    # Overall metrics
    overall = metrics["overall"]
    print("\n📊 OVERALL METRICS:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    # Classification metrics
    cls = metrics["classification"]
    print("\n🎯 CLASSIFICATION METRICS:")
    print(".4f")
    print(".4f")
    # Per-regime metrics
    print("\n📈 PER-REGIME METRICS:")
    print("<25")
    for regime in REGIME_KEYS:
        regime_metrics = metrics["per_regime"][regime]
        corr = regime_metrics["correlation"]
        corr_color = "🟢" if corr > 0.5 else "🟡" if corr > 0.3 else "🔴"
        top1_pct = regime_metrics["top1_contribution"] * 100
        print("<25"
              ".4f")
def save_results(metrics: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained regime classifier.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/regime_classifier",
        help="Directory containing the trained regime classifier",
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=["data/processed_datasets_unified"],
        help="Root folders containing evaluation datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ultrachat_trajectories", "ultrachat_synthetic_trajectories"],
        help="Names of dataset directories to evaluate on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output-file",
        default="artifacts/regime_evaluation.json",
        help="Where to save evaluation results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit evaluation to first N samples (0 = no limit)",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # Load evaluation data
    records = load_evaluation_data(args.data_roots, args.datasets)

    # Limit samples if requested
    if args.max_samples > 0:
        records = records[:args.max_samples]
        print(f"🔢 Limited evaluation to first {args.max_samples} samples")

    # Run predictions
    print("🔮 Running regime predictions...")
    predictions, ground_truth = predict_regimes(model, tokenizer, records, args.batch_size)

    # Compute metrics
    print("📊 Computing evaluation metrics...")
    metrics = compute_metrics(predictions, ground_truth)

    # Print results
    print_metrics(metrics)

    # Save results
    save_results(metrics, args.output_file)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
