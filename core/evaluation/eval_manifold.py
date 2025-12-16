#!/usr/bin/env python3
"""
Evaluate a trained emotion manifold head on labeled conversation data.

Loads the trained manifold model and evaluates predictions against ground truth labels,
computing metrics like MSE, MAE, and correlations for each emotion axis.
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
from core.lora_trainer.train_manifold import record_to_text, MANIFOLD_KEYS


def load_model_and_tokenizer(model_dir: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the trained manifold model and tokenizer."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    print(f"📦 Loading manifold model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Handle models without pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(MANIFOLD_KEYS),
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
    records, summary = loader.load_records(MANIFOLD_KEYS)

    if summary.usable_records == 0:
        raise ValueError("No usable evaluation records found.")

    print(f"📊 Loaded {summary.usable_records} evaluation records")
    return records


def predict_manifold(model: AutoModelForSequenceClassification,
                    tokenizer: AutoTokenizer,
                    records: List[Dict],
                    batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on records and return predictions vs ground truth.

    Returns:
        predictions: (N, 6) array of predicted manifold values
        ground_truth: (N, 6) array of true manifold values
    """
    device = next(model.parameters()).device

    texts = []
    ground_truth = []

    for rec in records:
        texts.append(record_to_text(rec))

        # Extract ground truth labels
        labels = rec.get("labels", {})
        gt_values = [float(labels.get(key, 0.0)) for key in MANIFOLD_KEYS]
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
    return predictions, ground_truth


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    metrics = {}

    # Overall metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))

    metrics["overall"] = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
    }

    # Per-axis metrics
    metrics["per_axis"] = {}
    for i, axis in enumerate(MANIFOLD_KEYS):
        axis_pred = predictions[:, i]
        axis_true = ground_truth[:, i]

        mse_axis = np.mean((axis_pred - axis_true) ** 2)
        mae_axis = np.mean(np.abs(axis_pred - axis_true))
        corr = np.corrcoef(axis_true, axis_pred)[0, 1] if np.std(axis_true) > 0 and np.std(axis_pred) > 0 else 0.0

        metrics["per_axis"][axis] = {
            "mse": float(mse_axis),
            "mae": float(mae_axis),
            "rmse": float(np.sqrt(mse_axis)),
            "correlation": float(corr),
            "mean_error": float(np.mean(axis_pred - axis_true)),
            "std_error": float(np.std(axis_pred - axis_true)),
        }

    # Distribution analysis
    metrics["distribution"] = {
        "predictions": {
            "mean": predictions.mean(axis=0).tolist(),
            "std": predictions.std(axis=0).tolist(),
            "min": predictions.min(axis=0).tolist(),
            "max": predictions.max(axis=0).tolist(),
        },
        "ground_truth": {
            "mean": ground_truth.mean(axis=0).tolist(),
            "std": ground_truth.std(axis=0).tolist(),
            "min": ground_truth.min(axis=0).tolist(),
            "max": ground_truth.max(axis=0).tolist(),
        }
    }

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print evaluation results in a nice format."""
    print("\n" + "="*60)
    print("🎯 EMOTION MANIFOLD EVALUATION RESULTS")
    print("="*60)

    # Overall metrics
    overall = metrics["overall"]
    print("\n📊 OVERALL METRICS:")
    print(".4f")
    print(".4f")
    print(".4f")
    # Per-axis metrics
    print("\n📈 PER-AXIS METRICS:")
    print("<12")
    for axis in MANIFOLD_KEYS:
        axis_metrics = metrics["per_axis"][axis]
        corr = axis_metrics["correlation"]
        corr_color = "🟢" if corr > 0.7 else "🟡" if corr > 0.5 else "🔴"
        print("<12"
              ".4f")

    # Distribution summary
    print("\n📊 PREDICTION DISTRIBUTIONS:")
    print("<12")
    pred_mean = metrics["distribution"]["predictions"]["mean"]
    true_mean = metrics["distribution"]["ground_truth"]["mean"]

    for i, axis in enumerate(MANIFOLD_KEYS):
        pred_val = pred_mean[i]
        true_val = true_mean[i]
        diff = abs(pred_val - true_val)
        status = "✅" if diff < 0.1 else "⚠️" if diff < 0.3 else "❌"
        print("<12")
def save_results(metrics: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained emotion manifold head.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/manifold_head",
        help="Directory containing the trained manifold model",
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
        default="artifacts/manifold_evaluation.json",
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
    print("🔮 Running manifold predictions...")
    predictions, ground_truth = predict_manifold(model, tokenizer, records, args.batch_size)

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
