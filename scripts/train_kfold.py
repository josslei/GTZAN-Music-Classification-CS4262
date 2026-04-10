"""
train_kfold.py — Main script for training music genre classification models.

Performs stratified k-fold cross-validation or single-fold training using a
fixed 10% test set (Fold 0) and 5 cross-validation folds (Folds 1-5).
Saves final results to outputs/logs/<experiment_name>/results.yaml.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.append(root_dir)

import argparse
from typing import Any, Dict, Type

import torch.nn as nn
import yaml
from src.data.mel_dataset import get_dataloaders
from src.models.cnn import CNN2D
from src.models.rnn import SimpleRNNModel, LSTMModel
from src.models.crnn import CRNNModel

from src.training.train_manager import train_one_fold

# Mapping of model names to their classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn2d": CNN2D,
    "rnn_simple": SimpleRNNModel,
    "lstm": LSTMModel,
    "crnn": CRNNModel,
}



def load_config(paths: list[str]) -> Dict[str, Any]:
    """Loads and merges configuration parameters from multiple YAML files."""
    config: Dict[str, Any] = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                # Deep merge for the 'training' and 'model' sub-dictionaries if they exist
                for key, value in loaded.items():
                    if (
                        key in config
                        and isinstance(config[key], dict)
                        and isinstance(value, dict)
                    ):
                        config[key].update(value)
                    else:
                        config[key] = value
    return config


def main(args: argparse.Namespace) -> None:
    """Main execution flow for running K-fold training."""
    # 1. Load base configurations
    base_configs = ["configs/data.yaml", "configs/train.yaml"]

    # 2. Determine and load experiment configuration
    exp_config_path = os.path.join("configs", "experiments", args.exp)
    if not exp_config_path.endswith((".yaml", ".yml")):
        exp_config_path += ".yaml"

    if not os.path.exists(exp_config_path):
        print(f"Error: Experiment config not found at {exp_config_path}")
        return

    # Merge base and experiment configs
    config = load_config(base_configs + [exp_config_path])

    # 3. Apply CLI overrides to config
    if args.epochs:
        config["training"]["max_epochs"] = int(args.epochs)
    if args.batch_size:
        config["batch_size"] = int(args.batch_size)
    if args.lr:
        config["training"]["lr"] = float(args.lr)

    # Ensure critical numeric parameters are correctly typed
    if "training" in config:
        if "lr" in config["training"]:
            config["training"]["lr"] = float(config["training"]["lr"])
        if "weight_decay" in config["training"]:
            config["training"]["weight_decay"] = float(
                config["training"]["weight_decay"]
            )
        if "max_epochs" in config["training"]:
            config["training"]["max_epochs"] = int(config["training"]["max_epochs"])
    if "batch_size" in config:
        config["batch_size"] = int(config["batch_size"])

    # Determine model from config
    model_name = config.get("model")
    if not model_name or model_name not in MODEL_REGISTRY:
        print(f"Error: Model '{model_name}' not specified or not in registry.")
        return

    # 4. Determine which folds to run
    folds_to_run = [args.fold] if args.fold else range(1, 6)

    exp_name = config["training"].get("experiment_name", "exp")
    print(f"Experiment Config: {args.exp}")
    print(f"Model Architecture: {model_name}")
    print(f"Experiment Name:    {exp_name}")
    print(f"Folds to Process:   {list(folds_to_run)}")

    # 5. Training Loop
    all_fold_metrics: Dict[int, Dict[str, float]] = {}

    for fold in folds_to_run:
        print(f"\n" + "=" * 50)
        print(f" PROCESSING FOLD {fold} ")
        print("=" * 50)

        # A. Setup Dataloaders
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            metadata_path=config["metadata_path"],
            mel_dir=config["mel_dir"],
            val_fold=fold,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            segment_seconds=config.get("segment_seconds"),
            sample_rate=config.get("sample_rate", 22050),
            hop_length=config.get("hop_length", 512),
        )

        # B. Initialize Model
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(num_classes=config["model"]["num_classes"])

        # C. Execute Training
        fold_results = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader if fold == 1 or args.fold else None,
            fold_idx=fold,
            **config["training"],
        )

        all_fold_metrics[fold] = fold_results
        print(f"Fold {fold} Result: Val Acc = {fold_results['val_acc']:.4f}")

    # 6. Final Summary and Saving Results
    summary_results: Dict[str, Any] = {
        "experiment_name": exp_name,
        "model": model_name,
        "folds": all_fold_metrics,
    }

    if not args.fold and len(all_fold_metrics) > 1:
        avg_val_acc = sum(m["val_acc"] for m in all_fold_metrics.values()) / len(
            all_fold_metrics
        )
        summary_results["avg_val_acc"] = avg_val_acc

        test_accs = [
            m["test_acc"] for m in all_fold_metrics.values() if "test_acc" in m
        ]

        print("\n" + "=" * 50)
        print(f" K-FOLD SUMMARY FOR {model_name.upper()} ")
        print(f" Average Validation Accuracy: {avg_val_acc:.4f}")
        if test_accs:
            avg_test_acc = sum(test_accs) / len(test_accs)
            summary_results["avg_test_acc"] = avg_test_acc
            print(f" Average Test Accuracy:       {avg_test_acc:.4f}")
        print("=" * 50)

    # 7. Save results.yaml to the output directory
    results_path = os.path.join(config["training"]["log_dir"], exp_name, "results.yaml")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.dump(summary_results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music Genre Classification Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment Selection
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Name of the experiment config file in configs/experiments/ (e.g., cnn2d_baseline).",
    )

    # Fold Selection
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Specific fold to train (1-5). If not provided, runs all 5 folds sequentially.",
    )

    # Hyperparameter Overrides
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the maximum number of training epochs from config.",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Override the batch size from config."
    )
    parser.add_argument(
        "--lr", type=float, help="Override the learning rate from config."
    )

    cli_args = parser.parse_args()
    main(cli_args)
