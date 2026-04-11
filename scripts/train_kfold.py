"""
train_kfold.py — Main script for training music genre classification models.

Performs stratified k-fold cross-validation or single-fold training using a
fixed 10% test set (Fold 0) and 5 cross-validation folds (Folds 1-5).
Saves final results to outputs/logs/<experiment_name>/results.yaml.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Add project root to sys.path
root_dir = str(Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.data.mel_dataset import get_dataloaders
from src.models.cnn import CNN2D, CNN2D3C
from src.models.rnn import RNN, RNNAttention, LSTM, LSTMAttention
from src.models.crnn import CRNN, CRNNAttention, CRNN3C, CRNN3CAttention
from src.training.lightning_module import GenreClassifierModule
from src.training.train_manager import train_one_fold
from scripts.evaluate_confusion import generate_confusion_matrix

# Mapping of model names to their classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn2d": CNN2D,
    "cnn2d_3c": CNN2D3C,
    "rnn": RNN,
    "rnna": RNNAttention,
    "lstm": LSTM,
    "lstma": LSTMAttention,
    "crnn": CRNN,
    "crnna": CRNNAttention,
    "crnn3c": CRNN3C,
    "crnn3ca": CRNN3CAttention,
}


def load_config(paths: List[str]) -> Dict[str, Any]:
    """Loads and merges configuration parameters from multiple YAML files."""
    config: Dict[str, Any] = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
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
    # Set seed for reproducibility
    import pytorch_lightning as pl
    pl.seed_everything(42, workers=True)

    # Set float32 matmul precision for performance on Tensor Cores
    import torch
    torch.set_float32_matmul_precision("medium")

    # 1. Load experiment configuration
    exp_config_path = os.path.join("configs", args.exp)
    if not exp_config_path.endswith((".yaml", ".yml")):
        exp_config_path += ".yaml"

    if not os.path.exists(exp_config_path):
        print(f"Error: Experiment config not found at {exp_config_path}")
        return

    # Load only the experiment config
    config = load_config([exp_config_path])

    # Determine model architecture name
    model_name = config.get("model_arch") or config.get("model")
    if isinstance(model_name, dict) or not model_name:
        print("Error: Model name not found or invalid. Check your experiment config.")
        return
    
    if model_name not in MODEL_REGISTRY:
        print(f"Error: Model architecture '{model_name}' not in registry.")
        return

    # 3. Apply CLI overrides and ensure types
    if args.epochs:
        config["training"]["max_epochs"] = int(args.epochs)
    if args.batch_size:
        config["batch_size"] = int(args.batch_size)
    if args.lr:
        config["training"]["lr"] = float(args.lr)

    if "training" in config:
        for k in ["lr", "weight_decay"]:
            if k in config["training"]:
                config["training"][k] = float(config["training"][k])
        if "max_epochs" in config["training"]:
            config["training"]["max_epochs"] = int(config["training"]["max_epochs"])
    if "batch_size" in config:
        config["batch_size"] = int(config["batch_size"])

    # 4. Determine which folds to run
    folds_to_run = [args.fold] if args.fold else list(range(1, 6))
    exp_name = config["training"].get("experiment_name", "exp")

    print(f"Experiment Config: {args.exp}")
    print(f"Model Architecture: {model_name}")
    print(f"Experiment Name:    {exp_name}")
    print(f"Folds to Process:   {folds_to_run}")

    # 5. Training Loop
    all_fold_results: Dict[int, Dict[str, Any]] = {}

    for fold in folds_to_run:
        print(f"\n" + "=" * 50)
        print(f" PROCESSING FOLD {fold} ")
        print("=" * 50)

        # A. Setup Dataloaders (Train & Val)
        train_loader, val_loader, _, _ = get_dataloaders(
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
            fold_idx=fold,
            **config["training"],
        )

        all_fold_results[fold] = fold_results
        print(f"Fold {fold} Result: Val Acc = {fold_results['val_acc']:.4f}")

    # 6. Post-Training Evaluation (Individual Test + Ensemble)
    print("\n" + "=" * 50)
    print(" STARTING TEST EVALUATION ")
    print("=" * 50)

    # Get the test loader (Fold 0)
    _, _, test_loader, _ = get_dataloaders(
        metadata_path=config["metadata_path"],
        mel_dir=config["mel_dir"],
        val_fold=1, # val_fold doesn't matter for test_loader (Fold 0)
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        segment_seconds=config.get("segment_seconds"),
        sample_rate=config.get("sample_rate", 22050),
        hop_length=config.get("hop_length", 512),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    individual_test_accs: Dict[int, float] = {}
    
    all_logits: List[torch.Tensor] = []
    true_labels: List[torch.Tensor] = []

    for fold, results in all_fold_results.items():
        best_path = results["best_model_path"]
        print(f"Evaluating Fold {fold} using {best_path}...")
        
        # Load model from checkpoint
        model_arch = MODEL_REGISTRY[model_name](num_classes=config["model"]["num_classes"])
        lit_module = GenreClassifierModule.load_from_checkpoint(
            best_path, model=model_arch
        )
        lit_module.to(device)
        lit_module.eval()

        fold_logits = []
        fold_correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                logits = lit_module(x)
                fold_logits.append(logits.cpu())
                
                if fold == list(all_fold_results.keys())[0]: # Collect labels only once
                    true_labels.append(y.cpu())

                preds = torch.argmax(logits, dim=1)
                fold_correct += (preds == y).sum().item()
                total += y.size(0)

        acc = fold_correct / total
        individual_test_accs[fold] = acc
        all_logits.append(torch.cat(fold_logits, dim=0))
        print(f"Fold {fold} Test Acc: {acc:.4f}")

    # 7. Model Ensemble (Averaging Logits)
    ensemble_acc = 0.0
    if len(all_logits) > 0:
        stacked_logits = torch.stack(all_logits, dim=0) # [folds, samples, classes]
        avg_logits = torch.mean(stacked_logits, dim=0) # [samples, classes]
        
        y_true = torch.cat(true_labels, dim=0)
        ensemble_preds = torch.argmax(avg_logits, dim=1)
        ensemble_acc = (ensemble_preds == y_true).float().mean().item()
        
        print("\n" + "=" * 50)
        print(f" ENSEMBLE TEST ACCURACY: {ensemble_acc:.4f}")
        print("=" * 50)

    # 8. Final Summary and Saving Results
    summary_results: Dict[str, Any] = {
        "experiment_name": exp_name,
        "model": model_name,
        "folds": {
            f: {"val_acc": r["val_acc"], "test_acc": individual_test_accs[f]} 
            for f, r in all_fold_results.items()
        },
        "ensemble_test_acc": ensemble_acc
    }
    
    if len(all_fold_results) > 0:
        summary_results["avg_val_acc"] = sum(r["val_acc"] for r in all_fold_results.values()) / len(all_fold_results)
        
    if len(individual_test_accs) > 1:
        summary_results["avg_test_acc"] = sum(individual_test_accs.values()) / len(individual_test_accs)

    # Save results.yaml and predictions
    exp_log_dir = os.path.join(config["training"]["log_dir"], exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # Save ensemble predictions and true labels for confusion matrix
    if len(all_logits) > 0:
        prediction_data = {
            "y_true": y_true,
            "y_pred": ensemble_preds,
            "avg_logits": avg_logits
        }
        torch.save(prediction_data, os.path.join(exp_log_dir, "test_predictions.pt"))

        # Generate Confusion Matrix
        generate_confusion_matrix(
            y_true=y_true, 
            y_pred=ensemble_preds, 
            log_dir=exp_log_dir, 
            exp_name=exp_name
        )

    results_path = os.path.join(exp_log_dir, "results.yaml")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.dump(summary_results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music Genre Classification Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp", type=str, required=True, help="Experiment config name.")
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], help="Specific fold.")
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    
    main(parser.parse_args())
