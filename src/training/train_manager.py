"""
train_manager.py — Core training logic for music genre classification.

This module provides the train_one_fold function, which handles the execution
of the PyTorch Lightning training process for a single fold.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.training.lightning_module import GenreClassifierModule


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_idx: int = 1,
    max_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    mixup_alpha: float = 0.2,
    optimizer_params: Optional[Dict[str, Any]] = None,
    scheduler_params: Optional[Dict[str, Any]] = None,
    patience: int = 10,
    log_dir: str = "outputs/logs",
    checkpoint_dir: str = "outputs/checkpoints",
    experiment_name: str = "genre_classification",
    accelerator: str = "auto",
    devices: Union[str, int] = "auto",
) -> Dict[str, Any]:
    """Trains a model for a single fold using PyTorch Lightning.

    Args:
        model: PyTorch model for classification.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        fold_idx: The current fold index (used for logging).
        max_epochs: Maximum number of training epochs.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        mixup_alpha: Beta distribution parameter for Mixup (0 = disabled).
        optimizer_params: Parameters for AdamW.
        scheduler_params: Parameters for ReduceLROnPlateau.
        patience: Number of epochs to wait for improvement before early stopping.
        log_dir: Directory to save logs.
        checkpoint_dir: Directory to save model checkpoints.
        experiment_name: Name of the current experiment.
        accelerator: Type of accelerator to use (e.g., 'auto', 'gpu', 'cpu').
        devices: Number or list of devices to use.

    Returns:
        A dictionary containing the best validation metrics and model path.
    """
    # 1. Initialize LightningModule
    module = GenreClassifierModule(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        mixup_alpha=mixup_alpha,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
    )

    # 2. Setup Loggers (TensorBoard)
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=f"fold_{fold_idx}",
    )

    # 3. Setup Callbacks
    # Custom RichProgressBar with simplified metrics and time format
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
    from rich.progress import (
        TimeElapsedColumn,
        TimeRemainingColumn,
        TextColumn,
        BarColumn,
        Progress,
    )

    class CustomRichProgressBar(RichProgressBar):
        def get_metrics(self, trainer, pl_module):
            items = super().get_metrics(trainer, pl_module)
            # Only keep specified metrics
            allowed_metrics = {"train_loss", "train_acc", "val_loss", "val_acc"}
            return {k: v for k, v in items.items() if k in allowed_metrics}

        def configure_columns(self, trainer):
            # Custom column list to control time format (implicitly handled by Rich's default string repr for small durations)
            # or we can keep it standard but simplified
            return [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn("/"),
                TimeRemainingColumn(),
            ]

    callbacks = [
        # Custom Rich UI for the terminal progress bar
        CustomRichProgressBar(leave=False),
        # Model checkpointing
        ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, experiment_name, f"fold_{fold_idx}"),
            filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
        ),
    ]

    # 4. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )

    # 5. Execute Training
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 6. Return results and checkpoint path
    results: Dict[str, Any] = {}
    results["val_loss"] = trainer.callback_metrics["val_loss"].item()
    results["val_acc"] = trainer.callback_metrics["val_acc"].item()
    results["best_model_path"] = getattr(trainer.checkpoint_callback, "best_model_path", "")

    return results
