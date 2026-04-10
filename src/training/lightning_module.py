"""
lightning_module.py — PyTorch Lightning wrapper for music genre classification.

This module provides a generic LightningModule that handles training, validation,
and optimization for any provided PyTorch model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from src.data.augment import mixup_batch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GenreClassifierModule(LightningModule):
    """Generic LightningModule for music genre classification.

    Attributes:
        model: The underlying PyTorch model (CNN, RNN, etc.).
        lr: Learning rate for the AdamW optimizer.
        weight_decay: Weight decay for the AdamW optimizer.
        criterion: Loss function (defaults to CrossEntropyLoss).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        mixup_alpha: float = 0.2,
        optimizer_params: Optional[Dict[str, Any]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the GenreClassifierModule.

        Args:
            model: PyTorch model for classification.
            lr: Initial learning rate.
            weight_decay: L2 regularization coefficient.
            mixup_alpha: Beta distribution parameter for Mixup (0 = disabled).
            optimizer_params: Parameters for AdamW (e.g., betas).
            scheduler_params: Parameters for ReduceLROnPlateau (e.g., factor, patience).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.mixup_alpha = mixup_alpha
        self.optimizer_params = optimizer_params or {}
        self.scheduler_params = scheduler_params or {}
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Handles a single training step with Mixup augmentation."""
        x, y = batch

        # Apply Mixup: blend pairs of samples and compute blended loss
        if self.mixup_alpha > 0.0:
            x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=self.mixup_alpha)
            logits = self(x_mixed)
            loss = lam * self.criterion(logits, y_a) + (1.0 - lam) * self.criterion(
                logits, y_b
            )
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        # Accuracy is computed against original (unmixed) labels for readability
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Handles a single validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Handles a single test step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        """Logs the current learning rate at the end of each training epoch."""
        sch = self.lr_schedulers()
        if sch is not None:
            # For ReduceLROnPlateau, the LR is in optimizer.param_groups
            opt = self.optimizers()
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> Any:
        """Sets up the AdamW optimizer and ReduceLROnPlateau scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.optimizer_params,
        )

        scheduler = ReduceLROnPlateau(optimizer, mode="min", **self.scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
