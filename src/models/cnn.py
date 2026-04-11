"""
cnn.py — Convolutional Neural Network models for music genre classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.utils.audio import DeltaFeatures


class CNN2D(nn.Module):
    """
    CNN2D architecture for music genre classification.

    Structure:
    - Block 1: Conv(3x3, 64) -> BN -> ReLU -> MaxPool(2x2)
    - Block 2: Conv(3x3, 128) -> BN -> ReLU -> MaxPool(2x2)
    - Block 3: Conv(3x3, 256) -> BN -> ReLU -> MaxPool(2x2)
    - Block 4: Conv(3x3, 512) -> BN -> ReLU -> MaxPool(2x2)
    - Global Average Pooling
    - Dense(512) -> Dropout(0.3)
    - Dense(num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initializes the CNN2D model.

        Args:
            num_classes: Number of target genres.
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            # Note: Softmax is typically handled by nn.CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Logits for each class.
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNN2D3C(nn.Module):
    """
    CNN2D architecture with Delta and Delta-Delta channels.
    Same as CNN2D, but with 3 input channels from mel, delta, delta-delta.
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initializes the CNN2D3C model."""
        super().__init__()
        
        # 1. Delta Features module
        self.delta_features = DeltaFeatures()

        # 2. CNN Features (now starting with 3 channels)
        self.features = nn.Sequential(
            # Block 1: 3 channels input
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input mel tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Logits for each class.
        """
        # 1. Compute delta and delta-delta features
        # (batch, 1, n_mels, time) -> (batch, 3, n_mels, time)
        x = self.delta_features(x)
        
        # 2. Standard CNN2D pass
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
