"""
crnn.py — Convolutional Recurrent Neural Network for music genre classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNNModel(nn.Module):
    """
    CRNN architecture for music genre classification.

    Structure:
    - CNN Part:
        - Conv(64, 3x3) -> BN -> ELU -> MaxPool(2, 1)
        - Conv(128, 3x3) -> BN -> ELU -> MaxPool(4, 1)
    - Reshape:
        - (Batch, 128, 16, Width) -> (Batch, Width, 128 * 16)
    - RNN Part:
        - Bidirectional LSTM(64, return_sequences=False, dropout=0.3)
    - Classifier:
        - Dense(10)
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the CRNNModel.

        Args:
            num_classes: Number of target genres.
            input_height: The number of mel bands (N_MELS).
        """
        super().__init__()

        # 1. CNN Part
        self.features = nn.Sequential(
            # Block 1: Reduce height by 2, keep width
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 2: Reduce height by 2, keep width
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 3: Reduce height by 4, keep width
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
        )
        self.cnn_feature_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate height after CNN
        cnn_out_height = input_height // 2 // 2 // 4
        rnn_input_size = 256 * cnn_out_height

        # 2. RNN Part
        # Note: dropout in LSTM is only applied between layers,
        # but for num_layers=1 we handle it manually or just define it.
        # PyTorch ignores dropout if num_layers=1.
        self.lstm1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.ln1 = nn.LayerNorm(128 * 2)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, height, width).

        Returns:
            Logits for each class.
        """
        # CNN Part: (batch, 1, 128, width) -> (batch, 128, 16, width)
        features: torch.Tensor = self.features(x)

        # Reshape for LSTM: (batch, C, H, W) -> (batch, W, H * C)
        batch_size, channels, height, width = features.shape
        x = features.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C)
        x = x.view(batch_size, width, height * channels)

        # RNN Part
        # x shape: (batch, width, 128)
        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = x.mean(dim=1)

        # Residual connection
        features = self.cnn_feature_pool(features).squeeze()
        x = x + features

        # Classifier
        x = self.classifier(x)
        return x
