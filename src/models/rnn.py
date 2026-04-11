"""
rnn.py — RNN and LSTM models for music genre classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.crnn import TemporalAttention


class RNN(nn.Module):
    """
    RNN architecture for music genre classification.
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the RNN model.

        Args:
            num_classes: Number of target genres.
            input_height: The number of mel bands (height of the spectrogram).
        """
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_height,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
            nonlinearity="tanh",
        )
        self.projection = nn.Linear(input_height, 256 * 2)

        self.ln1 = nn.LayerNorm(input_height)
        self.ln2 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.squeeze(1).transpose(1, 2)

        residual = self.projection(x)
        x = self.ln1(x)
        x, _ = self.rnn(x)
        x = x + residual

        # Global average pooling over time
        x = x.mean(dim=1)

        x = self.ln2(x)
        x = self.dropout(x)

        x = self.classifier(x)
        return x


class RNNAttention(nn.Module):
    """
    RNN architecture with Temporal Attention for music genre classification.
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the RNNAttention model."""
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_height,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
            nonlinearity="tanh",
        )
        self.projection = nn.Linear(input_height, 256 * 2)

        # Attention layer for 512-dimensional output (256 hidden * 2 directions)
        self.attention = TemporalAttention(hidden_dim=512)

        self.ln1 = nn.LayerNorm(input_height)
        self.ln2 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.squeeze(1).transpose(1, 2)

        residual = self.projection(x)
        x = self.ln1(x)
        x, _ = self.rnn(x)
        x = x + residual

        # Use Attention instead of mean pooling
        x, _ = self.attention(x)

        x = self.ln2(x)
        x = self.dropout(x)

        x = self.classifier(x)
        return x


class LSTM(nn.Module):
    """
    LSTM architecture for music genre classification (~1.7M parameters).
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the LSTM model."""
        super().__init__()

        self.ln1 = nn.LayerNorm(input_height)
        self.lstm1 = nn.LSTM(
            input_size=input_height,
            hidden_size=128,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(0.3)

        self.ln2 = nn.LayerNorm(256)
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.projection2 = nn.Linear(256, 256 * 2)

        self.ln3 = nn.LayerNorm(512)
        self.lstm3 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )
        self.dropout3 = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.squeeze(1).transpose(1, 2)

        x = self.ln1(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        residual = self.projection2(x)
        x = self.ln2(x)
        x, _ = self.lstm2(x)
        x = x + residual
        x = self.dropout2(x)

        residual = x
        x = self.ln3(x)
        x, _ = self.lstm3(x)
        x = x + residual

        # Global average pooling over time
        x = x.mean(dim=1)

        x = self.dropout3(x)

        x = self.classifier(x)
        return x


class LSTMAttention(nn.Module):
    """
    LSTM architecture with Temporal Attention for music genre classification.
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the LSTMAttention model."""
        super().__init__()

        self.ln1 = nn.LayerNorm(input_height)
        self.lstm1 = nn.LSTM(
            input_size=input_height,
            hidden_size=128,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(0.3)

        self.ln2 = nn.LayerNorm(256)
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.projection2 = nn.Linear(256, 256 * 2)

        self.ln3 = nn.LayerNorm(512)
        self.lstm3 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )
        self.dropout3 = nn.Dropout(0.3)

        # Attention layer for 512-dimensional output
        self.attention = TemporalAttention(hidden_dim=512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.squeeze(1).transpose(1, 2)

        x = self.ln1(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        residual = self.projection2(x)
        x = self.ln2(x)
        x, _ = self.lstm2(x)
        x = x + residual
        x = self.dropout2(x)

        residual = x
        x = self.ln3(x)
        x, _ = self.lstm3(x)
        x = x + residual

        #  Use Attention instead of mean pooling
        x, _ = self.attention(x)

        x = self.dropout3(x)

        x = self.classifier(x)
        return x
