"""
crnn.py — Convolutional Recurrent Neural Network for music genre classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.audio import DeltaFeatures


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism for weighting sequence frames.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initializes the TemporalAttention module.

        Args:
            hidden_dim: Dimension of the input hidden states.
        """
        super().__init__()
        # Compress the hidden state into a single score per frame
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            - context_vector: Weighted sum of frames (batch, hidden_dim).
            - weights: Attention weights (batch, seq_len, 1).
        """
        # 1. Compute scores for each frame
        scores = self.attention_weights(x)  # (batch, seq_len, 1)

        # 2. Normalize to a probability distribution (sum of weights = 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # 3. Weighted sum
        # (batch, seq_len, hidden_dim) * (batch, seq_len, 1) -> (batch, hidden_dim)
        context_vector = torch.sum(x * weights, dim=1)

        return context_vector, weights


class CRNN(nn.Module):
    """
    CRNN architecture for music genre classification.
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the CRNN model."""
        super().__init__()

        # 1. CNN Part
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
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
        self.lstm1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln1 = nn.LayerNorm(128 * 2)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.3)

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features: torch.Tensor = self.features(x)

        batch_size, channels, height, width = features.shape
        x = features.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, width, height * channels)

        # RNN Part
        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Average pooling over time
        x = x.mean(dim=1)

        # Residual connection from CNN features
        features_pooled = self.cnn_feature_pool(features).squeeze(-1).squeeze(-1)
        x = x + features_pooled

        # Classifier
        x = self.classifier(x)
        return x


class CRNNAttention(nn.Module):
    """
    CRNN architecture with Temporal Attention for music genre classification.
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the CRNNAttention model."""
        super().__init__()

        # 1. CNN Part
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
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
        self.lstm1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln1 = nn.LayerNorm(128 * 2)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.3)

        # 3. Attention Part
        self.attention = TemporalAttention(hidden_dim=256)

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features: torch.Tensor = self.features(x)

        batch_size, channels, height, width = features.shape
        x = features.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, width, height * channels)

        # RNN Part
        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Use Attention instead of mean pooling
        x, _attn_weights = self.attention(x)

        # Residual connection from CNN features
        features_pooled = self.cnn_feature_pool(features).squeeze(-1).squeeze(-1)
        x = x + features_pooled

        # Classifier
        x = self.classifier(x)
        return x


class CRNN3C(nn.Module):
    """
    CRNN architecture with Mel, Delta, Delta-Delta features.
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the CRNN3C model."""
        super().__init__()
        
        self.delta_features = DeltaFeatures()

        # 1. CNN Part (3 channels input)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
        )
        self.cnn_feature_pool = nn.AdaptiveAvgPool2d((1, 1))

        cnn_out_height = input_height // 2 // 2 // 4
        rnn_input_size = 256 * cnn_out_height

        # 2. RNN Part
        self.lstm1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln1 = nn.LayerNorm(128 * 2)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.3)

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.delta_features(x)
        features: torch.Tensor = self.features(x)

        batch_size, channels, height, width = features.shape
        x = features.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, width, height * channels)

        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x = x.mean(dim=1)

        features_pooled = self.cnn_feature_pool(features).squeeze(-1).squeeze(-1)
        x = x + features_pooled

        x = self.classifier(x)
        return x


class CRNN3CAttention(nn.Module):
    """
    CRNN architecture with Attention and 3-channel input.
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the CRNN3CAttention model."""
        super().__init__()
        
        self.delta_features = DeltaFeatures()

        # 1. CNN Part (3 channels input)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
        )
        self.cnn_feature_pool = nn.AdaptiveAvgPool2d((1, 1))

        cnn_out_height = input_height // 2 // 2 // 4
        rnn_input_size = 256 * cnn_out_height

        # 2. RNN Part
        self.lstm1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln1 = nn.LayerNorm(128 * 2)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.3)

        # 3. Attention Part
        self.attention = TemporalAttention(hidden_dim=256)

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.delta_features(x)
        features: torch.Tensor = self.features(x)

        batch_size, channels, height, width = features.shape
        x = features.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, width, height * channels)

        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _attn_weights = self.attention(x)

        features_pooled = self.cnn_feature_pool(features).squeeze(-1).squeeze(-1)
        x = x + features_pooled

        x = self.classifier(x)
        return x
