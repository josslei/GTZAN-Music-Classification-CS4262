"""
rnn.py — RNN and LSTM models for music genre classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRNNModel(nn.Module):
    """
    Simple RNN architecture for music genre classification.
    
    Structure:
    - Reshape: (batch, 1, height, width) -> (batch, width, height)
    - SimpleRNN(64, return_sequences=True)
    - LayerNorm -> Dropout(0.3)
    - SimpleRNN(64, return_sequences=False)
    - LayerNorm -> Dropout(0.3)
    - Dense(32, ReLU)
    - Dense(num_classes)
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the SimpleRNNModel.

        Args:
            num_classes: Number of target genres.
            input_height: The number of mel bands (height of the spectrogram).
        """
        super().__init__()
        
        # First RNN layer
        self.rnn1 = nn.RNN(
            input_size=input_height, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            nonlinearity='tanh'
        )
        self.ln1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second RNN layer
        self.rnn2 = nn.RNN(
            input_size=64, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True,
            nonlinearity='tanh'
        )
        self.ln2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Reshape: (batch, 1, height, width) -> (batch, width, height)
        x = x.squeeze(1).transpose(1, 2)

        # First RNN layer (return sequences = True)
        x, _ = self.rnn1(x)
        x = self.ln1(x)
        x = self.dropout1(x)

        # Second RNN layer (return sequences = False)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]           # (batch, 64)
        x = self.ln2(x)
        x = self.dropout2(x)

        x = self.fc(x)
        return x


class LSTMModel(nn.Module):
    """
    LSTM architecture for music genre classification (~1.7M parameters).
    
    Structure:
    - Reshape: (batch, 1, height, width) -> (batch, width, height)
    - LSTM(256, return_sequences=True) -> Dropout(0.3)
    - LayerNorm
    - LSTM(256, return_sequences=True) -> Dropout(0.3)
    - LayerNorm
    - LSTM(256, return_sequences=False) -> Dropout(0.3)
    - Dense(128, ReLU) -> Dropout(0.4)
    - Dense(num_classes)
    """

    def __init__(self, num_classes: int = 10, input_height: int = 128) -> None:
        """Initializes the LSTMModel.

        Args:
            num_classes: Number of target genres.
            input_height: The number of mel bands (height of the spectrogram).
        """
        super().__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_height,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.3)
        self.ln1 = nn.LayerNorm(256)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.3)
        self.ln2 = nn.LayerNorm(256)

        # Third LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.dropout3 = nn.Dropout(0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Reshape: (batch, 1, height, width) -> (batch, width, height)
        x = x.squeeze(1).transpose(1, 2)

        # First LSTM layer (return sequences = True)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = self.ln1(x)

        # Second LSTM layer (return sequences = True)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.ln2(x)

        # Third LSTM layer (return sequences = False)
        x, (h_n, _) = self.lstm3(x)
        # Using the last hidden state of the top layer
        x = h_n[-1]
        x = self.dropout3(x)

        # Classifier
        x = self.classifier(x)
        return x
