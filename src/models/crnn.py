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
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 2: Reduce height by 4, keep width
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
        )

        # Calculate height after CNN: 128 // 2 // 4 = 16
        cnn_out_height = input_height // 2 // 4
        rnn_input_size = 128 * cnn_out_height # 128 channels * 16 height bins = 2048

        # 2. RNN Part
        # Note: dropout in LSTM is only applied between layers, 
        # but for num_layers=1 we handle it manually or just define it.
        # PyTorch ignores dropout if num_layers=1.
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)

        # 3. Classifier
        # Bidirectional hidden_size=64 -> 128
        self.classifier = nn.Linear(64 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width).
            
        Returns:
            Logits for each class.
        """
        # CNN Part: (batch, 1, 128, width) -> (batch, 128, 16, width)
        x = self.features(x)

        # Reshape for LSTM: (batch, C, H, W) -> (batch, W, H * C)
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 2, 1).contiguous() # (batch, width, height, channels)
        x = x.view(batch_size, width, height * channels)

        # RNN Part
        # x shape: (batch, width, 128)
        _, (h_n, _) = self.lstm(x)
        
        # Concatenate final hidden states of forward and backward directions
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) # (batch, 128)
        
        # Apply dropout (as requested)
        x = self.dropout(x)

        # Classifier
        x = self.classifier(x)
        return x
