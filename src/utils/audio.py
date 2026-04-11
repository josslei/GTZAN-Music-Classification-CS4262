"""
audio.py — Audio feature extraction utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class DeltaFeatures(nn.Module):
    """
    Computes Delta and Delta-Delta coefficients for a mel-spectrogram.
    
    Input shape:  (batch, 1, n_mels, time)
    Output shape: (batch, 3, n_mels, time)
    """

    def __init__(self) -> None:
        """Initializes the DeltaFeatures module."""
        super().__init__()
        # win_length=5 is a standard default for deltas
        self.compute_deltas = T.ComputeDeltas(win_length=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input mel-spectrogram of shape (batch, 1, n_mels, time).

        Returns:
            Tensor of shape (batch, 3, n_mels, time) containing 
            [Mel, Delta, Delta-Delta].
        """
        # x is (batch, 1, n_mels, time)
        
        # 1. Delta
        delta = self.compute_deltas(x)
        
        # 2. Delta-Delta
        delta_delta = self.compute_deltas(delta)
        
        # 3. Concatenate along the channel dimension
        # (batch, 1, n_mels, time), (batch, 1, n_mels, time), (batch, 1, n_mels, time)
        # -> (batch, 3, n_mels, time)
        features = torch.cat([x, delta, delta_delta], dim=1)
        
        return features
