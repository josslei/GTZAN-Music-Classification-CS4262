"""
augment.py — Mel-spectrogram data augmentations for training.

Provides SpecAugment (frequency + time masking) and random gain perturbation.
These augmentations are applied only to training data, never to val/test.
"""

from __future__ import annotations

import torch


class SpecAugment:
    """Applies frequency and time masking (SpecAugment) to a mel-spectrogram.

    Reference: Park et al., "SpecAugment", 2019.

    Attributes:
        freq_mask_param: Maximum width of a frequency mask (in mel bins).
        time_mask_param: Maximum width of a time mask (in frames).
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Applies masking to a mel-spectrogram tensor.

        Args:
            mel: Tensor of shape (1, n_mels, time_frames).

        Returns:
            Augmented tensor with the same shape.
        """
        _, n_mels, n_frames = mel.shape

        # Frequency masking — zeroes out horizontal bands
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            if f == 0 or n_mels - f <= 0:
                continue
            f0 = torch.randint(0, n_mels - f, (1,)).item()
            mel[:, f0 : f0 + f, :] = 0.0

        # Time masking — zeroes out vertical bands
        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            if t == 0 or n_frames - t <= 0:
                continue
            t0 = torch.randint(0, n_frames - t, (1,)).item()
            mel[:, :, t0 : t0 + t] = 0.0

        return mel


class RandomGain:
    """Randomly scales the amplitude of a mel-spectrogram.

    Simulates volume variation between recordings.

    Attributes:
        min_gain: Minimum gain multiplier.
        max_gain: Maximum gain multiplier.
    """

    def __init__(self, min_gain: float = 0.8, max_gain: float = 1.2) -> None:
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Applies random gain to a mel-spectrogram tensor.

        Args:
            mel: Tensor of shape (1, n_mels, time_frames).

        Returns:
            Scaled tensor with the same shape.
        """
        gain = torch.empty(1).uniform_(self.min_gain, self.max_gain).item()
        return mel * gain


class MelAugment:
    """Composes SpecAugment and RandomGain into a single callable.

    This is the recommended default augmentation pipeline for training.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        min_gain: float = 0.8,
        max_gain: float = 1.2,
    ) -> None:
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_freq_masks=num_freq_masks,
            num_time_masks=num_time_masks,
        )
        self.random_gain = RandomGain(min_gain=min_gain, max_gain=max_gain)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Applies SpecAugment then RandomGain.

        Args:
            mel: Tensor of shape (1, n_mels, time_frames).

        Returns:
            Augmented tensor with the same shape.
        """
        mel = self.spec_augment(mel)
        mel = self.random_gain(mel)
        return mel
