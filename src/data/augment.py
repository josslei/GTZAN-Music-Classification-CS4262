"""
augment.py — Mel-spectrogram data augmentations for training.

Provides per-sample augmentations (SpecAugment, RandomGain, GaussianNoise,
TimeShift, PitchShift, RandomErasing) and a batch-level Mixup function.

Per-sample augmentations are composed in MelAugment and passed as `transform`
to the training Dataset.  Mixup is called inside the LightningModule's
training_step because it operates on full batches.
"""

from __future__ import annotations

from typing import Tuple

import torch


# ============================================================================
# Per-sample augmentations (plugged in via Dataset.transform)
# ============================================================================


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
            f = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if f == 0 or n_mels - f <= 0:
                continue
            f0 = int(torch.randint(0, int(n_mels - f), (1,)).item())
            mel[:, f0 : f0 + f, :] = 0.0

        # Time masking — zeroes out vertical bands
        for _ in range(self.num_time_masks):
            t = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            if t == 0 or n_frames - t <= 0:
                continue
            t0 = int(torch.randint(0, int(n_frames - t), (1,)).item())
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
        gain = torch.empty(1).uniform_(self.min_gain, self.max_gain).item()
        return mel * gain


class GaussianNoise:
    """Adds random Gaussian noise to a mel-spectrogram.

    Attributes:
        std: Standard deviation of the noise.
    """

    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(mel) * self.std
        return mel + noise


class TimeShift:
    """Cyclically shifts the mel-spectrogram along the time axis.

    Attributes:
        max_shift_fraction: Maximum fraction of total frames to shift.
    """

    def __init__(self, max_shift_fraction: float = 0.1) -> None:
        self.max_shift_fraction = max_shift_fraction

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        _, _, n_frames = mel.shape
        max_shift = int(n_frames * self.max_shift_fraction)
        if max_shift == 0:
            return mel
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        return torch.roll(mel, shifts=shift, dims=2)


class PitchShift:
    """Cyclically shifts the mel-spectrogram along the frequency axis.

    Simulates small pitch variations.

    Attributes:
        max_shift: Maximum number of mel bins to shift up or down.
    """

    def __init__(self, max_shift: int = 4) -> None:
        self.max_shift = max_shift

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        return torch.roll(mel, shifts=shift, dims=1)


class RandomErasing:
    """Replaces a random rectangular patch with random values.

    Unlike SpecAugment (which zeroes out), this fills with random noise,
    preventing the model from using zero-regions as a signal.

    Attributes:
        max_freq: Maximum height of the erased patch (mel bins).
        max_time: Maximum width of the erased patch (frames).
    """

    def __init__(self, max_freq: int = 15, max_time: int = 25) -> None:
        self.max_freq = max_freq
        self.max_time = max_time

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        _, n_mels, n_frames = mel.shape
        f = int(torch.randint(1, self.max_freq + 1, (1,)).item())
        t = int(torch.randint(1, self.max_time + 1, (1,)).item())
        if n_mels - f <= 0 or n_frames - t <= 0:
            return mel
        f0 = int(torch.randint(0, int(n_mels - f), (1,)).item())
        t0 = int(torch.randint(0, int(n_frames - t), (1,)).item())
        mel[:, f0 : f0 + f, t0 : t0 + t] = torch.randn(1, f, t)
        return mel


# ============================================================================
# Composed per-sample pipeline
# ============================================================================


class MelAugment:
    """Composes proven per-sample augmentations into a single callable.

    Applied in this order:
        1. SpecAugment (frequency + time masking) — 1 mask each
        2. Random Gain

    Note: GaussianNoise, TimeShift, PitchShift, and RandomErasing are
    available as individual classes above but excluded from the default
    pipeline to avoid underfitting on small datasets.
    """

    def __init__(
        self,
        # SpecAugment
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        # Random Gain
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


# ============================================================================
# Batch-level augmentation (called inside training_step)
# ============================================================================


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies Mixup augmentation at the batch level.

    Blends pairs of samples and their labels so the model learns smoother
    decision boundaries.

    Args:
        x: Input batch of shape (B, 1, n_mels, time_frames).
        y: Label batch of shape (B,) with integer class indices.
        alpha: Beta distribution parameter (higher = more mixing).

    Returns:
        A tuple (x_mixed, y_a, y_b, lam) where:
            x_mixed: blended input batch,
            y_a: original labels,
            y_b: permuted labels,
            lam: mixing coefficient (used to blend the loss).
    """
    lam: float = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_a = y
    y_b = y[perm]

    return x_mixed, y_a, y_b, lam
