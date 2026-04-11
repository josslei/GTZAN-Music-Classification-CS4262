"""
mel_dataset.py — PyTorch Dataset and DataLoader for Mel-spectrograms.

This module provides the MelSpectrogramDataset class to load .npy files and
the get_dataloaders function to create training, validation, and testing splits
based on a fixed test set (fold 0) and 5-fold cross-validation (folds 1-5).
"""

from __future__ import annotations

import csv

from src.data.augment import MelAugment
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MelSpectrogramDataset(Dataset):
    """PyTorch Dataset for loading Mel-spectrogram .npy files.

    Attributes:
        data_list: List of dictionaries with 'filename' and 'genre' keys.
        mel_dir: Root directory where .npy files are stored.
        class_mapping: Mapping from genre names (str) to integer labels (int).
        transform: Optional callable for data augmentation or normalization.
        max_frames: Fixed number of time frames to crop or pad to.
    """

    def __init__(
        self,
        data_list: List[Dict[str, str]],
        mel_dir: Union[str, Path],
        class_mapping: Dict[str, int],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """Initializes the MelSpectrogramDataset.

        Args:
            data_list: List of dictionaries containing 'filename' and 'genre'.
            mel_dir: Path to the directory with .npy files.
            class_mapping: Dictionary mapping string genre labels to integer IDs.
            transform: Optional transform to be applied on a sample.
            max_frames: Fixed frame length to crop/pad to if variable lengths exist.
        """
        self.data_list = data_list
        self.mel_dir = Path(mel_dir)
        self.class_mapping = class_mapping
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and processes a single sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple (mel_tensor, label) where mel_tensor is a torch.Tensor
            of shape (1, n_mels, max_frames) and label is a long torch.Tensor.
        """
        item = self.data_list[idx]
        filename = item["filename"]
        genre_str = item["genre"]

        # 1. Get the integer label
        label_int = self.class_mapping[genre_str]
        label = torch.tensor(label_int, dtype=torch.long)

        # 2. Load the .npy file
        file_path = self.mel_dir / filename
        mel_spec: np.ndarray = np.load(file_path)

        # 3. Convert to torch tensor (n_mels, time_frames)
        mel_tensor = torch.from_numpy(mel_spec).float()

        # 4. Handle variable lengths (Pad or Crop)
        if self.max_frames is not None:
            _, time_frames = mel_tensor.shape
            if time_frames > self.max_frames:
                # Random crop for training if needed, or simple slice
                start = torch.randint(0, time_frames - self.max_frames + 1, (1,)).item()
                mel_tensor = mel_tensor[:, start : start + self.max_frames]
            elif time_frames < self.max_frames:
                # Zero pad at the end
                pad_amount = self.max_frames - time_frames
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_amount))

        # 5. Add channel dimension -> (1, n_mels, time_frames)
        mel_tensor = mel_tensor.unsqueeze(0)

        # 6. Apply additional transforms
        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, label


def get_dataloaders(
    metadata_path: Union[str, Path],
    mel_dir: Union[str, Path],
    val_fold: int,
    batch_size: int = 32,
    num_workers: int = 4,
    segment_seconds: Optional[float] = None,
    sample_rate: int = 22050,
    hop_length: int = 512,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Reads metadata and returns PyTorch DataLoaders for train, val, and test.

    Split logic:
    - Test Set: Always Fold 0 (fixed 10%).
    - Validation Set: The specified `val_fold` (1 to 5).
    - Training Set: All remaining folds from {1, 2, 3, 4, 5} excluding `val_fold`.

    Args:
        metadata_path: Path to the metadata.csv file.
        mel_dir: Path to the directory containing .npy files.
        val_fold: The fold number (1 to 5) to use for validation.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses to use for data loading.
        segment_seconds: If provided, slices mel-spectrograms to this duration.
        sample_rate: Audio sample rate (needed to calculate max_frames).
        hop_length: FFT hop length (needed to calculate max_frames).

    Returns:
        A tuple (train_loader, val_loader, test_loader, class_mapping).
    """
    train_data: List[Dict[str, str]] = []
    val_data: List[Dict[str, str]] = []
    test_data: List[Dict[str, str]] = []
    unique_genres: set[str] = set()

    # Calculate max_frames dynamically if segment_seconds is provided
    max_frames: Optional[int] = None
    if segment_seconds is not None:
        max_frames = math.ceil((segment_seconds * sample_rate) / hop_length)

    # Define test fold
    TEST_FOLD = 0

    with open(metadata_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            genre = row["genre"]
            fold = int(row["fold"])

            unique_genres.add(genre)
            item_data = {"filename": filename, "genre": genre}

            if fold == TEST_FOLD:
                test_data.append(item_data)
            elif fold == val_fold:
                val_data.append(item_data)
            else:
                train_data.append(item_data)

    # Create class mapping alphabetically
    genres = sorted(list(unique_genres))
    class_mapping = {genre: idx for idx, genre in enumerate(genres)}

    # Initialize datasets (augmentation applied to training set only)
    train_dataset = MelSpectrogramDataset(
        train_data,
        mel_dir,
        class_mapping,
        max_frames=max_frames,
        transform=MelAugment(),
    )
    val_dataset = MelSpectrogramDataset(
        val_data, mel_dir, class_mapping, max_frames=max_frames
    )
    test_dataset = MelSpectrogramDataset(
        test_data, mel_dir, class_mapping, max_frames=max_frames
    )

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_mapping
