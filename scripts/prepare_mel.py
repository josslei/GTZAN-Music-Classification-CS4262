"""
prepare_mel.py — Convert WAV audio files to mel-spectrograms (.npy) and
generate a metadata.csv with genre labels and cross-validation fold numbers.

=== Input & Output ===
Input:  dataset/raw/Data/genres_original/<genre>/<genre>.NNNNN.wav
Output: dataset/mel/mel/<genre>.NNNNN.npy          (mel-spectrogram arrays)
        dataset/mel/metadata.csv                    (filename, genre, label, fold)

- Genre labels are integer-encoded in **lexicographical** order of genre names.
- metadata.csv entries are sorted **alphabetically** by filename.
- Fold numbers are assigned as follows:
    - Fold 0: Fixed Test Set (10% of total data).
    - Folds 1-5: Cross-validation folds (remaining 90% split into 5).
- All splits use StratifiedGroupKFold (scikit-learn) to ensure:
    1. Leakage prevention — all chunks from the same source stay in the same fold.
    2. Stratification  — class balance is preserved across folds.
    3. Mutual exclusion — every file belongs to exactly one fold.
    4. Determinism      — results are reproducible via a fixed random seed.

If metadata.csv already exists it will **not** be regenerated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.append(root_dir)

import csv
from pathlib import Path
from typing import Final

import librosa
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================================
# Random seed (determinism)
# ============================================================================
RANDOM_SEED: Final[int] = 42

# ============================================================================
# Hyperparameters — mel-spectrogram extraction
# ============================================================================
SAMPLE_RATE: Final[int] = 22050  # Target sample rate for loading audio
N_FFT: Final[int] = 2048  # FFT window size
HOP_LENGTH: Final[int] = 512  # Hop length between frames
N_MELS: Final[int] = 128  # Number of mel filter-bank bands
F_MIN: Final[float] = 0.0  # Minimum frequency for mel filter-bank
F_MAX: Final[float | None] = None  # Maximum frequency (None → SR / 2)
POWER: Final[float] = 2.0  # Exponent for magnitude spectrogram (2.0 = power)
TOP_DB: Final[float] = 80.0  # Threshold for log-scaling (dB below peak)

# ============================================================================
# Hyperparameters — cross-validation
# ============================================================================
N_CV_FOLDS: Final[int] = 5  # Number of CV folds (excluding test set)
TEST_SIZE: Final[float] = 0.1  # 10% fixed test set

# ============================================================================
# Paths
# ============================================================================
RAW_DIR: Final[Path] = Path("dataset/raw/Data/genres_original")
MEL_ROOT: Final[Path] = Path("dataset/mel")
MEL_OUT_DIR: Final[Path] = MEL_ROOT
METADATA_PATH: Final[Path] = MEL_ROOT / "metadata.csv"


# ============================================================================
# Core functions
# ============================================================================


def wav_to_mel(wav_path: Path) -> NDArray[np.float32]:
    """Loads a WAV file and returns its log-scaled mel-spectrogram.

    Args:
        wav_path: Path to the source .wav file.

    Returns:
        A 2-D float32 array of shape (N_MELS, time_frames).
    """
    y: NDArray[np.float32]
    try:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        raise e

    mel_spec: NDArray[np.float64] = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=POWER,
    )

    log_mel: NDArray[np.float32] = librosa.power_to_db(
        mel_spec, ref=np.max, top_db=TOP_DB
    ).astype(np.float32)

    return log_mel


def discover_wav_files(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """Walks the genre directories and collects (npy_filename, genre, wav_path).

    Args:
        raw_dir: Path to the root directory containing genre subfolders.

    Returns:
        A list of tuples (npy_filename, genre, wav_path) sorted alphabetically
        by npy_filename.
    """
    records: list[tuple[str, str, Path]] = []

    if not raw_dir.exists():
        return records

    for genre_dir in sorted(raw_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre_name: str = genre_dir.name
        for wav_file in sorted(genre_dir.glob("*.wav")):
            npy_name: str = wav_file.stem + ".npy"
            records.append((npy_name, genre_name, wav_file))

    records.sort(key=lambda r: r[0])
    return records


def build_label_map(genres: list[str]) -> dict[str, int]:
    """Maps genre names to integer labels in lexicographical order.

    Args:
        genres: List of genre names.

    Returns:
        A dictionary mapping genre name to integer label.
    """
    unique_genres: list[str] = sorted(set(genres))
    return {g: i for i, g in enumerate(unique_genres)}


def assign_folds_with_test(
    labels: list[int],
    groups: list[str],
) -> list[int]:
    """Assigns fold 0 for test and folds 1-5 for cross-validation.

    Args:
        labels: Integer-encoded label per sample.
        groups: Group identifier per sample.

    Returns:
        A list of fold numbers for each sample.
    """
    y = np.array(labels)
    g = np.array(groups)
    X = np.zeros((len(labels), 1))

    fold_assignments = np.zeros(len(labels), dtype=int)

    # 1. Split 10% as test (Fold 0)
    # Using StratifiedGroupKFold with 10 splits and taking 1 as test set.
    sgkf_test = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=RANDOM_SEED
    )
    train_idx, test_idx = next(sgkf_test.split(X, y, groups=g))

    # fold_assignments is already initialized to 0, so samples in test_idx are fold 0.
    # We only need to assign folds 1-5 to the train_idx samples.

    y_train = y[train_idx]
    g_train = g[train_idx]
    X_train = X[train_idx]

    # 2. Split remaining 90% into 5 folds (Folds 1-5)
    sgkf_cv = StratifiedGroupKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    for i, (_, val_idx_in_train) in enumerate(
        sgkf_cv.split(X_train, y_train, groups=g_train), start=1
    ):
        # Map back to original indices
        original_val_idx = train_idx[val_idx_in_train]
        fold_assignments[original_val_idx] = i

    return fold_assignments.tolist()


def write_metadata(
    records: list[tuple[str, str]],
    labels: list[int],
    folds: list[int],
    path: Path,
) -> None:
    """Writes metadata.csv with columns: filename, genre, label, fold.

    Args:
        records: List of (filename, genre) tuples.
        labels: List of integer labels corresponding to records.
        folds: List of fold assignments corresponding to records.
        path: Path where the CSV file will be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "genre", "label", "fold"])
        i: int
        for i, (npy_name, genre) in enumerate(records):
            writer.writerow([npy_name, genre, labels[i], folds[i]])


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main execution flow for audio processing and metadata generation."""
    wav_records: list[tuple[str, str, Path]] = discover_wav_files(RAW_DIR)
    if not wav_records:
        print(f"Error: No .wav files found under {RAW_DIR}.")
        return

    npy_names: list[str] = [r[0] for r in wav_records]
    genres: list[str] = [r[1] for r in wav_records]
    wav_paths: list[Path] = [r[2] for r in wav_records]

    print(f"Found {len(wav_records)} WAV files across {len(set(genres))} genres.")

    label_map = build_label_map(genres)
    print(f"Label mapping: {label_map}")
    labels: list[int] = [label_map[g] for g in genres]

    MEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    total: int = len(wav_paths)
    for i, wav_path in enumerate(wav_paths):
        out_path: Path = MEL_OUT_DIR / npy_names[i]
        if out_path.exists():
            continue
        try:
            mel: NDArray[np.float32] = wav_to_mel(wav_path)
            np.save(out_path, mel)
        except Exception:
            print(f"  Skipping corrupted file: {wav_path.name}")
            continue

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i + 1:>{len(str(total))}}/{total}] Saved {npy_names[i]}")

    print("Mel-spectrogram conversion complete.")

    if METADATA_PATH.exists():
        print(f"metadata.csv already exists at {METADATA_PATH} — skipping.")
    else:
        existing_records: list[tuple[str, str]] = []
        existing_labels: list[int] = []

        for name, genre, label in zip(npy_names, genres, labels):
            if (MEL_OUT_DIR / name).exists():
                existing_records.append((name, genre))
                existing_labels.append(label)

        groups: list[str] = [Path(n).stem for n, _ in existing_records]
        folds: list[int] = assign_folds_with_test(existing_labels, groups)

        write_metadata(existing_records, existing_labels, folds, METADATA_PATH)
        print(f"Wrote {METADATA_PATH} with {len(existing_records)} entries.")
        print("Fold 0: Test set (10%), Folds 1-5: CV sets (90%).")


if __name__ == "__main__":
    main()
