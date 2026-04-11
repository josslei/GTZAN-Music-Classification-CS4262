"""
prepare_mel.py — Convert WAV audio files to mel-spectrograms (.npy) and
generate a metadata.csv with genre labels and cross-validation fold numbers.

=== Input & Output ===
Input:  dataset/raw/Data/genres_original/<genre>/<genre>.NNNNN.wav
Output: dataset/mel/<genre>.NNNNN.npy          (mel-spectrogram arrays)
        dataset/mel/metadata.csv               (filename, genre, label, fold)

If --clap-mode is used:
Output: dataset/mel_clap/<genre>.NNNNN.npy     (1D raw audio arrays at 48000 Hz)
        dataset/mel_clap/metadata.csv

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
import argparse
import csv
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.append(root_dir)

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
# Hyperparameters — mel-spectrogram extraction (Default)
# ============================================================================
DEFAULT_SAMPLE_RATE: Final[int] = 22050
N_FFT: Final[int] = 2048
HOP_LENGTH: Final[int] = 512
N_MELS: Final[int] = 128
F_MIN: Final[float] = 0.0
F_MAX: Final[float | None] = 8000.0
POWER: Final[float] = 2.0
TOP_DB: Final[float] = 80.0

# ============================================================================
# Hyperparameters — cross-validation
# ============================================================================
N_CV_FOLDS: Final[int] = 5
TEST_SIZE: Final[float] = 0.1

# ============================================================================
# Paths
# ============================================================================
_RAW_DIR_FILTERED = Path("dataset/gtzan-fault-filtered/raw/Data/genres_original")
RAW_DIR: Final[Path] = _RAW_DIR_FILTERED if _RAW_DIR_FILTERED.exists() else Path("dataset/raw/Data/genres_original")


# ============================================================================
# Core functions
# ============================================================================


def process_audio(wav_path: Path, sample_rate: int, clap_mode: bool) -> NDArray[np.float32]:
    """Loads a WAV file and returns its processed representation.
    
    If clap_mode is False, returns log-scaled mel-spectrogram (N_MELS, time_frames).
    If clap_mode is True, returns 1D raw audio array (time_frames,) at target sample rate.
    """
    y: NDArray[np.float32]
    try:
        y, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        raise e

    if clap_mode:
        return y.astype(np.float32)

    mel_spec: NDArray[np.float64] = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
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
    unique_genres: list[str] = sorted(set(genres))
    return {g: i for i, g in enumerate(unique_genres)}


def assign_folds_with_test(labels: list[int], groups: list[str]) -> list[int]:
    y = np.array(labels)
    g = np.array(groups)
    X = np.zeros((len(labels), 1))

    fold_assignments = np.zeros(len(labels), dtype=int)

    sgkf_test = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sgkf_test.split(X, y, groups=g))

    y_train = y[train_idx]
    g_train = g[train_idx]
    X_train = X[train_idx]

    sgkf_cv = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for i, (_, val_idx_in_train) in enumerate(sgkf_cv.split(X_train, y_train, groups=g_train), start=1):
        original_val_idx = train_idx[val_idx_in_train]
        fold_assignments[original_val_idx] = i

    return fold_assignments.tolist()


def write_metadata(records: list[tuple[str, str]], labels: list[int], folds: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "genre", "label", "fold"])
        for i, (npy_name, genre) in enumerate(records):
            writer.writerow([npy_name, genre, labels[i], folds[i]])


# ============================================================================
# Main
# ============================================================================


def main(args: argparse.Namespace) -> None:
    clap_mode = args.clap_mode
    sample_rate = 48000 if clap_mode else DEFAULT_SAMPLE_RATE
    out_dir_name = "mel_clap" if clap_mode else "mel"
    
    mel_root = Path(f"dataset/{out_dir_name}")
    mel_out_dir = mel_root
    metadata_path = mel_root / "metadata.csv"

    print(f"Mode: {'CLAP (Raw Audio @ 48000Hz)' if clap_mode else 'Mel-Spectrogram (@ 22050Hz)'}")
    print(f"Output Directory: {mel_out_dir}")

    wav_records = discover_wav_files(RAW_DIR)
    if not wav_records:
        print(f"Error: No .wav files found under {RAW_DIR}.")
        return

    npy_names = [r[0] for r in wav_records]
    genres = [r[1] for r in wav_records]
    wav_paths = [r[2] for r in wav_records]

    print(f"Found {len(wav_records)} WAV files across {len(set(genres))} genres.")

    label_map = build_label_map(genres)
    labels = [label_map[g] for g in genres]

    mel_out_dir.mkdir(parents=True, exist_ok=True)
    total = len(wav_paths)
    for i, wav_path in enumerate(wav_paths):
        out_path = mel_out_dir / npy_names[i]
        if out_path.exists():
            continue
        try:
            processed = process_audio(wav_path, sample_rate, clap_mode)
            np.save(out_path, processed)
        except Exception:
            print(f"  Skipping corrupted file: {wav_path.name}")
            continue

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i + 1:>{len(str(total))}}/{total}] Saved {npy_names[i]}")

    print("Data processing complete.")

    if metadata_path.exists():
        print(f"metadata.csv already exists at {metadata_path} — skipping.")
    else:
        existing_records = []
        existing_labels = []

        for name, genre, label in zip(npy_names, genres, labels):
            if (mel_out_dir / name).exists():
                existing_records.append((name, genre))
                existing_labels.append(label)

        groups = [Path(n).stem for n, _ in existing_records]
        folds = assign_folds_with_test(existing_labels, groups)

        write_metadata(existing_records, existing_labels, folds, metadata_path)
        print(f"Wrote {metadata_path} with {len(existing_records)} entries.")
        print("Fold 0: Test set (10%), Folds 1-5: CV sets (90%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WAV audio to model inputs.")
    parser.add_argument("--clap-mode", action="store_true", help="Extract raw 1D audio arrays at 48000Hz for CLAP instead of mel-spectrograms.")
    args = parser.parse_args()
    main(args)
