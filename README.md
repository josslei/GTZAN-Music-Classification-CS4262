# Music Genre Classification using Deep Learning

A mini-research project for ML-CS4262 exploring various deep learning models for music genre classification.

## Project Structure

- `configs/`: Experiment configuration files (YAML, JSON).
- `data/`: (Renamed from `dataset/`) - Raw, processed, and external data.
- `reports/`: Analysis, figures, and final reports.
- `scripts/`: Training and evaluation entry points (e.g., `train_kfold.py`).
- `src/`: Reusable source code.
  - `data/`: Data loading and preprocessing.
  - `models/`: Model architectures (CNN, RNN, etc.).
  - `training/`: Training loops, losses, and trainers.
  - `utils/`: Helper functions.
- `tests/`: Unit tests for models and data loaders.
- `outputs/`: Saved models, checkpoints, and logs.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare the dataset: (instructions for GTZAN)
3. Train models: `python scripts/train_kfold.py`

## Authors
- Josslei
