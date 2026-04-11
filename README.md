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

## Zero-Shot Classification (LAION CLAP)

You can evaluate the zero-shot classification performance of the pretrained LAION CLAP model on the GTZAN dataset.

1. **Prepare the CLAP Dataset (48kHz Raw Audio):**
   ```bash
   python scripts/prepare_mel.py --clap-mode
   ```
   This will process the raw audio files into 1D `.npy` arrays at 48000 Hz and save them to `dataset/mel_clap/`.

2. **Run Zero-Shot Evaluation:**
   ```bash
   python scripts/clap_zeroshot.py --exp template_0
   ```
   - `--exp`: The name of the experiment configuration file (e.g., `template_0`, `template_1`) located in `configs/clap_zeroshot/`.
   - `--template`: Override the prompt template string directly (e.g. `"This is a {genre} song."`).
   - `--batch-size`: Override the number of samples to process at once.
   - `--max-duration`: Override the maximum audio duration (in seconds) to feed into the model to avoid Out-Of-Memory errors.

## Authors
- Josslei
