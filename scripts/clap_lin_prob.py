"""
clap_lin_prob.py — Linear probing using pretrained embeddings (e.g. CLAP, MERT).

Extracts features from pretrained models and classifies them using SVM, KNN, or MLP.
"""

import os
import sys
import argparse
import csv
import shutil
import yaml
import random
import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from typing import Any, Dict, List
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModel, AutoProcessor

# Add project root to sys.path
root_dir = str(Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.append(root_dir)


def set_seed(seed: int = 42):
    """Sets the random seed for everything to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


MODEL_MAPPING = {
    "mert": "m-a-p/MERT-v1-330M",
    "larger": "laion/larger_clap_music",
    "lukewys": "lukewys/laion_clap",
    "microsoft": "microsoft/clap-2023",
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MLP(nn.Module):
    """Shallow MLP for linear probing classification."""

    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.5):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(np.concatenate((y_train, y_val))))
    input_dim = X_train.shape[1]

    mlp_config = config.get("classifier", {}).get("mlp", {})
    hidden_layers = mlp_config.get("hidden_layers", [512, 256])
    dropout = mlp_config.get("dropout", 0.5)
    lr = mlp_config.get("lr", 0.001)
    l2 = mlp_config.get("l2", 0.0001)
    epochs = mlp_config.get("epochs", 50)
    batch_size = mlp_config.get("batch_size", 32)
    opt_type = mlp_config.get("optimizer", "adam").lower()
    momentum = mlp_config.get("momentum", 0.9)
    scheduler_type = mlp_config.get("scheduler", "none").lower()

    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_layers,
        num_classes=num_classes,
        dropout_rate=dropout,
    ).to(device)

    if opt_type == "nag":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=l2,
            momentum=momentum,
            nesterov=True,
        )
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=l2, momentum=momentum
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    if scheduler_type in ["cosine", "cosineannealing"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    last_train_loss = 0.0
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        last_train_loss = epoch_loss / len(dataloader)
        if scheduler is not None:
            scheduler.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(
            val_outputs, torch.tensor(y_val, dtype=torch.long).to(device)
        ).item()
        preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

    final_lr = optimizer.param_groups[0]["lr"]
    print(
        f"[MLP Stats] Train Loss: {last_train_loss:.4f} | Val Loss: {val_loss:.4f} | Final LR: {final_lr:.6f}"
    )

    return preds


def prepare_embeddings(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_key = str(config.get("model", "mert"))
    model_id = str(MODEL_MAPPING.get(model_key, model_key))
    print(f"Preparing embeddings using model: {model_key} ({model_id})")

    processor: Any = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model: Any = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    mel_dir = Path("dataset/mel_clap")
    metadata_path = mel_dir / "metadata.csv"

    if not metadata_path.exists():
        print(
            f"Error: {metadata_path} not found. Run python scripts/prepare_mel.py --clap-mode first."
        )
        return

    out_dir = Path(f"dataset/clap_embeddings/{model_key}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata
    shutil.copy(metadata_path, out_dir / "metadata.csv")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    batch_size = config.get("batch_size", 16)
    max_duration = config.get("max_duration", 10)

    total_batches = (len(records) + batch_size - 1) // batch_size
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Extracting Embeddings...", total=total_batches)
        for i in range(0, len(records), batch_size):
            batch_records = records[i : i + batch_size]
            batch_audios = []
            for row in batch_records:
                audio = np.load(mel_dir / row["filename"])
                max_samples = max_duration * 48000
                if len(audio) > max_samples:
                    start = (len(audio) - max_samples) // 2
                    audio = audio[start : start + max_samples]
                batch_audios.append(audio)

            target_sr = 48000
            if hasattr(processor, "sampling_rate"):
                target_sr = processor.sampling_rate
            elif hasattr(processor, "feature_extractor") and hasattr(
                processor.feature_extractor, "sampling_rate"
            ):
                target_sr = processor.feature_extractor.sampling_rate

            if target_sr != 48000:
                batch_audios = [
                    librosa.resample(y=audio, orig_sr=48000, target_sr=target_sr)
                    for audio in batch_audios
                ]

            if "mert" in model_key.lower():
                inputs = processor(
                    raw_speech=batch_audios,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=target_sr,
                )
            else:
                inputs = processor(
                    audio=batch_audios,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=target_sr,
                )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if hasattr(model, "get_audio_features"):
                    outputs = model.get_audio_features(**inputs, return_dict=True)
                else:
                    outputs = model(**inputs, return_dict=True)

                # Identify the correct embedding output
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    embeds = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    embeds = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                elif hasattr(outputs, "audio_embeds"):
                    embeds = outputs.audio_embeds
                else:
                    raise ValueError(
                        "Model output format not supported. Cannot find embeddings."
                    )

                embeds = embeds.cpu().numpy()

            for j, row in enumerate(batch_records):
                np.save(out_dir / row["filename"], embeds[j])

            progress.advance(task)

    print(f"Embeddings saved to {out_dir}")


def run_classification(config, args):
    model_key = config.get("model", "mert")
    embed_dir = Path(f"dataset/clap_embeddings/{model_key}")
    metadata_path = embed_dir / "metadata.csv"

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Run with --prepare first.")
        return

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    X_all_list, Y_all_list, folds_all_list = [], [], []
    genres = sorted(list(set(row["genre"] for row in records)))

    print(f"Loading embeddings from {embed_dir}...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Loading Data...", total=len(records))
        for row in records:
            X_all_list.append(np.load(embed_dir / row["filename"]))
            Y_all_list.append(int(row["label"]))
            folds_all_list.append(int(row["fold"]))
            progress.advance(task)

    X_all = np.array(X_all_list)
    Y_all = np.array(Y_all_list)
    folds_all = np.array(folds_all_list)

    cv_acc = []

    # Combined loop for CV and final Test evaluation
    folds_to_run = list(range(1, 6))
    if args.test:
        folds_to_run.append(0)

    for fold in folds_to_run:
        print(f"\n" + "-" * 30)
        if fold == 0:
            print(" FINAL TEST EVALUATION (Fold 0)")
            train_idx = folds_all != 0
            test_idx = folds_all == 0
            X_train, y_train = X_all[train_idx], Y_all[train_idx]
            X_eval, y_eval = X_all[test_idx], Y_all[test_idx]
        else:
            print(f" PROCESSING FOLD {fold}")
            train_idx = (folds_all != 0) & (folds_all != fold)
            val_idx = folds_all == fold
            X_train, y_train = X_all[train_idx], Y_all[train_idx]
            X_eval, y_eval = X_all[val_idx], Y_all[val_idx]
        print("-" * 30)

        # Z-Score Normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        # PCA Dimensionality Reduction
        if config.get("pca", {}).get("enabled", False):
            pca = PCA(n_components=config["pca"].get("n_components", 256))
            X_train = pca.fit_transform(X_train)
            X_eval = pca.transform(X_eval)

        clf_type = config.get("classifier", {}).get("type", "svm").lower()

        if clf_type == "svm":
            C_param = config.get("classifier", {}).get("svm", {}).get("C", 1.0)
            clf = SVC(kernel="linear", C=C_param)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_eval)
            print(f"[SVM Stats] C: {C_param} | Support Vectors: {len(clf.support_)}")
        elif clf_type == "knn":
            k_param = config.get("classifier", {}).get("knn", {}).get("k", 5)
            clf = KNeighborsClassifier(n_neighbors=k_param, metric="cosine")
            clf.fit(X_train, y_train)
            preds = clf.predict(X_eval)
            print(f"[KNN Stats] k: {k_param} | Metric: cosine")
        elif clf_type == "rf":
            rf_config = config.get("classifier", {}).get("rf", {})
            clf = RandomForestClassifier(
                n_estimators=rf_config.get("n_estimators", 100),
                criterion=rf_config.get("criterion", "gini"),
                max_depth=rf_config.get("max_depth", None),
                min_samples_split=rf_config.get("min_samples_split", 2),
                min_samples_leaf=rf_config.get("min_samples_leaf", 1),
                max_features=rf_config.get("max_features", "sqrt"),
                bootstrap=rf_config.get("bootstrap", True),
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_eval)
            clf_any: Any = clf
            print(
                f"[RF Stats] Estimators: {clf_any.n_estimators} | Max Depth: {clf_any.max_depth}"
            )
        elif clf_type == "logreg":
            lr_config = config.get("classifier", {}).get("logreg", {})
            lr_kwargs: Dict[str, Any] = {
                "C": lr_config.get("C", 1.0),
                "solver": lr_config.get("solver", "lbfgs"),
                "max_iter": lr_config.get("max_iter", 1000),
                "random_state": 42,
            }
            penalty = lr_config.get("penalty", "l2")

            # Map penalty to l1_ratio to avoid FutureWarnings in sklearn 1.8+
            if penalty == "l1":
                lr_kwargs["l1_ratio"] = 1.0
            elif penalty == "l2":
                lr_kwargs["l1_ratio"] = 0.0

            clf_base = LogisticRegression(**lr_kwargs)

            # Wrap in OneVsRest if using liblinear for multiclass
            if lr_kwargs["solver"] == "liblinear":
                clf = OneVsRestClassifier(clf_base)
            else:
                clf = clf_base

            clf.fit(X_train, y_train)
            preds = clf.predict(X_eval)

            # Diagnostic prints (handle wrapped vs raw model)
            clf_any: Any = clf
            actual_model: Any = (
                clf_any.estimators_[0] if hasattr(clf_any, "estimators_") else clf_any
            )
            print(
                f"[LogReg Stats] C: {actual_model.C} | Solver: {lr_kwargs['solver']} | Iterations: {actual_model.n_iter_[0]}"
            )
        elif clf_type == "nb":
            nb_config = config.get("classifier", {}).get("nb", {})
            clf = GaussianNB(var_smoothing=float(nb_config.get("var_smoothing", 1e-9)))
            clf.fit(X_train, y_train)
            preds = clf.predict(X_eval)
            clf_any: Any = clf
            print(f"[NB Stats] var_smoothing: {clf_any.var_smoothing}")
        elif clf_type == "mlp":
            preds = train_mlp(X_train, y_train, X_eval, y_eval, config)
        else:
            raise ValueError(f"Unknown classifier type: {clf_type}")

        acc = accuracy_score(y_eval, preds)
        if fold == 0:
            print(f"\nTest Accuracy: {acc:.4f}")
            print("Classification Report:")
            print(classification_report(y_eval, preds, target_names=genres))
        else:
            cv_acc.append(acc)
            print(f"Fold {fold} Val Accuracy: {acc:.4f}")

    print(f"\nAverage CV Accuracy: {np.mean(cv_acc):.4f}")


def main(args):
    # Set seed for reproducibility
    set_seed(42)

    if args.prepare:
        if not args.model:
            print(
                "Error: --model is required when using --prepare (e.g., --model mert)"
            )
            return

        config = {
            "model": args.model,
            "batch_size": args.batch_size if args.batch_size else 16,
            "max_duration": args.max_duration if args.max_duration else 10,
        }
        prepare_embeddings(config, args)
        return

    if not args.exp:
        print("Error: --exp is required for classification (e.g., --exp mlp)")
        return

    config_path = f"configs/clap_lin_prob/{args.exp}"
    if not config_path.endswith((".yaml", ".yml")):
        config_path += ".yaml"

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    config = load_config(config_path)

    if args.model:
        config["model"] = args.model

    run_classification(config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLAP/MERT Linear Probing Pipeline")
    parser.add_argument(
        "--exp",
        type=str,
        help="Experiment config name for classification (e.g., svm, mlp)",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run model inference to extract and save embeddings",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation on the held-out test set (Fold 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_MAPPING.keys()),
        help="Pretrained model to use (required for --prepare, optional override for --exp)",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Override batch size for extraction"
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        help="Override max duration in seconds for extraction",
    )
    args = parser.parse_args()
    main(args)
