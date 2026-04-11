"""
evaluate_confusion.py — Generate confusion matrices for trained models.
"""

from __future__ import annotations

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import csv
from pathlib import Path

def get_genres(metadata_path="dataset/mel/metadata.csv"):
    """Loads unique genres from metadata in alphabetical order."""
    if not os.path.exists(metadata_path):
        return ["blues", "classical", "country", "disco", "hiphop", 
                "jazz", "metal", "pop", "reggae", "rock"]
    
    genres = set()
    with open(metadata_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genres.add(row["genre"])
    return sorted(list(genres))

def generate_confusion_matrix(y_true, y_pred, log_dir, exp_name, genres=None):
    """
    Calculates, saves, and visualizes the confusion matrix.
    """
    if genres is None:
        genres = get_genres()
        
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    # 1. Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Save raw matrix as numpy file and text file
    np.save(os.path.join(log_dir, "confusion_matrix.npy"), cm)
    
    # Also save classification report
    report = classification_report(y_true, y_pred, target_names=genres)
    with open(os.path.join(log_dir, "classification_report.txt"), "w") as f:
        f.write(str(report))

    # 3. Visualize
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title(f"Confusion Matrix: {exp_name}")
    plt.colorbar()
    
    tick_marks = np.arange(len(genres))
    plt.xticks(tick_marks, genres, rotation=45)
    plt.yticks(tick_marks, genres)

    # Normalize matrix for display text
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # 4. Save visualization
    plot_path = os.path.join(log_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\n[✔] Confusion matrix generated for {exp_name}")
    print(f"    - Matrix saved to: {os.path.join(log_dir, 'confusion_matrix.npy')}")
    print(f"    - Plot saved to:   {plot_path}")
    print(f"    - Report saved to: {os.path.join(log_dir, 'classification_report.txt')}")

def scan_and_generate_missing():
    """
    Scans the outputs/logs directory for experiments that have 
    test_predictions.pt but no confusion_matrix.png.
    """
    log_base_dir = "outputs/logs"
    if not os.path.exists(log_base_dir):
        print(f"Log directory {log_base_dir} not found.")
        return

    print(f"Scanning {log_base_dir} for results...")
    
    experiments = sorted([d for d in os.listdir(log_base_dir) 
                   if os.path.isdir(os.path.join(log_base_dir, d))])
    
    processed_count = 0
    for exp_name in experiments:
        exp_dir = os.path.join(log_base_dir, exp_name)
        pred_path = os.path.join(exp_dir, "test_predictions.pt")
        plot_path = os.path.join(exp_dir, "confusion_matrix.png")
        
        if os.path.exists(pred_path):
            if not os.path.exists(plot_path):
                print(f"\n[!] Generating missing confusion matrix for: {exp_name}")
                try:
                    data = torch.load(pred_path, weights_only=False, map_location="cpu")
                    generate_confusion_matrix(
                        y_true=data["y_true"], 
                        y_pred=data["y_pred"], 
                        log_dir=exp_dir, 
                        exp_name=exp_name
                    )
                    processed_count += 1
                except Exception as e:
                    print(f"    Error processing {exp_name}: {e}")
            else:
                # Optional: print that it's already there if running manually
                pass

    if processed_count == 0:
        print("No new confusion matrices were needed.")
    else:
        print(f"\nDone! Processed {processed_count} experiments.")

if __name__ == "__main__":
    scan_and_generate_missing()
