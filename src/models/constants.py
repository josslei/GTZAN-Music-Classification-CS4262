"""
constants.py — Global architectural constants for models.

This file computes derived dimensions (like MAX_FRAMES) from the central 
configuration so that all models stay in sync with the data processing.
"""

import math
import yaml
from pathlib import Path

# Locate the config relative to this file (src/models/constants.py)
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "data.yaml"

def _load_dims():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    sr = cfg.get("sample_rate", 22050)
    hop = cfg.get("hop_length", 512)
    sec = cfg.get("segment_seconds")
    
    # If segment_seconds is None, default to GTZAN full length (30s)
    if sec is None:
        sec = 30.0
        
    frames = math.ceil((sec * sr) / hop)
    return frames

# Constants visible to all models
MAX_FRAMES: int = _load_dims()
N_MELS: int = 128
CHANNELS: int = 1
