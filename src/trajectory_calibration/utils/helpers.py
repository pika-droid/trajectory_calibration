"""
Core helper utilities for trajectory calibration.

Provides safe checkpoint loading, reproducibility seed management,
text normalization, and configuration dataclasses.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class Config:
    """Centralized configuration for feature extraction and calibration benchmarks."""

    model_path: str = "mucai/llava-v1.5-7b-m3"
    arch: str = "m3"  # "m3" or "mqt", auto-resolved in __post_init__
    precision: str = "fp16"
    features_dir: Path = Path("results/features")
    output_dir: Path = Path("results/experiments")
    seed: int = 42
    max_new_tokens: int = 16
    subset_size: int | None = None
    token_scales: list[int] = field(default_factory=lambda: [1, 9, 36, 144, 576])

    def __post_init__(self) -> None:
        if "mqt" in self.model_path.lower() or self.arch.lower() == "mqt":
            self.arch = "mqt"
            self.token_scales = [1, 9, 36, 144, 256]
        else:
            self.arch = "m3"
            self.token_scales = [1, 9, 36, 144, 576]
        self.fine_scale = self.token_scales[-1]
        self.features_dir = Path(self.features_dir)
        self.output_dir = Path(self.output_dir)


def safe_torch_load(path: str | Path, map_location: str = "cpu") -> Any:
    """
    Robust PyTorch checkpoint loader with fallback across PyTorch versions.

    Handles PyTorch >= 2.4 default weights_only=True restriction for complex
    dictionary payloads.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path_obj.resolve()}")

    try:
        # PyTorch >= 2.4 supports weights_only parameter
        return torch.load(path_obj, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch < 2.4 does not accept weights_only
        return torch.load(path_obj, map_location=map_location)


def set_seed(seed: int = 42) -> None:
    """Lock reproducibility seeds across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(text: Any) -> str:
    """
    Normalize and clean text for exact and soft answer matching.

    Lowers case, strips punctuation, and removes leading/trailing articles (a, an, the).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    return " ".join(text.split())
