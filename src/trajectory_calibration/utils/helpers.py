"""
Core helper utilities for trajectory calibration.

Provides safe checkpoint loading, reproducibility seed management,
and text normalization.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trajectory_calibration.utils.config import ARCH_SCALES, Config


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


__all__ = ["ARCH_SCALES", "Config", "clean_text", "safe_torch_load", "set_seed"]
