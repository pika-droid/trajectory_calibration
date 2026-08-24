"""
Mathematical and numerical primitives for trajectory calibration.

Provides numerically stable sigmoid, bounded inverse-sigmoid logits,
and probability clipping guards.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Numerically stable sigmoid function bounded to avoid float overflow.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_clipped = np.clip(x_arr, -35.0, 35.0)
    res = 1.0 / (1.0 + np.exp(-x_clipped))
    return float(res) if np.isscalar(x) or (isinstance(res, np.ndarray) and res.ndim == 0) else res


def get_logits(confs: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    """
    Computes bounded inverse-sigmoid logits: ln(c / (1 - c)).
    """
    confs_arr = np.asarray(confs, dtype=np.float64)
    c = np.clip(confs_arr, eps, 1.0 - eps)
    res = np.log(c / (1.0 - c))
    return float(res) if np.isscalar(confs) or (isinstance(res, np.ndarray) and res.ndim == 0) else res


def safe_clip_probs(p: np.ndarray | float, eps: float = 1e-7) -> np.ndarray | float:
    """
    Clips probabilities to [eps, 1 - eps] for numerical stability.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    res = np.clip(p_arr, eps, 1.0 - eps)
    return float(res) if np.isscalar(p) or (isinstance(res, np.ndarray) and res.ndim == 0) else res
