"""
Mathematical and numerical primitives for trajectory calibration.

Provides numerically stable sigmoid, bounded inverse-sigmoid logits,
and probability clipping guards.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from scipy.special import expit, logit


@overload
def sigmoid(x: float) -> float: ...


@overload
def sigmoid(x: np.ndarray) -> np.ndarray: ...


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Numerically stable sigmoid function using scipy.special.expit.
    """
    res = expit(np.asarray(x, dtype=np.float64))
    return float(res) if np.isscalar(x) or (isinstance(res, np.ndarray) and res.ndim == 0) else res


@overload
def get_logits(confs: float, eps: float = ...) -> float: ...


@overload
def get_logits(confs: np.ndarray, eps: float = ...) -> np.ndarray: ...


def get_logits(confs: np.ndarray | float, eps: float = 1e-15) -> np.ndarray | float:
    """
    Computes bounded inverse-sigmoid logits using scipy.special.logit.
    """
    c = np.clip(np.asarray(confs, dtype=np.float64), eps, 1.0 - eps)
    res = logit(c)
    return (
        float(res) if np.isscalar(confs) or (isinstance(res, np.ndarray) and res.ndim == 0) else res
    )


@overload
def safe_clip_probs(p: float, eps: float = ...) -> float: ...


@overload
def safe_clip_probs(p: np.ndarray, eps: float = ...) -> np.ndarray: ...


def safe_clip_probs(p: np.ndarray | float, eps: float = 1e-7) -> np.ndarray | float:
    """
    Clips probabilities to [eps, 1 - eps] for numerical stability.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    res = np.clip(p_arr, eps, 1.0 - eps)
    return float(res) if np.isscalar(p) or (isinstance(res, np.ndarray) and res.ndim == 0) else res
