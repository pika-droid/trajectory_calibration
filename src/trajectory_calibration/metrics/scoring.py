"""
Probability scoring rules and discrimination metrics.

Implements Brier Score, Binary Negative Log-Likelihood (NLL),
Prediction Standard Deviation (collapse check), and AUROC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def compute_brier(confidences: np.ndarray | list[float], accuracies: np.ndarray | list[float]) -> float:
    """
    Brier Score (Mean Squared Error between probabilities and binary outcomes).

    Formula: (1/N) * sum_i (c_i - y_i)^2
    """
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)
    if len(confs) == 0:
        return 0.0
    return float(brier_score_loss(accs, confs))


def compute_nll(confidences: np.ndarray | list[float], accuracies: np.ndarray | list[float], eps: float = 1e-12) -> float:
    """
    Binary Negative Log-Likelihood (Log Loss / Cross Entropy).

    Formula: - (1/N) * sum_i [y_i * ln(c_i) + (1 - y_i) * ln(1 - c_i)]
    """
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)
    if len(confs) == 0:
        return 0.0
    confs_clipped = np.clip(confs, eps, 1.0 - eps)
    if len(np.unique(accs)) < 2:
        return float(log_loss(accs, confs_clipped, labels=[0, 1]))
    return float(log_loss(accs, confs_clipped))


def compute_prediction_std(probs: np.ndarray | list[float]) -> float:
    """
    Standard deviation of predicted probabilities.

    Used as a collapse diagnostic: sigma_p < 0.02 indicates prediction collapse.
    """
    p = np.asarray(probs, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    return float(np.std(p))


def compute_auroc(probs: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    """
    Area Under the Receiver Operating Characteristic Curve (AUROC).

    Measures model discrimination power (ranking correct vs incorrect answers).
    """
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, p))
    except Exception:
        return 0.5
