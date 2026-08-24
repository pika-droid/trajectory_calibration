"""
Model health and diagnostic status evaluators (ADR-012).
"""

from __future__ import annotations

from typing import Any
import numpy as np
import scipy.stats as stats

from trajectory_calibration.metrics.ece import compute_ece, compute_mce
from trajectory_calibration.metrics.scoring import (
    compute_brier,
    compute_nll,
    compute_prediction_std,
)


def evaluate_model_diagnostics(
    probs_test: np.ndarray | list[float],
    y_test: np.ndarray | list[float],
    c_test: np.ndarray | list[float],
    y_train: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """
    Evaluates ECE, MCE, Brier, Brier Gain, Prediction Std, Spearman Rank Correlation,
    and assigns ADR-012 Diagnostic Status (VALID, COLLAPSED, SCRAMBLED).
    """
    probs_arr = np.asarray(probs_test, dtype=np.float64)
    y_arr = np.asarray(y_test, dtype=np.float64)
    c_arr = np.asarray(c_test, dtype=np.float64)

    ece = compute_ece(probs_arr, y_arr)
    mce = compute_mce(probs_arr, y_arr)
    brier = compute_brier(probs_arr, y_arr)
    nll = compute_nll(probs_arr, y_arr)
    prob_std = compute_prediction_std(probs_arr)

    base_rate = float(np.mean(y_train)) if y_train is not None and len(y_train) > 0 else float(np.mean(y_arr))
    base_brier = float(np.mean((base_rate - y_arr) ** 2))
    brier_gain = float(base_brier - brier)

    if len(probs_arr) > 5 and np.std(probs_arr) > 1e-6 and np.std(c_arr) > 1e-6:
        rho, _ = stats.spearmanr(probs_arr, c_arr)
        rho = float(rho) if not np.isnan(rho) else 1.0
    else:
        rho = 1.0

    if prob_std < 0.02 or brier_gain <= 0.001:
        status = "COLLAPSED"
    elif rho < 0.10:
        status = "SCRAMBLED"
    else:
        status = "VALID"

    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "brier_gain": brier_gain,
        "nll": nll,
        "prob_std": prob_std,
        "spearman_rho": rho,
        "status": status,
    }
