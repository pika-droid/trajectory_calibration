"""
Two-stage residual trajectory calibration and standard metric evaluation panel.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from scipy.integrate import trapezoid as _trapezoid
except ImportError:
    _trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from trajectory_calibration.features.diagnostics import evaluate_model_diagnostics
from trajectory_calibration.metrics.ece import (
    compute_adaptive_ece,
    compute_ece,
    compute_kde_ece,
    compute_mce,
)
from trajectory_calibration.metrics.murphy import compute_murphy_brier_decomposition
from trajectory_calibration.metrics.scoring import (
    compute_auroc,
    compute_brier,
    compute_nll,
    compute_prediction_std,
)
from trajectory_calibration.utils.math import get_logits, sigmoid

logger = logging.getLogger("trajectory_calibration.calibrators.residual")


def compute_aurc(probs: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    """Area Under the Risk-Coverage Curve (AURC)."""
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0

    order = np.argsort(-p)
    y_sorted = labels[order]

    cum_correct = np.cumsum(y_sorted)
    coverage = np.arange(1, n + 1) / n
    precision = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - precision

    if _trapezoid is not None:
        aurc = float(_trapezoid(risk, coverage))
    else:
        aurc = float(np.sum((risk[:-1] + risk[1:]) / 2.0 * np.diff(coverage)))
    return aurc


class ResidualTrajectoryCalibrator:
    """
    Two-stage residual trajectory calibrator.
    - Stage 1: Fits 1D Platt scaling on x1 -> l1.
    - Stage 2: Fits L2 logistic regression on trajectory features with l1 as fixed offset.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.stage1_lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
        self.stage2_lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000, random_state=self.random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ResidualTrajectoryCalibrator:
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int64)
        x1 = X[:, [0]]

        # Stage 1: Platt on x1
        self.stage1_lr.fit(x1, y)

        # Stage 2: Fit residual on Z
        Z = X[:, 1:] if X.shape[1] > 1 else X
        Z_norm = self.scaler.fit_transform(Z)
        self.stage2_lr.fit(Z_norm, y)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X = np.asarray(X_test, dtype=np.float64)
        x1 = X[:, [0]]
        p1 = self.stage1_lr.predict_proba(x1)[:, 1]
        l1 = get_logits(p1)

        Z = X[:, 1:] if X.shape[1] > 1 else X
        Z_norm = self.scaler.transform(Z)
        res_logits = self.stage2_lr.decision_function(Z_norm)
        return sigmoid(l1 + res_logits)


def evaluate_full_metric_panel(
    probs: np.ndarray, y: np.ndarray, c_576: np.ndarray, y_train: np.ndarray | None = None
) -> dict[str, Any]:
    """
    Evaluates the complete scientific metric panel across 10 distinct metrics.
    """
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    c = np.asarray(c_576, dtype=np.float64)

    ece = compute_ece(p, labels)
    ada_ece = compute_adaptive_ece(p, labels)
    kde_ece = compute_kde_ece(p, labels)
    brier = compute_brier(p, labels)
    nll = compute_nll(p, labels)
    auroc = compute_auroc(p, labels)
    aurc = compute_aurc(p, labels)
    murphy = compute_murphy_brier_decomposition(p, labels)
    diag = evaluate_model_diagnostics(p, labels, c, y_train=y_train)

    return {
        "ece": ece,
        "ece_percent": ece * 100.0,
        "adaptive_ece": ada_ece,
        "adaptive_ece_percent": ada_ece * 100.0,
        "kde_ece": kde_ece,
        "brier": brier,
        "brier_gain": diag["brier_gain"],
        "murphy_uncertainty": murphy["uncertainty"],
        "murphy_resolution": murphy["resolution"],
        "murphy_reliability": murphy["reliability"],
        "auroc": auroc,
        "nll": nll,
        "aurc": aurc,
        "prob_std": diag["prob_std"],
        "spearman_rho": diag["spearman_rho"],
        "status": diag["status"],
    }
