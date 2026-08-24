"""
Two-stage residual trajectory calibration and standard metric evaluation panel.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.features.trajectory import evaluate_model_diagnostics, get_logits, sigmoid
from trajectory_calibration.metrics.calibration import (
    compute_adaptive_ece,
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_kde_ece,
    compute_mce,
    compute_murphy_brier_decomposition,
    compute_nll,
    compute_prediction_std,
)

logger = logging.getLogger("trajectory_calibration.calibrators.residual")


def compute_aurc(probs: np.ndarray, y: np.ndarray) -> float:
    """Area Under the Risk-Coverage Curve (AURC)."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return 0.0

    order = np.argsort(-probs)
    y_sorted = y[order]

    cum_correct = np.cumsum(y_sorted)
    coverage = np.arange(1, n + 1) / n
    precision = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - precision

    aurc = float(np.trapz(risk, coverage))
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
        self.stage1_lr = LogisticRegression(C=1000.0, solver="lbfgs")
        self.stage2_lr = LogisticRegression(C=self.C, solver="lbfgs", random_state=self.random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ResidualTrajectoryCalibrator:
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int64)
        x1 = X[:, [0]]

        # Stage 1: Platt on x1
        self.stage1_lr.fit(x1, y)
        p1 = self.stage1_lr.predict_proba(x1)[:, 1]
        l1 = get_logits(p1).reshape(-1, 1)

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
    Evaluates the complete scientific metric panel across 10 distinct metrics:
    - ECE, ECE Percent
    - Adaptive ECE, Adaptive ECE Percent
    - KDE ECE
    - Brier Score & Murphy Decomposition (Uncertainty, Resolution, Reliability)
    - AUROC, NLL, AURC, Prediction Std, Spearman Rho, Diagnostic Status
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c_576 = np.asarray(c_576, dtype=np.float64)

    ece = compute_ece(probs, y)
    ada_ece = compute_adaptive_ece(probs, y)
    kde_ece = compute_kde_ece(probs, y)
    brier = compute_brier(probs, y)
    nll = compute_nll(probs, y)
    auroc = compute_auroc(probs, y)
    aurc = compute_aurc(probs, y)
    murphy = compute_murphy_brier_decomposition(probs, y)
    diag = evaluate_model_diagnostics(probs, y, c_576, y_train=y_train)

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
