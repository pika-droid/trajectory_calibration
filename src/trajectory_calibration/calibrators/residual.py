"""
Two-stage residual trajectory calibration and standard metric evaluation panel.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.features.diagnostics import evaluate_model_diagnostics
from trajectory_calibration.metrics.ece import (
    compute_adaptive_ece,
    compute_ece,
    compute_kde_ece,
)
from trajectory_calibration.metrics.murphy import compute_murphy_brier_decomposition
from trajectory_calibration.metrics.scoring import (
    compute_auroc,
    compute_brier,
    compute_nll,
)
from trajectory_calibration.utils.math import sigmoid

logger = logging.getLogger("trajectory_calibration.calibrators.residual")


def compute_aurc(probs: np.ndarray | list[float], y: np.ndarray | list[float]) -> float:
    """Area Under the Risk-Coverage Curve (AURC)."""
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    if n == 1:
        pred = 1.0 if p[0] >= 0.5 else 0.0
        return 0.0 if labels[0] == pred else 1.0

    confidence = np.maximum(p, 1.0 - p)  # Confidence for binary classification
    order = np.argsort(-confidence)  # Sort by descending confidence
    y_sorted = labels[order]
    p_sorted = p[order]

    # Risk = 1 - accuracy at each coverage level
    cum_correct = np.cumsum(y_sorted == (p_sorted >= 0.5).astype(float))
    coverage = np.arange(1, n + 1)
    risk = 1.0 - cum_correct / coverage

    return float(np.mean(risk))


class ResidualTrajectoryCalibrator:
    """
    Two-stage residual trajectory calibrator.
    - Stage 1: Fits 1D Platt scaling on x1 -> l1.
    - Stage 2: Fits L2 logistic regression on trajectory features Z with l1 as fixed GLM offset.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = float(C)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.stage1_lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
        self.weights: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ResidualTrajectoryCalibrator:
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        n, d = X.shape
        x1 = X[:, [0]]

        # Stage 1: Platt on x1
        self.stage1_lr.fit(x1, y)
        l1 = self.stage1_lr.decision_function(x1)

        # Stage 2: Fit residual weights on Z with l1 as fixed GLM offset
        Z = X[:, 1:] if d > 1 else X
        Z_norm = self.scaler.fit_transform(Z)
        k = Z_norm.shape[1]

        def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
            logits = l1 + np.dot(Z_norm, w)
            p = sigmoid(logits)
            nll = compute_nll(p, y)
            reg = (0.5 / self.C) * np.sum(w**2)
            total_loss = nll + reg

            r = (p - y) / n
            grad_w = np.dot(Z_norm.T, r) + (w / self.C)
            return float(total_loss), grad_w

        res = minimize(loss_and_grad, x0=np.zeros(k), jac=True, method="L-BFGS-B")
        self.weights = res.x
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X = np.asarray(X_test, dtype=np.float64)
        x1 = X[:, [0]]
        l1 = self.stage1_lr.decision_function(x1)

        Z = X[:, 1:] if X.shape[1] > 1 else X
        Z_norm = np.asarray(self.scaler.transform(Z), dtype=np.float64)
        res_logits = np.dot(Z_norm, self.weights) if self.weights is not None else 0.0
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
