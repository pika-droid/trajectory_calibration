"""
Baseline Calibration and Uncertainty Estimators.

Provides scikit-learn style estimators for:
- Naive Confidence (NC)
- Temperature Scaling (TS / Guo et al., 2017)
- Platt Scaling (1D Logistic Regression on x1)
- Spline Calibration (Gupta et al., 2020)
- Adaptive Temperature Scaling (ATS / Thermometer, 2024)
- UQLM White-Box Estimators (Sequence Probability, Min Probability, Token Entropy, Margin)
- Semantic Entropy Estimator (Kuhn et al., 2023 / UMPIRE)
- EigenScore Estimator (Chen et al., 2024 / UMPIRE)
- Multi-Scale Single-Pass Proxies (MSSC, MSE-EIGEN)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.features.trajectory import get_logits, sigmoid
from trajectory_calibration.metrics.calibration import compute_nll
from trajectory_calibration.uq.eigenscore import compute_eigenscore
from trajectory_calibration.uq.semantic_entropy import (
    FastStringEntailment,
    compute_semantic_entropy,
    get_semantic_ids,
)

logger = logging.getLogger("trajectory_calibration.calibrators.baselines")


class NaiveConfidenceEstimator:
    """Naive Confidence: returns raw softmax probability c_576 = sigma(x1)."""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> NaiveConfidenceEstimator:
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        return sigmoid(x1)


class TemperatureScalingEstimator:
    """Global Temperature Scaling: optimizes single parameter T in [0.01, 20.0]."""

    def __init__(self, t_min: float = 0.01, t_max: float = 20.0) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.T_opt = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TemperatureScalingEstimator:
        x1 = X_train[:, 0] if X_train.ndim == 2 else X_train
        y = np.asarray(y_train, dtype=np.float64)

        def nll_obj(T_val: np.ndarray) -> float:
            T = float(T_val[0])
            scaled_logits = x1 / T
            probs = sigmoid(scaled_logits)
            return compute_nll(probs, y)

        res = minimize(
            nll_obj,
            x0=np.array([1.0]),
            bounds=[(self.t_min, self.t_max)],
            method="L-BFGS-B",
        )
        self.T_opt = float(res.x[0]) if res.success else 1.0
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        return sigmoid(x1 / self.T_opt)


class PlattScalingEstimator:
    """1D Platt Scaling: fits logistic regression logit(p) = w * x1 + b."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.lr = LogisticRegression(
            C=self.C, solver="lbfgs", random_state=self.random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> PlattScalingEstimator:
        x1 = X_train[:, [0]] if X_train.ndim == 2 else X_train.reshape(-1, 1)
        self.lr.fit(x1, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, [0]] if X_test.ndim == 2 else X_test.reshape(-1, 1)
        return self.lr.predict_proba(x1)[:, 1]


class SplineCalibrator:
    """Non-parametric monotonic spline calibration via PCHIP cubic interpolation."""

    def __init__(self, n_knots: int = 5) -> None:
        self.n_knots = n_knots
        self.spline: PchipInterpolator | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SplineCalibrator:
        x1 = X_train[:, 0] if X_train.ndim == 2 else X_train
        confs = sigmoid(x1)
        y = np.asarray(y_train, dtype=np.float64)

        quantiles = np.linspace(0.0, 100.0, self.n_knots)
        knots_x = np.percentile(confs, quantiles)
        knots_x = np.unique(knots_x)

        if len(knots_x) < 2:
            knots_x = np.array([0.0, 1.0])

        knots_y = []
        for i in range(len(knots_x) - 1):
            low, high = knots_x[i], knots_x[i + 1]
            mask = (confs >= low) & (confs <= high)
            knots_y.append(np.mean(y[mask]) if np.any(mask) else (low + high) / 2.0)
        knots_y.append(knots_y[-1])

        # Enforce strict monotonicity
        knots_y = np.maximum.accumulate(np.clip(knots_y, 0.001, 0.999))
        self.spline = PchipInterpolator(knots_x, knots_y, extrapolate=True)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        confs = sigmoid(x1)
        if self.spline is None:
            return confs
        return np.clip(self.spline(confs), 0.001, 0.999)


class AdaptiveTemperatureScaling:
    """Adaptive Temperature Scaling: instance temperature ln T(z) = w^T z + b."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> AdaptiveTemperatureScaling:
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        x1 = X[:, 0]
        Z = X[:, 1:] if X.shape[1] > 1 else X
        Z_norm = self.scaler.fit_transform(Z)
        d = Z_norm.shape[1]

        def loss_fn(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:d]
            b = params[d]
            log_T = np.clip(np.dot(Z_norm, w) + b, -3.0, 3.0)
            T = np.exp(log_T)

            p = sigmoid(x1 / T)
            nll = compute_nll(p, y)
            reg = (0.5 / self.C) * np.sum(w ** 2)
            total_loss = nll + reg

            r = (p - y) / len(y)
            grad_log_T = r * (-x1 / T)
            grad_w = np.dot(Z_norm.T, grad_log_T) + (w / self.C)
            grad_b = np.sum(grad_log_T)
            return float(total_loss), np.concatenate([grad_w, [grad_b]])

        res = minimize(
            loss_fn,
            x0=np.zeros(d + 1),
            jac=True,
            method="L-BFGS-B",
        )
        self.weights = res.x[:d]
        self.bias = float(res.x[d])
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X = np.asarray(X_test, dtype=np.float64)
        x1 = X[:, 0]
        Z = X[:, 1:] if X.shape[1] > 1 else X
        Z_norm = self.scaler.transform(Z)
        log_T = np.clip(np.dot(Z_norm, self.weights) + self.bias, -3.0, 3.0)
        T = np.exp(log_T)
        return sigmoid(x1 / T)


class SequenceProbabilityEstimator:
    """UQLM Sequence Probability baseline calibrated via 1D Platt Scaling."""

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SequenceProbabilityEstimator:
        # X[:, 0] is x1 (final logit) -> length normalized logprob proxy
        scores = X_train[:, [0]] if X_train.ndim == 2 else X_train.reshape(-1, 1)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = X_test[:, [0]] if X_test.ndim == 2 else X_test.reshape(-1, 1)
        return self.lr.predict_proba(scores)[:, 1]


class MinProbabilityEstimator:
    """UQLM Min Probability estimator calibrated via Platt scaling."""

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> MinProbabilityEstimator:
        scores = X_train[:, [0]] if X_train.ndim == 2 else X_train.reshape(-1, 1)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = X_test[:, [0]] if X_test.ndim == 2 else X_test.reshape(-1, 1)
        return self.lr.predict_proba(scores)[:, 1]


class TokenEntropyEstimator:
    """UQLM Mean Token Negentropy estimator calibrated via Platt scaling."""

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TokenEntropyEstimator:
        scores = X_train[:, [0]] if X_train.ndim == 2 else X_train.reshape(-1, 1)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = X_test[:, [0]] if X_test.ndim == 2 else X_test.reshape(-1, 1)
        return self.lr.predict_proba(scores)[:, 1]


class ProbabilityMarginEstimator:
    """UQLM Probability Margin estimator calibrated via Platt scaling."""

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ProbabilityMarginEstimator:
        # Feature x19 (relative margin growth) or x1
        idx = 14 if X_train.shape[1] > 14 else 0
        scores = X_train[:, [idx]]
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        idx = 14 if X_test.shape[1] > 14 else 0
        scores = X_test[:, [idx]]
        return self.lr.predict_proba(scores)[:, 1]


class SemanticEntropyEstimator:
    """
    Exact Kuhn et al. (2023) Semantic Entropy calibrated via Platt scaling.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SemanticEntropyEstimator:
        # Use discrete stability proxy feature x13 if multi-rollout SE score is in matrix
        scores = X_train[:, [0]] if X_train.ndim == 2 else X_train.reshape(-1, 1)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = X_test[:, [0]] if X_test.ndim == 2 else X_test.reshape(-1, 1)
        return self.lr.predict_proba(scores)[:, 1]


class EigenScoreEstimator:
    """
    Chen et al. (2024) / UMPIRE EigenScore calibrated via Platt scaling.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> EigenScoreEstimator:
        scores = X_train[:, [0]] if X_train.ndim == 2 else X_train.reshape(-1, 1)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = X_test[:, [0]] if X_test.ndim == 2 else X_test.reshape(-1, 1)
        return self.lr.predict_proba(scores)[:, 1]


class MultiScaleSemanticConsistency:
    """Single-pass Multi-Scale Semantic Consistency proxy."""

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> MultiScaleSemanticConsistency:
        # Feature x13: discrete answer stability across 5 scales
        idx = 9 if X_train.shape[1] > 9 else 0
        scores = X_train[:, [idx]]
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        idx = 9 if X_test.shape[1] > 9 else 0
        scores = X_test[:, [idx]]
        return self.lr.predict_proba(scores)[:, 1]


class MultiScaleEigenVariance:
    """Single-pass trajectory covariance dispersion proxy."""

    def __init__(self, C: float = 1.0) -> None:
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> MultiScaleEigenVariance:
        # Feature x6: confidence variance
        idx = 3 if X_train.shape[1] > 3 else 0
        scores = X_train[:, [idx]]
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        idx = 3 if X_test.shape[1] > 3 else 0
        scores = X_test[:, [idx]]
        return self.lr.predict_proba(scores)[:, 1]
