"""
Varying-Coefficient Platt Scaling (VCPS).

Dynamically modulates both the slope a(z) and intercept b(z) as generalized
linear functions of multi-scale trajectory signatures z:

    logit(p(x)) = a(z) * x1 + b(z)
    a(z) = exp(a0 + gamma^T z_slope)  (strictly positive dynamic slope)
    b(z) = b0 + w^T z_intercept

Includes exact analytical gradients and L-BFGS-B optimization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.metrics.scoring import compute_nll
from trajectory_calibration.utils.math import get_logits, sigmoid

logger = logging.getLogger("trajectory_calibration.calibrators.vcps")


class VaryingCoefficientPlattScaler:
    """Varying-Coefficient Platt Scaler (VCPS)."""

    def __init__(
        self,
        C_slope: float = 1.0,
        C_intercept: float = 1.0,
        slope_features: list[str] | None = None,
        intercept_features: list[str] | None = None,
        mode: str = "full",  # "full", "slope_only", "intercept_only", "1d_platt"
        random_state: int = 42,
    ) -> None:
        self.C_slope = float(C_slope)
        self.C_intercept = float(C_intercept)
        self.slope_features = slope_features or ["x13", "x6"]
        self.intercept_features = intercept_features or ["x13", "x6", "x8", "x4"]
        self.mode = mode
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.a0: float = 1.0
        self.gamma: np.ndarray | None = None
        self.b0: float = 0.0
        self.w: np.ndarray | None = None
        self.feature_names: list[str] | None = None

    def _build_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> tuple[Any, np.ndarray]:
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        n, d = X.shape
        self.feature_names = feature_names or [f"x{i}" for i in range(1, d + 1)]
        x1 = X[:, 0]
        X_norm = self.scaler.fit_transform(X)

        matched_slope = [f for f in self.slope_features if f in self.feature_names]
        if len(matched_slope) < len(self.slope_features):
            missing = set(self.slope_features) - set(self.feature_names)
            logger.warning(
                f"VCPS slope features {missing} not found in feature_names {self.feature_names}. "
                f"Falling back to positional indices. Pass feature_names to fit()."
            )

        matched_int = [f for f in self.intercept_features if f in self.feature_names]
        if len(matched_int) < len(self.intercept_features):
            missing = set(self.intercept_features) - set(self.feature_names)
            logger.warning(
                f"VCPS intercept features {missing} not found in feature_names {self.feature_names}. "
                f"Falling back to positional indices. Pass feature_names to fit()."
            )

        slope_idx = [self.feature_names.index(f) for f in matched_slope] or list(range(1, min(3, d)))
        int_idx = [self.feature_names.index(f) for f in matched_int] or list(range(1, min(5, d)))
        self._slope_idx, self._int_idx = slope_idx, int_idx
        k_slope, k_int = len(slope_idx), len(int_idx)

        lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000).fit(x1.reshape(-1, 1), y)
        init_a0, init_b0 = float(lr.coef_[0][0]), float(lr.intercept_[0])

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            if self.mode == "1d_platt":
                a0, b0 = params[0], params[1]
                gamma, w = np.zeros(k_slope), np.zeros(k_int)
            elif self.mode == "slope_only":
                a0, gamma, b0 = params[0], params[1 : 1 + k_slope], params[1 + k_slope]
                w = np.zeros(k_int)
            elif self.mode == "intercept_only":
                a0, gamma, b0, w = params[0], np.zeros(k_slope), params[1], params[2:]
            else:  # full
                a0, gamma = params[0], params[1 : 1 + k_slope]
                b0, w = params[1 + k_slope], params[2 + k_slope :]

            if self.mode in ["full", "slope_only"]:
                slope_log = np.clip(np.log(max(a0, 0.1)) + np.dot(X_norm[:, slope_idx], gamma), -3.0, 3.0)
                a_x = np.exp(slope_log)
            else:
                a_x = np.full(n, max(a0, 0.1))

            b_x = b0 + np.dot(X_norm[:, int_idx], w) if self.mode in ["full", "intercept_only"] else np.full(n, b0)
            logits = a_x * x1 + b_x
            p = sigmoid(logits)

            nll = compute_nll(p, y)
            reg_gamma = (0.5 / self.C_slope) * np.sum(gamma ** 2) if self.mode in ["full", "slope_only"] else 0.0
            reg_w = (0.5 / self.C_intercept) * np.sum(w ** 2) if self.mode in ["full", "intercept_only"] else 0.0
            total_loss = nll + reg_gamma + reg_w

            r = (p - y) / n
            grad_a0 = np.sum(r * x1 * a_x * (1.0 / max(a0, 0.1))) if a0 > 0.1 else 0.0
            grad_gamma = np.dot(X_norm[:, slope_idx].T, r * x1 * a_x) + (gamma / self.C_slope)
            grad_b0 = np.sum(r)
            grad_w = np.dot(X_norm[:, int_idx].T, r) + (w / self.C_intercept)

            if self.mode == "1d_platt":
                grad = np.array([grad_a0, grad_b0])
            elif self.mode == "slope_only":
                grad = np.concatenate([[grad_a0], grad_gamma, [grad_b0]])
            elif self.mode == "intercept_only":
                grad = np.concatenate([[grad_a0], [grad_b0], grad_w])
            else:
                grad = np.concatenate([[grad_a0], grad_gamma, [grad_b0], grad_w])

            return float(total_loss), grad

        if self.mode == "1d_platt":
            x0 = np.array([init_a0, init_b0])
        elif self.mode == "slope_only":
            x0 = np.concatenate([[init_a0], np.zeros(k_slope), [init_b0]])
        elif self.mode == "intercept_only":
            x0 = np.concatenate([[init_a0], [init_b0], np.zeros(k_int)])
        else:
            x0 = np.concatenate([[init_a0], np.zeros(k_slope), [init_b0], np.zeros(k_int)])

        return objective, x0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> VaryingCoefficientPlattScaler:
        objective, x0 = self._build_objective(X_train, y_train, feature_names=feature_names)
        res = minimize(objective, x0=x0, jac=True, method="L-BFGS-B")
        p_opt = res.x
        k_slope, k_int = len(self._slope_idx), len(self._int_idx)
        self.a0 = float(p_opt[0])
        if self.mode == "full":
            self.gamma, self.b0, self.w = p_opt[1 : 1 + k_slope], float(p_opt[1 + k_slope]), p_opt[2 + k_slope :]
        elif self.mode == "slope_only":
            self.gamma, self.b0, self.w = p_opt[1 : 1 + k_slope], float(p_opt[1 + k_slope]), np.zeros(k_int)
        elif self.mode == "intercept_only":
            self.gamma, self.b0, self.w = np.zeros(k_slope), float(p_opt[1]), p_opt[2:]
        else:
            self.gamma, self.b0, self.w = np.zeros(k_slope), float(p_opt[1]), np.zeros(k_int)
        return self

    def compute_dynamic_slope(self, X: np.ndarray) -> np.ndarray:
        """Calculates dynamic slope a(z) for each sample."""
        X_norm = self.scaler.transform(X)
        if self.mode in ["full", "slope_only"] and self.gamma is not None:
            slope_log = np.clip(np.log(max(self.a0, 0.1)) + np.dot(X_norm[:, self._slope_idx], self.gamma), -3.0, 3.0)
            return np.exp(slope_log)
        return np.full(len(X), max(self.a0, 0.1))

    def compute_dynamic_intercept(self, X: np.ndarray) -> np.ndarray:
        """Calculates dynamic intercept b(z) for each sample."""
        X_norm = self.scaler.transform(X)
        if self.mode in ["full", "intercept_only"] and self.w is not None:
            return self.b0 + np.dot(X_norm[:, self._int_idx], self.w)
        return np.full(len(X), self.b0)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Computes calibrated probabilities: sigma(a(z) * x1 + b(z))."""
        X = np.asarray(X_test, dtype=np.float64)
        x1 = X[:, 0]
        a_x = self.compute_dynamic_slope(X)
        b_x = self.compute_dynamic_intercept(X)
        return sigmoid(a_x * x1 + b_x)

    def get_effective_temperature(self, X: np.ndarray) -> np.ndarray:
        """Returns instance effective temperature T_eff(z) = 1 / a(z)."""
        a_x = self.compute_dynamic_slope(X)
        return 1.0 / np.maximum(a_x, 1e-4)
