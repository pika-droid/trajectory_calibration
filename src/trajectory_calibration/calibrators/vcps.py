"""
Varying-Coefficient Platt Scaling (VCPS).

Our primary proposed method (Method 29). Dynamically modulates both the
slope a(z) and intercept b(z) as generalized linear functions of multi-scale
trajectory signatures z:

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

from trajectory_calibration.features.trajectory import get_logits, sigmoid
from trajectory_calibration.metrics.calibration import compute_nll

logger = logging.getLogger("trajectory_calibration.calibrators.vcps")


class VaryingCoefficientPlattScaler:
    """
    Varying-Coefficient Platt Scaler (VCPS).
    """

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

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> VaryingCoefficientPlattScaler:
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        n, d = X.shape

        self.feature_names = feature_names or [f"x{i}" for i in range(1, d + 1)]
        x1 = X[:, 0]

        # Standardize features
        X_norm = self.scaler.fit_transform(X)

        # Map feature subsets
        slope_idx = [self.feature_names.index(f) for f in self.slope_features if f in self.feature_names]
        int_idx = [self.feature_names.index(f) for f in self.intercept_features if f in self.feature_names]

        if not slope_idx:
            slope_idx = list(range(1, min(3, d)))
        if not int_idx:
            int_idx = list(range(1, min(5, d)))

        self._slope_idx = slope_idx
        self._int_idx = int_idx

        k_slope = len(slope_idx)
        k_int = len(int_idx)

        # Initialize via 1D Platt
        lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=200)
        lr.fit(x1.reshape(-1, 1), y)
        init_a0 = float(lr.coef_[0][0])
        init_b0 = float(lr.intercept_[0])

        # Objective function with exact analytical gradients
        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            if self.mode == "1d_platt":
                a0 = params[0]
                gamma = np.zeros(k_slope)
                b0 = params[1]
                w = np.zeros(k_int)
            elif self.mode == "slope_only":
                a0 = params[0]
                gamma = params[1 : 1 + k_slope]
                b0 = params[1 + k_slope]
                w = np.zeros(k_int)
            elif self.mode == "intercept_only":
                a0 = params[0]
                gamma = np.zeros(k_slope)
                b0 = params[1]
                w = params[2:]
            else:  # full
                a0 = params[0]
                gamma = params[1 : 1 + k_slope]
                b0 = params[1 + k_slope]
                w = params[2 + k_slope :]

            # Slope and intercept
            if self.mode in ["full", "slope_only"]:
                slope_log = np.clip(np.log(max(init_a0, 0.1)) + np.dot(X_norm[:, slope_idx], gamma), -3.0, 3.0)
                a_x = np.exp(slope_log)
            else:
                a_x = np.full(n, max(init_a0, 0.1))

            if self.mode in ["full", "intercept_only"]:
                b_x = b0 + np.dot(X_norm[:, int_idx], w)
            else:
                b_x = np.full(n, b0)

            logits = a_x * x1 + b_x
            p = sigmoid(logits)

            # Loss: NLL + L2 penalty
            nll = compute_nll(p, y)
            reg_gamma = (0.5 / self.C_slope) * np.sum(gamma ** 2) if self.mode in ["full", "slope_only"] else 0.0
            reg_w = (0.5 / self.C_intercept) * np.sum(w ** 2) if self.mode in ["full", "intercept_only"] else 0.0
            total_loss = nll + reg_gamma + reg_w

            # Gradients
            r = (p - y) / n
            grad_a0 = np.sum(r * x1 * a_x)
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

        # Pack initial parameter vector
        if self.mode == "1d_platt":
            x0 = np.array([init_a0, init_b0])
        elif self.mode == "slope_only":
            x0 = np.concatenate([[init_a0], np.zeros(k_slope), [init_b0]])
        elif self.mode == "intercept_only":
            x0 = np.concatenate([[init_a0], [init_b0], np.zeros(k_int)])
        else:
            x0 = np.concatenate([[init_a0], np.zeros(k_slope), [init_b0], np.zeros(k_int)])

        res = minimize(objective, x0=x0, jac=True, method="L-BFGS-B")

        # Unpack solution
        p_opt = res.x
        self.a0 = float(p_opt[0])
        if self.mode == "full":
            self.gamma = p_opt[1 : 1 + k_slope]
            self.b0 = float(p_opt[1 + k_slope])
            self.w = p_opt[2 + k_slope :]
        elif self.mode == "slope_only":
            self.gamma = p_opt[1 : 1 + k_slope]
            self.b0 = float(p_opt[1 + k_slope])
            self.w = np.zeros(k_int)
        elif self.mode == "intercept_only":
            self.gamma = np.zeros(k_slope)
            self.b0 = float(p_opt[1])
            self.w = p_opt[2:]
        else:
            self.gamma = np.zeros(k_slope)
            self.b0 = float(p_opt[1])
            self.w = np.zeros(k_int)

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
