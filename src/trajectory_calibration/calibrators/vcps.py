"""
Varying-Coefficient Platt Scaling (VCPS).

Dynamically modulates both the slope a(z) and intercept b(z) as generalized
linear functions of multi-scale trajectory signatures z:

    logit(p(x)) = a(z) * x1 + b(z)
    a(z) = exp(a0 + gamma^T z_slope)  (strictly positive dynamic slope)
    b(z) = b0 + w^T z_intercept

Includes dynamic binding across all 16 trajectory signatures (VCPS-17D, 34 params)
or stepwise selected subsets (VCPS-5D, 10 params), with analytical L-BFGS-B gradients.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.metrics.scoring import compute_nll
from trajectory_calibration.utils.math import sigmoid

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
        feature_set: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.C_slope = float(C_slope)
        self.C_intercept = float(C_intercept)
        self.slope_features = slope_features
        self.intercept_features = intercept_features
        self.mode = mode
        self.feature_set = feature_set.lower().strip() if feature_set is not None else None
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.alpha0: float = 0.0
        self.a0: float = 1.0
        self.gamma: np.ndarray | None = None
        self.b0: float = 0.0
        self.w: np.ndarray | None = None
        self.feature_names: list[str] | None = None
        self.n_features_in_: int | None = None

    @property
    def n_params(self) -> int:
        """Total number of optimized parameters based on mode and active features."""
        if self.mode == "1d_platt":
            return 2
        k_slope = len(self._slope_idx) if hasattr(self, "_slope_idx") else 0
        k_int = len(self._int_idx) if hasattr(self, "_int_idx") else 0
        if self.mode == "slope_only":
            return 1 + k_slope + 1
        if self.mode == "intercept_only":
            return 1 + 1 + k_int
        return 1 + k_slope + 1 + k_int

    def _build_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> tuple[Any, np.ndarray]:
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        n, d = X.shape
        self.n_features_in_ = d
        self.feature_names = feature_names or [f"x{i}" for i in range(1, d + 1)]
        x1 = X[:, 0]
        X_norm = self.scaler.fit_transform(X)

        non_anchor = [f for f in self.feature_names if f != "x1"]

        # Dynamic slope binding
        if self.slope_features is not None:
            matched_slope = [f for f in self.slope_features if f in self.feature_names]
            if len(matched_slope) < len(self.slope_features):
                missing = set(self.slope_features) - set(self.feature_names)
                logger.warning(
                    f"VCPS slope features {missing} not in feature_names {self.feature_names}"
                )
            if matched_slope:
                slope_features = matched_slope
                slope_idx = [self.feature_names.index(f) for f in slope_features]
            else:
                slope_idx = list(range(1, min(3, d)))
                slope_features = [self.feature_names[i] for i in slope_idx]
        elif self.feature_set == "5d":
            slope_features = non_anchor[:4]
            slope_idx = [self.feature_names.index(f) for f in slope_features]
        elif self.feature_set == "17d" or d >= 17:
            slope_features = non_anchor
            slope_idx = [self.feature_names.index(f) for f in slope_features]
        else:
            slope_features = non_anchor
            slope_idx = [self.feature_names.index(f) for f in slope_features]

        # Dynamic intercept binding
        if self.intercept_features is not None:
            matched_int = [f for f in self.intercept_features if f in self.feature_names]
            if len(matched_int) < len(self.intercept_features):
                missing = set(self.intercept_features) - set(self.feature_names)
                logger.warning(
                    f"VCPS intercept features {missing} not in feature_names {self.feature_names}"
                )
            if matched_int:
                int_features = matched_int
                int_idx = [self.feature_names.index(f) for f in int_features]
            else:
                int_idx = list(range(1, min(5, d)))
                int_features = [self.feature_names[i] for i in int_idx]
        elif self.feature_set == "5d":
            int_features = non_anchor[:4]
            int_idx = [self.feature_names.index(f) for f in int_features]
        elif self.feature_set == "17d" or d >= 17:
            int_features = non_anchor
            int_idx = [self.feature_names.index(f) for f in int_features]
        else:
            int_features = non_anchor
            int_idx = [self.feature_names.index(f) for f in int_features]

        self.slope_features_ = slope_features
        self.intercept_features_ = int_features
        self._slope_idx, self._int_idx = slope_idx, int_idx
        k_slope, k_int = len(slope_idx), len(int_idx)

        lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000).fit(x1.reshape(-1, 1), y)
        init_a0, init_b0 = float(lr.coef_[0][0]), float(lr.intercept_[0])
        init_alpha0 = float(np.log(max(init_a0, 1e-4)))

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            if self.mode == "1d_platt":
                alpha0, b0 = params[0], params[1]
                gamma, w = np.zeros(k_slope), np.zeros(k_int)
            elif self.mode == "slope_only":
                alpha0, gamma, b0 = params[0], params[1 : 1 + k_slope], params[1 + k_slope]
                w = np.zeros(k_int)
            elif self.mode == "intercept_only":
                alpha0, gamma, b0, w = params[0], np.zeros(k_slope), params[1], params[2:]
            else:  # full
                alpha0, gamma = params[0], params[1 : 1 + k_slope]
                b0, w = params[1 + k_slope], params[2 + k_slope :]

            if self.mode in ["full", "slope_only"]:
                slope_raw = alpha0 + np.dot(X_norm[:, slope_idx], gamma)
                active_mask = (slope_raw >= -3.0) & (slope_raw <= 3.0)
                slope_log = np.clip(slope_raw, -3.0, 3.0)
                a_x = np.exp(slope_log)
            else:
                slope_raw = np.full(n, alpha0)
                active_mask = np.ones(n, dtype=bool)
                a_x = np.full(n, np.exp(alpha0))

            b_x = (
                b0 + np.dot(X_norm[:, int_idx], w)
                if self.mode in ["full", "intercept_only"]
                else np.full(n, b0)
            )
            logits = a_x * x1 + b_x
            p = sigmoid(logits)

            nll = compute_nll(p, y)
            reg_gamma = (
                (0.5 / self.C_slope) * np.sum(gamma**2)
                if self.mode in ["full", "slope_only"]
                else 0.0
            )
            reg_w = (
                (0.5 / self.C_intercept) * np.sum(w**2)
                if self.mode in ["full", "intercept_only"]
                else 0.0
            )
            total_loss = nll + reg_gamma + reg_w

            r = (p - y) / n
            r_slope = (
                r * x1 * a_x * active_mask if self.mode in ["full", "slope_only"] else r * x1 * a_x
            )
            grad_alpha0 = float(np.sum(r_slope))
            grad_gamma = np.dot(X_norm[:, slope_idx].T, r_slope) + (gamma / self.C_slope)
            grad_b0 = float(np.sum(r))
            grad_w = np.dot(X_norm[:, int_idx].T, r) + (w / self.C_intercept)

            if self.mode == "1d_platt":
                grad = np.array([grad_alpha0, grad_b0])
            elif self.mode == "slope_only":
                grad = np.concatenate([[grad_alpha0], grad_gamma, [grad_b0]])
            elif self.mode == "intercept_only":
                grad = np.concatenate([[grad_alpha0], [grad_b0], grad_w])
            else:
                grad = np.concatenate([[grad_alpha0], grad_gamma, [grad_b0], grad_w])

            return float(total_loss), grad

        if self.mode == "1d_platt":
            x0 = np.array([init_alpha0, init_b0])
        elif self.mode == "slope_only":
            x0 = np.concatenate([[init_alpha0], np.zeros(k_slope), [init_b0]])
        elif self.mode == "intercept_only":
            x0 = np.concatenate([[init_alpha0], [init_b0], np.zeros(k_int)])
        else:
            x0 = np.concatenate([[init_alpha0], np.zeros(k_slope), [init_b0], np.zeros(k_int)])

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
        self.alpha0 = float(p_opt[0])
        self.a0 = float(np.exp(self.alpha0))
        if self.mode == "full":
            self.gamma, self.b0, self.w = (
                p_opt[1 : 1 + k_slope],
                float(p_opt[1 + k_slope]),
                p_opt[2 + k_slope :],
            )
        elif self.mode == "slope_only":
            self.gamma, self.b0, self.w = (
                p_opt[1 : 1 + k_slope],
                float(p_opt[1 + k_slope]),
                np.zeros(k_int),
            )
        elif self.mode == "intercept_only":
            self.gamma, self.b0, self.w = np.zeros(k_slope), float(p_opt[1]), p_opt[2:]
        else:
            self.gamma, self.b0, self.w = np.zeros(k_slope), float(p_opt[1]), np.zeros(k_int)
        return self

    def compute_dynamic_slope(self, X: np.ndarray) -> np.ndarray:
        """Calculates dynamic slope a(z) for each sample."""
        X_norm = np.asarray(self.scaler.transform(X), dtype=np.float64)
        if self.mode in ["full", "slope_only"] and self.gamma is not None:
            slope_log = np.clip(
                self.alpha0 + np.dot(X_norm[:, self._slope_idx], self.gamma), -3.0, 3.0
            )
            return np.asarray(np.exp(slope_log))
        return np.full(len(X), np.exp(self.alpha0))

    def compute_dynamic_intercept(self, X: np.ndarray) -> np.ndarray:
        """Calculates dynamic intercept b(z) for each sample."""
        X_norm = np.asarray(self.scaler.transform(X), dtype=np.float64)
        if self.mode in ["full", "intercept_only"] and self.w is not None:
            return np.asarray(self.b0 + np.dot(X_norm[:, self._int_idx], self.w))
        return np.full(len(X), self.b0)

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Computes calibrated logits: a(z) * x1 + b(z)."""
        X_arr = np.asarray(X, dtype=np.float64)
        x1 = X_arr[:, 0]
        a_x = self.compute_dynamic_slope(X_arr)
        b_x = self.compute_dynamic_intercept(X_arr)
        return a_x * x1 + b_x

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Computes calibrated probabilities: sigma(a(z) * x1 + b(z))."""
        return sigmoid(self._predict_logits(X_test))

    def get_effective_temperature(self, X: np.ndarray) -> np.ndarray:
        """Returns instance effective temperature T_eff(z) = 1 / a(z)."""
        a_x = self.compute_dynamic_slope(X)
        return 1.0 / np.maximum(a_x, 1e-4)
