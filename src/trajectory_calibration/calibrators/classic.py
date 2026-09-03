"""
Standard post-hoc calibration estimators.

Provides scikit-learn style estimators for:
- Naive Confidence (NC)
- Global Temperature Scaling (TS / Guo et al., 2017)
- 1D Platt Scaling (Platt, 1999)
- Monotonic Spline Calibration (PCHIP / Gupta et al., 2020)
- Adaptive Temperature Scaling (ATS / Thermometer, 2024)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.metrics.scoring import compute_nll
from trajectory_calibration.utils.math import sigmoid


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
        self.t_min, self.t_max, self.T_opt = t_min, t_max, 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TemperatureScalingEstimator:
        x1 = X_train[:, 0] if X_train.ndim == 2 else X_train
        y = np.asarray(y_train, dtype=np.float64)

        def nll_obj(T_val: np.ndarray) -> float:
            return compute_nll(sigmoid(x1 / float(T_val[0])), y)

        res = minimize(nll_obj, x0=np.array([1.0]), bounds=[(self.t_min, self.t_max)], method="L-BFGS-B")
        self.T_opt = float(res.x[0]) if res.success else 1.0
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        return sigmoid(x1 / self.T_opt)


class PlattScalingEstimator:
    """1D Platt Scaling: fits logistic regression logit(p) = w * x1 + b."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C, self.random_state = C, random_state
        self.lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000, random_state=self.random_state)

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
        knots_x = np.unique(np.percentile(confs, quantiles))
        if len(knots_x) < 2:
            knots_x = np.array([0.0, 1.0])

        knots_y = [
            np.mean(y[(confs >= knots_x[i]) & (confs <= knots_x[i + 1])])
            if np.any((confs >= knots_x[i]) & (confs <= knots_x[i + 1]))
            else (knots_x[i] + knots_x[i + 1]) / 2.0
            for i in range(len(knots_x) - 1)
        ]
        knots_y.append(knots_y[-1])
        knots_y = np.maximum.accumulate(np.clip(knots_y, 0.001, 0.999))
        self.spline = PchipInterpolator(knots_x, knots_y, extrapolate=True)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        confs = sigmoid(x1)
        return confs if self.spline is None else np.clip(self.spline(confs), 0.001, 0.999)


class AdaptiveTemperatureScaling:
    """Adaptive Temperature Scaling: instance temperature ln T(z) = w^T z + b."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C, self.random_state = C, random_state
        self.scaler = StandardScaler()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> AdaptiveTemperatureScaling:
        X, y = np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64)
        x1 = X[:, 0]
        Z = X[:, 1:] if X.shape[1] > 1 else X
        Z_norm = self.scaler.fit_transform(Z)
        d = Z_norm.shape[1]

        def loss_fn(params: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = params[:d], params[d]
            log_T = np.clip(np.dot(Z_norm, w) + b, -3.0, 3.0)
            T = np.exp(log_T)
            p = sigmoid(x1 / T)
            r = (p - y) / len(y)
            grad_w = np.dot(Z_norm.T, r * (-x1 / T)) + (w / self.C)
            grad_b = np.sum(r * (-x1 / T))
            total_loss = compute_nll(p, y) + (0.5 / self.C) * np.sum(w ** 2)
            return float(total_loss), np.concatenate([grad_w, [grad_b]])

        res = minimize(loss_fn, x0=np.zeros(d + 1), jac=True, method="L-BFGS-B")
        self.weights, self.bias = res.x[:d], float(res.x[d])
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X = np.asarray(X_test, dtype=np.float64)
        x1 = X[:, 0]
        Z = X[:, 1:] if X.shape[1] > 1 else X
        Z_norm = self.scaler.transform(Z)
        log_T = np.clip(np.dot(Z_norm, self.weights) + self.bias, -3.0, 3.0)
        return sigmoid(x1 / np.exp(log_T))


class QuadraticPlattScaler:
    """Quadratic / Logit-Only Platt Scaling: logit(p) = gamma * x1^2 + (a0 + w) * x1 + b0."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C, self.random_state = C, random_state
        self.lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000, random_state=self.random_state)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        x1 = arr[:, 0] if arr.ndim == 2 else arr
        return np.column_stack([x1, x1 ** 2])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> QuadraticPlattScaler:
        self.lr.fit(self._transform(X_train), np.asarray(y_train, dtype=np.float64))
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._transform(X_test))[:, 1]

    def compute_dynamic_slope(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        x1 = arr[:, 0] if arr.ndim == 2 else arr
        return float(self.lr.coef_[0][0]) + 2.0 * float(self.lr.coef_[0][1]) * x1

    def compute_dynamic_intercept(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return np.full(len(arr) if arr.ndim == 2 else 1, float(self.lr.intercept_[0]))

    def get_effective_temperature(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / np.maximum(np.abs(self.compute_dynamic_slope(X)), 1e-4)



class TrajectoryLREstimator:
    """Trajectory Logistic Regression: raw logit anchor x1 + standardized trajectory features z."""

    def __init__(
        self,
        fit_intercept: bool = True,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.C = float(C)
        self.random_state = random_state
        self._scaler = ColumnTransformer(
            transformers=[
                ("logit", "passthrough", [0]),
                ("traj", StandardScaler(with_mean=self.fit_intercept), slice(1, None)),
            ],
            remainder="drop",
        )
        self._lr: LogisticRegression | None = None

    @property
    def lr(self) -> LogisticRegression | None:
        return self._lr

    @lr.setter
    def lr(self, val: LogisticRegression | None) -> None:
        self._lr = val

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TrajectoryLREstimator:
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_trans = self._scaler.fit_transform(X) if X.shape[1] > 1 else X
        self._lr = LogisticRegression(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state,
        ).fit(X_trans, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self._lr is None:
            raise RuntimeError("TrajectoryLREstimator must be fitted before predict_proba.")
        X_trans = self._scaler.transform(X) if X.shape[1] > 1 else X
        return self._lr.predict_proba(X_trans)[:, 1]
