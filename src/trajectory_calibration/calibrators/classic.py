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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
        self.t_min = t_min
        self.t_max = t_max
        self.T_opt = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TemperatureScalingEstimator:
        x1 = X_train[:, 0] if X_train.ndim == 2 else X_train
        y = np.asarray(y_train, dtype=np.float64)

        def nll_obj(T_val: np.ndarray) -> float:
            return compute_nll(sigmoid(x1 / float(T_val[0])), y)

        res = minimize(
            nll_obj, x0=np.array([1.0]), bounds=[(self.t_min, self.t_max)], method="L-BFGS-B"
        )
        self.T_opt = float(res.x[0]) if res.success else 1.0
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        return sigmoid(x1 / self.T_opt)


class PlattScalingEstimator:
    """1D Platt Scaling: fits logistic regression logit(p) = w * x1 + b."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C, self.random_state = C, random_state
        self.lr = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=1000, random_state=self.random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> PlattScalingEstimator:
        x1 = X_train[:, [0]] if X_train.ndim == 2 else X_train.reshape(-1, 1)
        self.lr.fit(x1, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, [0]] if X_test.ndim == 2 else X_test.reshape(-1, 1)
        return self.lr.predict_proba(x1)[:, 1]


class SplineCalibrator:
    """Monotonic cubic spline calibration via PCHIP (Gupta et al., 2020)."""

    def __init__(self, n_knots: int = 5, y_min: float = 0.0, y_max: float = 1.0) -> None:
        self.n_knots = n_knots
        self.y_min = y_min
        self.y_max = y_max
        self.spline: PchipInterpolator | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SplineCalibrator:
        x1 = X_train[:, 0] if X_train.ndim == 2 else X_train
        confs = sigmoid(x1)
        y = np.asarray(y_train, dtype=np.float64)

        n = len(confs)
        k = max(2, self.n_knots)

        mean_y = float(np.mean(y)) if n > 0 else 0.5

        if n < 2:
            x_knots = np.linspace(0.0, 1.0, k + 1)
            y_knots = np.full(k + 1, mean_y)
            self.spline = PchipInterpolator(x_knots, y_knots)
            return self

        # 1. Partition [0, 1] into K knot bins using empirical quantiles of training confidences
        quantiles = np.linspace(0.0, 1.0, k + 1)
        raw_knots = np.quantile(confs, quantiles)

        # Strictly enforce x_0 = 0.0, x_K = 1.0, and x_0 < x_1 < ... < x_K
        eps = 1e-4
        x_knots = np.zeros(k + 1)
        x_knots[0] = 0.0
        x_knots[-1] = 1.0

        # Forward pass: ensure x_knots[i] >= x_knots[i-1] + eps
        for i in range(1, k):
            x_knots[i] = max(raw_knots[i], x_knots[i - 1] + eps)

        # Backward pass: ensure x_knots[i] <= x_knots[i+1] - eps
        for i in range(k - 1, 0, -1):
            x_knots[i] = min(x_knots[i], x_knots[i + 1] - eps)

        # Fallback to uniform grid if quantile collapse occurred
        if np.any(np.diff(x_knots) <= 0):
            x_knots = np.linspace(0.0, 1.0, k + 1)

        # 2. Estimate empirical accuracy in each bin
        bin_accs = []
        for i in range(k):
            low, high = x_knots[i], x_knots[i + 1]
            mask = (confs >= low) & (confs <= high) if i == 0 else (confs > low) & (confs <= high)
            if np.any(mask):
                bin_accs.append(float(np.mean(y[mask])))
            else:
                bin_accs.append(mean_y)

        raw_y = np.zeros(k + 1)
        raw_y[0] = bin_accs[0]
        raw_y[-1] = bin_accs[-1]
        for i in range(1, k):
            raw_y[i] = 0.5 * (bin_accs[i - 1] + bin_accs[i])

        # Apply IsotonicRegression to guarantee strict monotonicity y_0 <= y_1 <= ... <= y_K
        iso = IsotonicRegression(y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip")
        y_knots = iso.fit_transform(x_knots, raw_y)

        # 3. Fit PchipInterpolator to strictly preserve monotonicity without overshoot
        self.spline = PchipInterpolator(x_knots, y_knots)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.spline is None:
            raise RuntimeError("SplineCalibrator must be fitted before predict_proba.")
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        confs = sigmoid(x1)
        preds = self.spline(confs)
        return np.clip(preds, self.y_min, self.y_max)


class AdaptiveTemperatureScaling:
    """Adaptive Temperature Scaling: instance temperature ln T(z) = w^T z + b."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C, self.random_state = C, random_state
        self.scaler = StandardScaler()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.ts_fallback: TemperatureScalingEstimator | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> AdaptiveTemperatureScaling:
        X, y = np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64)
        x1 = X[:, 0] if X.ndim == 2 else X
        if X.ndim == 1 or X.shape[1] <= 1:
            self.ts_fallback = TemperatureScalingEstimator().fit(x1, y)
            self.bias = float(np.log(self.ts_fallback.T_opt))
            self.weights = np.zeros(0)
            return self

        # Initialize bias to scalar log-temperature from standard temperature scaling
        base_ts = TemperatureScalingEstimator().fit(x1, y)
        init_b = float(np.log(base_ts.T_opt))

        Z = X[:, 1:]
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
            total_loss = compute_nll(p, y) + (0.5 / self.C) * np.sum(w**2)
            return float(total_loss), np.concatenate([grad_w, [grad_b]])

        x0 = np.concatenate([np.zeros(d), [init_b]])
        res = minimize(loss_fn, x0=x0, jac=True, method="L-BFGS-B")
        self.weights, self.bias = res.x[:d], float(res.x[d])
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.ts_fallback is not None:
            return self.ts_fallback.predict_proba(X_test)
        X = np.asarray(X_test, dtype=np.float64)
        x1 = X[:, 0] if X.ndim == 2 else X
        if X.ndim == 1 or X.shape[1] <= 1:
            return sigmoid(x1 / np.exp(self.bias))
        Z = X[:, 1:]
        Z_norm = np.asarray(self.scaler.transform(Z), dtype=np.float64)
        weights = self.weights if self.weights is not None else np.zeros(Z.shape[1])
        log_T = np.clip(np.dot(Z_norm, weights) + self.bias, -3.0, 3.0)
        return sigmoid(x1 / np.exp(log_T))


class QuadraticPlattScaler:
    """Polynomial / Quadratic Platt Scaling: logit(p) = gamma * x1^2 + (a0 + w) * x1 + b0."""

    def __init__(self, degree: int = 2, C: float = 1.0, random_state: int = 42) -> None:
        self.degree = degree
        self.C, self.random_state = C, random_state
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.lr = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=1000, random_state=self.random_state
        )

    def _extract_x1(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return (arr[:, 0] if arr.ndim == 2 else arr).reshape(-1, 1)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.poly.transform(self._extract_x1(X)))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> QuadraticPlattScaler:
        x1 = self._extract_x1(X_train)
        X_poly = self.poly.fit_transform(x1)
        self.lr.fit(X_poly, np.asarray(y_train, dtype=np.float64))
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = self._extract_x1(X_test)
        X_poly = self.poly.transform(x1)
        return np.asarray(self.lr.predict_proba(X_poly)[:, 1])

    def compute_dynamic_slope(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        x1 = arr[:, 0] if arr.ndim == 2 else arr
        c0 = float(self.lr.coef_[0][0])
        c1 = float(self.lr.coef_[0][1]) if self.lr.coef_.shape[1] > 1 else 0.0
        return c0 + 2.0 * c1 * x1

    def compute_dynamic_intercept(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        intercept_val = float(np.asarray(self.lr.intercept_).ravel()[0])
        return np.full(len(arr) if arr.ndim == 2 else 1, intercept_val)

    def get_effective_temperature(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / np.maximum(np.abs(self.compute_dynamic_slope(X)), 1e-4)


PolynomialCalibrator = QuadraticPlattScaler


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
