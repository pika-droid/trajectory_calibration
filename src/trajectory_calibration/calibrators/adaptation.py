"""
Domain adaptation and prior shift correction routines.

Implements Saerens EM (2002) for unsupervised target prior shift adaptation,
target intercept shifting, and Beta calibration (Kull et al., 2017).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, root_scalar

from trajectory_calibration.metrics.scoring import compute_nll
from trajectory_calibration.utils.math import get_logits, safe_clip_probs, sigmoid


def run_saerens_em_binary(
    p_source: np.ndarray | list[float], pi_source: float, max_iter: int = 100, tol: float = 1e-6
) -> tuple[np.ndarray, float]:
    """
    Saerens et al. (2002) Expectation-Maximization for unsupervised target prior shift adaptation.
    """
    p_s = safe_clip_probs(np.asarray(p_source, dtype=np.float64))
    pi_s = float(np.clip(pi_source, 1e-4, 1.0 - 1e-4))
    pi_t = float(np.mean(p_s))

    for _ in range(max_iter):
        w1 = pi_t / pi_s
        w0 = (1.0 - pi_t) / (1.0 - pi_s)

        num = w1 * p_s
        den = num + w0 * (1.0 - p_s)
        q = safe_clip_probs(num / den)

        pi_t_new = float(np.mean(q))
        if abs(pi_t_new - pi_t) < tol:
            pi_t = pi_t_new
            break
        pi_t = pi_t_new

    w1 = pi_t / pi_s
    w0 = (1.0 - pi_t) / (1.0 - pi_s)
    num = w1 * p_s
    den = num + w0 * (1.0 - p_s)
    p_adapted = safe_clip_probs(num / den)

    return p_adapted, pi_t


def fit_target_intercept_adaptation(
    preds_source: np.ndarray | list[float], target_unlabeled_guess: float | np.ndarray
) -> np.ndarray:
    """
    Shifts the calibrated logit intercept to align mean probability with the target base rate.
    Uses Brent's root-finding method on [-30.0, 30.0].
    """
    preds_s = safe_clip_probs(np.asarray(preds_source, dtype=np.float64))
    source_logits = get_logits(preds_s)
    target_mean = float(np.mean(target_unlabeled_guess))

    def f(alpha_d: float) -> float:
        shifted_p = sigmoid(source_logits + alpha_d)
        return float(np.mean(shifted_p) - target_mean)

    f_neg = f(-30.0)
    f_pos = f(30.0)
    if f_neg * f_pos <= 0:
        sol = root_scalar(f, bracket=[-30.0, 30.0], method="brentq")
        best_alpha = float(sol.root)
    elif f_neg > 0:
        best_alpha = -30.0
    else:
        best_alpha = 30.0

    return sigmoid(source_logits + best_alpha)


def fit_beta_calibration(
    confs_train: np.ndarray | list[float], y_train: np.ndarray | list[int | float]
) -> tuple[float, float, float]:
    """
    Fits Beta Calibration (Kull et al., 2017): logit(p) = a * ln(c) - b * ln(1 - c) + c_param.
    Monotonicity constraint requires a >= 0 and b >= 0.
    """
    c = safe_clip_probs(np.asarray(confs_train, dtype=np.float64))
    y = np.asarray(y_train, dtype=np.float64)
    n = len(c)
    if n == 0:
        return 1.0, 1.0, 0.0

    z1 = np.log(c)
    z2 = -np.log(1.0 - c)

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        a_val, b_val, c_p = params[0], params[1], params[2]
        logits = a_val * z1 + b_val * z2 + c_p
        p = sigmoid(logits)
        loss = compute_nll(p, y)
        r = (p - y) / n
        grad_a = float(np.dot(z1, r))
        grad_b = float(np.dot(z2, r))
        grad_c = float(np.sum(r))
        return loss, np.array([grad_a, grad_b, grad_c])

    res = minimize(
        loss_and_grad,
        x0=np.array([1.0, 1.0, 0.0]),
        jac=True,
        bounds=[(0.0, None), (0.0, None), (None, None)],
        method="L-BFGS-B",
    )
    a = float(max(res.x[0], 0.0))
    b = float(max(res.x[1], 0.0))
    c_param = float(res.x[2])
    return a, b, c_param


def apply_beta_calibration(
    confs: np.ndarray | list[float], a: float, b: float, c_param: float
) -> np.ndarray:
    """Applies Beta calibration parameters to input confidences."""
    c = safe_clip_probs(np.asarray(confs, dtype=np.float64))
    calib_logits = a * np.log(c) - b * np.log(1.0 - c) + c_param
    return sigmoid(calib_logits)


class BetaCalibrator:
    """Beta Calibration estimator (Kull et al., 2017)."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 1.0
        self.c_param: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> BetaCalibrator:
        x1 = X_train[:, 0] if X_train.ndim == 2 else X_train
        confs = sigmoid(x1)
        self.a, self.b, self.c_param = fit_beta_calibration(confs, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x1 = X_test[:, 0] if X_test.ndim == 2 else X_test
        confs = sigmoid(x1)
        return apply_beta_calibration(confs, self.a, self.b, self.c_param)
