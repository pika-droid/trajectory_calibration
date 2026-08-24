"""
Domain adaptation and prior shift correction routines.

Implements Saerens EM (2002) for unsupervised target prior shift adaptation,
target intercept shifting, and Beta calibration (Kull et al., 2017).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from trajectory_calibration.features.trajectory import get_logits, sigmoid


def safe_clip_probs(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Clips probabilities to [eps, 1 - eps] for numerical stability."""
    return np.clip(p, eps, 1.0 - eps)


def run_saerens_em_binary(
    p_source: np.ndarray, pi_source: float, max_iter: int = 100, tol: float = 1e-6
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
    preds_source: np.ndarray, target_unlabeled_guess: float | np.ndarray
) -> np.ndarray:
    """
    Shifts the calibrated logit intercept to align mean probability with the target base rate.
    """
    preds_s = safe_clip_probs(np.asarray(preds_source, dtype=np.float64))
    source_logits = get_logits(preds_s)
    target_mean = float(np.mean(target_unlabeled_guess))

    def obj(alpha_d: float) -> float:
        shifted_p = sigmoid(source_logits + alpha_d)
        return float((np.mean(shifted_p) - target_mean) ** 2)

    res = minimize_scalar(obj, bounds=(-10.0, 10.0), method="bounded")
    best_alpha = float(res.x)
    return sigmoid(source_logits + best_alpha)


def fit_beta_calibration(
    confs_train: np.ndarray, y_train: np.ndarray
) -> tuple[float, float, float]:
    """
    Fits Beta Calibration (Kull et al., 2017): logit(p) = a * ln(c) - b * ln(1 - c) + c_param.
    """
    c = safe_clip_probs(np.asarray(confs_train, dtype=np.float64))
    y = np.asarray(y_train, dtype=np.int64)

    X_beta = np.column_stack([np.log(c), -np.log(1.0 - c)])
    lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=200)
    lr.fit(X_beta, y)

    a = float(lr.coef_[0][0])
    b = float(lr.coef_[0][1])
    c_param = float(lr.intercept_[0])
    return a, b, c_param


def apply_beta_calibration(
    confs: np.ndarray, a: float, b: float, c_param: float
) -> np.ndarray:
    """Applies Beta calibration parameters to input confidences."""
    c = safe_clip_probs(np.asarray(confs, dtype=np.float64))
    calib_logits = a * np.log(c) - b * np.log(1.0 - c) + c_param
    return sigmoid(calib_logits)
