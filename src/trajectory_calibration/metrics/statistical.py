"""
Statistical validation and diagnostic estimation utilities.

Provides logistic calibration slope/intercept regression and
empirical bootstrap confidence interval estimation.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_calibration_slope_intercept(
    probs: np.ndarray | list[float], y: np.ndarray | list[int | float], eps: float = 1e-12
) -> tuple[float, float]:
    """
    Calculates calibration slope and intercept via logistic regression on logit(p).

    Ideal calibration: Intercept alpha = 0.0, Slope beta = 1.0.
    """
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return 0.0, 1.0

    c = np.clip(p, eps, 1.0 - eps)
    logits = np.log(c / (1.0 - c)).reshape(-1, 1)

    try:
        lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
        lr.fit(logits, labels)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        return slope, intercept
    except Exception:
        return 1.0, 0.0


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    probs: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Empirical bootstrap confidence interval for any calibration metric.

    Returns: (lower_bound, upper_bound)
    """
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    bootstrap_values = []
    alpha = (1.0 - ci) / 2.0

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        val = metric_fn(p[idx], labels[idx])
        bootstrap_values.append(val)

    lower = float(np.percentile(bootstrap_values, 100.0 * alpha))
    upper = float(np.percentile(bootstrap_values, 100.0 * (1.0 - alpha)))
    return lower, upper
