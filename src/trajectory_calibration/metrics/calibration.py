"""
Evaluation metrics for model calibration and uncertainty estimation.

Implements ECE, Adaptive ECE, Kernel Density ECE, Brier Score,
Murphy Brier Decomposition (Uncertainty - Resolution + Reliability),
AUROC, NLL, and Empirical Bootstrap Confidence Intervals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width binning.

    Formula: ECE = sum_k (|B_k| / N) * |acc(B_k) - conf(B_k)|
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_mce(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
    """
    Maximum Calibration Error (MCE) across equal-width bins.

    Formula: MCE = max_k |acc(B_k) - conf(B_k)|
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if np.any(in_bin):
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = max(mce, error)

    return float(mce)


def compute_brier(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """
    Brier Score (Mean Squared Error between probabilities and binary outcomes).

    Formula: (1/N) * sum_i (c_i - y_i)^2
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)
    if len(confidences) == 0:
        return 0.0
    return float(np.mean((confidences - accuracies) ** 2))


def compute_nll(confidences: np.ndarray, accuracies: np.ndarray, eps: float = 1e-12) -> float:
    """
    Binary Negative Log-Likelihood (Log Loss / Cross Entropy).

    Formula: - (1/N) * sum_i [y_i * ln(c_i) + (1 - y_i) * ln(1 - c_i)]
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)
    if len(confidences) == 0:
        return 0.0
    c = np.clip(confidences, eps, 1.0 - eps)
    return float(-np.mean(accuracies * np.log(c) + (1.0 - accuracies) * np.log(1.0 - c)))


def compute_prediction_std(probs: np.ndarray) -> float:
    """
    Standard deviation of predicted probabilities.

    Used as a collapse diagnostic: sigma_p < 0.02 indicates prediction collapse.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if len(probs) == 0:
        return 0.0
    return float(np.std(probs))


def compute_murphy_brier_decomposition(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """
    Murphy (1973) Brier Score Decomposition.

    Decomposes Brier into three orthogonal, scientifically interpretable components:
      Brier = Uncertainty - Resolution + Reliability

    - Uncertainty: U = base_rate * (1 - base_rate)
    - Resolution:  Res = sum_k (N_k/N) * (acc_k - base_rate)^2 (sorting discrimination)
    - Reliability: Rel = sum_k (N_k/N) * (conf_k - acc_k)^2 (calibration error)
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return {"brier": 0.0, "uncertainty": 0.0, "resolution": 0.0, "reliability": 0.0}

    base_rate = np.mean(y)
    uncertainty = base_rate * (1.0 - base_rate)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    resolution = 0.0
    reliability = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        n_k = np.sum(in_bin)
        if n_k > 0:
            prop_k = n_k / n
            acc_k = np.mean(y[in_bin])
            conf_k = np.mean(probs[in_bin])

            resolution += prop_k * ((acc_k - base_rate) ** 2)
            reliability += prop_k * ((conf_k - acc_k) ** 2)

    brier = uncertainty - resolution + reliability

    return {
        "brier": float(brier),
        "uncertainty": float(uncertainty),
        "resolution": float(resolution),
        "reliability": float(reliability),
    }


def compute_adaptive_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """
    Adaptive ECE with equal-mass (quantile) binning.

    Ensures every bin has approximately N / n_bins samples, eliminating distortion
    from sparse bins in equal-width ECE.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return 0.0

    # Determine quantile bin edges
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.percentile(probs, quantiles)
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0

    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1
    if actual_bins <= 0:
        return float(np.abs(np.mean(probs) - np.mean(y)))

    ada_ece = 0.0
    for i in range(actual_bins):
        b_low, b_high = bin_edges[i], bin_edges[i + 1]
        if i == actual_bins - 1:
            in_bin = (probs >= b_low) & (probs <= b_high)
        else:
            in_bin = (probs >= b_low) & (probs < b_high)

        n_k = np.sum(in_bin)
        if n_k > 0:
            prop_k = n_k / n
            acc_k = np.mean(y[in_bin])
            conf_k = np.mean(probs[in_bin])
            ada_ece += prop_k * np.abs(conf_k - acc_k)

    return float(ada_ece)


def compute_kde_ece(probs: np.ndarray, y: np.ndarray, n_grid: int = 100) -> float:
    """
    Continuous Kernel Density ECE using Gaussian kernel density estimation.

    Integrates calibration error |hat{y}(p) - p| weighted by probability density f(p).
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(probs)
    if n < 5 or np.all(probs == probs[0]):
        return float(np.abs(np.mean(probs) - np.mean(y)))

    # Scott rule bandwidth with lower bound
    std = np.std(probs)
    if std < 1e-6:
        return float(np.abs(np.mean(probs) - np.mean(y)))
    bandwidth = max(std * (n ** (-0.2)), 0.01)

    grid = np.linspace(0.001, 0.999, n_grid)
    kde_ece = 0.0

    # Vectorized Gaussian kernel weights
    diff = grid[:, None] - probs[None, :]  # (n_grid, n)
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)  # (n_grid, n)
    sum_weights = np.sum(weights, axis=1) + 1e-12

    # Local accuracy estimate
    local_acc = np.sum(weights * y[None, :], axis=1) / sum_weights
    density = sum_weights / (n * bandwidth * np.sqrt(2 * np.pi))
    density_norm = density / np.sum(density)

    kde_ece = np.sum(density_norm * np.abs(local_acc - grid))
    return float(kde_ece)


def compute_auroc(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Area Under the Receiver Operating Characteristic Curve (AUROC).

    Measures model discrimination power (ranking correct vs incorrect answers).
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, probs))
    except Exception:
        return 0.5


def fit_calibration_slope_intercept(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    """
    Calculates calibration slope and intercept via logistic regression on logit(p).

    Ideal calibration: Intercept alpha = 0.0, Slope beta = 1.0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return 0.0, 1.0

    c = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(c / (1.0 - c)).reshape(-1, 1)

    try:
        lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=200)
        lr.fit(logits, y)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        return slope, intercept
    except Exception:
        return 1.0, 0.0


def bootstrap_ci(
    metric_fn,
    probs: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Empirical bootstrap confidence interval for any calibration metric.

    Returns: (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(probs)
    if n == 0:
        return 0.0, 0.0

    bootstrap_values = []
    alpha = (1.0 - ci) / 2.0

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample_probs = probs[idx]
        sample_y = y[idx]
        val = metric_fn(sample_probs, sample_y)
        bootstrap_values.append(val)

    lower = float(np.percentile(bootstrap_values, 100.0 * alpha))
    upper = float(np.percentile(bootstrap_values, 100.0 * (1.0 - alpha)))
    return lower, upper
