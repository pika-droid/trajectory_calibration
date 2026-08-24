"""
Expected Calibration Error (ECE) and variant binning metrics.

Implements equal-width ECE, Maximum Calibration Error (MCE),
Adaptive (equal-mass quantile) ECE, and Continuous Kernel Density ECE.
"""

from __future__ import annotations

import numpy as np


def compute_ece(confidences: np.ndarray | list[float], accuracies: np.ndarray | list[float], n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width binning.

    Formula: ECE = sum_k (|B_k| / N) * |acc(B_k) - conf(B_k)|
    """
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)

    if len(confs) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accs[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_mce(confidences: np.ndarray | list[float], accuracies: np.ndarray | list[float], n_bins: int = 15) -> float:
    """
    Maximum Calibration Error (MCE) across equal-width bins.

    Formula: MCE = max_k |acc(B_k) - conf(B_k)|
    """
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)

    if len(confs) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        if np.any(in_bin):
            accuracy_in_bin = np.mean(accs[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = max(mce, error)

    return float(mce)


def compute_adaptive_ece(probs: np.ndarray | list[float], y: np.ndarray | list[float], n_bins: int = 15) -> float:
    """
    Adaptive ECE with equal-mass (quantile) binning.

    Ensures every bin has approximately N / n_bins samples, eliminating distortion
    from sparse bins in equal-width ECE.
    """
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0

    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.percentile(p, quantiles)
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0

    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1
    if actual_bins <= 0:
        return float(np.abs(np.mean(p) - np.mean(labels)))

    ada_ece = 0.0
    for i in range(actual_bins):
        b_low, b_high = bin_edges[i], bin_edges[i + 1]
        if i == actual_bins - 1:
            in_bin = (p >= b_low) & (p <= b_high)
        else:
            in_bin = (p >= b_low) & (p < b_high)

        n_k = np.sum(in_bin)
        if n_k > 0:
            prop_k = n_k / n
            acc_k = np.mean(labels[in_bin])
            conf_k = np.mean(p[in_bin])
            ada_ece += prop_k * np.abs(conf_k - acc_k)

    return float(ada_ece)


def compute_kde_ece(probs: np.ndarray | list[float], y: np.ndarray | list[float], n_grid: int = 100) -> float:
    """
    Continuous Kernel Density ECE using Gaussian kernel density estimation.

    Integrates calibration error |hat{y}(p) - p| weighted by probability density f(p).
    """
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n < 5 or np.all(p == p[0]):
        return float(np.abs(np.mean(p) - np.mean(labels)))

    std = np.std(p)
    if std < 1e-6:
        return float(np.abs(np.mean(p) - np.mean(labels)))
    bandwidth = max(std * (n ** (-0.2)), 0.01)

    grid = np.linspace(0.001, 0.999, n_grid)

    diff = grid[:, None] - p[None, :]
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
    sum_weights = np.sum(weights, axis=1) + 1e-12

    local_acc = np.sum(weights * labels[None, :], axis=1) / sum_weights
    density = sum_weights / (n * bandwidth * np.sqrt(2 * np.pi))
    density_norm = density / np.sum(density)

    kde_ece = np.sum(density_norm * np.abs(local_acc - grid))
    return float(kde_ece)
