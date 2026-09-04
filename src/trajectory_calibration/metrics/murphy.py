"""
Murphy (1973) Brier Score Decomposition.

Decomposes the Brier score into three orthogonal, scientifically interpretable components:
    Brier = Uncertainty - Resolution + Reliability
"""

from __future__ import annotations

import numpy as np


def compute_murphy_brier_decomposition(
    probs: np.ndarray | list[float], y: np.ndarray | list[float], n_bins: int = 15
) -> dict[str, float]:
    """
    Murphy (1973) Brier Score Decomposition.

    - Uncertainty: U = base_rate * (1 - base_rate) (inherent problem difficulty)
    - Resolution:  Res = sum_k (N_k/N) * (acc_k - base_rate)^2 (sorting discrimination)
    - Reliability: Rel = sum_k (N_k/N) * (conf_k - acc_k)^2 (calibration error)
    """
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {"brier": 0.0, "uncertainty": 0.0, "resolution": 0.0, "reliability": 0.0}

    base_rate = np.mean(labels)
    uncertainty = base_rate * (1.0 - base_rate)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    resolution = 0.0
    reliability = 0.0
    within = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (p >= bin_lower) & (p <= bin_upper)
        else:
            in_bin = (p >= bin_lower) & (p < bin_upper)

        n_k = np.sum(in_bin)
        if n_k > 0:
            prop_k = n_k / n
            acc_k = np.mean(labels[in_bin])
            conf_k = np.mean(p[in_bin])

            resolution += prop_k * ((acc_k - base_rate) ** 2)
            reliability += prop_k * ((conf_k - acc_k) ** 2)
            within += np.sum((p[in_bin] - conf_k) ** 2) / n

    brier = uncertainty - resolution + reliability + within

    return {
        "brier": float(brier),
        "uncertainty": float(uncertainty),
        "resolution": float(resolution),
        "reliability": float(reliability),
        "within": float(within),
    }
