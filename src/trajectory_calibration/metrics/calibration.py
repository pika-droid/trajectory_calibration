"""
Unified Calibration and Uncertainty Evaluation Metrics.

Exposes ECE, Adaptive ECE, KDE-ECE, MCE, Brier Score, Murphy Decomposition,
AUROC, NLL, Prediction Std, and Bootstrap Confidence Intervals.
"""

from trajectory_calibration.metrics.ece import (
    compute_adaptive_ece,
    compute_ece,
    compute_kde_ece,
    compute_mce,
)
from trajectory_calibration.metrics.murphy import compute_murphy_brier_decomposition
from trajectory_calibration.metrics.scoring import (
    compute_auroc,
    compute_brier,
    compute_nll,
    compute_prediction_std,
)
from trajectory_calibration.metrics.statistical import (
    bootstrap_ci,
    fit_calibration_slope_intercept,
)

__all__ = [
    "compute_ece",
    "compute_mce",
    "compute_adaptive_ece",
    "compute_kde_ece",
    "compute_brier",
    "compute_nll",
    "compute_prediction_std",
    "compute_auroc",
    "compute_murphy_brier_decomposition",
    "fit_calibration_slope_intercept",
    "bootstrap_ci",
]
