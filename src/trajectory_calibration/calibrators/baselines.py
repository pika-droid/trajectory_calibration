"""
Baseline Calibrators and Uncertainty Quantification Estimators Facade.

Re-exports classic post-hoc calibrators (NC, TS, Platt, Spline, ATS) and
genuine single-pass multi-scale trajectory proxies (MSSC, MSE-EIGEN, Margin).
"""

from trajectory_calibration.calibrators.adaptation import BetaCalibrator
from trajectory_calibration.calibrators.classic import (
    AdaptiveTemperatureScaling,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    PolynomialCalibrator,
    QuadraticPlattScaler,
    SplineCalibrator,
    TemperatureScalingEstimator,
    TrajectoryLREstimator,
)
from trajectory_calibration.calibrators.proxies import (
    MultiScaleEigenVariance,
    MultiScaleSemanticConsistency,
    ProbabilityMarginEstimator,
)
from trajectory_calibration.calibrators.trajectory_platt import TrajectoryPlattScaler

__all__ = [
    "NaiveConfidenceEstimator",
    "TemperatureScalingEstimator",
    "PlattScalingEstimator",
    "QuadraticPlattScaler",
    "PolynomialCalibrator",
    "SplineCalibrator",
    "AdaptiveTemperatureScaling",
    "TrajectoryLREstimator",
    "TrajectoryPlattScaler",
    "BetaCalibrator",
    "ProbabilityMarginEstimator",
    "MultiScaleSemanticConsistency",
    "MultiScaleEigenVariance",
]
