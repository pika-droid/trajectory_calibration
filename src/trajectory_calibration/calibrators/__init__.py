"""
Trajectory Calibration Estimators Module.
"""

from trajectory_calibration.calibrators.adaptation import (
    BetaCalibrator,
    apply_beta_calibration,
    fit_beta_calibration,
    fit_target_intercept_adaptation,
    run_saerens_em_binary,
    safe_clip_probs,
)
from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    MultiScaleEigenVariance,
    MultiScaleSemanticConsistency,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    ProbabilityMarginEstimator,
    QuadraticPlattScaler,
    SplineCalibrator,
    TemperatureScalingEstimator,
    TrajectoryLREstimator,
    TrajectoryPlattScaler,
)
from trajectory_calibration.calibrators.residual import (
    ResidualTrajectoryCalibrator,
    compute_aurc,
    evaluate_full_metric_panel,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler

__all__ = [
    "AdaptiveTemperatureScaling",
    "BetaCalibrator",
    "MultiScaleEigenVariance",
    "MultiScaleSemanticConsistency",
    "NaiveConfidenceEstimator",
    "PlattScalingEstimator",
    "ProbabilityMarginEstimator",
    "QuadraticPlattScaler",
    "ResidualTrajectoryCalibrator",
    "SplineCalibrator",
    "TemperatureScalingEstimator",
    "TrajectoryLREstimator",
    "TrajectoryPlattScaler",
    "VaryingCoefficientPlattScaler",
    "apply_beta_calibration",
    "compute_aurc",
    "evaluate_full_metric_panel",
    "fit_beta_calibration",
    "fit_target_intercept_adaptation",
    "run_saerens_em_binary",
    "safe_clip_probs",
]
