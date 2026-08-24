"""
Trajectory Calibration Estimators Module.
"""

from trajectory_calibration.calibrators.adaptation import (
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
    SplineCalibrator,
    TemperatureScalingEstimator,
)
from trajectory_calibration.calibrators.classic import (
    AdaptiveTemperatureScaling,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    SplineCalibrator,
    TemperatureScalingEstimator,
)
from trajectory_calibration.calibrators.proxies import (
    MultiScaleEigenVariance,
    MultiScaleSemanticConsistency,
    ProbabilityMarginEstimator,
)
from trajectory_calibration.calibrators.residual import (
    ResidualTrajectoryCalibrator,
    compute_aurc,
    evaluate_full_metric_panel,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler

__all__ = [
    "NaiveConfidenceEstimator",
    "TemperatureScalingEstimator",
    "PlattScalingEstimator",
    "SplineCalibrator",
    "AdaptiveTemperatureScaling",
    "ProbabilityMarginEstimator",
    "MultiScaleSemanticConsistency",
    "MultiScaleEigenVariance",
    "VaryingCoefficientPlattScaler",
    "ResidualTrajectoryCalibrator",
    "compute_aurc",
    "evaluate_full_metric_panel",
    "run_saerens_em_binary",
    "fit_target_intercept_adaptation",
    "fit_beta_calibration",
    "apply_beta_calibration",
    "safe_clip_probs",
]
