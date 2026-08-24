"""Calibration methods, baselines, and varying-coefficient models."""

from trajectory_calibration.calibrators.adaptation import (
    apply_beta_calibration,
    fit_beta_calibration,
    fit_target_intercept_adaptation,
    run_saerens_em_binary,
    safe_clip_probs,
)
from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    EigenScoreEstimator,
    MinProbabilityEstimator,
    MultiScaleEigenVariance,
    MultiScaleSemanticConsistency,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    ProbabilityMarginEstimator,
    SemanticEntropyEstimator,
    SequenceProbabilityEstimator,
    SplineCalibrator,
    TemperatureScalingEstimator,
    TokenEntropyEstimator,
)
from trajectory_calibration.calibrators.residual import (
    ResidualTrajectoryCalibrator,
    compute_aurc,
    evaluate_full_metric_panel,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler

__all__ = [
    "AdaptiveTemperatureScaling",
    "EigenScoreEstimator",
    "MinProbabilityEstimator",
    "MultiScaleEigenVariance",
    "MultiScaleSemanticConsistency",
    "NaiveConfidenceEstimator",
    "PlattScalingEstimator",
    "ProbabilityMarginEstimator",
    "ResidualTrajectoryCalibrator",
    "SemanticEntropyEstimator",
    "SequenceProbabilityEstimator",
    "SplineCalibrator",
    "TemperatureScalingEstimator",
    "TokenEntropyEstimator",
    "VaryingCoefficientPlattScaler",
    "apply_beta_calibration",
    "compute_aurc",
    "evaluate_full_metric_panel",
    "fit_beta_calibration",
    "fit_target_intercept_adaptation",
    "run_saerens_em_binary",
    "safe_clip_probs",
]
