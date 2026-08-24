"""Tests for all post-hoc calibrators and VCPS."""

import numpy as np
import pytest
from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    SplineCalibrator,
    TemperatureScalingEstimator,
)
from trajectory_calibration.calibrators.residual import ResidualTrajectoryCalibrator
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.trajectory import generate_mock_df


def test_all_calibrators_fit_predict():
    df = generate_mock_df("test_mock", n_samples=80, seed=42)
    X = df[["x1", "x13", "x6", "x8", "x4"]].values
    y = df["is_correct"].values

    models = [
        NaiveConfidenceEstimator(),
        TemperatureScalingEstimator(),
        PlattScalingEstimator(),
        SplineCalibrator(),
        AdaptiveTemperatureScaling(),
        ResidualTrajectoryCalibrator(),
        VaryingCoefficientPlattScaler(mode="full"),
        VaryingCoefficientPlattScaler(mode="slope_only"),
        VaryingCoefficientPlattScaler(mode="intercept_only"),
        VaryingCoefficientPlattScaler(mode="1d_platt"),
    ]

    for model in models:
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert len(probs) == len(y)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert not np.isnan(probs).any()
