"""Tests for all post-hoc calibrators and VCPS."""

import numpy as np
import pytest
from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    BetaCalibrator,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    QuadraticPlattScaler,
    SplineCalibrator,
    TemperatureScalingEstimator,
    TrajectoryLREstimator,
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
        QuadraticPlattScaler(),
        TrajectoryLREstimator(fit_intercept=False),
        BetaCalibrator(),
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


def test_quadratic_platt_properties():
    df = generate_mock_df("test_mock", n_samples=100, seed=42)
    X = df[["x1", "x13", "x6", "x8", "x4"]].values
    y = df["is_correct"].values

    quad = QuadraticPlattScaler()
    quad.fit(X, y)

    probs = quad.predict_proba(X)
    assert len(probs) == len(y)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.isnan(probs).any()

    slopes = quad.compute_dynamic_slope(X)
    assert len(slopes) == len(y)
    assert not np.isnan(slopes).any()

    intercepts = quad.compute_dynamic_intercept(X)
    assert len(intercepts) == len(y)

    teff = quad.get_effective_temperature(X)
    assert len(teff) == len(y)
    assert np.all(teff > 0.0)


def test_proxy_calibrators_validation():
    from trajectory_calibration.calibrators.proxies import (
        MultiScaleSemanticConsistency,
        MultiScaleEigenVariance,
        ProbabilityMarginEstimator,
    )
    df = generate_mock_df("test_mock", n_samples=30, seed=42)
    X_5d = df[["x1", "x13", "x6", "x8", "x4"]].values
    y = df["is_correct"].values

    # Passing feature_names should succeed
    mssc = MultiScaleSemanticConsistency(feature_names=["x1", "x13", "x6", "x8", "x4"])
    mssc.fit(X_5d, y)
    preds = mssc.predict_proba(X_5d)
    assert len(preds) == len(y)

    # Missing column in feature_names should raise ValueError
    with pytest.raises(ValueError):
        mssc_bad = MultiScaleSemanticConsistency(feature_names=["x1", "x6", "x8"])
        mssc_bad.fit(X_5d[:, :3], y)


def test_word_boundary_accuracy_evaluation():
    from trajectory_calibration.vlm.evaluators import evaluate_accuracy

    # Substring false positive prevention
    sample_tired = {"answer": "tired"}
    assert evaluate_accuracy("red", sample_tired, "chartqa") == 0.0

    sample_phone = {"answer": "phone"}
    assert evaluate_accuracy("one", sample_phone, "chartqa") == 0.0

    # Legitimate word boundary match
    sample_dog = {"answer": "a brown dog"}
    assert evaluate_accuracy("dog", sample_dog, "chartqa") == 1.0
    assert evaluate_accuracy("brown", sample_dog, "chartqa") == 1.0


def test_load_image_from_filepath(tmp_path):
    from PIL import Image
    from trajectory_calibration.vlm.formatting import load_image_from_sample

    img_file = tmp_path / "test_img.png"
    Image.new("RGB", (32, 32), color="blue").save(img_file)

    sample = {"image_path": str(img_file)}
    loaded = load_image_from_sample(sample)
    assert loaded is not None
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (32, 32)


def test_trajectory_lr_zero_bias_properties():
    df = generate_mock_df("test_mock", n_samples=100, seed=42)
    X = df[["x1", "x13", "x6", "x8", "x4"]].values
    y = df["is_correct"].values

    lr_no_bias = TrajectoryLREstimator(fit_intercept=False)
    lr_no_bias.fit(X, y)

    # 1. Verify intercept is explicitly 0.0
    assert lr_no_bias.lr.intercept_ == 0.0

    # 2. Verify zero-bias property: at input X=0, logit=0 -> prob=0.5
    zero_input = np.zeros((3, X.shape[1]))
    probs_zero = lr_no_bias.predict_proba(zero_input)
    assert np.allclose(probs_zero, 0.5)

    # 3. Verify valid bounded probabilities on test data
    probs = lr_no_bias.predict_proba(X)
    assert len(probs) == len(y)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.isnan(probs).any()


