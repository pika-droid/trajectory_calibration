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
    TrajectoryPlattScaler,
)
from trajectory_calibration.calibrators.residual import ResidualTrajectoryCalibrator
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.trajectory import (
    CANONICAL_5D_KEYS,
    FEATURE_KEYS,
    generate_mock_df,
)


def test_all_calibrators_fit_predict():
    df = generate_mock_df("test_mock", n_samples=80, seed=42)
    X = df[CANONICAL_5D_KEYS].values
    y = df["is_correct"].values

    models = [
        NaiveConfidenceEstimator(),
        TemperatureScalingEstimator(),
        PlattScalingEstimator(),
        QuadraticPlattScaler(),
        TrajectoryLREstimator(fit_intercept=False),
        TrajectoryPlattScaler(n_features=5),
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
    X = df[CANONICAL_5D_KEYS].values
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
        MultiScaleEigenVariance,
        MultiScaleSemanticConsistency,
        ProbabilityMarginEstimator,
    )

    df = generate_mock_df("test_mock", n_samples=40, seed=42)
    X_17d = df[FEATURE_KEYS].values
    y = df["is_correct"].values

    # 1. MultiScaleSemanticConsistency: targets x3 (idx 2)
    mssc = MultiScaleSemanticConsistency(feature_names=FEATURE_KEYS)
    mssc.fit(X_17d, y)
    preds_mssc = mssc.predict_proba(X_17d)
    assert len(preds_mssc) == len(y)
    assert np.all(preds_mssc >= 0.0) and np.all(preds_mssc <= 1.0)

    # MSSC fallback without feature_names (uses fallback_idx=2)
    mssc_fallback = MultiScaleSemanticConsistency()
    mssc_fallback.fit(X_17d, y)
    preds_mssc_fb = mssc_fallback.predict_proba(X_17d)
    assert np.allclose(preds_mssc, preds_mssc_fb)

    # 2. MultiScaleEigenVariance: targets x13 (idx 12)
    msev = MultiScaleEigenVariance(feature_names=FEATURE_KEYS)
    msev.fit(X_17d, y)
    preds_msev = msev.predict_proba(X_17d)
    assert len(preds_msev) == len(y)
    assert np.all(preds_msev >= 0.0) and np.all(preds_msev <= 1.0)

    # MSE-EIGEN fallback without feature_names (uses fallback_idx=12)
    msev_fallback = MultiScaleEigenVariance()
    msev_fallback.fit(X_17d, y)
    preds_msev_fb = msev_fallback.predict_proba(X_17d)
    assert np.allclose(preds_msev, preds_msev_fb)

    # 3. ProbabilityMarginEstimator: targets x17 (idx 16)
    pme = ProbabilityMarginEstimator(feature_names=FEATURE_KEYS)
    pme.fit(X_17d, y)
    preds_pme = pme.predict_proba(X_17d)
    assert len(preds_pme) == len(y)
    assert np.all(preds_pme >= 0.0) and np.all(preds_pme <= 1.0)

    # ProbabilityMargin fallback without feature_names (uses fallback_idx=16)
    pme_fallback = ProbabilityMarginEstimator()
    pme_fallback.fit(X_17d, y)
    preds_pme_fb = pme_fallback.predict_proba(X_17d)
    assert np.allclose(preds_pme, preds_pme_fb)

    # 4. Error cases: missing feature name raises ValueError
    with pytest.raises(ValueError, match="not found in provided feature_names"):
        mssc_bad = MultiScaleSemanticConsistency(feature_names=["x1", "x4", "x5"])
        mssc_bad.fit(X_17d[:, :3], y)

    # Fallback exceeds column count raises ValueError
    with pytest.raises(ValueError, match="fallback index 16 exceeds column count"):
        pme_short = ProbabilityMarginEstimator()
        pme_short.fit(X_17d[:, :5], y)


def test_word_boundary_accuracy_evaluation():
    from trajectory_calibration.vlm.evaluators import evaluate_accuracy

    # Substring false positive prevention
    sample_tired = {"answer": "tired"}
    assert evaluate_accuracy("red", sample_tired, "chartqa") == 0.0

    sample_phone = {"answer": "phone"}
    assert evaluate_accuracy("one", sample_phone, "chartqa") == 0.0

    # Ground-truth containment in longer prediction
    sample_dog = {"answer": "dog"}
    assert evaluate_accuracy("a brown dog", sample_dog, "chartqa") == 1.0
    assert evaluate_accuracy("brown dog", sample_dog, "chartqa") == 1.0

    # Substring of ground truth is rejected
    sample_phrase = {"answer": "a brown dog"}
    assert evaluate_accuracy("dog", sample_phrase, "chartqa") == 0.0


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
    X = df[["x1", "x2", "x3", "x4", "x5"]].values
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
