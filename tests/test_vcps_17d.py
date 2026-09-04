"""
Unit test suite for VCPS-17D Dynamic Signature Binding and Full 36-Parameter Optimization.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import check_grad

from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import FEATURE_KEYS
from trajectory_calibration.features.trajectory import generate_mock_df


def test_vcps_17d_signature_binding() -> None:
    """Verify that VCPS-17D binds all 17 non-anchor trajectory signatures (36 parameters)."""
    df = generate_mock_df("test_17d_mock", n_samples=120, seed=42)
    X = df[FEATURE_KEYS].values
    y = df["is_correct"].values
    assert X.shape[1] == 18

    vcps = VaryingCoefficientPlattScaler(feature_set="17d", random_state=42)
    vcps.fit(X, y, feature_names=FEATURE_KEYS)

    expected_signatures = [k for k in FEATURE_KEYS if k != "x1"]
    assert len(expected_signatures) == 17
    assert vcps.n_features_in_ == 18
    assert vcps.n_params == 36

    assert len(vcps._slope_idx) == 17
    assert len(vcps._int_idx) == 17
    assert vcps.slope_features_ == expected_signatures
    assert vcps.intercept_features_ == expected_signatures

    assert vcps.gamma is not None and len(vcps.gamma) == 17
    assert vcps.w is not None and len(vcps.w) == 17
    assert isinstance(vcps.a0, float)
    assert isinstance(vcps.b0, float)


def test_vcps_17d_gradient_finite_difference() -> None:
    """Verify analytical gradients match finite-difference approximations for full 17D binding."""
    np.random.seed(42)
    n, d = 150, 18
    X = np.random.randn(n, d)
    # Ensure realistic anchor logit range and non-trivial labels
    y = (X[:, 0] + 0.2 * X[:, 1] - 0.15 * X[:, 2] > 0).astype(float)
    feature_names = FEATURE_KEYS

    modes = ["full", "slope_only", "intercept_only", "1d_platt"]

    for mode in modes:
        vcps = VaryingCoefficientPlattScaler(
            mode=mode,
            feature_set="17d",
            C_slope=1.0,
            C_intercept=1.0,
            random_state=42,
        )

        objective, x0 = vcps._build_objective(X, y, feature_names=feature_names)

        if mode == "full":
            assert len(x0) == 36
            assert vcps.n_params == 36
        elif mode in ["slope_only", "intercept_only"]:
            assert len(x0) == 19
            assert vcps.n_params == 19
        else:  # 1d_platt
            assert len(x0) == 2
            assert vcps.n_params == 2

        def func(params: np.ndarray) -> float:
            val, _ = objective(params)
            return val

        def grad(params: np.ndarray) -> np.ndarray:
            _, g = objective(params)
            return g

        # 1. Scipy check_grad at initial point
        cg_err = check_grad(func, grad, x0)
        assert cg_err < 1e-5, f"check_grad error {cg_err} >= 1e-5 for mode={mode} at x0"

        # 2. Scipy check_grad at a perturbed point
        np.random.seed(999)
        x_perturbed = x0 + 0.05 * np.random.randn(len(x0))

        cg_err_perturbed = check_grad(func, grad, x_perturbed)
        assert cg_err_perturbed < 1e-5, f"check_grad error {cg_err_perturbed} >= 1e-5 for mode={mode} at perturbed"

        # 3. Explicit finite-difference check across all dimensions
        analytical_val, analytical_grad = objective(x_perturbed)
        eps = 1e-6
        numerical_grad = np.zeros_like(x_perturbed)

        for i in range(len(x_perturbed)):
            p_plus = x_perturbed.copy()
            p_minus = x_perturbed.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            f_plus, _ = objective(p_plus)
            f_minus, _ = objective(p_minus)
            numerical_grad[i] = (f_plus - f_minus) / (2.0 * eps)

        denom = np.maximum(np.abs(analytical_grad) + np.abs(numerical_grad), 1e-6)
        rel_errors = np.abs(analytical_grad - numerical_grad) / denom
        max_rel_error = np.max(rel_errors)
        assert max_rel_error < 1e-5, f"Max relative error {max_rel_error} >= 1e-5 for mode={mode}"


def test_vcps_17d_predict_proba_validity() -> None:
    """Verify predict_proba produces valid, strictly non-degenerate probabilities without collapse."""
    df = generate_mock_df("test_17d_predict", n_samples=200, seed=123)
    X = df[FEATURE_KEYS].values
    y = df["is_correct"].values

    train_n = 150
    X_train, y_train = X[:train_n], y[:train_n]
    X_test, y_test = X[train_n:], y[train_n:]

    vcps = VaryingCoefficientPlattScaler(feature_set="17d", random_state=42)
    vcps.fit(X_train, y_train, feature_names=FEATURE_KEYS)

    probs = vcps.predict_proba(X_test)
    assert len(probs) == len(y_test)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()

    # Verify absence of prediction collapse (sigma_p > 0.05)
    sigma_p = float(np.std(probs))
    assert sigma_p > 0.05, f"Prediction collapsed: sigma_p = {sigma_p:.4f} <= 0.05"

    # Verify dynamic slope and effective temperature validity
    slopes = vcps.compute_dynamic_slope(X_test)
    assert len(slopes) == len(y_test)
    assert np.all(slopes > 0.0)

    teff = vcps.get_effective_temperature(X_test)
    assert len(teff) == len(y_test)
    assert np.all(teff > 0.0)


def test_vcps_5d_binding_and_explicit_override() -> None:
    """Verify feature_set='5d' binds all 5 signatures for slope and intercept (12 params)."""
    df = generate_mock_df("test_5d_override", n_samples=80, seed=42)
    five_keys = ["x1", "x13", "x6", "x8", "x4", "x20"]  # x1 anchor + 5 trajectory signatures
    X_5d = df[five_keys].values
    y = df["is_correct"].values

    # Test 5D dynamic binding
    vcps_5d = VaryingCoefficientPlattScaler(feature_set="5d", random_state=42)
    vcps_5d.fit(X_5d, y, feature_names=five_keys)
    assert vcps_5d.slope_features_ == ["x13", "x6", "x8", "x4", "x20"]
    assert vcps_5d.intercept_features_ == ["x13", "x6", "x8", "x4", "x20"]
    assert vcps_5d.n_params == 12  # 1 + 5 + 1 + 5

    # Test explicit override
    vcps_override = VaryingCoefficientPlattScaler(
        feature_set="17d",
        slope_features=["x6"],
        intercept_features=["x8", "x4"],
        random_state=42,
    )
    vcps_override.fit(X_5d, y, feature_names=five_keys)
    assert vcps_override.slope_features_ == ["x6"]
    assert vcps_override.intercept_features_ == ["x8", "x4"]
    assert vcps_override.n_params == 5  # 1 + 1 + 1 + 2

    # Test 5D dynamic binding when passed an 18-D matrix directly
    df_18 = generate_mock_df("test_5d_on_18d", n_samples=80, seed=42)
    X_18 = df_18[FEATURE_KEYS].values
    y_18 = df_18["is_correct"].values
    vcps_5d_on_18 = VaryingCoefficientPlattScaler(feature_set="5d", random_state=42)
    vcps_5d_on_18.fit(X_18, y_18, feature_names=FEATURE_KEYS)
    assert len(vcps_5d_on_18.slope_features_) == 5
    assert len(vcps_5d_on_18.intercept_features_) == 5
    assert vcps_5d_on_18.n_params == 12  # 1 + 5 + 1 + 5

