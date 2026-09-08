"""
Property-based tests for calibration math, numerical stability, and metric axioms.

Uses Hypothesis to stress-test estimators, dynamic coefficients, and metric invariants
across a wide range of arbitrary floats, boundary values, and feature spaces.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from trajectory_calibration.calibrators.classic import (
    PlattScalingEstimator,
    QuadraticPlattScaler,
    SplineCalibrator,
    TemperatureScalingEstimator,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.metrics.calibration import compute_adaptive_ece, compute_ece
from trajectory_calibration.metrics.murphy import compute_murphy_brier_decomposition
from trajectory_calibration.metrics.scoring import compute_brier, compute_nll
from trajectory_calibration.utils.math import get_logits, safe_clip_probs, sigmoid

# Common hypothesis test configuration
HYPOTHESIS_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# =====================================================================
# 1. Primitives & Math Invariants
# =====================================================================


class TestMathematicalPrimitives:
    """Property tests for core scalar/array math transformations."""

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-500.0, max_value=500.0))
    @HYPOTHESIS_SETTINGS
    def test_sigmoid_bounded_and_monotonic(self, x: float):
        """Sigmoid must strictly map any finite real to (0, 1) and be monotonically non-decreasing."""
        s = sigmoid(x)
        assert 0.0 <= s <= 1.0
        assert not np.isnan(s)
        assert not np.isinf(s)

        # Monotonicity with a small delta
        delta = 0.1
        s_higher = sigmoid(x + delta)
        assert s_higher >= s - 1e-9

    @given(st.floats(min_value=1e-12, max_value=1.0 - 1e-12))
    @HYPOTHESIS_SETTINGS
    def test_get_logits_invertibility(self, p: float):
        """Inverse sigmoid (logit) followed by sigmoid must round-trip back to p."""
        logit_val = get_logits(p)
        assert not np.isnan(logit_val)
        p_reconstructed = sigmoid(logit_val)
        assert np.isclose(p, p_reconstructed, atol=1e-6)

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        )
    )
    @HYPOTHESIS_SETTINGS
    def test_safe_clip_probs_array_invariants(self, arr: np.ndarray):
        """safe_clip_probs must constrain all values to [eps, 1-eps] strictly."""
        eps = 1e-5
        clipped = safe_clip_probs(arr, eps=eps)
        assert np.all(clipped >= eps)
        assert np.all(clipped <= 1.0 - eps)
        assert not np.any(np.isnan(clipped))


# =====================================================================
# 2. Metric Invariants & Algebraic Identities
# =====================================================================


class TestMetricInvariants:
    """Property tests for metric boundedness, Murphy decomposition, and loss convexity."""

    @given(
        st.integers(min_value=10, max_value=100).flatmap(
            lambda n: st.tuples(
                arrays(
                    dtype=np.float64,
                    shape=n,
                    elements=st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
                ),
                arrays(dtype=np.int32, shape=n, elements=st.integers(min_value=0, max_value=1)),
            )
        )
    )
    @HYPOTHESIS_SETTINGS
    def test_murphy_decomposition_exact_identity(self, inputs: tuple[np.ndarray, np.ndarray]):
        """Murphy decomposition identity must strictly satisfy: Brier = Rel - Res + Unc + Within."""
        probs, y = inputs
        decomp = compute_murphy_brier_decomposition(probs, y, n_bins=10)

        brier = decomp["brier"]
        rel = decomp["reliability"]
        res = decomp["resolution"]
        unc = decomp["uncertainty"]
        within = decomp["within"]

        # Invariant 1: Individual metric components are non-negative and finite
        assert brier >= 0.0
        assert rel >= 0.0
        assert res >= -1e-7
        assert unc >= 0.0
        assert within >= 0.0

        # Invariant 2: Murphy algebraic identity holds within numerical tolerance
        reconstructed_brier = rel - res + unc + within
        assert np.isclose(brier, reconstructed_brier, atol=1e-5)

    @given(
        st.integers(min_value=10, max_value=80).flatmap(
            lambda n: st.tuples(
                arrays(
                    dtype=np.float64,
                    shape=n,
                    elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                ),
                arrays(dtype=np.int32, shape=n, elements=st.integers(min_value=0, max_value=1)),
            )
        )
    )
    @HYPOTHESIS_SETTINGS
    def test_calibration_metrics_boundedness(self, inputs: tuple[np.ndarray, np.ndarray]):
        """ECE, Adaptive ECE, Brier, and NLL must remain strictly bounded and non-negative."""
        probs, y = inputs
        ece = compute_ece(probs, y, n_bins=10)
        aece = compute_adaptive_ece(probs, y, n_bins=10)
        brier = compute_brier(probs, y)
        nll = compute_nll(probs, y)

        assert 0.0 <= ece <= 1.0
        assert 0.0 <= aece <= 1.0
        assert 0.0 <= brier <= 1.0
        assert nll >= 0.0
        assert not np.isnan(ece)
        assert not np.isnan(aece)
        assert not np.isnan(brier)
        assert not np.isnan(nll)

    @given(st.integers(min_value=5, max_value=50))
    @HYPOTHESIS_SETTINGS
    def test_perfect_calibration_limit(self, n: int):
        """When predicted probabilities equal true binary labels, ECE and Brier score must be zero."""
        y = np.random.choice([0, 1], size=n)
        p = y.astype(np.float64)

        brier = compute_brier(p, y)
        ece = compute_ece(p, y, n_bins=5)
        assert np.isclose(brier, 0.0, atol=1e-9)
        assert np.isclose(ece, 0.0, atol=1e-9)


# =====================================================================
# 3. Estimator Stability & Invariants
# =====================================================================


class TestEstimatorInvariants:
    """Property tests verifying calibrators produce strictly bounded valid probabilities."""

    @given(
        st.integers(min_value=25, max_value=60).flatmap(
            lambda n: st.tuples(
                arrays(
                    dtype=np.float64,
                    shape=n,
                    elements=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
                ),
                arrays(dtype=np.int32, shape=n, elements=st.integers(min_value=0, max_value=1)),
            )
        )
    )
    @HYPOTHESIS_SETTINGS
    def test_classic_calibrators_probability_bounds(self, inputs: tuple[np.ndarray, np.ndarray]):
        """Classic post-hoc estimators must output probabilities in [0, 1] without NaNs or Infs."""
        confs, y = inputs
        # Ensure binary classes present to fit logistic estimators
        if len(np.unique(y)) < 2:
            y[0] = 0
            y[1] = 1

        calibrators = [
            PlattScalingEstimator(),
            TemperatureScalingEstimator(),
            SplineCalibrator(),
            QuadraticPlattScaler(degree=2),
        ]

        test_confs = np.linspace(0.01, 0.99, 20)

        for cal in calibrators:
            cal.fit(confs, y)
            preds = cal.predict_proba(test_confs)

            assert isinstance(preds, np.ndarray)
            assert len(preds) == len(test_confs)
            assert np.all(preds >= 0.0)
            assert np.all(preds <= 1.0)
            assert not np.any(np.isnan(preds))
            assert not np.any(np.isinf(preds))


# =====================================================================
# 4. VCPS Numerical & Gradient Stability
# =====================================================================


class TestVCPSNumericalStability:
    """Property tests for Varying-Coefficient Platt Scaling stability and gradient validity."""

    @given(
        st.sampled_from(["full", "slope_only", "intercept_only", "1d_platt"]),
        st.integers(min_value=25, max_value=50),
        st.sampled_from([2, 5, 8]),
    )
    @HYPOTHESIS_SETTINGS
    def test_vcps_dynamic_coefficients_and_probability_invariants(
        self, mode: str, n_samples: int, n_features: int
    ):
        """VCPS dynamic slopes and probabilities must remain strictly bounded for random inputs."""
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, n_features)
        # First column is x1 (log-odds), remaining are trajectory features
        X[:, 0] = np.clip(X[:, 0], -5.0, 5.0)
        y = rng.choice([0, 1], size=n_samples)
        if len(np.unique(y)) < 2:
            y[0] = 0
            y[1] = 1

        vcps = VaryingCoefficientPlattScaler(
            mode=mode,
            C_slope=1.0,
            C_intercept=1.0,
            random_state=42,
        )
        vcps.fit(X, y)

        X_test = rng.randn(15, n_features)
        probs = vcps.predict_proba(X_test)
        slopes = vcps.compute_dynamic_slope(X_test)
        intercepts = vcps.compute_dynamic_intercept(X_test)

        # Invariant 1: Output probabilities in [0, 1]
        assert len(probs) == len(X_test)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert not np.any(np.isnan(probs))

        # Invariant 2: Dynamic slopes are strictly positive and finite
        assert np.all(slopes > 0.0)
        assert np.all(np.isfinite(slopes))
        if mode in ["full", "slope_only"]:
            assert np.all(slopes <= np.exp(3.0) + 1e-5)
            assert np.all(slopes >= np.exp(-3.0) - 1e-5)

        # Invariant 3: Intercepts are finite
        assert not np.any(np.isnan(intercepts))
        assert not np.any(np.isinf(intercepts))

    @given(
        st.sampled_from(["full", "slope_only", "intercept_only", "1d_platt"]),
        st.integers(min_value=20, max_value=40),
    )
    @HYPOTHESIS_SETTINGS
    def test_vcps_analytical_gradient_finite_differences(self, mode: str, n_samples: int):
        """VCPS analytical gradient must agree with finite difference numerical gradient."""
        n_features = 3
        rng = np.random.RandomState(123)
        X = rng.randn(n_samples, n_features)
        y = rng.choice([0, 1], size=n_samples)
        if len(np.unique(y)) < 2:
            y[0] = 0
            y[1] = 1

        vcps = VaryingCoefficientPlattScaler(
            mode=mode,
            C_slope=1.0,
            C_intercept=1.0,
            random_state=123,
        )
        objective, x0 = vcps._build_objective(X, y)

        # Test at a randomized point near optimum
        x_eval = x0 + rng.randn(len(x0)) * 0.1
        val_eval, grad_analytical = objective(x_eval)

        # Directional derivative finite difference check
        direction = rng.randn(len(x0))
        direction /= np.linalg.norm(direction)

        eps = 1e-6
        val_plus, _ = objective(x_eval + eps * direction)
        val_minus, _ = objective(x_eval - eps * direction)
        grad_numerical_directional = (val_plus - val_minus) / (2.0 * eps)
        grad_analytical_directional = float(np.dot(grad_analytical, direction))

        abs_diff = abs(grad_numerical_directional - grad_analytical_directional)
        rel_diff = abs_diff / (abs(grad_numerical_directional) + 1e-8)
        assert rel_diff < 1e-3 or abs_diff < 1e-4
