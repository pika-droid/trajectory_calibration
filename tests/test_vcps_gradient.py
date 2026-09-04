"""Tests for verifying VCPS analytical gradients against finite differences."""

import numpy as np
import pytest
from scipy.optimize import check_grad
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler


def test_vcps_gradient_consistency():
    """Verify analytical gradients match finite-difference approximation across all modes."""
    np.random.seed(42)
    n, d = 200, 5
    X = np.random.randn(n, d)
    # Ensure x1 has reasonable dynamic range and labels are binary
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(float)
    feature_names = [f"x{i}" for i in range(1, d + 1)]

    modes = ["full", "slope_only", "intercept_only", "1d_platt"]

    for mode in modes:
        vcps = VaryingCoefficientPlattScaler(
            mode=mode,
            slope_features=["x2", "x3"],
            intercept_features=["x2", "x3", "x4", "x5"],
            C_slope=1.0,
            C_intercept=1.0,
            random_state=42,
        )

        objective, x0 = vcps._build_objective(X, y, feature_names=feature_names)

        def func(params):
            val, _ = objective(params)
            return val

        def grad(params):
            _, g = objective(params)
            return g

        # 1. Check with scipy.optimize.check_grad at initial point
        cg_err = check_grad(func, grad, x0)
        assert cg_err < 1e-5, f"check_grad error {cg_err} >= 1e-5 for mode={mode} at x0"

        # 2. Check at a perturbed point
        np.random.seed(123)
        x_perturbed = x0 + 0.1 * np.random.randn(len(x0))

        cg_err_perturbed = check_grad(func, grad, x_perturbed)
        assert cg_err_perturbed < 1e-5, f"check_grad error {cg_err_perturbed} >= 1e-5 for mode={mode} at x_perturbed"

        # Check specifically when base slope a0 < 0.1 (alpha0 = -2.5 -> a0 ~ 0.082)
        x_small = x_perturbed.copy()
        x_small[0] = -2.5
        cg_err_small = check_grad(func, grad, x_small)
        assert cg_err_small < 1e-5, f"check_grad error {cg_err_small} >= 1e-5 for mode={mode} at small a0 < 0.1"

        # 3. Explicit finite-difference relative error check
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
