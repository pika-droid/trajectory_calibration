"""
Unit tests for TrajectoryPlattScaler calibrator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.calibrators.baselines import TrajectoryPlattScaler
from trajectory_calibration.metrics.scoring import compute_nll


def test_fit_predict_basic():
    """Create synthetic data (N=200, K=17), fit, predict_proba, assert shapes and valid range."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(200, 17)
    y = (np.random.rand(200) > 0.5).astype(np.float64)

    scaler = TrajectoryPlattScaler(n_features=17)
    scaler.fit(X, y)
    probs = scaler.predict_proba(X)

    assert probs.shape == (200,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.isnan(probs).any()


def test_fit_predict_5d():
    """Same as basic but with K=5 features."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(200, 5)
    y = (np.random.rand(200) > 0.5).astype(np.float64)

    scaler = TrajectoryPlattScaler(n_features=5)
    scaler.fit(X, y)
    probs = scaler.predict_proba(X)

    assert probs.shape == (200,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.isnan(probs).any()


def test_parameter_count():
    """Assert a has shape (K,) and b has shape (1,) for various K."""
    for K in [5, 17]:
        model = TrajectoryPlattScaler(n_features=K)
        assert model.a.shape == (K,)
        assert model.b.shape == (1,)
        assert isinstance(model.a, torch.nn.Parameter)
        assert isinstance(model.b, torch.nn.Parameter)


def test_convergence():
    """Fit on linearly separable data, assert NLL decreases and accuracy > 0.8."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(200, 17)
    w_true = np.random.randn(17)
    y = (X @ w_true > 0).astype(np.float64)

    # Initial NLL with untrained model (p = 0.5)
    nll_before = compute_nll(np.full(len(y), 0.5), y)

    scaler = TrajectoryPlattScaler(n_features=17)
    scaler.fit(X, y)
    probs = scaler.predict_proba(X)

    nll_after = compute_nll(probs, y)
    accuracy = float(np.mean((probs >= 0.5) == y))

    assert nll_after < nll_before
    assert accuracy > 0.8


def test_matches_sklearn_lr():
    """Fit both TrajectoryPlattScaler and LogisticRegression on the same data, assert close predictions."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(200, 17)
    w_true = np.random.randn(17) * 0.5
    logits = X @ w_true
    probs_true = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(200) < probs_true).astype(np.float64)

    # 1. Compare with C=1000 (weak regularization)
    platt_c1000 = TrajectoryPlattScaler(n_features=17, C=1000.0)
    platt_c1000.fit(X, y)
    preds_platt = platt_c1000.predict_proba(X)

    X_std = StandardScaler().fit_transform(X)
    lr_c1000 = LogisticRegression(C=1000.0, solver="lbfgs")
    lr_c1000.fit(X_std, y)
    preds_lr = lr_c1000.predict_proba(X_std)[:, 1]

    assert np.allclose(preds_platt, preds_lr, rtol=0.1, atol=0.05)

    # 2. Compare default TrajectoryPlattScaler(n_features=17) with LogisticRegression(C=1000)
    platt_default = TrajectoryPlattScaler(n_features=17)
    platt_default.fit(X, y)
    preds_platt_default = platt_default.predict_proba(X)
    assert np.allclose(preds_platt_default, preds_lr, rtol=0.1, atol=0.05)


def test_unfitted_raises_runtime_error():
    """Calling predict_proba on unfitted model raises RuntimeError."""
    scaler = TrajectoryPlattScaler(n_features=5)
    with pytest.raises(RuntimeError):
        scaler.predict_proba(np.zeros((10, 5)))


def test_dimension_mismatch_raises_value_error():
    """Dimension mismatch between n_features and input raises ValueError."""
    scaler = TrajectoryPlattScaler(n_features=17)
    with pytest.raises(ValueError, match="Expected 17 features"):
        scaler.fit(np.zeros((10, 5)), np.zeros(10))


def test_single_sample_prediction():
    """Predicting on a single 1D or 2D sample succeeds and returns valid probabilities."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = (np.random.rand(50) > 0.5).astype(np.float64)

    scaler = TrajectoryPlattScaler(n_features=5).fit(X, y)

    # Single 2D sample (1, 5)
    single_2d = np.random.randn(1, 5)
    p_2d = scaler.predict_proba(single_2d)
    assert p_2d.shape == (1,)
    assert 0.0 <= p_2d[0] <= 1.0

    # Single 1D sample (5,)
    single_1d = np.random.randn(5)
    p_1d = scaler.predict_proba(single_1d)
    assert p_1d.shape == (1,)
    assert 0.0 <= p_1d[0] <= 1.0


def test_predict_proba_empty_input():
    """Predicting on an empty 2D array returns an empty 1D array."""
    scaler = TrajectoryPlattScaler(n_features=5).fit(np.random.randn(10, 5), np.ones(10))
    p_empty = scaler.predict_proba(np.zeros((0, 5)))
    assert isinstance(p_empty, np.ndarray)
    assert p_empty.shape == (0,)


def test_fit_empty_raises_value_error():
    """Fitting on empty data raises ValueError."""
    scaler = TrajectoryPlattScaler(n_features=5)
    with pytest.raises(ValueError, match="Cannot fit TrajectoryPlattScaler with 0 samples"):
        scaler.fit(np.zeros((0, 5)), np.zeros(0))


def test_fit_sample_count_mismatch_raises_value_error():
    """Mismatched sample counts between X and y raise ValueError."""
    scaler = TrajectoryPlattScaler(n_features=5)
    with pytest.raises(ValueError, match="same number of samples"):
        scaler.fit(np.random.randn(10, 5), np.ones(5))


def test_invalid_init_parameters_raise_value_error():
    """Invalid n_features or C raise ValueError."""
    with pytest.raises(ValueError, match="n_features must be positive"):
        TrajectoryPlattScaler(n_features=0)
    with pytest.raises(ValueError, match="n_features must be positive"):
        TrajectoryPlattScaler(n_features=-2)
    with pytest.raises(ValueError, match="C must be positive or None"):
        TrajectoryPlattScaler(n_features=5, C=0.0)
    with pytest.raises(ValueError, match="C must be positive or None"):
        TrajectoryPlattScaler(n_features=5, C=-1.0)


def test_fit_idempotence():
    """Calling fit a second time on new data resets parameters to zero and converges identically to fresh instance."""
    np.random.seed(42)
    torch.manual_seed(42)

    X1 = np.random.randn(50, 3)
    y1 = (X1[:, 0] > 0).astype(np.float64)

    m_reused = TrajectoryPlattScaler(n_features=3)
    m_reused.fit(X1, y1)

    X2 = np.random.randn(50, 3)
    y2 = (X2[:, 1] > 0).astype(np.float64)

    m_reused.fit(X2, y2)
    preds_reused = m_reused.predict_proba(X2)

    m_fresh = TrajectoryPlattScaler(n_features=3)
    m_fresh.fit(X2, y2)
    preds_fresh = m_fresh.predict_proba(X2)

    assert np.allclose(preds_reused, preds_fresh, atol=1e-5)
    assert np.allclose(m_reused.a.detach().numpy(), m_fresh.a.detach().numpy(), atol=1e-5)
    assert np.allclose(m_reused.b.detach().numpy(), m_fresh.b.detach().numpy(), atol=1e-5)


def test_constant_features_robustness():
    """Degenerate zero-variance features are handled stably without NaN or optimizer failure."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(50, 5)
    X[:, 2] = 10.0  # Constant column
    y = (X[:, 0] > 0).astype(np.float64)

    scaler = TrajectoryPlattScaler(n_features=5)
    scaler.fit(X, y)
    probs = scaler.predict_proba(X)

    assert not np.isnan(probs).any()
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_homogeneous_labels_robustness():
    """Model fits stably on 100% positive and 100% negative training sets."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(30, 5)

    # All positive labels
    scaler_pos = TrajectoryPlattScaler(n_features=5).fit(X, np.ones(30))
    p_pos = scaler_pos.predict_proba(X)
    assert not np.isnan(p_pos).any()
    assert np.all(p_pos > 0.99)

    # All negative labels
    scaler_neg = TrajectoryPlattScaler(n_features=5).fit(X, np.zeros(30))
    p_neg = scaler_neg.predict_proba(X)
    assert not np.isnan(p_neg).any()
    assert np.all(p_neg < 0.01)
