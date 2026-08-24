"""Tests for calibration metrics and statistical evaluators."""

import numpy as np
import pytest
from trajectory_calibration.metrics.calibration import (
    compute_adaptive_ece,
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_kde_ece,
    compute_mce,
    compute_murphy_brier_decomposition,
    compute_nll,
    compute_prediction_std,
)


def test_ece_perfect_calibration():
    # Perfectly calibrated predictions: conf=0.8, acc=0.8
    confs = np.array([0.8] * 100)
    accs = np.array([1.0] * 80 + [0.0] * 20)
    ece = compute_ece(confs, accs, n_bins=10)
    assert pytest.approx(ece, abs=1e-3) == 0.0


def test_brier_score():
    confs = np.array([1.0, 0.0, 0.5])
    accs = np.array([1.0, 0.0, 1.0])
    # Errors: (1-1)^2 = 0, (0-0)^2 = 0, (0.5-1)^2 = 0.25 -> Mean = 0.25 / 3 = 0.0833
    brier = compute_brier(confs, accs)
    assert pytest.approx(brier, abs=1e-4) == 0.25 / 3.0


def test_murphy_decomposition_identity():
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.1, 0.9, size=200)
    y = (rng.rand(200) > 0.4).astype(float)

    decomp = compute_murphy_brier_decomposition(probs, y, n_bins=10)
    expected_brier = decomp["uncertainty"] - decomp["resolution"] + decomp["reliability"]
    assert pytest.approx(decomp["brier"], abs=1e-4) == expected_brier


def test_adaptive_ece_bounds():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    ada_ece = compute_adaptive_ece(probs, y, n_bins=2)
    assert 0.0 <= ada_ece <= 1.0


def test_kde_ece_bounds():
    probs = np.linspace(0.1, 0.9, 50)
    y = (probs > 0.5).astype(float)
    kde_ece = compute_kde_ece(probs, y)
    assert 0.0 <= kde_ece <= 1.0


def test_auroc():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    auroc = compute_auroc(probs, y)
    assert auroc == 1.0
