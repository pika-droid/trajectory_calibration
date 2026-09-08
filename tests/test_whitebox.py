"""Tests for UQLM white-box scorers."""

import numpy as np
import pytest

from trajectory_calibration.uq.whitebox import WhiteBoxScorers


def test_sequence_probability():
    token_lps = [-0.1, -0.2, -0.3]
    # Geometric mean: exp((-0.1 - 0.2 - 0.3) / 3) = exp(-0.2)
    p_ln = WhiteBoxScorers.sequence_probability(token_lps, length_normalize=True)
    assert pytest.approx(p_ln, abs=1e-4) == np.exp(-0.2)

    p_raw = WhiteBoxScorers.sequence_probability(token_lps, length_normalize=False)
    assert pytest.approx(p_raw, abs=1e-4) == np.exp(-0.6)


def test_min_probability():
    token_probs = [0.9, 0.4, 0.85]
    min_p = WhiteBoxScorers.min_probability(token_probs)
    assert min_p == 0.4


def test_probability_margin():
    margin = WhiteBoxScorers.probability_margin(0.85, 0.15)
    assert pytest.approx(margin, abs=1e-5) == 0.70


def test_token_entropy_and_negentropy():
    # Uniform distribution (highest entropy)
    uniform_p = np.array([0.5, 0.5])
    h = WhiteBoxScorers.token_entropy(uniform_p, base=2.0)
    assert pytest.approx(h, abs=1e-5) == 1.0

    negentropy = WhiteBoxScorers.mean_token_negentropy([uniform_p], num_classes=2)
    # Negentropy: 1 - 1.0 / log2(2) = 0.0
    assert pytest.approx(negentropy, abs=1e-4) == 0.0
