"""Tests for 17-D trajectory feature extraction and 5-D selection."""

import numpy as np
import pytest
from trajectory_calibration.features.trajectory import (
    FEATURE_KEYS,
    compute_features_from_sample,
    generate_mock_df,
    get_logits,
    select_best_5d_subset,
    sigmoid,
)


def test_compute_features_sample():
    sample = {
        "question_id": 101,
        "features": {
            1: {"conf_softmax": 0.5, "margin": 0.1, "answer": "cat", "vqa_accuracy": 0.0},
            9: {"conf_softmax": 0.6, "margin": 0.2, "answer": "dog", "vqa_accuracy": 1.0},
            36: {"conf_softmax": 0.7, "margin": 0.3, "answer": "dog", "vqa_accuracy": 1.0},
            144: {"conf_softmax": 0.8, "margin": 0.4, "answer": "dog", "vqa_accuracy": 1.0},
            576: {"conf_softmax": 0.9, "margin": 0.5, "answer": "dog", "vqa_accuracy": 1.0},
        },
    }
    feats = compute_features_from_sample(sample, fine_scale=576)
    assert "x1" in feats
    assert "x13" in feats
    assert feats["is_correct"] == 1
    assert feats["vqa_accuracy"] == 1.0
    assert feats["x3"] == pytest.approx(0.9 - 0.6, abs=1e-5)  # conf gain: 576 - 9


def test_sigmoid_get_logits_roundtrip():
    confs = np.array([0.1, 0.5, 0.9])
    logits = get_logits(confs)
    recov = sigmoid(logits)
    np.testing.assert_allclose(confs, recov, atol=1e-5)


def test_select_best_5d_subset():
    df = generate_mock_df("test_mock", n_samples=60, seed=42)
    X = df[FEATURE_KEYS].values
    y = df["is_correct"].values

    subset = select_best_5d_subset(X, y, FEATURE_KEYS)
    assert len(subset) == 5
    assert subset[0] == "x1"  # x1 is always root anchor
