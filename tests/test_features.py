"""Tests for 17-D trajectory feature extraction (1 anchor + 16 signatures) and 5-D selection."""

import numpy as np
import pytest

from trajectory_calibration.features.trajectory import (
    CANONICAL_5D_KEYS,
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

    # 1. Verify FEATURE_KEYS is contiguous x1 through x17
    assert [f"x{i}" for i in range(1, 18)] == FEATURE_KEYS
    assert len(FEATURE_KEYS) == 17

    # 2. Verify all 17 contiguous feature keys are present in extracted dict
    for k in FEATURE_KEYS:
        assert k in feats, f"Missing feature key: {k}"

    # 3. Verify exactly 17 'x' features and absence of ablated/legacy keys
    x_keys = [k for k in feats if k.startswith("x")]
    assert len(x_keys) == 17
    for old_key in ["x18", "x19", "x20", "x21", "x22"]:
        assert old_key not in feats, f"Ablated/legacy key {old_key} should not be in features"

    # 4. Verify specific mathematical values in new importance ranking
    assert feats["is_correct"] == 1
    assert feats["vqa_accuracy"] == 1.0
    assert feats["x1"] == pytest.approx(np.log(9.0), abs=1e-5)  # Final Logit: logit(0.9)
    assert feats["x2"] == pytest.approx(
        0.5, abs=1e-5
    )  # Discrete Answer Stability: 2 unique answers ("cat", "dog") -> 1/2
    assert feats["x4"] == pytest.approx(
        0.25, abs=1e-5
    )  # Answer Flip Frequency: 1 flip / 4 transitions
    assert (
        feats["x5"] == 4.0
    )  # Monotonicity count: all 4 transitions increase (0.5 < 0.6 < 0.7 < 0.8 < 0.9)
    assert feats["x7"] == pytest.approx(
        -0.1, abs=1e-5
    )  # Mid-Fine Gain Contrast: (0.9-0.8) - (0.8-0.6)
    assert feats["x8"] == pytest.approx(
        0.02, abs=1e-5
    )  # Confidence Variance: Var([0.5,0.6,0.7,0.8,0.9])
    assert feats["x9"] == pytest.approx(0.25, abs=1e-5)  # End-Scale Spike Ratio: 0.9 - 0.65
    assert feats["x10"] == pytest.approx(0.0, abs=1e-5)  # Scale Dip Depth: max(0, 0.6 - 0.7) = 0
    assert feats["x11"] == pytest.approx(
        4.0 / 9.0, abs=1e-5
    )  # First-to-Final Jump Ratio: (0.9 - 0.5) / 0.9
    assert feats["x13"] == pytest.approx(1.5, abs=1e-5)  # Relative Gain Ratio: 0.9 / 0.6
    assert feats["x15"] == pytest.approx(0.9 - 0.6, abs=1e-5)  # Conf gain: 576 - 9 (now x15)
    assert feats["x16"] == pytest.approx(
        2.5, abs=1e-5
    )  # Relative Margin Growth: 0.5 / 0.2 (now x16)
    assert feats["x17"] == pytest.approx(
        np.log(1.5), abs=1e-5
    )  # Logprob Gain: ln(0.9) - ln(0.6) (now x17)


def test_sigmoid_get_logits_roundtrip():
    confs = np.array([0.1, 0.5, 0.9])
    logits = get_logits(confs)
    recov = sigmoid(logits)
    np.testing.assert_allclose(confs, recov, atol=1e-5)


def test_canonical_5d_keys():
    """Verify universal static canonical 5D feature set invariants and exports."""
    import trajectory_calibration.features as feats
    import trajectory_calibration.features.definitions as defs
    import trajectory_calibration.features.trajectory as traj

    # 1. Structural invariants
    assert CANONICAL_5D_KEYS == ["x1", "x2", "x3", "x4", "x5"]
    assert len(CANONICAL_5D_KEYS) == 5
    assert CANONICAL_5D_KEYS[0] == "x1"
    assert CANONICAL_5D_KEYS[1:] == ["x2", "x3", "x4", "x5"]
    assert FEATURE_KEYS[:5] == CANONICAL_5D_KEYS

    # 2. Cross-module export invariants
    assert defs.CANONICAL_5D_KEYS == CANONICAL_5D_KEYS
    assert "CANONICAL_5D_KEYS" in defs.__all__
    assert feats.CANONICAL_5D_KEYS == CANONICAL_5D_KEYS
    assert "CANONICAL_5D_KEYS" in feats.__all__
    assert traj.CANONICAL_5D_KEYS == CANONICAL_5D_KEYS
    assert "CANONICAL_5D_KEYS" in traj.__all__


def test_select_best_5d_subset():
    df = generate_mock_df("test_mock", n_samples=60, seed=42)
    X = df[FEATURE_KEYS].values
    y = df["is_correct"].values

    subset = select_best_5d_subset(X, y, FEATURE_KEYS)
    assert len(subset) == 5  # 1 anchor + 4 trajectory signatures
    assert subset[0] == "x1"  # x1 is always root anchor
    assert len([k for k in subset if k != "x1"]) == 4


def test_stratified_split_fallback_small_sample():
    import pandas as pd

    from trajectory_calibration.features.loader import get_stratified_split

    # Single-class dataset should not crash train_test_split
    df_single = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "is_correct": [1, 1, 1, 1, 1],
            "answer_type": ["open"] * 5,
        }
    )
    tr_idx, te_idx = get_stratified_split(df_single, test_size=0.2, random_state=42)
    assert len(tr_idx) + len(te_idx) == 5

    # Dataset with minority class count = 1 should fall back gracefully
    df_rare = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "is_correct": [1, 1, 1, 1, 0],
            "answer_type": ["open"] * 5,
        }
    )
    tr_idx, te_idx = get_stratified_split(df_rare, test_size=0.2, random_state=42)
    assert len(tr_idx) + len(te_idx) == 5
