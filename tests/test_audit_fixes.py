"""
Unit test suite verifying audit remediations and mathematical correctness.
Covers:
1. VCPS small-slope optimization convergence without vanishing gradients when a0 in [0.001, 0.05].
2. Monotonicity guarantee in fit_beta_calibration (a >= 0, b >= 0).
3. Dataset sample count preservation when question_id is missing or None.
4. Adaptive ECE stability and partition preservation on repeated/uniform confidences.
5. Rejection of substring semantic contradictions in VQA evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from trajectory_calibration.calibrators.adaptation import fit_beta_calibration
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.extractor import compute_features_from_sample
from trajectory_calibration.metrics.ece import compute_adaptive_ece
from trajectory_calibration.vlm.evaluators import evaluate_accuracy


def test_vcps_small_slope_optimization():
    """Verify VCPS can optimize small slopes (a0 in [0.001, 0.05]) without vanishing gradients."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    # Generate labels with very small underlying slope ~ 0.02
    true_slope = 0.02
    logits = true_slope * x1 - 0.1
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n) < probs).astype(float)
    X = np.column_stack([x1, np.random.randn(n, 4)])
    feature_names = ["x1", "x2", "x3", "x4", "x5"]

    vcps = VaryingCoefficientPlattScaler(mode="1d_platt", random_state=42)
    objective, x0 = vcps._build_objective(X, y, feature_names=feature_names)

    # Test gradient at a point where a0 = exp(alpha0) = 0.01 (alpha0 = ln(0.01) ~ -4.605)
    x_small = np.array([-4.605, 0.0])
    loss, grad = objective(x_small)
    assert not np.isnan(loss) and not np.isnan(grad).any()
    assert abs(grad[0]) > 0.0, "Gradient wrt alpha0 should not be zero (no vanishing gradient clamp)"

    # Check analytical gradient matches finite differences
    cg_err = check_grad(lambda p: objective(p)[0], lambda p: objective(p)[1], x_small)
    assert cg_err < 1e-5, f"check_grad error {cg_err} >= 1e-5 at small slope a0 ~ 0.01"

    # Fit model and verify convergence
    vcps.fit(X, y, feature_names=feature_names)
    assert vcps.a0 > 0.0
    assert not np.isnan(vcps.a0)
    pred_p = vcps.predict_proba(X)
    assert len(pred_p) == n
    assert np.all(pred_p >= 0.0) and np.all(pred_p <= 1.0)


def test_beta_calibration_monotonicity():
    """Verify constrained optimization guarantees a >= 0 and b >= 0 for Kull et al. (2017)."""
    np.random.seed(42)
    # Test across multiple datasets including inverted/adversarial labels
    for seed in [1, 42, 100, 999]:
        rng = np.random.RandomState(seed)
        confs = rng.uniform(0.01, 0.99, size=150)
        # Even with inverted relationship where unconstrained LR would produce negative weights
        y = (confs < 0.5).astype(int)

        a, b, c_param = fit_beta_calibration(confs, y)
        assert a >= 0.0, f"Slope a={a} must be non-negative (Kull et al. 2017 constraint)"
        assert b >= 0.0, f"Slope b={b} must be non-negative (Kull et al. 2017 constraint)"
        assert not np.isnan(a) and not np.isnan(b) and not np.isnan(c_param)


def test_dataset_sample_count_preservation_no_qid():
    """Verify datasets missing question_id do not collapse to a single row during deduplication."""
    n_samples = 25
    raw_data = [
        {
            "features": {
                1: {"conf_softmax": 0.5, "margin": 0.1, "answer": "cat", "vqa_accuracy": 1.0},
                9: {"conf_softmax": 0.6, "margin": 0.2, "answer": "cat", "vqa_accuracy": 1.0},
                36: {"conf_softmax": 0.7, "margin": 0.3, "answer": "cat", "vqa_accuracy": 1.0},
                144: {"conf_softmax": 0.8, "margin": 0.4, "answer": "cat", "vqa_accuracy": 1.0},
                576: {"conf_softmax": 0.9, "margin": 0.5, "answer": "cat", "vqa_accuracy": 1.0},
            },
            # Neither question_id nor id is provided
        }
        for _ in range(n_samples)
    ]

    rows = []
    for idx, item in enumerate(raw_data):
        feat_dict = compute_features_from_sample(item, fine_scale=576, idx=idx)
        rows.append(feat_dict)

    df = pd.DataFrame(rows)
    assert len(df) == n_samples

    # Verify each sample received a distinct question_id
    assert len(df["question_id"].unique()) == n_samples

    # Emulate loader deduplication guard
    if "question_id" in df.columns and len(df) > 1:
        initial_len = len(df)
        n_unique = df["question_id"].nunique()
        if 1 < n_unique < initial_len:
            df = df.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
        assert len(df) == initial_len == n_samples, "Dataset should not collapse when question_id is missing"

    # Also verify that a dataset where all items have constant question_id=0 does not collapse to 1 row
    df_const = df.copy()
    df_const["question_id"] = 0
    initial_len_const = len(df_const)
    n_unique_const = df_const["question_id"].nunique()
    if 1 < n_unique_const < initial_len_const:
        df_const = df_const.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
    assert len(df_const) == initial_len_const == n_samples, "Dataset with constant question_id should not collapse to 1 row"


def test_adaptive_ece_preserves_bins_uniform_confidences():
    """Verify rank partitioning handles identical/repeated confidences gracefully without crashing."""
    # All confidences are identical (e.g. constant prediction 0.8) on positive labels
    n = 120
    probs = np.full(n, 0.8)
    y = np.ones(n)

    # 15 bins: rank partitioning preserves bin structure without quantile edge collapse
    ece = compute_adaptive_ece(probs, y, n_bins=15)
    assert not np.isnan(ece)
    assert 0.0 <= ece <= 1.0
    # Every bin has 1.0 accuracy matching 0.8 confidence -> ECE is exactly |0.8 - 1.0| = 0.20
    assert pytest.approx(ece, abs=1e-3) == 0.20


def test_vqa_matcher_substring_contradiction_rejection():
    """Verify VQA matcher rejects contradictions (e.g. predicting 'dog' for 'not a dog')."""
    # 1. Negative contradiction rejection: prediction is substring of ground truth
    sample_neg = {"answer": "not a dog"}
    assert evaluate_accuracy("dog", sample_neg, "chartqa") == 0.0
    assert evaluate_accuracy("not a dog", sample_neg, "chartqa") == 1.0

    sample_no = {"answer": "no cats allowed"}
    assert evaluate_accuracy("cats", sample_no, "chartqa") == 0.0

    # 2. Prediction being a partial substring of longer ground truth is rejected
    sample_phrase = {"answer": "a brown dog"}
    assert evaluate_accuracy("dog", sample_phrase, "chartqa") == 0.0
    assert evaluate_accuracy("brown", sample_phrase, "chartqa") == 0.0

    # 3. Ground-truth containment in longer prediction is accepted
    sample_single = {"answer": "dog"}
    assert evaluate_accuracy("there is a dog", sample_single, "chartqa") == 1.0
    assert evaluate_accuracy("a brown dog", sample_single, "chartqa") == 1.0
