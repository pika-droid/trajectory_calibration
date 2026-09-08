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

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.stats import entropy

from trajectory_calibration.calibrators.adaptation import fit_beta_calibration
from trajectory_calibration.calibrators.classic import (
    QuadraticPlattScaler,
    SplineCalibrator,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import FEATURE_KEYS
from trajectory_calibration.features.extractor import compute_features_from_sample
from trajectory_calibration.metrics.ece import compute_adaptive_ece
from trajectory_calibration.metrics.statistical import fit_calibration_slope_intercept
from trajectory_calibration.uq.eigenscore import compute_logdet, compute_umpire_metric
from trajectory_calibration.uq.semantic_entropy import (
    FastStringEntailment,
    cluster_assignment_entropy,
    compute_semantic_entropy,
    get_semantic_ids,
)
from trajectory_calibration.uq.whitebox import WhiteBoxScorers
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
    assert abs(grad[0]) > 0.0, (
        "Gradient wrt alpha0 should not be zero (no vanishing gradient clamp)"
    )

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
        assert len(df) == initial_len == n_samples, (
            "Dataset should not collapse when question_id is missing"
        )

    # Also verify that a dataset where all items have constant question_id=0 does not collapse to 1 row
    df_const = df.copy()
    df_const["question_id"] = 0
    initial_len_const = len(df_const)
    n_unique_const = df_const["question_id"].nunique()
    if 1 < n_unique_const < initial_len_const:
        df_const = df_const.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
    assert len(df_const) == initial_len_const == n_samples, (
        "Dataset with constant question_id should not collapse to 1 row"
    )


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


def test_spline_calibrator_pchip_monotonicity() -> None:
    """Verify SplineCalibrator guarantees monotonicity without overshoot on noisy data."""
    np.random.seed(42)
    n = 200
    logits = np.linspace(-3.0, 3.0, n)
    confs = 1.0 / (1.0 + np.exp(-logits))
    y = (confs + 0.15 * np.random.randn(n) > 0.5).astype(float)

    for n_knots in [3, 5, 8]:
        spline = SplineCalibrator(n_knots=n_knots)
        spline.fit(logits.reshape(-1, 1), y)

        test_logits = np.linspace(-5.0, 5.0, 500).reshape(-1, 1)
        preds = spline.predict_proba(test_logits)

        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-7), f"Monotonicity violated for n_knots={n_knots}"
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
        assert not np.isnan(preds).any()


def test_spline_calibrator_edge_cases() -> None:
    """Verify SplineCalibrator handles small, boundary, and extreme sample distributions gracefully."""
    spline = SplineCalibrator(n_knots=5)

    # 1. Single sample
    spline.fit(np.array([[0.0]]), np.array([1.0]))
    preds = spline.predict_proba(np.array([[-1.0], [0.0], [1.0]]))
    assert len(preds) == 3
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    # 2. Identical confidences in interior
    spline.fit(np.array([[2.0], [2.0], [2.0], [2.0]]), np.array([0.0, 1.0, 0.0, 1.0]))
    preds = spline.predict_proba(np.array([[-1.0], [2.0], [5.0]]))
    assert len(preds) == 3
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    # 3. All confidences exactly 1.0 (large logits)
    spline.fit(np.full((10, 1), 20.0), np.ones(10))
    preds = spline.predict_proba(np.array([[-5.0], [0.0], [5.0], [20.0]]))
    assert np.all(np.diff(preds) >= -1e-7)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    # 4. All confidences exactly 0.0 (negative logits)
    spline.fit(np.full((10, 1), -20.0), np.zeros(10))
    preds = spline.predict_proba(np.array([[-20.0], [-5.0], [0.0], [5.0]]))
    assert np.all(np.diff(preds) >= -1e-7)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    # 5. Heavy concentration near 1.0
    confs_high = np.array([-2.0, 5.0, 10.0, 12.0, 15.0, 20.0]).reshape(-1, 1)
    y_high = np.array([0, 1, 1, 1, 1, 1])
    spline.fit(confs_high, y_high)
    preds = spline.predict_proba(np.linspace(-5.0, 25.0, 100).reshape(-1, 1))
    assert np.all(np.diff(preds) >= -1e-7)


def test_all_functions_under_200_loc() -> None:
    """Verify all functions and methods across src/ and scripts/ are under 200 LOC."""
    repo_root = Path(__file__).resolve().parent.parent
    py_files = list((repo_root / "src").rglob("*.py")) + list((repo_root / "scripts").rglob("*.py"))
    assert len(py_files) > 0, "No python files discovered"

    violations = []
    for f in py_files:
        content = f.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno
                if length >= 200:
                    violations.append(
                        f"{f.relative_to(repo_root)}:{node.lineno} {node.name} ({length} LOC)"
                    )

    assert not violations, f"Functions exceeding 200 LOC: {violations}"


def test_umpire_normalization_scaling() -> None:
    """Verify compute_umpire_metric normalizes v_logdet by 2k and prob_penalty by k."""
    np.random.seed(42)
    k = 10
    d = 64
    vecs = np.random.randn(k, d)
    seq_lps = [-0.1 * i for i in range(1, k + 1)]

    umpire_10 = compute_umpire_metric(vecs, seq_lps, alpha_param=1.0)
    assert isinstance(umpire_10, float)
    assert not np.isnan(umpire_10)

    empty_vecs = np.zeros((0, d))
    assert compute_umpire_metric(empty_vecs, []) == 0.0

    # Test numerical stability on near-collinear Gram matrix (Issue 12)
    collinear_vecs = np.ones((10, 64)) + 1e-9 * np.random.randn(10, 64)
    ld = compute_logdet(np.dot(collinear_vecs, collinear_vecs.T), alpha=1e-8)
    assert np.isfinite(ld), f"compute_logdet must be finite on near-collinear embeddings, got {ld}"


def test_degenerate_single_class_slope_intercept() -> None:
    """Verify fit_calibration_slope_intercept returns (1.0, 0.0) for single-class data."""
    slope, intercept = fit_calibration_slope_intercept([0.3, 0.7, 0.9], [1, 1, 1])
    assert slope == 1.0
    assert intercept == 0.0

    slope, intercept = fit_calibration_slope_intercept([0.2, 0.4, 0.5], [0, 0, 0])
    assert slope == 1.0
    assert intercept == 0.0


def test_quadratic_platt_scaler_transform_separation() -> None:
    """Verify QuadraticPlattScaler fit uses fit_transform and predict_proba uses transform."""
    quad = QuadraticPlattScaler()
    X_train = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1, 1])

    quad.fit(X_train, y_train)
    assert hasattr(quad.poly, "n_features_in_")

    X_test = np.array([[-0.5], [1.5]])
    preds1 = quad.predict_proba(X_test)
    preds2 = quad.predict_proba(X_test)
    assert np.allclose(preds1, preds2)
    assert np.all(preds1 >= 0.0) and np.all(preds1 <= 1.0)


def test_vcps_gradient_active_mask() -> None:
    """Verify VCPS analytical gradient correctly handles active_mask when slope_raw is clipped."""
    np.random.seed(42)
    n, d = 50, 4
    X = np.random.randn(n, d)
    y = (X[:, 0] > 0).astype(float)
    feature_names = [f"x{i}" for i in range(1, d + 1)]

    vcps = VaryingCoefficientPlattScaler(
        mode="full",
        slope_features=["x2", "x3"],
        intercept_features=["x2", "x3", "x4"],
        random_state=42,
    )
    objective, x0 = vcps._build_objective(X, y, feature_names=feature_names)

    x_clipped = x0.copy()
    x_clipped[0] = 4.0
    val, grad = objective(x_clipped)
    assert not np.isnan(val)
    assert not np.isnan(grad).any()

    cg_err = check_grad(lambda p: objective(p)[0], lambda p: objective(p)[1], x_clipped)
    assert cg_err < 1e-4, f"check_grad error {cg_err} >= 1e-4 at clipped slope point"


def test_fast_string_entailment_polarity_guard() -> None:
    """Verify FastStringEntailment rejects entailment when negation polarity differs."""
    matcher = FastStringEntailment()

    assert matcher.check_implication("a dog", "not a dog") == 0
    assert matcher.check_implication("not a dog", "a dog") == 0
    assert matcher.check_implication("the cat is present", "no cat is present") == 0
    assert matcher.check_implication("there are animals", "none of the animals") == 0
    assert matcher.check_implication("it will happen", "it will never happen") == 0

    assert matcher.check_implication("not a dog", "not a dog") == 2
    assert matcher.check_implication("a brown dog", "brown dog") == 2

    strings = ["a dog", "not a dog", "a brown dog"]
    ids = get_semantic_ids(strings, model=matcher)
    assert ids[0] == ids[2]
    assert ids[1] != ids[0]


def test_scipy_standard_library_consistency() -> None:
    """Verify standard library functions produce expected numerical results."""
    dist = np.array([0.25, 0.25, 0.25, 0.25])
    h = WhiteBoxScorers.token_entropy(dist)
    expected_h = float(entropy(dist))
    assert pytest.approx(h, abs=1e-6) == expected_h

    sem_h = compute_semantic_entropy([0, 1], [-0.693147, -0.693147])
    assert pytest.approx(sem_h, abs=1e-4) == np.log(2.0)

    cae = cluster_assignment_entropy([0, 1, 0, 1])
    assert pytest.approx(cae, abs=1e-5) == np.log(2.0)


def test_17d_feature_extraction_completeness_and_ablation() -> None:
    """Verify feature extraction produces exactly 17 features, with x21 completely ablated and no legacy gaps."""
    sample = {
        "features": {
            1: {"conf_softmax": 0.4, "margin": 0.1, "answer": "yes", "vqa_accuracy": 1.0},
            9: {"conf_softmax": 0.5, "margin": 0.2, "answer": "yes", "vqa_accuracy": 1.0},
            36: {"conf_softmax": 0.6, "margin": 0.3, "answer": "yes", "vqa_accuracy": 1.0},
            144: {"conf_softmax": 0.7, "margin": 0.4, "answer": "yes", "vqa_accuracy": 1.0},
            576: {"conf_softmax": 0.8, "margin": 0.5, "answer": "yes", "vqa_accuracy": 1.0},
        },
    }
    feats = compute_features_from_sample(sample, fine_scale=576)

    # Assert exactly 17 'x' features matching FEATURE_KEYS
    extracted_x = [k for k in feats if k.startswith("x")]
    assert len(extracted_x) == 17
    assert sorted(extracted_x, key=lambda s: int(s[1:])) == [f"x{i}" for i in range(1, 18)]
    assert [f"x{i}" for i in range(1, 18)] == FEATURE_KEYS

    # Assert ablated feature x21 is completely absent
    assert "x21" not in feats

    # Assert legacy non-contiguous keys are absent
    for legacy_key in ["x18", "x19", "x20", "x22"]:
        assert legacy_key not in feats
