"""
Unit and Integration Test Suite for Visual Token Budget Study (ADR 0005).

Covers:
1. Scale hierarchy, single-pass token counts, and cumulative token sums for M3 and MQT.
2. Multi-depth prefix feature extraction across k in {1, 2, 3, 4, 5}.
3. Zero-division protections and degenerate handling at k=1.
4. Estimator fitting and full metric panel evaluation at arbitrary token depth.
5. Core 7 Macro-averaging logic and isolation from adversarial datasets.
6. Option A Universal Ranking and publication LaTeX subtable formatting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.extract_features import generate_mock_extraction
from scripts.run_token_budget_study import (
    CORE_7_DATASETS,
    METHODS_ORDER,
    compute_core7_macro,
    fit_and_eval_level,
    format_token_budget_latex_subtable,
    get_scale_levels,
    rank_and_format_latex,
)
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS, FEATURE_KEYS
from trajectory_calibration.features.extractor import compute_features_from_sample


def test_scale_levels_and_cumulative_tokens() -> None:
    """Verify single-pass and cumulative token accounting for M3 and MQT."""
    # M3-LLaVA (7B): scales [1, 9, 36, 144, 576]
    m3_levels = get_scale_levels("m3")
    assert len(m3_levels) == 5
    assert [lvl["level"] for lvl in m3_levels] == [1, 2, 3, 4, 5]
    assert [lvl["scale"] for lvl in m3_levels] == [1, 9, 36, 144, 576]
    assert [lvl["single_tokens"] for lvl in m3_levels] == [1, 9, 36, 144, 576]
    assert [lvl["cum_tokens"] for lvl in m3_levels] == [1, 10, 46, 190, 766]
    assert m3_levels[0]["prefix_scales"] == [1]
    assert m3_levels[2]["prefix_scales"] == [1, 9, 36]
    assert m3_levels[4]["prefix_scales"] == [1, 9, 36, 144, 576]

    # MQT-LLaVA (7B): scales [1, 9, 36, 144, 256]
    mqt_levels = get_scale_levels("mqt")
    assert len(mqt_levels) == 5
    assert [lvl["level"] for lvl in mqt_levels] == [1, 2, 3, 4, 5]
    assert [lvl["scale"] for lvl in mqt_levels] == [1, 9, 36, 144, 256]
    assert [lvl["single_tokens"] for lvl in mqt_levels] == [1, 9, 36, 144, 256]
    assert [lvl["cum_tokens"] for lvl in mqt_levels] == [1, 10, 46, 190, 446]
    assert mqt_levels[4]["prefix_scales"] == [1, 9, 36, 144, 256]


def test_prefix_feature_extraction_all_depths() -> None:
    """Verify compute_features_from_sample across prefix lengths k=1..5 without errors, NaNs, or Infs."""
    mock_samples = generate_mock_extraction("chartqa", num_samples=10, arch="m3")
    sample = mock_samples[0]

    for k in range(1, 6):
        prefix = [1, 9, 36, 144, 576][:k]
        feats = compute_features_from_sample(sample, fine_scale=576, scales=prefix)

        # 1. Canonical 5D Keys must exist and be finite
        for key in CANONICAL_5D_KEYS:
            assert key in feats
            val = feats[key]
            assert isinstance(val, (int, float))
            assert np.isfinite(val), f"Non-finite value for {key} at k={k}"

        # 2. All 17D Keys must exist and be finite
        for key in FEATURE_KEYS:
            assert key in feats
            val = feats[key]
            assert isinstance(val, (int, float))
            assert np.isfinite(val), f"Non-finite value for {key} at k={k}"

        # 3. Ground truth target and final confidence
        assert "c_fine" in feats
        assert "is_correct" in feats
        assert feats["is_correct"] in [0, 1]
        assert 0.0 <= float(feats["c_fine"]) <= 1.0

        # 4. For k=1: zero division guards
        if k == 1:
            assert feats["x2"] == 1.0  # single answer -> 1.0
            assert feats["x3"] == 0.0  # zero denom slope -> 0.0
            assert feats["x4"] == 0.0  # 0 flips -> 0.0
            assert feats["x5"] == 0.0  # 0 monotonicity comparisons -> 0.0


def test_fit_and_eval_level_on_mock_data() -> None:
    """Verify fit_and_eval_level trains all 4 calibrators and produces valid metric panels."""
    np.random.seed(42)
    mock_samples = generate_mock_extraction("ai2d", num_samples=80, arch="m3")
    prefix = [1, 9, 36]
    rows = [compute_features_from_sample(s, fine_scale=576, scales=prefix) for s in mock_samples]
    df = pd.DataFrame(rows)

    train_df = df.iloc[:50].reset_index(drop=True)
    test_df = df.iloc[50:].reset_index(drop=True)

    evals = fit_and_eval_level(train_df, test_df, seed=42)

    assert set(evals.keys()) == set(METHODS_ORDER)
    for panel in evals.values():
        assert "ece_percent" in panel
        assert "adaptive_ece_percent" in panel
        assert "brier" in panel
        assert "auroc" in panel
        assert np.isfinite(panel["ece_percent"])
        assert np.isfinite(panel["adaptive_ece_percent"])
        assert np.isfinite(panel["brier"])
        assert np.isfinite(panel["auroc"])


def test_core7_macro_aggregation_isolation() -> None:
    """Verify compute_core7_macro aggregates strictly over Core 7 datasets and ignores adversarial ones."""
    rows = []
    # Generate mock raw rows for 7 core + 1 adversarial dataset
    datasets = [*CORE_7_DATASETS, "avqa"]
    for ds in datasets:
        for lvl in range(1, 6):
            for seed in [42, 43]:
                for m_name in METHODS_ORDER:
                    rows.append(
                        {
                            "dataset": ds,
                            "arch": "m3",
                            "level": lvl,
                            "scale": 10 * lvl,
                            "single_tokens": 10 * lvl,
                            "cum_tokens": 10 * lvl * lvl,
                            "tokens_used": 10 * lvl,
                            "seed": seed,
                            "method": m_name,
                            "ece_percent": 10.0 if ds != "avqa" else 99.0,
                            "adaptive_ece_percent": 8.0 if ds != "avqa" else 99.0,
                            "brier": 0.15 if ds != "avqa" else 0.99,
                            "auroc": 0.80 if ds != "avqa" else 0.10,
                        }
                    )

    raw_df = pd.DataFrame(rows)
    macro_df = compute_core7_macro(raw_df)

    assert len(macro_df) == 5 * len(METHODS_ORDER)
    # The macro mean across Core 7 should be exactly 10.0 for ece and 8.0 for ada_ece (not polluted by avqa's 99.0)
    for lvl in range(1, 6):
        sub = macro_df[macro_df["level"] == lvl]
        for _, row in sub.iterrows():
            assert np.isclose(row["ece_mean"], 10.0)
            assert np.isclose(row["ada_ece_mean"], 8.0)
            assert np.isclose(row["brier_mean"], 0.15)
            assert np.isclose(row["auroc_mean"], 0.80)


def test_rank_and_format_latex_option_a() -> None:
    """Verify rank_and_format_latex formats Rank 1 as bold and Rank 2 as italic."""
    # Lower is better (e.g. ECE)
    vals_low = [12.50, 4.20, 8.30, 4.20]
    formatted_low = rank_and_format_latex(vals_low, higher_is_better=False)
    assert formatted_low[0] == "12.50"
    assert formatted_low[1] == "\\textbf{4.20}"
    assert formatted_low[2] == "\\textit{8.30}"
    assert formatted_low[3] == "\\textbf{4.20}"

    # Higher is better (e.g. AUROC)
    vals_high = [0.650, 0.780, 0.720, 0.500]
    formatted_high = rank_and_format_latex(vals_high, higher_is_better=True, decimals=3)
    assert formatted_high[0] == "0.650"
    assert formatted_high[1] == "\\textbf{0.780}"
    assert formatted_high[2] == "\\textit{0.720}"
    assert formatted_high[3] == "0.500"


def test_latex_subtable_generation(tmp_path: Path) -> None:
    """Verify format_token_budget_latex_subtable generates syntactically valid LaTeX tables."""
    rows = []
    for lvl in range(1, 6):
        for m_name in METHODS_ORDER:
            rows.append(
                {
                    "level": lvl,
                    "scale": 10 * lvl,
                    "cum_tokens": 10 * lvl * lvl,
                    "tokens_used": 10 * lvl * lvl if "5D" in m_name else 10 * lvl,
                    "method": m_name,
                    "ece_percent_mean": 5.0 + lvl,
                    "adaptive_ece_percent_mean": 4.0 + lvl,
                    "brier_mean": 0.12,
                    "auroc_mean": 0.75,
                }
            )
    df = pd.DataFrame(rows)
    tex_str = format_token_budget_latex_subtable(
        df, "Sample Caption", "tab:sample_test", is_macro=False
    )

    assert "\\begin{table}[t]" in tex_str
    assert "\\end{table}" in tex_str
    assert "\\label{tab:sample_test}" in tex_str
    assert "\\rowcolor{gray!10}" in tex_str
    assert "\\textbf{Trajectory Platt (5D)}" in tex_str
