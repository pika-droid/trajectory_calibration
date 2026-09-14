"""
Unit and Integration Tests for 14-Dataset Combinatorial 5D Trajectory Feature Ablation Study.

Verifies:
1. Combinatorial generation: Exactly 2^5 - 1 = 31 subsets with proper cardinality counts.
2. Decoupled Scaling Invariant: ColumnTransformer correctly isolates raw x1 vs standardized z.
3. Metric calculations and aggregation formulas.
4. Generated CSV and LaTeX table structural validity and Decimal Option A ranking.
5. Function LOC constraint (< 200 LOC per function).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from scripts.generate_ablation_latex_tables import (
    rank_and_format_decimal,
    rank_and_format_delta,
)
from scripts.run_combinatorial_ablation_5d import (
    ALL_14_DATASETS,
    build_ablation_pipeline,
    compute_loo_sensitivity_summary,
    compute_macro_progression_summary,
    compute_per_dataset_best_subsets,
    evaluate_single_subset,
    fit_predict_oof,
    generate_all_5d_subsets,
)
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_generate_all_5d_subsets() -> None:
    """Verify exactly 31 non-empty subsets are generated with correct binomial counts."""
    subsets = generate_all_5d_subsets()
    assert len(subsets) == 31, f"Expected 31 subsets, got {len(subsets)}"

    # Check counts per cardinality: C(5,1)=5, C(5,2)=10, C(5,3)=10, C(5,4)=5, C(5,5)=1
    cardinalities = [len(s) for s in subsets]
    for k, expected_count in [(1, 5), (2, 10), (3, 10), (4, 5), (5, 1)]:
        assert cardinalities.count(k) == expected_count, (
            f"For k={k}, expected {expected_count} subsets, got {cardinalities.count(k)}"
        )

    # Verify all feature keys in subsets are from CANONICAL_5D_KEYS
    for s in subsets:
        assert all(f in CANONICAL_5D_KEYS for f in s)


def test_decoupled_scaling_pipeline_structure() -> None:
    """Verify pipeline ColumnTransformer adheres to Decoupled Scaling Invariant."""
    # Case 1: 1D anchor alone
    pipe_1d = build_ablation_pipeline(["x1"])
    prep_1d = pipe_1d.named_steps["prep"]
    assert isinstance(prep_1d, ColumnTransformer)
    assert prep_1d.transformers[0][0] == "x1_pass"
    assert prep_1d.transformers[0][1] == "passthrough"

    # Case 2: Multi-D with x1 anchor
    pipe_multi = build_ablation_pipeline(["x1", "x2", "x3"])
    prep_multi = pipe_multi.named_steps["prep"]
    assert isinstance(prep_multi, ColumnTransformer)
    assert prep_multi.transformers[0][0] == "x1_pass"
    assert prep_multi.transformers[0][1] == "passthrough"
    assert prep_multi.transformers[1][0] == "z_scale"
    assert prep_multi.transformers[1][2] == ["x2", "x3"]

    # Case 3: Pure trajectory without x1
    pipe_no_x1 = build_ablation_pipeline(["x2", "x4", "x5"])
    prep_no_x1 = pipe_no_x1.named_steps["prep"]
    assert isinstance(prep_no_x1, ColumnTransformer)
    assert prep_no_x1.transformers[0][0] == "z_scale"
    assert prep_no_x1.transformers[0][2] == ["x2", "x4", "x5"]


def test_fit_and_evaluate_single_subset() -> None:
    """Verify out-of-fold cross-validation and metric panel on synthetic DataFrame."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
            "x4": np.random.randn(n),
            "x5": np.random.randn(n),
            "is_correct": np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
        }
    )

    res = evaluate_single_subset(df, ds_name="test_ds", subset=["x1", "x2"], n_splits=3, seed=42)
    assert res["dataset"] == "test_ds"
    assert res["cardinality"] == 2
    assert res["subset"] == "x1+x2"
    assert res["has_x1"] is True
    assert 0.0 <= float(res["ada_ece"]) <= 1.0
    assert 0.0 <= float(res["ece"]) <= 1.0
    assert 0.0 <= float(res["brier"]) <= 1.0
    assert 0.0 <= float(res["auroc"]) <= 1.0


def test_summaries_and_loo_calculations() -> None:
    """Verify macro progression, LOO sensitivity, and per-dataset best subsets calculations."""
    datasets = ["ds_a", "ds_b"]
    subsets = generate_all_5d_subsets()
    rows = []
    np.random.seed(42)
    for ds in datasets:
        for s in subsets:
            card = len(s)
            rows.append(
                {
                    "dataset": ds,
                    "cardinality": card,
                    "subset": "+".join(s),
                    "has_x1": "x1" in s,
                    "ada_ece": 0.05 - 0.005 * card + np.random.uniform(0, 0.002),
                    "ece": 0.05 - 0.004 * card,
                    "brier": 0.15 - 0.005 * card,
                    "auroc": 0.60 + 0.03 * card,
                }
            )
    df_raw = pd.DataFrame(rows)

    df_macro = compute_macro_progression_summary(df_raw)
    assert len(df_macro) == 5
    assert list(df_macro["cardinality"]) == [1, 2, 3, 4, 5]

    df_loo = compute_loo_sensitivity_summary(df_raw)
    assert len(df_loo) == 5
    assert set(df_loo["feature_dropped"]) == set(CANONICAL_5D_KEYS)

    df_best = compute_per_dataset_best_subsets(df_raw)
    assert len(df_best) == len(datasets) * 5


def test_rank_and_format_decimal() -> None:
    """Verify Decimal Option A formatting (bold Rank 1, italic Rank 2)."""
    vals = [0.052, 0.031, 0.045, 0.060]
    # Lower is better (e.g. Ada-ECE)
    formatted = rank_and_format_decimal(vals, higher_is_better=False, decimals=3)
    assert formatted[1] == r"\textbf{0.031}"  # Rank 1
    assert formatted[2] == r"\textit{0.045}"  # Rank 2
    assert formatted[0] == "0.052"
    assert formatted[3] == "0.060"

    # Higher is better (e.g. AUROC)
    formatted_h = rank_and_format_decimal(vals, higher_is_better=True, decimals=3)
    assert formatted_h[3] == r"\textbf{0.060}"  # Rank 1
    assert formatted_h[0] == r"\textit{0.052}"  # Rank 2


def test_rank_and_format_delta() -> None:
    """Verify signed Option A formatting for LOO sensitivity deltas."""
    # Case 1: Ada-ECE degradation (higher positive is worse/more sensitive)
    vals_ada = [0.0031, -0.0004, 0.0006, -0.0009]
    res_ada = rank_and_format_delta(vals_ada, higher_degradation_is_worse=True, decimals=4)
    assert res_ada[0] == r"\textbf{+0.0031}"  # Rank 1 degradation
    assert res_ada[2] == r"\textit{+0.0006}"  # Rank 2 degradation
    assert res_ada[1] == "-0.0004"
    assert res_ada[3] == "-0.0009"

    # Case 2: AUROC degradation (more negative is worse/more sensitive)
    vals_auroc = [-0.075, +0.003, -0.004, +0.000]
    res_auroc = rank_and_format_delta(vals_auroc, higher_degradation_is_worse=False, decimals=3)
    assert res_auroc[0] == r"\textbf{-0.075}"  # Rank 1 degradation
    assert res_auroc[2] == r"\textit{-0.004}"  # Rank 2 degradation
    assert res_auroc[1] == "+0.003"
    assert res_auroc[3] == "+0.000"


def test_fit_predict_oof_single_class_edge_case() -> None:
    """Verify fit_predict_oof gracefully handles single-class edge cases without crashing."""
    n = 20
    df_all_ones = pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "is_correct": np.ones(n, dtype=int),
        }
    )
    preds = fit_predict_oof(df_all_ones, subset=["x1", "x2"], n_splits=3)
    assert len(preds) == n
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
    assert not np.isnan(preds).any()


def test_loo_latex_table_structure() -> None:
    """Verify table_ablation_5d_loo.tex has exactly 7 columns in tabular and rows, and subscripted math."""
    tbl_path = ROOT_DIR / "dataset_tables" / "table_ablation_5d_loo.tex"
    if tbl_path.exists():
        content = tbl_path.read_text(encoding="utf-8")
        assert r"\begin{tabular}{clccccl}" in content
        # Ensure no un-subscripted $x1$, $x2$, etc.
        for feat in ["x1", "x2", "x3", "x4", "x5"]:
            assert f"${feat}$" not in content, f"Found un-subscripted ${feat}$ in LOO table"
            assert f"$x_{{{feat[1]}}}$" in content, (
                f"Expected subscripted $x_{{{feat[1]}}}$ in LOO table"
            )


def test_ablation_artifacts_exist_and_valid() -> None:
    """Verify that real CSV, plot, and LaTeX table artifacts exist in repository."""
    res_dir = ROOT_DIR / "results" / "experiments" / "ablation_5d"
    figs_dir = res_dir / "figures"
    tbls_dir = ROOT_DIR / "dataset_tables"

    for arch in ["m3", "mqt"]:
        raw_csv = res_dir / f"ablation_5d_{arch}_raw_all_combinations.csv"
        macro_csv = res_dir / f"ablation_5d_{arch}_macro_progression.csv"
        loo_csv = res_dir / f"ablation_5d_{arch}_loo_sensitivity.csv"
        best_csv = res_dir / f"ablation_5d_{arch}_per_dataset_best_subsets.csv"

        assert raw_csv.exists(), f"Missing {raw_csv}"
        assert macro_csv.exists(), f"Missing {macro_csv}"
        assert loo_csv.exists(), f"Missing {loo_csv}"
        assert best_csv.exists(), f"Missing {best_csv}"

        df_raw = pd.read_csv(raw_csv)
        assert len(df_raw) == 14 * 31, f"Expected 434 rows in {raw_csv}, got {len(df_raw)}"
        assert set(df_raw["dataset"].unique()) == set(ALL_14_DATASETS)

        df_macro = pd.read_csv(macro_csv)
        assert len(df_macro) == 5

        df_loo = pd.read_csv(loo_csv)
        assert len(df_loo) == 5

    # Figures
    assert (figs_dir / "figure_ablation_5d_cardinality_progression.png").exists()
    assert (figs_dir / "figure_ablation_5d_loo_marginal_utility.png").exists()

    # LaTeX Tables
    assert (tbls_dir / "table_ablation_5d_macro_progression.tex").exists()
    assert (tbls_dir / "table_ablation_5d_loo.tex").exists()
    assert (tbls_dir / "table_ablation_5d_14ds_grid.tex").exists()


def test_all_functions_under_200_loc() -> None:
    """Verify all functions across all 3 new ablation scripts are strictly under 200 LOC."""
    scripts_to_check = [
        ROOT_DIR / "scripts" / "run_combinatorial_ablation_5d.py",
        ROOT_DIR / "scripts" / "plot_ablation_5d.py",
        ROOT_DIR / "scripts" / "generate_ablation_latex_tables.py",
    ]

    violations: list[str] = []
    for script_path in scripts_to_check:
        assert script_path.exists(), f"File {script_path} does not exist"
        tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_line = getattr(node, "end_lineno", node.lineno)
                loc = end_line - node.lineno + 1
                if loc >= 200:
                    violations.append(f"{script_path.name}::{node.name} ({loc} LOC)")

    assert not violations, f"Functions exceeding 200 LOC constraint: {violations}"


def test_generate_macro_progression_table_arbitrary_index() -> None:
    """Verify generate_macro_progression_table does not crash on non-contiguous DataFrame indices."""
    from scripts.generate_ablation_latex_tables import generate_macro_progression_table

    data = {
        "cardinality": [1, 2, 3, 4, 5],
        "n_subsets": [5, 10, 10, 5, 1],
        "best_subset": ["x5", "x2+x4", "x2+x4+x5", "x1+x2+x3+x5", "x1+x2+x3+x4+x5"],
        "best_has_x1": [False, False, False, True, True],
        "best_ada_ece": [0.0249, 0.0241, 0.0233, 0.0275, 0.0284],
        "best_ece": [0.0077, 0.0085, 0.0138, 0.0159, 0.0220],
        "best_brier": [0.1521, 0.1447, 0.1434, 0.1412, 0.1324],
        "best_auroc": [0.544, 0.603, 0.626, 0.639, 0.714],
        "worst_subset": ["x3", "x3+x4", "x1+x2+x4", "x1+x2+x3+x4", "x1+x2+x3+x4+x5"],
        "worst_ada_ece": [0.0487, 0.0383, 0.0329, 0.0315, 0.0284],
        "mean_ada_ece": [0.0329, 0.0313, 0.0297, 0.0287, 0.0284],
        "std_ada_ece": [0.0088, 0.0043, 0.0025, 0.0015, 0.0000],
        "min_ada_ece": [0.0249, 0.0241, 0.0233, 0.0275, 0.0284],
        "max_ada_ece": [0.0487, 0.0383, 0.0329, 0.0315, 0.0284],
        "mean_auroc": [0.597, 0.648, 0.679, 0.698, 0.714],
        "std_auroc": [0.052, 0.046, 0.037, 0.030, 0.000],
    }
    # Non-standard sliced/shuffled index
    df_m3 = pd.DataFrame(data, index=[10, 25, 42, 99, 105])
    df_mqt = pd.DataFrame(data, index=[5, 4, 3, 2, 1])

    tex = generate_macro_progression_table(df_m3, df_mqt)
    assert r"\begin{table*}[t]" in tex
    assert r"\end{table*}" in tex
    assert "M3-LLaVA (7B)" in tex
    assert "MQT-LLaVA (7B)" in tex


def test_all_tables_compendium_includes_ablation_group() -> None:
    """Verify all_tables_compendium.tex integrates Group 8 and contains valid LaTeX math syntax."""
    compendium_path = ROOT_DIR / "dataset_tables" / "all_tables_compendium.tex"
    if compendium_path.exists():
        content = compendium_path.read_text(encoding="utf-8")
        assert "Group 8: Combinatorial 5D Trajectory Feature Ablation Study" in content
        assert "table_ablation_5d_macro_progression" in content
        assert "table_ablation_5d_loo" in content
        assert "table_ablation_5d_14ds_grid" in content
        # Check no corrupted math tokens
        assert "at  = 0.0$" not in content
        assert "across  \\in" not in content
        assert "$T = 0.0$" in content
        assert "$T \\in" in content
