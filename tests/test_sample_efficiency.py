"""
Unit tests for Sample Efficiency Study (RQ2).

Verifies:
1. Exact subsample sizing and reproducible random state.
2. Label stratification and two-class preservation under extreme imbalance.
3. Capping and full-partition tagging when budget >= len(train_df).
4. Multi-estimator fitting and metric panel validity (NC, TS, Platt 1D, TP 5D, VCPS-5D).
5. Option A universal ranking formatting for LaTeX tables.
6. Macro aggregation consistency across budget tiers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_sample_efficiency_study import (
    METHODS_ORDER,
    compute_macro_aggregates,
    fit_and_evaluate_methods,
    rank_and_format_latex,
    subsample_train_split,
)
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS


def create_synthetic_dataframe(
    n_samples: int = 500, pos_ratio: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    """Creates a synthetic DataFrame with canonical 5D features."""
    rng = np.random.RandomState(seed)
    n_pos = int(n_samples * pos_ratio)
    n_neg = n_samples - n_pos
    labels = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(labels)

    x1 = rng.randn(n_samples) + (labels - 0.5) * 1.5
    c_fine = 1.0 / (1.0 + np.exp(-x1))

    data: dict[str, np.ndarray | list[str]] = {
        "x1": x1,
        "x2": rng.randn(n_samples),
        "x3": rng.randn(n_samples),
        "x4": rng.randn(n_samples),
        "x5": rng.randn(n_samples),
        "c_576": c_fine,
        "c_fine": c_fine,
        "is_correct": labels,
        "question_id": [f"q_{i}" for i in range(n_samples)],
    }
    return pd.DataFrame(data)


def test_subsample_train_split_exact_counts() -> None:
    """Validates exact target sample counts for different budget tiers."""
    df = create_synthetic_dataframe(n_samples=500, pos_ratio=0.5, seed=42)
    for budget in [50, 100, 200]:
        sub_df, eff_n, is_capped = subsample_train_split(df, budget=budget, seed=42)
        assert len(sub_df) == budget
        assert eff_n == budget
        assert not is_capped
        assert set(CANONICAL_5D_KEYS).issubset(sub_df.columns)


def test_subsample_train_split_stratification() -> None:
    """Validates both classes are present even under extreme class imbalance."""
    df = create_synthetic_dataframe(n_samples=800, pos_ratio=0.01, seed=42)
    assert df["is_correct"].sum() == 8

    for seed in [42, 43, 44, 45, 46]:
        sub_df, eff_n, is_capped = subsample_train_split(df, budget=50, seed=seed)
        assert len(sub_df) == 50
        assert eff_n == 50
        assert not is_capped
        assert sub_df["is_correct"].nunique() == 2, (
            f"Seed {seed} failed to preserve both classes: {sub_df['is_correct'].value_counts().to_dict()}"
        )


def test_subsample_train_split_capping() -> None:
    """Validates capping at len(train_df) when budget >= len(train_df) or 'Full'."""
    df = create_synthetic_dataframe(n_samples=300, pos_ratio=0.5, seed=42)

    # Budget 500 > 300
    sub_df, eff_n, is_capped = subsample_train_split(df, budget=500, seed=42)
    assert len(sub_df) == 300
    assert eff_n == 300
    assert is_capped

    # Budget 'Full'
    sub_df_full, eff_n_full, is_capped_full = subsample_train_split(df, budget="Full", seed=42)
    assert len(sub_df_full) == 300
    assert eff_n_full == 300
    assert is_capped_full


def test_fit_and_evaluate_methods_synthetic() -> None:
    """Validates multi-estimator fitting and output metric panel on synthetic data."""
    train_df = create_synthetic_dataframe(n_samples=50, pos_ratio=0.5, seed=42)
    test_df = create_synthetic_dataframe(n_samples=100, pos_ratio=0.5, seed=99)

    results = fit_and_evaluate_methods(train_df, test_df, seed=42)

    assert set(results.keys()) == set(METHODS_ORDER)

    for _m_name, panel in results.items():
        assert "adaptive_ece_percent" in panel
        assert "ece_percent" in panel
        assert "auroc" in panel
        assert "brier" in panel

        ada = float(panel["adaptive_ece_percent"])
        ece = float(panel["ece_percent"])
        auroc = float(panel["auroc"])
        brier = float(panel["brier"])

        assert not np.isnan(ada) and ada >= 0.0
        assert not np.isnan(ece) and ece >= 0.0
        assert not np.isnan(auroc) and 0.0 <= auroc <= 1.0
        assert not np.isnan(brier) and 0.0 <= brier <= 1.0


def test_latex_ranking_option_a() -> None:
    """Validates Option A formatting (bold rank 1, italic rank 2)."""
    # Lower is better (e.g. Ada-ECE)
    vals_low = [12.5, 5.2, 8.1, 15.0, 5.2]
    formatted_low = rank_and_format_latex(vals_low, higher_is_better=False, decimals=2)
    assert formatted_low[1] == "\\textbf{5.20}"
    assert formatted_low[4] == "\\textbf{5.20}"
    assert formatted_low[2] == "\\textit{8.10}"
    assert formatted_low[0] == "12.50"

    # Higher is better (e.g. AUROC)
    vals_high = [0.650, 0.720, 0.710, 0.580, None]
    formatted_high = rank_and_format_latex(vals_high, higher_is_better=True, decimals=3)
    assert formatted_high[1] == "\\textbf{0.720}"
    assert formatted_high[2] == "\\textit{0.710}"
    assert formatted_high[0] == "0.650"
    assert formatted_high[4] == "-"

    # Rounded float collision edge cases: values differing in raw float but equal at display precision
    vals_collision = [0.6714, 0.6712, 0.6711, 0.6560]
    fmt_coll = rank_and_format_latex(vals_collision, higher_is_better=True, decimals=3)
    assert fmt_coll[0] == "\\textbf{0.671}"
    assert fmt_coll[1] == "\\textbf{0.671}"
    assert fmt_coll[2] == "\\textbf{0.671}"
    assert fmt_coll[3] == "\\textit{0.656}"


def test_macro_aggregation_consistency() -> None:
    """Validates compute_macro_aggregates correctly summarizes per-seed evaluations."""
    records = []
    for arch in ["m3"]:
        for ds in ["ds1", "ds2"]:
            for b in ["50", "Full"]:
                for s in [42, 43]:
                    for m in METHODS_ORDER:
                        records.append(
                            {
                                "dataset": ds,
                                "arch": arch,
                                "budget": b,
                                "effective_n": 50 if b == "50" else 200,
                                "is_capped": b == "Full",
                                "seed": s,
                                "method": m,
                                "adaptive_ece_percent": 6.0
                                if m == "VCPS-5D (Our Method)"
                                else 10.0,
                                "ece_percent": 5.0,
                                "auroc": 0.70,
                                "brier": 0.20,
                            }
                        )

    raw_df = pd.DataFrame(records)
    macro_df = compute_macro_aggregates(raw_df)

    assert not macro_df.empty
    vcps_50 = macro_df[
        (macro_df["budget"] == "50") & (macro_df["method"] == "VCPS-5D (Our Method)")
    ]
    assert len(vcps_50) == 1
    assert abs(float(vcps_50["ada_ece_mean"].iloc[0]) - 6.0) < 1e-6
    assert abs(float(vcps_50["ada_ece_std"].iloc[0]) - 0.0) < 1e-6


def test_generate_sample_efficiency_sheets(tmp_path: Path) -> None:
    """Verifies Excel workbook generation, sheet count, and table structure."""
    from sheets.scripts.generate_sample_efficiency_sheets import (
        ALL_14_DATASETS,
        generate_sheets,
    )

    out_file = tmp_path / "sample_efficiency_test.xlsx"
    results_dir = Path("results/experiments/sample_efficiency")
    generated_path = generate_sheets(results_dir=results_dir, output_file=out_file)

    assert generated_path.exists()
    xl = pd.ExcelFile(generated_path)
    sheet_names = xl.sheet_names
    assert len(sheet_names) == 15
    assert "Macro_Mean" in sheet_names
    for ds in ALL_14_DATASETS:
        assert ds in sheet_names

    df_macro = pd.read_excel(generated_path, sheet_name="Macro_Mean")
    expected_cols = [
        "Model",
        "Budget Tier",
        "Method",
        "Adaptive ECE (%)",
        "AUROC",
        "Brier Score",
    ]
    for col in expected_cols:
        assert col in df_macro.columns

    # Also build the official production workbook in sheets/
    prod_path = generate_sheets()
    assert prod_path.exists()


def test_functions_under_200_loc() -> None:
    """Verifies all functions and methods in sample efficiency modules are under 200 LOC."""
    import ast

    repo_root = Path(__file__).resolve().parent.parent
    target_files = [
        repo_root / "scripts" / "run_sample_efficiency_study.py",
        repo_root / "sheets" / "scripts" / "generate_sample_efficiency_sheets.py",
        repo_root / "tests" / "test_sample_efficiency.py",
    ]

    violations = []
    for f in target_files:
        assert f.exists(), f"Target file not found: {f}"
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno
                if length >= 200:
                    violations.append(f"{f.name}:{node.lineno} {node.name} ({length} LOC)")

    assert not violations, f"Functions exceeding 200 LOC: {violations}"


def test_format_macro_latex_table_syntax() -> None:
    """Verifies format_macro_latex_table generates valid LaTeX without misplaced rowcolor."""
    from scripts.run_sample_efficiency_study import format_macro_latex_table

    macro_path = Path("results/experiments/sample_efficiency/sample_efficiency_m3_macro.csv")
    if macro_path.exists():
        macro_df = pd.read_csv(macro_path)
        latex_code = format_macro_latex_table(macro_df, "M3-LLaVA", "tab:test_m3")
        assert "\\begin{table}[t]" in latex_code
        assert "\\end{table}" in latex_code
        assert "\\toprule" in latex_code
        assert "\\bottomrule" in latex_code
        for line in latex_code.splitlines():
            if "\\rowcolor" in line:
                stripped = line.strip()
                assert stripped.startswith("\\rowcolor"), (
                    f"\\rowcolor must be at start of row, found: {line}"
                )


def test_plot_sample_efficiency_curves_generation(tmp_path: Path) -> None:
    """Verifies plot_sample_efficiency_curves generates non-empty PNG and PDF artifacts."""
    from scripts.run_sample_efficiency_study import plot_sample_efficiency_curves

    macro_path = Path("results/experiments/sample_efficiency/sample_efficiency_m3_macro.csv")
    if macro_path.exists():
        macro_df = pd.read_csv(macro_path)
        png_path = tmp_path / "test_curve.png"
        pdf_path = tmp_path / "test_curve.pdf"
        plot_sample_efficiency_curves(macro_df, "M3-LLaVA", png_path, pdf_path)
        assert png_path.exists() and png_path.stat().st_size > 0
        assert pdf_path.exists() and pdf_path.stat().st_size > 0
