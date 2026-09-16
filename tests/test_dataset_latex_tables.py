"""
Unit tests for Group 1-4 & Group 9 LaTeX table generation and Option A ranking invariants.
Covers:
1. rank_and_format handles ties and precision collision gracefully.
2. Group 1: 14 per-dataset tables in dataset_tables/dataset_wise_results/ with 4 methods & 4 metrics.
3. Group 2: macro_mean.tex with 4 methods & 4 metrics across Core 6 datasets.
4. Group 9: vqav2_multirollout_comparison.tex with single-pass and multi-rollout methods.
5. Group 3: temp_ablation tables with stacked temperature blocks and 4 methods & 4 metrics.
6. Group 4: lodo_cross_dataset.tex with 4 methods & 4 metrics across both protocols.
7. Strict LOC constraint (< 200 LOC).
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.generate_dataset_latex_tables import (
    DATASET_FILE_MAP,
    DATASET_WISE_DIR,
    TABLES_DIR,
    TARGET_METHODS_ORDER,
    TEMP_DIR,
    VQAV2_METHODS_ORDER,
    rank_and_format,
)

ROOT = Path(__file__).resolve().parent.parent


def test_rank_and_format_precision_and_ties() -> None:
    """Verify rank_and_format handles identical rounded display values without conflicting ranks."""
    # Lower is better (e.g. ECE / Brier)
    vals = [10.50, 4.251, 4.254, 7.80]
    res = rank_and_format(vals, higher_is_better=False, decimals=2)
    # Both 4.251 and 4.254 round to 4.25 and must share Rank 1 bold
    assert res[1] == "\\textbf{4.25}"
    assert res[2] == "\\textbf{4.25}"
    assert res[3] == "\\textit{7.80}"
    assert res[0] == "10.50"

    # Higher is better (e.g. AUROC)
    aurocs = [0.698751, 0.698510, 0.698000, 0.629000]
    res_auc = rank_and_format(aurocs, higher_is_better=True, decimals=3)
    # Both 0.698751 and 0.698510 round to 0.699 and must share Rank 1 bold
    assert res_auc[0] == "\\textbf{0.699}"
    assert res_auc[1] == "\\textbf{0.699}"
    # 0.698 is Rank 2 italic
    assert res_auc[2] == "\\textit{0.698}"
    assert res_auc[3] == "0.629"

    # None and NaN handling
    none_vals = [None, 0.5, float("nan")]
    res_none = rank_and_format(none_vals, higher_is_better=False, decimals=2)
    assert res_none == ["-", "\\textbf{0.50}", "-"]


def test_group1_dataset_wise_tables_exist_and_structure() -> None:
    """Verify Group 1 tables exist in dataset_wise_results and older copies removed from root."""
    assert DATASET_WISE_DIR.exists()
    assert len(DATASET_FILE_MAP) == 14

    for filename in DATASET_FILE_MAP.values():
        out_f = DATASET_WISE_DIR / filename
        assert out_f.exists(), f"Missing dataset table: {out_f}"

        # Ensure old root-level copy is deleted
        old_f = TABLES_DIR / filename
        assert not old_f.exists(), f"Old root-level table not removed: {old_f}"

        content = out_f.read_text(encoding="utf-8")
        assert "\\begin{table}[t]" in content
        assert "\\begin{tabular}{lccccc}" in content
        assert "ECE (\\%)" in content
        assert "Ada-ECE (\\%)" in content
        assert "Brier" in content
        assert "AUROC" in content

        # Check all 4 methods present in both M3 and MQT sections
        for m_key, _, _ in TARGET_METHODS_ORDER:
            assert m_key in content


def test_group2_macro_mean_table() -> None:
    """Verify Group 2 macro_mean.tex exists and contains all 4 methods and 4 metrics across Core 6."""
    macro_f = TABLES_DIR / "macro_mean.tex"
    assert macro_f.exists(), f"Missing macro table: {macro_f}"
    content = macro_f.read_text(encoding="utf-8")

    assert "M3-LLaVA 7B" in content
    assert "MQT-LLaVA 7B" in content
    assert "Across Core 7 Datasets" in content
    assert "\\begin{tabular}{lccccc}" in content
    assert "Macro ECE (\\%)" in content
    assert "Macro Ada-ECE (\\%)" in content
    assert "Macro Brier" in content
    assert "Macro AUROC" in content
    assert "\\rowcolor{gray!10} \\textbf{Trajectory Platt (5D)}" in content


def test_group9_vqav2_multirollout_table() -> None:
    """Verify Group 9 vqav2_multirollout_comparison.tex contains single-pass and multi-pass methods."""
    vqa_f = TABLES_DIR / "vqav2_multirollout_comparison.tex"
    assert vqa_f.exists(), f"Missing VQAv2 multirollout table: {vqa_f}"
    content = vqa_f.read_text(encoding="utf-8")

    assert "M3-LLaVA 7B" in content
    assert "MQT-LLaVA 7B" in content
    assert "\\begin{tabular}{lccccc}" in content
    assert "ECE (\\%)" in content
    assert "Ada-ECE (\\%)" in content
    assert "Brier" in content
    assert "AUROC" in content

    for m_key, _, _ in VQAV2_METHODS_ORDER:
        if m_key == "ln_entropy":
            assert "LN-Entropy" in content
        elif m_key == "semantic_entropy":
            assert "Semantic Entropy" in content
        elif m_key == "eigen_score":
            assert "EigenScore" in content
        elif m_key == "umpire":
            assert "UMPIRE" in content
        else:
            assert m_key in content


def test_group3_temp_ablation_tables() -> None:
    """Verify Group 3 temp_ablation tables exist with stacked temperature blocks."""
    expected_files = ["macro_mean.tex", "pope.tex", "scienceqa.tex", "textvqa.tex", "vizwizvqa.tex"]
    for fname in expected_files:
        p = TEMP_DIR / fname
        assert p.exists(), f"Missing temp ablation table: {p}"
        content = p.read_text(encoding="utf-8")

        assert "Sampling Temperature $T = 0.0$" in content
        assert "Sampling Temperature $T = 0.3$" in content
        assert "Sampling Temperature $T = 0.6$" in content
        assert "Sampling Temperature $T = 1.0$" in content
        assert "Sampling Temperature $T = 1.5$" in content
        assert "Mean (Averaged Across Temperatures)" in content
        assert "\\begin{tabular}{lcccc}" in content
        assert "Trajectory Platt (5D)" in content


def test_group4_lodo_cross_dataset_table() -> None:
    """Verify Group 4 lodo_cross_dataset.tex has 4 methods across both protocols."""
    lodo_f = TABLES_DIR / "lodo_cross_dataset.tex"
    assert lodo_f.exists(), f"Missing LODO table: {lodo_f}"
    content = lodo_f.read_text(encoding="utf-8")

    assert "Zero-Shot Base" in content
    assert "Target Adapted (Saerens-EM)" in content
    assert "\\begin{tabular}{llcccc}" in content
    assert "Macro ECE (\\%)" in content
    assert "Macro Ada-ECE (\\%)" in content
    assert "Macro Brier" in content
    assert "Macro AUROC" in content

    for m in [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Trajectory Platt (5D)",
    ]:
        assert m in content


def test_functions_under_200_loc() -> None:
    """Verify that all functions in scripts/generate_dataset_latex_tables.py are under 200 LOC."""
    target_script = ROOT / "scripts" / "generate_dataset_latex_tables.py"
    tree = ast.parse(target_script.read_text(encoding="utf-8"), filename=str(target_script))

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node.end_lineno - node.lineno + 1 if node.end_lineno else 0
            if length >= 200:
                violations.append(f"{node.name} ({length} LOC)")

    assert not violations, f"Functions exceeding 200 LOC: {violations}"
