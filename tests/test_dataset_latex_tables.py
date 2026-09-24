"""
Unit tests for Group 1-4 & Group 9 LaTeX table generation and Option A ranking invariants.
Covers:
1. rank_and_format handles ties and precision collision gracefully.
2. Group 1: 16 per-dataset tables in dataset_tables/dataset_wise_results/ with 4 methods & 4 metrics.
3. Group 2: macro_mean.tex with 4 methods & 4 metrics across Core 7 datasets.
4. Group 9: vqav2_multirollout_comparison.tex with single-pass and multi-rollout methods.
5. Group 3: temp_ablation tables with stacked temperature blocks and 4 methods & 4 metrics.
6. Group 4: lodo_cross_dataset.tex with 4 methods & 4 metrics across both protocols.
7. Strict LOC constraint (< 200 LOC).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from scripts.generate_dataset_latex_tables import (
    ADVERSARIAL_DATASETS,
    ALL_8_METHODS,
    BREAKDOWN_METHODS,
    CORE_DATASETS,
    DATASET_DISPLAY_MAP,
    DATASET_FILE_MAP,
    DATASET_WISE_DIR,
    RQ1_ADVERSARIAL_FILE,
    RQ1_CORE7_FILE,
    TABLES_DIR,
    TARGET_9_DATASETS,
    TARGET_METHODS_8_ORDER,
    TARGET_METHODS_ORDER,
    TEMP_DIR,
    VQAV2_METHODS_ORDER,
    build_core7_breakdown_table,
    build_rq1_adversarial_table,
    build_rq1_core7_table,
    generate_single_table,
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
    assert len(DATASET_FILE_MAP) == 16

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

        # Strictly unshaded across all 16 dataset tables
        assert r"\rowcolor" not in content, f"Found forbidden \\rowcolor in {filename}"


def test_adversarial_safety_latex_tables() -> None:
    """Verify avqa.tex and vllmsafety.tex exist with proper M3/MQT blocks, 8 methods, 4 metrics, unshaded Option A formatting, and macro_mean exclusion."""
    for ds_key, fname, label_suffix in [
        ("avqa", "avqa.tex", "avqa"),
        ("vllm-safety", "vllmsafety.tex", "vllmsafety"),
    ]:
        out_f = DATASET_WISE_DIR / fname
        assert out_f.exists(), f"Missing {fname} in {DATASET_WISE_DIR}"

        content = out_f.read_text(encoding="utf-8")

        # Table 1: M3-LLaVA (7B)
        assert f"\\label{{tab:benchmark_m3_{label_suffix}}}" in content
        assert "M3-LLaVA 7B" in content
        assert f"on \\texttt{{{ds_key}}}" in content

        # Table 2: MQT-LLaVA (7B)
        assert f"\\label{{tab:benchmark_mqt_{label_suffix}}}" in content
        assert "MQT-LLaVA 7B" in content

        # Formatting structure
        assert "\\begin{tabular}{lccccc}" in content
        assert "\\toprule" in content
        assert "\\bottomrule" in content
        assert "ECE (\\%)" in content
        assert "Ada-ECE (\\%)" in content
        assert "Brier" in content
        assert "AUROC" in content

        # Strictly unshaded: NO \rowcolor
        assert r"\rowcolor" not in content

        # All 8 Methods present in order
        for m_key, _, _ in TARGET_METHODS_8_ORDER:
            assert m_key in content

        # Verify Trajectory Platt (5D) is the crowning final method row
        last_m_pos = content.rfind("Trajectory Platt (5D)")
        ump_pos = content.rfind("UMPIRE")
        assert last_m_pos > ump_pos, "Trajectory Platt (5D) must be positioned as the final row"

        # Multi-pass baselines have placeholder dashes
        assert " - & - & - & - " in content or "- & - & - & -" in content

        # Option A ranking formatting (Rank 1 bold, Rank 2 italic)
        assert "\\textbf{" in content
        assert "\\textit{" in content

    # Verify macro_mean.tex does NOT include avqa or vllm-safety
    macro_f = TABLES_DIR / "macro_mean.tex"
    macro_content = macro_f.read_text(encoding="utf-8")
    assert "avqa" not in macro_content
    assert "vllm-safety" not in macro_content
    assert "vllmsafety" not in macro_content


def test_group2_macro_mean_table() -> None:
    """Verify Group 2 macro_mean.tex exists, unshaded, with 4 methods across Core 7."""
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
    assert "Trajectory Platt (5D)" in content
    assert r"\rowcolor" not in content


def test_group9_vqav2_multirollout_table() -> None:
    """Verify Group 9 vqav2_multirollout_comparison.tex contains single-pass and multi-pass methods, unshaded."""
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
    assert r"\rowcolor" not in content

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
    target_methods = [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Trajectory Platt (5D)",
    ]
    excluded_methods = ["VCPS", "MSSC", "Quadratic Platt", "Spline Calibration"]

    for fname in expected_files:
        p = TEMP_DIR / fname
        assert p.exists(), f"Missing temp ablation table: {p}"
        content = p.read_text(encoding="utf-8")

        assert "M3-LLaVA 7B" in content
        assert "MQT-LLaVA 7B" in content
        assert "Sampling Temperature $T = 0.0$" in content
        assert "Sampling Temperature $T = 0.3$" in content
        assert "Sampling Temperature $T = 0.6$" in content
        assert "Sampling Temperature $T = 1.0$" in content
        assert "Sampling Temperature $T = 1.5$" in content
        assert "Mean (Averaged Across Temperatures)" in content
        assert "\\begin{tabular}{lcccc}" in content
        assert "ECE (\\%)" in content
        assert "Ada-ECE (\\%)" in content
        assert "Brier" in content
        assert "AUROC" in content

        for m in target_methods:
            assert m in content, f"Missing method {m} in {fname}"
        for em in excluded_methods:
            assert em not in content, f"Unexpected method {em} found in {fname}"
        assert r"\rowcolor" not in content, f"Found forbidden \\rowcolor in {fname}"


def test_group4_lodo_cross_dataset_table() -> None:
    """Verify Group 4 lodo_cross_dataset.tex has 4 methods across both protocols, unshaded."""
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
    assert r"\rowcolor" not in content, "Found forbidden \\rowcolor in lodo_cross_dataset.tex"

    for m in [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Trajectory Platt (5D)",
    ]:
        assert m in content


def test_core7_benchmark_breakdown_table() -> None:
    """Verify core7_benchmark_breakdown.tex has stacked M3/MQT panels, 7 datasets + Average, 4 methods, dual metrics, Option A ranking, and NO shading."""
    out_f = TABLES_DIR / "core7_benchmark_breakdown.tex"
    assert out_f.exists(), f"Missing breakdown table: {out_f}"
    content = out_f.read_text(encoding="utf-8")

    # Table wrapper and structure
    assert r"\begin{table*}[t]" in content
    assert r"\begin{tabular}{l cccccccccccccccc}" in content
    assert r"\label{tab:core7_benchmark_breakdown}" in content
    assert r"\toprule" in content
    assert r"\bottomrule" in content

    # Panel headers
    assert "Panel A: M3-LLaVA (7B)" in content
    assert "Panel B: MQT-LLaVA (7B)" in content

    # Core 7 Datasets in header + Average
    for ds_key in CORE_DATASETS:
        display_name = DATASET_DISPLAY_MAP[ds_key]
        assert rf"\textbf{{{display_name}}}" in content, (
            f"Missing dataset header for {display_name}"
        )
    assert r"\textbf{Average}" in content

    # Sub-headers
    assert r"Ada $\downarrow$" in content
    assert r"AUC $\uparrow$" in content

    # 4 Methods present in both panels
    for m in BREAKDOWN_METHODS:
        assert content.count(m) >= 2, f"Method {m} should appear in both Panel A and Panel B"

    # Strictly unshaded: NO \rowcolor anywhere
    assert r"\rowcolor" not in content, "Found forbidden \\rowcolor in unshaded table"

    # Option A ranking applied
    assert r"\textbf{" in content
    assert r"\textit{" in content


def test_build_core7_breakdown_table_dynamic() -> None:
    """Verify build_core7_breakdown_table handles programmatic generation, real CSV data, and empty/missing data."""
    df_m3 = pd.read_csv(ROOT / "results/experiments/benchmark/benchmark_m3_summary.csv")
    df_mqt = pd.read_csv(ROOT / "results/experiments/benchmark/benchmark_mqt_summary.csv")
    table_str = build_core7_breakdown_table(df_m3, df_mqt)

    assert r"\begin{table*}[t]" in table_str
    assert r"\label{tab:core7_benchmark_breakdown}" in table_str
    assert "Panel A: M3-LLaVA (7B)" in table_str
    assert "Panel B: MQT-LLaVA (7B)" in table_str

    # Test with empty DataFrame (all missing values formatted with dashes)
    empty_df = pd.DataFrame(columns=["dataset", "method", "adaptive_ece_percent", "auroc"])
    empty_table = build_core7_breakdown_table(empty_df, empty_df)
    assert "-" in empty_table
    assert r"\begin{table*}[t]" in empty_table


def test_9_dataset_wise_tables_all_8_methods_and_unshaded() -> None:
    """Verify the 9 dataset-wise tables contain all 8 methods in canonical order, unshaded, with multi-pass placeholders."""
    assert len(TARGET_9_DATASETS) == 9
    for ds in TARGET_9_DATASETS:
        fname = DATASET_FILE_MAP[ds]
        p = DATASET_WISE_DIR / fname
        assert p.exists(), f"Missing dataset table: {p}"
        content = p.read_text(encoding="utf-8")

        # Strictly unshaded: NO \rowcolor
        assert r"\rowcolor" not in content, f"Found forbidden \\rowcolor in {fname}"

        # Both architectures present
        assert "M3-LLaVA 7B" in content
        assert "MQT-LLaVA 7B" in content

        # All 8 methods present in strict canonical order in both panels
        panels = content.split(r"\begin{table}[t]")
        assert len(panels) >= 3  # empty prefix + 2 tables
        for panel in panels[1:]:
            positions = [panel.find(m) for m in ALL_8_METHODS]
            assert all(pos != -1 for pos in positions), f"Missing method in panel of {fname}"
            assert positions == sorted(positions), (
                f"Methods not in canonical order in panel of {fname}: {positions}"
            )

        # Multi-pass baselines have placeholder dashes
        assert "- & - & - & -" in content


def test_rq1_core7_benchmark_table() -> None:
    """Verify rq1_core7_benchmark.tex structure, 33 columns, Core 7 isolation, 8 methods, unshaded Option A ranking."""
    assert RQ1_CORE7_FILE.exists(), f"Missing {RQ1_CORE7_FILE}"
    content = RQ1_CORE7_FILE.read_text(encoding="utf-8")

    # Table environment and column specification (1 method col + 32 metric cols = 33 cols)
    assert r"\begin{table*}[t]" in content
    assert r"\label{tab:rq1_core7_benchmark}" in content
    assert r"\begin{tabular}{l cccccccccccccccccccccccccccccccc}" in content
    assert r"\toprule" in content
    assert r"\bottomrule" in content

    # Panel headers
    assert "Panel A: M3-LLaVA (7B)" in content
    assert "Panel B: MQT-LLaVA (7B)" in content

    # Top headers: exactly the 7 Core datasets + Macro Average
    for ds in CORE_DATASETS:
        display_name = DATASET_DISPLAY_MAP[ds]
        assert rf"\textbf{{{display_name}}}" in content, f"Missing {display_name} in header"
    assert r"\textbf{Macro Average}" in content

    # Strict isolation: adversarial datasets must NOT be present
    assert "AVQA" not in content
    assert "VLLM-Safety" not in content

    # Sub-headers (4 metrics per dataset)
    assert r"ECE (\%) $\downarrow$" in content
    assert r"Ada-ECE (\%) $\downarrow$" in content
    assert r"Brier $\downarrow$" in content
    assert r"AUROC $\uparrow$" in content

    # All 8 methods present in both panels in strict canonical order
    panels = content.split(r"Panel B: MQT-LLaVA (7B)")
    assert len(panels) == 2
    for panel_idx, panel in enumerate(panels):
        positions = [panel.find(m) for m in ALL_8_METHODS]
        assert all(pos != -1 for pos in positions), f"Missing method in panel {panel_idx}"
        assert positions == sorted(positions), (
            f"Methods not in canonical order in panel {panel_idx} of rq1_core7: {positions}"
        )

    # Strictly unshaded: NO \rowcolor
    assert r"\rowcolor" not in content, "Found forbidden \\rowcolor in rq1_core7_benchmark.tex"

    # Option A ranking applied for single-pass methods
    assert r"\textbf{" in content
    assert r"\textit{" in content

    # Multi-pass baselines have placeholder dashes
    assert "- & - & - & -" in content


def test_rq1_adversarial_benchmark_table() -> None:
    """Verify rq1_adversarial_benchmark.tex structure, 13 columns, adversarial isolation, 8 methods, unshaded Option A ranking."""
    assert RQ1_ADVERSARIAL_FILE.exists(), f"Missing {RQ1_ADVERSARIAL_FILE}"
    content = RQ1_ADVERSARIAL_FILE.read_text(encoding="utf-8")

    # Table environment and column specification (1 method col + 12 metric cols = 13 cols)
    assert r"\begin{table*}[t]" in content
    assert r"\label{tab:rq1_adversarial_benchmark}" in content
    assert r"\begin{tabular}{l cccccccccccc}" in content
    assert r"\toprule" in content
    assert r"\bottomrule" in content

    # Panel headers
    assert "Panel A: M3-LLaVA (7B)" in content
    assert "Panel B: MQT-LLaVA (7B)" in content

    # Top headers: exactly the 2 Adversarial datasets + Macro Average
    for ds in ADVERSARIAL_DATASETS:
        display_name = DATASET_DISPLAY_MAP[ds]
        assert rf"\textbf{{{display_name}}}" in content, f"Missing {display_name} in header"
    assert r"\textbf{Macro Average}" in content

    # Strict isolation: Core 7 dataset display names must NOT be column headers
    for ds in ["ai2d", "chartqa", "docvqa", "scienceqa"]:
        assert rf"\textbf{{{DATASET_DISPLAY_MAP[ds]}}}" not in content

    # All 8 methods present in both panels in strict canonical order
    panels = content.split(r"Panel B: MQT-LLaVA (7B)")
    assert len(panels) == 2
    for panel_idx, panel in enumerate(panels):
        positions = [panel.find(m) for m in ALL_8_METHODS]
        assert all(pos != -1 for pos in positions), f"Missing method in panel {panel_idx}"
        assert positions == sorted(positions), (
            f"Methods not in canonical order in panel {panel_idx} of rq1_adv: {positions}"
        )

    # Strictly unshaded: NO \rowcolor
    assert r"\rowcolor" not in content, (
        "Found forbidden \\rowcolor in rq1_adversarial_benchmark.tex"
    )

    # Option A ranking applied
    assert r"\textbf{" in content
    assert r"\textit{" in content

    # Multi-pass baselines have placeholder dashes
    assert "- & - & - & -" in content


def test_all_tables_compendium_inputs() -> None:
    """Verify all_tables_compendium.tex inputs both rq1_core7_benchmark and rq1_adversarial_benchmark."""
    compendium_f = TABLES_DIR / "all_tables_compendium.tex"
    assert compendium_f.exists(), f"Missing {compendium_f}"
    content = compendium_f.read_text(encoding="utf-8")

    assert r"\input{dataset_tables/rq1_core7_benchmark}" in content
    assert r"\input{dataset_tables/rq1_adversarial_benchmark}" in content


def test_build_rq1_tables_dynamic() -> None:
    """Verify programmatic resilience of build_rq1_core7_table and build_rq1_adversarial_table with empty and partial DataFrames."""
    empty_df = pd.DataFrame(
        columns=["dataset", "method", "ece_percent", "adaptive_ece_percent", "brier", "auroc"]
    )
    t_core7 = build_rq1_core7_table(empty_df, empty_df)
    assert r"\begin{table*}[t]" in t_core7
    assert r"\label{tab:rq1_core7_benchmark}" in t_core7
    assert "-" in t_core7

    t_adv = build_rq1_adversarial_table(empty_df, empty_df)
    assert r"\begin{table*}[t]" in t_adv
    assert r"\label{tab:rq1_adversarial_benchmark}" in t_adv
    assert "-" in t_adv


def test_dynamic_multi_pass_data_ingestion() -> None:
    """Verify that when df_arch contains multi-pass baseline rows, table builders dynamically populate and rank them."""
    rows = []
    for ds in CORE_DATASETS:
        for m in ALL_8_METHODS:
            rows.append(
                {
                    "dataset": ds,
                    "method": m,
                    "ece_percent": 10.0,
                    "adaptive_ece_percent": 8.0 if "5D" in m else 9.0,
                    "brier": 0.15,
                    "auroc": 0.85 if "5D" in m else 0.80,
                }
            )
    df_full = pd.DataFrame(rows)

    # Core 7 table should format numerical values for multi-pass instead of dashes
    t_core7 = build_rq1_core7_table(df_full, df_full)
    assert r"\begin{table*}[t]" in t_core7
    for m in ["LN-Entropy", "Semantic Entropy", "EigenScore", "UMPIRE"]:
        assert m in t_core7
        for line in t_core7.splitlines():
            if line.startswith(m):
                # Verify cells do not contain placeholder dash "-"
                clean_line = line.strip().removesuffix(r"\\").strip()
                cells = [c.strip() for c in clean_line.split("&")[1:]]
                assert all(c != "-" for c in cells), f"Method {m} has placeholder dashes: {line}"
                assert "10.00" in line
                assert "0.1500" in line

    # Single table should also dynamically populate multi-pass rows
    t_single = generate_single_table(df_full, "ai2d", "m3", "M3-LLaVA", is_macro=False)
    for m in ["LN-Entropy", "Semantic Entropy", "EigenScore", "UMPIRE"]:
        for line in t_single.splitlines():
            if line.startswith(m):
                clean_line = line.strip().removesuffix(r"\\").strip()
                cells = [c.strip() for c in clean_line.split("&")[2:]]
                assert all(c != "-" for c in cells), f"Single table method {m} has dashes: {line}"
                assert "10.00" in line


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
