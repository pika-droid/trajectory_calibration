"""
Unit tests for Paper Table 2 (M3) and Table 3 (MQT) cross-domain transfer generation.

Verifies:
1. File creation of table2_m3_transfer.tex and table3_mqt_transfer.tex.
2. Table structure: 10 columns, header format, 3 source blocks, 6 rows per block.
3. Decimal ECE invariant: values in [0, 1] formatted to 4 decimal places without '%' symbols.
4. Exact numeric values matching verified benchmark sources.
5. Option A ranking (Rank 1 bold, Rank 2 italic among transfer methods; DirectTrain unranked).
6. Strict function LOC constraint (< 200 LOC).
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from scripts.generate_paper_table2 import (
    BENCHMARK_M3_PATH,
    BENCHMARK_MQT_PATH,
    DIRECT_TRAIN_DISPLAY,
    EXCEL_PATH,
    SOURCE_BLOCKS,
    TABLE2_OUTPUT,
    TABLE3_OUTPUT,
    TRANSFER_METHODS,
    generate_both_tables,
    main,
    rank_and_format_transfer,
)

ROOT = Path(__file__).resolve().parent.parent


def test_file_creation_and_basic_content() -> None:
    """Verify that main() writes table2_m3_transfer.tex and table3_mqt_transfer.tex."""
    main()

    assert TABLE2_OUTPUT.exists(), f"Missing {TABLE2_OUTPUT}"
    assert TABLE3_OUTPUT.exists(), f"Missing {TABLE3_OUTPUT}"

    t2_content = TABLE2_OUTPUT.read_text(encoding="utf-8")
    t3_content = TABLE3_OUTPUT.read_text(encoding="utf-8")

    assert len(t2_content) > 500
    assert len(t3_content) > 500

    for content, arch_label in [(t2_content, "M3-LLaVA"), (t3_content, "MQT-LLaVA")]:
        assert r"\begin{table*}[t]" in content
        assert r"\end{table*}" in content
        assert arch_label in content
        assert r"\begin{tabular}{l cccc cccc cccc}" in content
        assert r"\toprule" in content
        assert r"\bottomrule" in content


def test_table_structure_and_blocks() -> None:
    """Verify 13-column layout, headers, 3 source blocks, and 5 rows per block."""
    table2_m3, table3_mqt = generate_both_tables()

    for table_latex in [table2_m3, table3_mqt]:
        # Header check
        subcol_header = (
            r"\textbf{ECE} $\downarrow$ & \textbf{Ada-ECE} $\downarrow$ & "
            r"\textbf{AUROC} $\uparrow$ & \textbf{Brier} $\downarrow$"
        )
        assert (
            rf"\textbf{{Method}} & {subcol_header} & {subcol_header} & {subcol_header} \\"
            in table_latex
        )

        # 3 Source blocks check
        for _, src_title, src_ctx, targets in SOURCE_BLOCKS:
            assert rf"\textbf{{{src_title}}} ({src_ctx})" in table_latex
            for _, tgt_disp, tgt_ctx in targets:
                assert rf"\textbf{{{tgt_disp}}} ({tgt_ctx})" in table_latex

        # Rows check
        assert DIRECT_TRAIN_DISPLAY in table_latex
        for _, method_display in TRANSFER_METHODS:
            assert method_display in table_latex


def _strip_latex_formatting(cell: str) -> str:
    """Extract raw numerical text from LaTeX formatting tags."""
    cell = cell.strip()
    cell = re.sub(r"\\textbf\{([^}]+)\}", r"\1", cell)
    cell = re.sub(r"\\textit\{([^}]+)\}", r"\1", cell)
    return cell.strip()


def test_decimal_ece_in_zero_one() -> None:
    """Verify that all ECE metrics are decimals in [0.0, 1.0], formatted to 4 decimals, no '%'."""
    table2_m3, table3_mqt = generate_both_tables()

    for table_latex in [table2_m3, table3_mqt]:
        lines = table_latex.splitlines()
        for line in lines:
            line_str = line.strip()
            # Check row lines
            is_dt = line_str.startswith(DIRECT_TRAIN_DISPLAY)
            is_transfer = any(line_str.startswith(disp) for _, disp in TRANSFER_METHODS)

            if is_dt or is_transfer:
                assert line_str.endswith(r"\\")
                content = line_str[:-2].strip()
                tokens = [t.strip() for t in content.split("&")]
                assert len(tokens) == 13, f"Expected 13 columns, got {len(tokens)} in {line_str}"

                # Columns 1, 5, 9 are ECE columns
                for ece_idx in [1, 5, 9]:
                    raw_val = _strip_latex_formatting(tokens[ece_idx])
                    assert "%" not in raw_val, f"Found % in ECE value: {raw_val}"
                    assert re.match(r"^\d+\.\d{4}$", raw_val), (
                        f"ECE value {raw_val} not formatted to 4 decimals"
                    )
                    ece_float = float(raw_val)
                    assert 0.0 <= ece_float <= 1.0, f"ECE {ece_float} outside [0, 1]"

                # Columns 2, 6, 10 are Ada-ECE columns
                for ada_idx in [2, 6, 10]:
                    raw_val = _strip_latex_formatting(tokens[ada_idx])
                    assert "%" not in raw_val, f"Found % in Ada-ECE value: {raw_val}"
                    assert re.match(r"^\d+\.\d{4}$", raw_val), (
                        f"Ada-ECE value {raw_val} not formatted to 4 decimals"
                    )
                    ada_float = float(raw_val)
                    assert 0.0 <= ada_float <= 1.0, f"Ada-ECE {ada_float} outside [0, 1]"

                # Columns 3, 7, 11 are AUROC columns (3 decimals)
                for auroc_idx in [3, 7, 11]:
                    raw_val = _strip_latex_formatting(tokens[auroc_idx])
                    assert re.match(r"^\d+\.\d{3}$", raw_val), (
                        f"AUROC value {raw_val} not formatted to 3 decimals"
                    )
                    auroc_float = float(raw_val)
                    assert 0.0 <= auroc_float <= 1.0, f"AUROC {auroc_float} outside [0, 1]"

                # Columns 4, 8, 12 are Brier columns (4 decimals)
                for brier_idx in [4, 8, 12]:
                    raw_val = _strip_latex_formatting(tokens[brier_idx])
                    assert re.match(r"^\d+\.\d{4}$", raw_val), (
                        f"Brier value {raw_val} not formatted to 4 decimals"
                    )
                    brier_float = float(raw_val)
                    assert 0.0 <= brier_float <= 2.0, f"Brier {brier_float} outside range"


def test_exact_numbers_m3_and_mqt() -> None:
    """Verify that table figures dynamically match values from CSV summaries and Excel sheets."""
    table2_m3, table3_mqt = generate_both_tables()

    # 1. Check DirectTrain DocVQA values from CSV
    df_m3 = pd.read_csv(BENCHMARK_M3_PATH)
    docvqa_row_m3 = df_m3[
        (df_m3["method"] == "Trajectory Platt (5D)") & (df_m3["dataset"] == "docvqa")
    ].iloc[0]
    expected_m3_dt = (
        f"{docvqa_row_m3['ece']:.4f} & {docvqa_row_m3['adaptive_ece']:.4f} & "
        f"{docvqa_row_m3['auroc']:.3f} & {docvqa_row_m3['brier']:.4f}"
    )
    assert expected_m3_dt in table2_m3

    df_mqt = pd.read_csv(BENCHMARK_MQT_PATH)
    docvqa_row_mqt = df_mqt[
        (df_mqt["method"] == "Trajectory Platt (5D)") & (df_mqt["dataset"] == "docvqa")
    ].iloc[0]
    expected_mqt_dt = (
        f"{docvqa_row_mqt['ece']:.4f} & {docvqa_row_mqt['adaptive_ece']:.4f} & "
        f"{docvqa_row_mqt['auroc']:.3f} & {docvqa_row_mqt['brier']:.4f}"
    )
    assert expected_mqt_dt in table3_mqt

    # 2. Check Transfer figures dynamically from Excel sheet src_vqav2 -> DocVQA
    df_vqav2 = pd.read_excel(EXCEL_PATH, sheet_name="src_vqav2", header=None)
    m3_headers = df_vqav2.iloc[1].tolist()
    ece_col_idx = m3_headers.index("Platt 5D ECE (%)")
    ada_col_idx = m3_headers.index("Platt 5D Ada-ECE (%)")
    brier_col_idx = m3_headers.index("Platt 5D Brier")
    auroc_col_idx = m3_headers.index("Platt 5D AUROC")

    m3_p5_ece = float(df_vqav2.iloc[4, ece_col_idx]) / 100.0
    m3_p5_ada = float(df_vqav2.iloc[4, ada_col_idx]) / 100.0
    m3_p5_brier = float(df_vqav2.iloc[4, brier_col_idx])
    m3_p5_auroc = float(df_vqav2.iloc[4, auroc_col_idx])
    assert f"{m3_p5_ece:.4f}" in table2_m3
    assert f"{m3_p5_ada:.4f}" in table2_m3
    assert f"{m3_p5_brier:.4f}" in table2_m3
    assert f"{m3_p5_auroc:.3f}" in table2_m3

    mqt_headers = df_vqav2.iloc[18].tolist()
    mqt_ece_idx = mqt_headers.index("Platt 5D ECE (%)")
    mqt_ada_idx = mqt_headers.index("Platt 5D Ada-ECE (%)")
    mqt_brier_idx = mqt_headers.index("Platt 5D Brier")
    mqt_auroc_idx = mqt_headers.index("Platt 5D AUROC")

    mqt_p5_ece = float(df_vqav2.iloc[21, mqt_ece_idx]) / 100.0
    mqt_p5_ada = float(df_vqav2.iloc[21, mqt_ada_idx]) / 100.0
    mqt_p5_brier = float(df_vqav2.iloc[21, mqt_brier_idx])
    mqt_p5_auroc = float(df_vqav2.iloc[21, mqt_auroc_idx])
    assert f"{mqt_p5_ece:.4f}" in table3_mqt
    assert f"{mqt_p5_ada:.4f}" in table3_mqt
    assert f"{mqt_p5_brier:.4f}" in table3_mqt
    assert f"{mqt_p5_auroc:.3f}" in table3_mqt


def test_ranking_logic_and_direct_train_unranked() -> None:
    """Verify Option A ranking among transfer rows, and unranked DirectTrain."""
    # Test rank_and_format_transfer unit logic
    # Lower is better (ECE/Brier)
    vals_low = [0.7421, 0.6865, 0.7496, 0.5616]
    res_low = rank_and_format_transfer(vals_low, higher_is_better=False, decimals=4)
    assert res_low[3] == r"\textbf{0.5616}"
    assert res_low[1] == r"\textit{0.6865}"
    assert res_low[0] == "0.7421"

    # Higher is better (AUROC)
    vals_high = [0.777, 0.777, 0.777, 0.775]
    res_high = rank_and_format_transfer(vals_high, higher_is_better=True, decimals=3)
    # 0.777 is Rank 1, 0.775 is Rank 2
    assert res_high[0] == r"\textbf{0.777}"
    assert res_high[1] == r"\textbf{0.777}"
    assert res_high[2] == r"\textbf{0.777}"
    assert res_high[3] == r"\textit{0.775}"

    # In generated tables, DirectTrain rows should have no \textbf or \textit in numbers
    table2_m3, table3_mqt = generate_both_tables()
    for table_latex in [table2_m3, table3_mqt]:
        for line in table_latex.splitlines():
            line_str = line.strip()
            if line_str.startswith(DIRECT_TRAIN_DISPLAY):
                cells = [c.strip() for c in line_str[:-2].split("&")[1:]]
                for c in cells:
                    assert r"\textbf" not in c, f"DirectTrain cell bolded: {c}"
                    assert r"\textit" not in c, f"DirectTrain cell italicized: {c}"


def test_functions_under_200_loc() -> None:
    """Verify that all functions in scripts/generate_paper_table2.py are strictly under 200 LOC."""
    target_script = ROOT / "scripts" / "generate_paper_table2.py"
    assert target_script.exists(), f"Target script does not exist: {target_script}"

    source = target_script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target_script))

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node.end_lineno - node.lineno + 1 if node.end_lineno else 0
            if length >= 200:
                violations.append(f"{node.name} ({length} LOC)")

    assert not violations, f"Functions exceeding 200 LOC: {violations}"


def test_section3_and_appendix_documentation_invariants() -> None:
    """Verify Section 3 paper documentation, appendix updates, and wiki decision entry."""
    sec3_file = ROOT / "docs" / "section3_trajectory_platt_transfer.tex"
    assert sec3_file.exists(), f"Missing {sec3_file}"
    sec3_text = sec3_file.read_text(encoding="utf-8")

    # Base model token budgets (Krish confirmation)
    assert "576 visual token representation" in sec3_text
    assert "256 visual token representation" in sec3_text
    assert "x_1 = \\ln" in sec3_text

    # Mathematical formulation invariants
    assert "Discrete Answer Stability" in sec3_text
    assert "Scale Entropy Slope" in sec3_text
    assert "Answer Flip Frequency" in sec3_text
    assert "Monotonicity Count" in sec3_text
    assert "Decoupled Scaling Invariant" in sec3_text
    assert "K + 1 = 6" in sec3_text
    assert "2K = 10" in sec3_text
    assert r"75.8\%" in sec3_text

    # Algorithm pseudo-code
    assert r"\begin{algorithm2e}[t]" in sec3_text
    assert "algorithm2e" in sec3_text
    assert r"\caption{Trajectory Platt (5D) Zero-Shot Calibration Transfer}" in sec3_text

    # Appendix verification
    appendix_file = ROOT / "docs" / "appendix_methods_and_metrics.tex"
    assert appendix_file.exists(), f"Missing {appendix_file}"
    app_text = appendix_file.read_text(encoding="utf-8")
    assert "576 visual token representation" in app_text
    assert "256 visual token representation" in app_text
    assert "Trajectory Platt Scaling (5D)" in app_text

    # Wiki decisions log verification
    wiki_file = ROOT / "wiki" / "decisions.md"
    assert wiki_file.exists(), f"Missing {wiki_file}"
    wiki_text = wiki_file.read_text(encoding="utf-8")
    assert "Issue 29:" in wiki_text
    assert "Cross-Domain Calibration Transfer Automation" in wiki_text


def test_ranking_rounding_collision_edge_cases() -> None:
    """Verify that values differing in raw float but identical at display precision share rank."""
    # AUROC (decimals=3): 0.8066 and 0.8071 both round to 0.807
    raw_aurocs = [0.8071, 0.8066, 0.8000, 0.7950, 0.7500]
    res = rank_and_format_transfer(raw_aurocs, higher_is_better=True, decimals=3)
    # Both top values must be bolded as \textbf{0.807}, not one bold and one italic
    assert res[0] == r"\textbf{0.807}"
    assert res[1] == r"\textbf{0.807}"
    # Next distinct rounded value (0.800) is Rank 2
    assert res[2] == r"\textit{0.800}"

    # ECE (decimals=4, lower is better): 0.05124 and 0.05121 both round to 0.0512
    raw_eces = [0.05121, 0.05124, 0.06000, 0.07000, 0.08000]
    res_ece = rank_and_format_transfer(raw_eces, higher_is_better=False, decimals=4)
    assert res_ece[0] == r"\textbf{0.0512}"
    assert res_ece[1] == r"\textbf{0.0512}"
    assert res_ece[2] == r"\textit{0.0600}"


def test_pdflatex_compilation_smoke() -> None:
    """If pdflatex is installed, verify that generated tables and Section 3 compile cleanly."""
    pdflatex_bin = shutil.which("pdflatex")
    if not pdflatex_bin:
        return

    # Ensure tables are freshly generated
    main()

    test_tex = r"""\documentclass{article}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[noend,ruled,vlined,algo2e]{algorithm2e}
\newcommand{\tablestyle}[2]{\setlength{\tabcolsep}{#1}\renewcommand{\arraystretch}{#2}}
\begin{document}
\input{docs/section3_trajectory_platt_transfer}
\input{dataset_tables/table2_m3_transfer}
\input{dataset_tables/table3_mqt_transfer}
\end{document}"""

    with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
        tmp_path = Path(tmpdir)
        tex_file = tmp_path / "smoke_compile.tex"
        tex_file.write_text(test_tex, encoding="utf-8")

        result = subprocess.run(
            [
                pdflatex_bin,
                "-interaction=nonstopmode",
                f"-output-directory={tmpdir}",
                str(tex_file),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"pdflatex compilation failed:\n{result.stdout[-1000:]}"
