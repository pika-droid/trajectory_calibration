#!/usr/bin/env python3
"""
Generates publication-ready LaTeX tables in dataset_tables/ and dataset_tables/temp_ablation/
from benchmark and temperature study summary CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "dataset_tables"
DATASET_WISE_DIR = TABLES_DIR / "dataset_wise_results"
TEMP_DIR = TABLES_DIR / "temp_ablation"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DATASET_WISE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CORE_DATASETS = ["ai2d", "chartqa", "docvqa", "scienceqa", "textvqa", "vizwiz-vqa", "vqav2"]
ADVERSARIAL_DATASETS = ["avqa", "vllm-safety"]
TARGET_9_DATASETS = set(CORE_DATASETS) | set(ADVERSARIAL_DATASETS)

RQ1_CORE7_FILE = TABLES_DIR / "rq1_core7_benchmark.tex"
RQ1_ADVERSARIAL_FILE = TABLES_DIR / "rq1_adversarial_benchmark.tex"

DATASET_FILE_MAP = {
    "ai2d": "ai2d.tex",
    "avqa": "avqa.tex",
    "chartqa": "chartqa.tex",
    "docvqa": "docvqa.tex",
    "gqa": "gqa.tex",
    "infographicvqa": "infographicvqa.tex",
    "lego-puzzles": "legopuzzles.tex",
    "mmbench": "mmbench.tex",
    "mmmu": "mmmu.tex",
    "pope": "pope.tex",
    "scienceqa": "scienceqa.tex",
    "seedbench": "seedbench.tex",
    "textvqa": "textvqa.tex",
    "vizwiz-vqa": "vizwizvqa.tex",
    "vllm-safety": "vllmsafety.tex",
    "vqav2": "vqav2.tex",
}

DATASET_NAME_MAP = {
    "lego-puzzles": "legopuzzles",
    "vizwiz-vqa": "vizwizvqa",
    "vllm-safety": "vllmsafety",
    "vqav2_5scale": "vqav2",
}

SINGLE_PASS_BASELINES = [
    "Naive Confidence (NC)",
    "Temperature Scaling (TS)",
    "Platt Scaling (1D)",
]

MULTI_PASS_METHODS = [
    "LN-Entropy",
    "Semantic Entropy",
    "EigenScore",
    "UMPIRE",
]

PROPOSED_METHOD = "Trajectory Platt (5D)"

ALL_8_METHODS = SINGLE_PASS_BASELINES + MULTI_PASS_METHODS + [PROPOSED_METHOD]
SINGLE_PASS_METHODS = SINGLE_PASS_BASELINES + [PROPOSED_METHOD]

TARGET_METHODS_ORDER = [
    ("Naive Confidence (NC)", "Single-Pass ($T = 0.0$)", False),
    ("Temperature Scaling (TS)", "Single-Pass ($T = 0.0$)", False),
    ("Platt Scaling (1D)", "Single-Pass ($T = 0.0$)", False),
    ("Trajectory Platt (5D)", "Single-Pass ($T = 0.0$)", True),
]

TARGET_METHODS_8_ORDER = [
    ("Naive Confidence (NC)", "Single-Pass ($T = 0.0$)", False),
    ("Temperature Scaling (TS)", "Single-Pass ($T = 0.0$)", False),
    ("Platt Scaling (1D)", "Single-Pass ($T = 0.0$)", False),
    ("LN-Entropy", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("Semantic Entropy", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("EigenScore", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("UMPIRE", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("Trajectory Platt (5D)", "Single-Pass ($T = 0.0$)", True),
]

VQAV2_METHODS_ORDER = [
    ("Naive Confidence (NC)", "Single-Pass ($T = 0.0, K = 1$)", False),
    ("Temperature Scaling (TS)", "Single-Pass ($T = 0.0, K = 1$)", False),
    ("Platt Scaling (1D)", "Single-Pass ($T = 0.0, K = 1$)", False),
    ("ln_entropy", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("semantic_entropy", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("eigen_score", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("umpire", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("Trajectory Platt (5D)", "Single-Pass ($T = 0.0, K = 1$)", True),
]

UMP_DISPLAY_NAMES = {
    "ln_entropy": "LN-Entropy",
    "semantic_entropy": "Semantic Entropy",
    "eigen_score": "EigenScore",
    "umpire": "UMPIRE",
}

DATASET_DISPLAY_MAP = {
    "ai2d": "AI2D",
    "chartqa": "ChartQA",
    "docvqa": "DocVQA",
    "scienceqa": "ScienceQA",
    "textvqa": "TextVQA",
    "vizwiz-vqa": "VizWiz-VQA",
    "vqav2": "VQAv2",
    "avqa": "AVQA",
    "vllm-safety": "VLLM-Safety",
}

BREAKDOWN_METHODS = [
    "Naive Confidence (NC)",
    "Temperature Scaling (TS)",
    "Platt Scaling (1D)",
    "Trajectory Platt (5D)",
]


def rank_and_format(
    vals: list[float | None], higher_is_better: bool = False, decimals: int = 2
) -> list[str]:
    """Applies Option A ranking with bold Rank 1 and italic Rank 2."""
    valid_vals = [round(v, decimals) for v in vals if v is not None and not np.isnan(v)]
    if not valid_vals:
        return ["-" for _ in vals]

    sorted_unique = sorted(list(set(valid_vals)), reverse=higher_is_better)
    best_val = sorted_unique[0] if len(sorted_unique) > 0 else None
    second_val = sorted_unique[1] if len(sorted_unique) > 1 else None

    formatted = []
    for v in vals:
        if v is None or np.isnan(v):
            formatted.append("-")
            continue
        v_round = round(v, decimals)
        s = f"{v:.{decimals}f}"
        if best_val is not None and abs(v_round - best_val) < 1e-6:
            formatted.append(f"\\textbf{{{s}}}")
        elif second_val is not None and abs(v_round - second_val) < 1e-6:
            formatted.append(f"\\textit{{{s}}}")
        else:
            formatted.append(s)
    return formatted


def generate_single_table(
    df_bench: pd.DataFrame,
    ds: str,
    arch_label: str,
    arch_model: str,
    is_macro: bool = False,
) -> str:
    """Generates a standardized LaTeX benchmark table (8 methods for target 9 datasets, 4 for others)."""
    tab_label = (
        f"tab:benchmark_{arch_label}_{DATASET_NAME_MAP.get(ds, ds)}"
        if not is_macro
        else f"tab:benchmark_macro_{arch_label}"
    )
    macro_prefix = "Macro-Averaged " if is_macro else ""
    macro_col = "Macro " if is_macro else ""

    sub_df = (
        df_bench[df_bench["dataset"].isin(CORE_DATASETS)]
        if is_macro
        else df_bench[df_bench["dataset"] == ds]
    )

    methods_order = (
        TARGET_METHODS_8_ORDER
        if (ds in TARGET_9_DATASETS and not is_macro)
        else TARGET_METHODS_ORDER
    )

    rows = []
    for m_key, regime, is_ours in methods_order:
        sub = sub_df[sub_df["method"] == m_key]
        if not sub.empty:
            ece = float(sub["ece_percent"].mean())
            ada_ece = float(sub["adaptive_ece_percent"].mean())
            brier = float(sub["brier"].mean())
            auroc = float(sub["auroc"].mean())
            rows.append((m_key, regime, is_ours, ece, ada_ece, brier, auroc))
        else:
            rows.append((m_key, regime, is_ours, None, None, None, None))

    if not rows:
        return ""

    caption_name = (
        f"on \\texttt{{{ds}}}" if not is_macro else f"Across Core {len(CORE_DATASETS)} Datasets"
    )
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{\\textbf{{{macro_prefix}Calibration Benchmark {caption_name} ({arch_model} 7B).}} \\textbf{{Bold}}: best; \\textit{{italic}}: second best. $\\downarrow$/$\\uparrow$: lower/higher is better.}}",
        f"\\label{{{tab_label}}}",
        "\\tablestyle{4pt}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        f"\\textbf{{Calibration Method}} & \\textbf{{Regime / Sampling}} & \\textbf{{{macro_col}ECE (\\%)}} $\\downarrow$ & \\textbf{{{macro_col}Ada-ECE (\\%)}} $\\downarrow$ & \\textbf{{{macro_col}Brier}} $\\downarrow$ & \\textbf{{{macro_col}AUROC}} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    ece_formatted = rank_and_format([r[3] for r in rows], higher_is_better=False, decimals=2)
    ada_ece_formatted = rank_and_format([r[4] for r in rows], higher_is_better=False, decimals=2)
    brier_formatted = rank_and_format([r[5] for r in rows], higher_is_better=False, decimals=4)
    auroc_formatted = rank_and_format([r[6] for r in rows], higher_is_better=True, decimals=3)

    for i, (disp_name, regime, _, _, _, _, _) in enumerate(rows):
        ece_str = ece_formatted[i]
        ada_str = ada_ece_formatted[i]
        brier_str = brier_formatted[i]
        auc_str = auroc_formatted[i]
        row_str = f"{disp_name} & {regime} & {ece_str} & {ada_str} & {brier_str} & {auc_str} \\\\"
        lines.append(row_str)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def generate_vqav2_multirollout_latex_table(
    df_bench: pd.DataFrame,
    df_ump: pd.DataFrame,
    arch_label: str,
    arch_model: str,
) -> str:
    """Generates dedicated VQAv2 single-pass (1x) vs multi-rollout (10x) comparison table."""
    tab_label = f"tab:vqav2_multirollout_{arch_label}"
    rows = []

    for m_key, regime, is_ours in VQAV2_METHODS_ORDER:
        disp_name = UMP_DISPLAY_NAMES.get(m_key, m_key)
        if m_key in UMP_DISPLAY_NAMES:
            sub = df_ump[(df_ump["dataset"] == "vqav2") & (df_ump["method"] == m_key)]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(
                    (
                        disp_name,
                        regime,
                        is_ours,
                        float(r["cece"] * 100.0),
                        None,
                        None,
                        float(r["auc"]),
                    )
                )
        else:
            sub = df_bench[(df_bench["dataset"] == "vqav2") & (df_bench["method"] == m_key)]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(
                    (
                        disp_name,
                        regime,
                        is_ours,
                        float(r["ece_percent"]),
                        float(r["adaptive_ece_percent"]),
                        float(r["brier"]),
                        float(r["auroc"]),
                    )
                )

    if not rows:
        return ""

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{\\textbf{{VQAv2 Calibration: Single-Pass Trajectory Calibration ($1\\times$) vs. Multi-Rollout UMPIRE Suite ($10\\times$) ({arch_model} 7B).}} \\textbf{{Bold}}: best; \\textit{{italic}}: second best. $\\downarrow$/$\\uparrow$: lower/higher is better.}}",
        f"\\label{{{tab_label}}}",
        "\\tablestyle{4pt}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Calibration Method} & \\textbf{Regime / Sampling} & \\textbf{ECE (\\%)} $\\downarrow$ & \\textbf{Ada-ECE (\\%)} $\\downarrow$ & \\textbf{Brier} $\\downarrow$ & \\textbf{AUROC} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    ece_formatted = rank_and_format([r[3] for r in rows], higher_is_better=False, decimals=2)
    ada_ece_formatted = rank_and_format([r[4] for r in rows], higher_is_better=False, decimals=2)
    brier_formatted = rank_and_format([r[5] for r in rows], higher_is_better=False, decimals=4)
    auroc_formatted = rank_and_format([r[6] for r in rows], higher_is_better=True, decimals=3)

    for i, (disp_name, regime, _, _, _, _, _) in enumerate(rows):
        ece_str = ece_formatted[i]
        ada_str = ada_ece_formatted[i]
        brier_str = brier_formatted[i]
        auc_str = auroc_formatted[i]
        row_str = f"{disp_name} & {regime} & {ece_str} & {ada_str} & {brier_str} & {auc_str} \\\\"
        lines.append(row_str)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def _render_breakdown_panel_rows(df_arch: pd.DataFrame) -> list[str]:
    """Renders data rows for a single architecture panel across Core 7 datasets and Average."""
    ada_by_ds: dict[str, list[str]] = {}
    auc_by_ds: dict[str, list[str]] = {}

    for ds in CORE_DATASETS:
        ada_vals: list[float | None] = []
        auc_vals: list[float | None] = []
        for m in BREAKDOWN_METHODS:
            sub = df_arch[(df_arch["dataset"] == ds) & (df_arch["method"] == m)]
            ada_vals.append(float(sub["adaptive_ece_percent"].mean()) if not sub.empty else None)
            auc_vals.append(float(sub["auroc"].mean()) if not sub.empty else None)
        ada_by_ds[ds] = rank_and_format(ada_vals, higher_is_better=False, decimals=2)
        auc_by_ds[ds] = rank_and_format(auc_vals, higher_is_better=True, decimals=3)

    avg_ada_vals: list[float | None] = []
    avg_auc_vals: list[float | None] = []
    for m in BREAKDOWN_METHODS:
        sub = df_arch[(df_arch["dataset"].isin(CORE_DATASETS)) & (df_arch["method"] == m)]
        avg_ada_vals.append(float(sub["adaptive_ece_percent"].mean()) if not sub.empty else None)
        avg_auc_vals.append(float(sub["auroc"].mean()) if not sub.empty else None)

    avg_ada_formatted = rank_and_format(avg_ada_vals, higher_is_better=False, decimals=2)
    avg_auc_formatted = rank_and_format(avg_auc_vals, higher_is_better=True, decimals=3)

    lines: list[str] = []
    for i, m in enumerate(BREAKDOWN_METHODS):
        cells: list[str] = [m]
        for ds in CORE_DATASETS:
            cells.append(ada_by_ds[ds][i])
            cells.append(auc_by_ds[ds][i])
        cells.append(avg_ada_formatted[i])
        cells.append(avg_auc_formatted[i])
        lines.append(" & ".join(cells) + " \\\\")

    return lines


def build_core7_breakdown_table(df_m3: pd.DataFrame, df_mqt: pd.DataFrame) -> str:
    """Builds unified stacked 2-panel Core 7 benchmark breakdown table with Ada-ECE and AUROC."""
    tab_label = "tab:core7_benchmark_breakdown"

    top_ds_headers = " & ".join(
        [rf"\multicolumn{{2}}{{c}}{{\textbf{{{DATASET_DISPLAY_MAP[ds]}}}}}" for ds in CORE_DATASETS]
    )
    cmidrules = " ".join(
        [rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(len(CORE_DATASETS) + 1)]
    )
    sub_headers = " & ".join([r"Ada $\downarrow$ & AUC $\uparrow$"] * (len(CORE_DATASETS) + 1))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Comprehensive Calibration Benchmark Breakdown Across Core 7 Datasets.} Evaluated across Adaptive ECE (Ada-ECE (\%) $\downarrow$) and AUROC ($\uparrow$) on M3-LLaVA (7B) (Panel A) and MQT-LLaVA (7B) (Panel B). \textbf{Bold}: best; \textit{italic}: second best within each metric column. Clean unshaded presentation.}",
        rf"\label{{{tab_label}}}",
        r"\tablestyle{2.8pt}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l cccccccccccccccc}",
        r"\toprule",
        rf" & {top_ds_headers} & \multicolumn{{2}}{{c}}{{\textbf{{Average}}}} \\",
        cmidrules,
        rf"\textbf{{Calibration Method}} & {sub_headers} \\",
        r"\midrule",
        r"\multicolumn{17}{l}{\textbf{Panel A: M3-LLaVA (7B)}} \\",
        r"\midrule",
    ]

    lines.extend(_render_breakdown_panel_rows(df_m3))
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{17}{l}{\textbf{Panel B: MQT-LLaVA (7B)}} \\",
            r"\midrule",
        ]
    )
    lines.extend(_render_breakdown_panel_rows(df_mqt))
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def _render_rq1_panel_rows(
    df_arch: pd.DataFrame,
    datasets: list[str],
) -> list[str]:
    """Renders data rows for an RQ1 architecture panel across datasets and Macro Average."""
    ds_metrics: dict[str, tuple[list[str], list[str], list[str], list[str]]] = {}
    for ds in datasets:
        ece_vals: list[float | None] = []
        ada_vals: list[float | None] = []
        brier_vals: list[float | None] = []
        auc_vals: list[float | None] = []
        for m in ALL_8_METHODS:
            sub = df_arch[(df_arch["dataset"] == ds) & (df_arch["method"] == m)]
            if not sub.empty:
                ece_vals.append(float(sub["ece_percent"].mean()))
                ada_vals.append(float(sub["adaptive_ece_percent"].mean()))
                brier_vals.append(float(sub["brier"].mean()))
                auc_vals.append(float(sub["auroc"].mean()))
            else:
                ece_vals.append(None)
                ada_vals.append(None)
                brier_vals.append(None)
                auc_vals.append(None)
        ds_metrics[ds] = (
            rank_and_format(ece_vals, higher_is_better=False, decimals=2),
            rank_and_format(ada_vals, higher_is_better=False, decimals=2),
            rank_and_format(brier_vals, higher_is_better=False, decimals=4),
            rank_and_format(auc_vals, higher_is_better=True, decimals=3),
        )

    avg_ece_vals: list[float | None] = []
    avg_ada_vals: list[float | None] = []
    avg_brier_vals: list[float | None] = []
    avg_auc_vals: list[float | None] = []
    for m in ALL_8_METHODS:
        sub = df_arch[(df_arch["dataset"].isin(datasets)) & (df_arch["method"] == m)]
        if not sub.empty:
            avg_ece_vals.append(float(sub["ece_percent"].mean()))
            avg_ada_vals.append(float(sub["adaptive_ece_percent"].mean()))
            avg_brier_vals.append(float(sub["brier"].mean()))
            avg_auc_vals.append(float(sub["auroc"].mean()))
        else:
            avg_ece_vals.append(None)
            avg_ada_vals.append(None)
            avg_brier_vals.append(None)
            avg_auc_vals.append(None)

    avg_ece_fmt = rank_and_format(avg_ece_vals, higher_is_better=False, decimals=2)
    avg_ada_fmt = rank_and_format(avg_ada_vals, higher_is_better=False, decimals=2)
    avg_brier_fmt = rank_and_format(avg_brier_vals, higher_is_better=False, decimals=4)
    avg_auc_fmt = rank_and_format(avg_auc_vals, higher_is_better=True, decimals=3)

    lines: list[str] = []
    for i, m in enumerate(ALL_8_METHODS):
        cells: list[str] = [m]
        for ds in datasets:
            ece_f, ada_f, brier_f, auc_f = ds_metrics[ds]
            cells.extend([ece_f[i], ada_f[i], brier_f[i], auc_f[i]])
        cells.extend([avg_ece_fmt[i], avg_ada_fmt[i], avg_brier_fmt[i], avg_auc_fmt[i]])
        lines.append(" & ".join(cells) + " \\\\")

    return lines


def build_rq1_core7_table(df_m3: pd.DataFrame, df_mqt: pd.DataFrame) -> str:
    """Builds unified stacked 2-panel Core 7 master benchmark table for RQ1."""
    tab_label = "tab:rq1_core7_benchmark"
    num_cols = len(CORE_DATASETS) * 4 + 4 + 1  # 33 columns
    col_spec = "l " + "c" * (num_cols - 1)

    top_ds_headers = " & ".join(
        [rf"\multicolumn{{4}}{{c}}{{\textbf{{{DATASET_DISPLAY_MAP[ds]}}}}}" for ds in CORE_DATASETS]
    )
    cmidrules = " ".join(
        [rf"\cmidrule(lr){{{4 * i + 2}-{4 * i + 5}}}" for i in range(len(CORE_DATASETS) + 1)]
    )
    sub_headers = " & ".join(
        [
            r"ECE (\%) $\downarrow$ & Ada-ECE (\%) $\downarrow$ & Brier $\downarrow$ & AUROC $\uparrow$"
        ]
        * (len(CORE_DATASETS) + 1)
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{RQ1 Evaluation: Autoregressive Trajectory Features vs. Calibration Baselines Across Core 7 Vision-Language Benchmarks.} Comparison of single-pass calibrators ($T = 0.0, K = 1$) against multi-pass sampling baselines ($K = 10$) across M3-LLaVA (7B) (Panel A) and MQT-LLaVA (7B) (Panel B). Metrics: Expected Calibration Error (ECE (\%) $\downarrow$), Adaptive ECE (Ada-ECE (\%) $\downarrow$), Brier Score ($\downarrow$), and AUROC ($\uparrow$). \textbf{Bold}: Rank 1; \textit{italic}: Rank 2 within each metric column among single-pass methods. Multi-pass baselines report placeholders ($-$) pending rerun completion. Clean unshaded presentation.}",
        rf"\label{{{tab_label}}}",
        r"\tablestyle{2.0pt}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {top_ds_headers} & \multicolumn{{4}}{{c}}{{\textbf{{Macro Average}}}} \\",
        cmidrules,
        rf"\textbf{{Calibration Method}} & {sub_headers} \\",
        r"\midrule",
        rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{Panel A: M3-LLaVA (7B)}}}} \\",
        r"\midrule",
    ]

    lines.extend(_render_rq1_panel_rows(df_m3, CORE_DATASETS))
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{Panel B: MQT-LLaVA (7B)}}}} \\",
            r"\midrule",
        ]
    )
    lines.extend(_render_rq1_panel_rows(df_mqt, CORE_DATASETS))
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def build_rq1_adversarial_table(df_m3: pd.DataFrame, df_mqt: pd.DataFrame) -> str:
    """Builds unified stacked 2-panel Adversarial master benchmark table for RQ1."""
    tab_label = "tab:rq1_adversarial_benchmark"
    num_cols = len(ADVERSARIAL_DATASETS) * 4 + 4 + 1  # 13 columns
    col_spec = "l " + "c" * (num_cols - 1)

    top_ds_headers = " & ".join(
        [
            rf"\multicolumn{{4}}{{c}}{{\textbf{{{DATASET_DISPLAY_MAP[ds]}}}}}"
            for ds in ADVERSARIAL_DATASETS
        ]
    )
    cmidrules = " ".join(
        [rf"\cmidrule(lr){{{4 * i + 2}-{4 * i + 5}}}" for i in range(len(ADVERSARIAL_DATASETS) + 1)]
    )
    sub_headers = " & ".join(
        [
            r"ECE (\%) $\downarrow$ & Ada-ECE (\%) $\downarrow$ & Brier $\downarrow$ & AUROC $\uparrow$"
        ]
        * (len(ADVERSARIAL_DATASETS) + 1)
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{RQ1 Evaluation: Calibration Robustness Under Adversarial and Safety Stress Benchmarks.} Evaluated across Expected Calibration Error (ECE (\%) $\downarrow$), Adaptive ECE (Ada-ECE (\%) $\downarrow$), Brier Score ($\downarrow$), and AUROC ($\uparrow$) on M3-LLaVA (7B) (Panel A) and MQT-LLaVA (7B) (Panel B) across AVQA, VLLM-Safety, and Adversarial Macro Average. \textbf{Bold}: Rank 1; \textit{italic}: Rank 2 within each metric column among single-pass methods. Multi-pass baselines report placeholders ($-$) pending rerun completion. Clean unshaded presentation.}",
        rf"\label{{{tab_label}}}",
        r"\tablestyle{4.0pt}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {top_ds_headers} & \multicolumn{{4}}{{c}}{{\textbf{{Macro Average}}}} \\",
        cmidrules,
        rf"\textbf{{Calibration Method}} & {sub_headers} \\",
        r"\midrule",
        rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{Panel A: M3-LLaVA (7B)}}}} \\",
        r"\midrule",
    ]

    lines.extend(_render_rq1_panel_rows(df_m3, ADVERSARIAL_DATASETS))
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{Panel B: MQT-LLaVA (7B)}}}} \\",
            r"\midrule",
        ]
    )
    lines.extend(_render_rq1_panel_rows(df_mqt, ADVERSARIAL_DATASETS))
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def generate_benchmark_tables():
    """Generates individual dataset tables, macro mean table, RQ1 master tables, and VQAv2 multi-rollout table."""
    df_m3 = pd.read_csv(ROOT / "results/experiments/benchmark/benchmark_m3_summary.csv")
    df_mqt = pd.read_csv(ROOT / "results/experiments/benchmark/benchmark_mqt_summary.csv")
    df_m3["dataset"] = df_m3["dataset"].replace({"vqav2_5scale": "vqav2"})
    df_mqt["dataset"] = df_mqt["dataset"].replace({"vqav2_5scale": "vqav2"})

    ump_m3 = pd.read_csv(ROOT / "results/umpire_eval/m3_llava_cumulative_summary.csv")
    ump_mqt = pd.read_csv(ROOT / "results/umpire_eval/mqt_llava_cumulative_summary.csv")

    adv_csv = ROOT / "results/adversarial_safety_benchmark_results.csv"
    if adv_csv.exists():
        df_adv = pd.read_csv(adv_csv)
        df_adv_m3 = df_adv[df_adv["arch"] == "M3"].copy()
        df_adv_mqt = df_adv[df_adv["arch"] == "MQT"].copy()
        df_m3 = pd.concat([df_m3, df_adv_m3], ignore_index=True)
        df_mqt = pd.concat([df_mqt, df_adv_mqt], ignore_index=True)

    # 1. Per-dataset tables into dataset_wise_results/
    for ds_key, filename in DATASET_FILE_MAP.items():
        t_m3 = generate_single_table(df_m3, ds_key, "m3", "M3-LLaVA", is_macro=False)
        t_mqt = generate_single_table(df_mqt, ds_key, "mqt", "MQT-LLaVA", is_macro=False)
        content = t_m3 + "\n\n" + t_mqt + "\n"
        out_file = DATASET_WISE_DIR / filename
        out_file.write_text(content, encoding="utf-8")

        # Remove deprecated root-level copy if present
        old_file = TABLES_DIR / filename
        if old_file.exists():
            old_file.unlink()
        print(f"Generated: {out_file}")

    # 2. Macro mean table in dataset_tables/macro_mean.tex across Core 7 datasets
    macro_m3 = generate_single_table(df_m3, "macro_mean", "macro_m3", "M3-LLaVA", is_macro=True)
    macro_mqt = generate_single_table(df_mqt, "macro_mean", "macro_mqt", "MQT-LLaVA", is_macro=True)
    (TABLES_DIR / "macro_mean.tex").write_text(
        macro_m3 + "\n\n" + macro_mqt + "\n", encoding="utf-8"
    )
    print(f"Generated: {TABLES_DIR / 'macro_mean.tex'}")

    # 3. Dedicated VQAv2 Multi-Rollout Comparison table
    vqav2_m3 = generate_vqav2_multirollout_latex_table(df_m3, ump_m3, "m3", "M3-LLaVA")
    vqav2_mqt = generate_vqav2_multirollout_latex_table(df_mqt, ump_mqt, "mqt", "MQT-LLaVA")
    (TABLES_DIR / "vqav2_multirollout_comparison.tex").write_text(
        vqav2_m3 + "\n\n" + vqav2_mqt + "\n", encoding="utf-8"
    )
    print(f"Generated: {TABLES_DIR / 'vqav2_multirollout_comparison.tex'}")

    # 4. Core 7 comprehensive breakdown table in dataset_tables/core7_benchmark_breakdown.tex
    core7_breakdown = build_core7_breakdown_table(df_m3, df_mqt)
    out_core7 = TABLES_DIR / "core7_benchmark_breakdown.tex"
    out_core7.write_text(core7_breakdown + "\n", encoding="utf-8")
    print(f"Generated: {out_core7}")

    # 5. RQ1 Master Table: Core 7 Benchmark
    rq1_core7 = build_rq1_core7_table(df_m3, df_mqt)
    out_rq1_core7 = RQ1_CORE7_FILE
    out_rq1_core7.write_text(rq1_core7 + "\n", encoding="utf-8")
    print(f"Generated: {out_rq1_core7}")

    # 6. RQ1 Master Table: Adversarial Benchmark
    rq1_adv = build_rq1_adversarial_table(df_m3, df_mqt)
    out_rq1_adv = RQ1_ADVERSARIAL_FILE
    out_rq1_adv.write_text(rq1_adv + "\n", encoding="utf-8")
    print(f"Generated: {out_rq1_adv}")

    # 7. Calculate and display win statistics on Adaptive ECE across Core 7 datasets
    our_methods = {"Trajectory Platt (5D)"}
    target_methods = [m[0] for m in TARGET_METHODS_ORDER]
    for arch_name, df_arch in [("M3-LLaVA", df_m3), ("MQT-LLaVA", df_mqt)]:
        our_wins = 0
        total_ds = 0
        for ds in CORE_DATASETS:
            sub = df_arch[(df_arch["dataset"] == ds) & (df_arch["method"].isin(target_methods))]
            if sub.empty:
                continue
            total_ds += 1
            min_row = sub.loc[sub["adaptive_ece"].idxmin()]
            if min_row["method"] in our_methods:
                our_wins += 1
        print(
            f"[Win Statistics - {arch_name}] Trajectory Platt (5D) won on {our_wins}/{total_ds} Core datasets on Adaptive ECE."
        )


def _render_temp_block(
    block_title: str,
    raw_rows: list[tuple[str, float, float, float, float]],
) -> list[str]:
    """Renders one stacked temperature block with Option A ranking across 4 metrics."""
    lines = [
        rf"\multicolumn{{5}}{{l}}{{\textbf{{{block_title}}}}} \\",
        r"\midrule",
    ]
    ece_strs = rank_and_format([r[1] for r in raw_rows], higher_is_better=False, decimals=2)
    ada_strs = rank_and_format([r[2] for r in raw_rows], higher_is_better=False, decimals=2)
    brier_strs = rank_and_format([r[3] for r in raw_rows], higher_is_better=False, decimals=4)
    auc_strs = rank_and_format([r[4] for r in raw_rows], higher_is_better=True, decimals=3)

    for i, (m, _, _, _, _) in enumerate(raw_rows):
        ece_s, ada_s, brier_s, auc_s = ece_strs[i], ada_strs[i], brier_strs[i], auc_strs[i]
        lines.append(f"{m} & {ece_s} & {ada_s} & {brier_s} & {auc_s} \\\\")
    return lines


def build_temp_table(
    df: pd.DataFrame,
    ds_name: str | None,
    arch_label: str,
    arch_model: str,
    is_macro: bool = False,
) -> str:
    """Builds stacked temperature transfer table for 4 core methods."""
    temp_methods = [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Trajectory Platt (5D)",
    ]
    target_temps = [0.0, 0.3, 0.6, 1.0, 1.5]
    temp_datasets = ["pope", "scienceqa", "textvqa", "vizwiz-vqa"]

    caption_target = (
        f"on \\texttt{{{ds_name}}}" if not is_macro else "(Macro-Averaged Across 4 Benchmarks)"
    )
    tab_label = (
        f"tab:temp_transfer_{arch_label}_{'macro' if is_macro else str(ds_name).replace('-', '')}"
    )

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{\\textbf{{Temperature Transfer Robustness {caption_target} ({arch_model} 7B).}} Evaluated across sampling temperatures $T \\in \\{{0.0, 0.3, 0.6, 1.0, 1.5\\}}$ and Mean (trained at $T=0.0$). \\textbf{{Bold}}: best; \\textit{{italic}}: second best. $\\downarrow$/$\\uparrow$: lower/higher is better.}}",
        f"\\label{{{tab_label}}}",
        "\\tablestyle{4pt}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Calibration Method} & \\textbf{ECE (\\%)} $\\downarrow$ & \\textbf{Ada-ECE (\\%)} $\\downarrow$ & \\textbf{Brier} $\\downarrow$ & \\textbf{AUROC} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    sub = df if is_macro else df[df["dataset"] == ds_name]
    if is_macro:
        sub = sub[sub["dataset"].isin(temp_datasets)]

    for t in target_temps:
        raw_rows = []
        for m in temp_methods:
            m_sub = sub[(sub["method"] == m) & (sub["temperature"] == t)]
            ece = float(m_sub["ece_percent"].mean()) if not m_sub.empty else np.nan
            ada = float(m_sub["adaptive_ece_percent"].mean()) if not m_sub.empty else np.nan
            brier = float(m_sub["brier"].mean()) if not m_sub.empty else np.nan
            auc = float(m_sub["auroc"].mean()) if not m_sub.empty else np.nan
            raw_rows.append((m, ece, ada, brier, auc))
        lines.extend(_render_temp_block(f"Sampling Temperature $T = {t:.1f}$", raw_rows))
        lines.append(r"\midrule")

    # Mean block across temperatures
    mean_rows = []
    for m in temp_methods:
        m_sub = sub[sub["method"] == m]
        ece = float(m_sub["ece_percent"].mean()) if not m_sub.empty else np.nan
        ada = float(m_sub["adaptive_ece_percent"].mean()) if not m_sub.empty else np.nan
        brier = float(m_sub["brier"].mean()) if not m_sub.empty else np.nan
        auc = float(m_sub["auroc"].mean()) if not m_sub.empty else np.nan
        mean_rows.append((m, ece, ada, brier, auc))
    lines.extend(_render_temp_block("Mean (Averaged Across Temperatures)", mean_rows))

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def generate_temperature_tables():
    """Generates Group 3 temperature transfer ablation tables under dataset_tables/temp_ablation/."""
    df_m3 = pd.read_csv(
        ROOT / "results/experiments/temperature_study/temperature_transfer_m3_summary.csv"
    )
    df_mqt = pd.read_csv(
        ROOT / "results/experiments/temperature_study/temperature_transfer_mqt_summary.csv"
    )
    df_m3["dataset"] = df_m3["dataset"].replace({"vqav2_5scale": "vqav2"})
    df_mqt["dataset"] = df_mqt["dataset"].replace({"vqav2_5scale": "vqav2"})

    temp_datasets = ["pope", "scienceqa", "textvqa", "vizwiz-vqa"]
    for ds in temp_datasets:
        fname = f"{ds.replace('-', '')}.tex"
        t_m3 = build_temp_table(df_m3, ds, "m3", "M3-LLaVA", is_macro=False)
        t_mqt = build_temp_table(df_mqt, ds, "mqt", "MQT-LLaVA", is_macro=False)
        out_f = TEMP_DIR / fname
        out_f.write_text(t_m3 + "\n\n" + t_mqt + "\n", encoding="utf-8")
        print(f"Generated: {out_f}")

    m3_macro = build_temp_table(df_m3, None, "m3", "M3-LLaVA", is_macro=True)
    mqt_macro = build_temp_table(df_mqt, None, "mqt", "MQT-LLaVA", is_macro=True)
    out_macro = TEMP_DIR / "macro_mean.tex"
    out_macro.write_text(m3_macro + "\n\n" + mqt_macro + "\n", encoding="utf-8")
    print(f"Generated: {out_macro}")


def build_lodo_table(df: pd.DataFrame, arch_label: str, arch_model: str) -> str:
    """Builds LODO table for 4 core methods across 4 standardized metrics."""
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{\\textbf{{Leave-One-Dataset-Out (LODO) Cross-Domain Transfer ({arch_model} 7B).}} Macro-averaged across all 14 held-out target benchmarks. Trained on pooled 13 benchmarks. \\textbf{{Bold}}: best; \\textit{{italic}}: second best. $\\downarrow$/$\\uparrow$: lower/higher is better.}}",
        f"\\label{{tab:lodo_transfer_{arch_label}}}",
        "\\tablestyle{4pt}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Calibration Method} & \\textbf{Transfer Protocol} & \\textbf{Macro ECE (\\%)} $\\downarrow$ & \\textbf{Macro Ada-ECE (\\%)} $\\downarrow$ & \\textbf{Macro Brier} $\\downarrow$ & \\textbf{Macro AUROC} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    methods_order = [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Trajectory Platt (5D)",
    ]

    agg = (
        df.groupby(["method", "transfer_mode"])[
            ["ece_percent", "adaptive_ece_percent", "brier", "auroc"]
        ]
        .mean()
        .reset_index()
    )

    rows = []
    for m in methods_order:
        for mode in ["Zero-Shot Base", "Target Adapted (Saerens-EM)"]:
            sub = agg[(agg["method"] == m) & (agg["transfer_mode"] == mode)]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(
                    (
                        m,
                        mode,
                        float(r["ece_percent"]),
                        float(r["adaptive_ece_percent"]),
                        float(r["brier"]),
                        float(r["auroc"]),
                        "5D" in m,
                    )
                )

    ece_formatted = rank_and_format([r[2] for r in rows], higher_is_better=False, decimals=2)
    ada_formatted = rank_and_format([r[3] for r in rows], higher_is_better=False, decimals=2)
    brier_formatted = rank_and_format([r[4] for r in rows], higher_is_better=False, decimals=4)
    auc_formatted = rank_and_format([r[5] for r in rows], higher_is_better=True, decimals=3)

    for i, (m, mode, _, _, _, _, _) in enumerate(rows):
        ece_str = ece_formatted[i]
        ada_str = ada_formatted[i]
        brier_str = brier_formatted[i]
        auc_str = auc_formatted[i]
        lines.append(f"{m} & {mode} & {ece_str} & {ada_str} & {brier_str} & {auc_str} \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def generate_lodo_tables():
    """Generates Group 4 LODO table under dataset_tables/lodo_cross_dataset.tex."""
    df_m3 = pd.read_csv(ROOT / "results/experiments/lodo/lodo_m3_summary.csv")
    df_mqt = pd.read_csv(ROOT / "results/experiments/lodo/lodo_mqt_summary.csv")

    t_m3 = build_lodo_table(df_m3, "m3", "M3-LLaVA")
    t_mqt = build_lodo_table(df_mqt, "mqt", "MQT-LLaVA")
    out_f = TABLES_DIR / "lodo_cross_dataset.tex"
    out_f.write_text(t_m3 + "\n\n" + t_mqt + "\n", encoding="utf-8")
    print(f"Generated: {out_f}")


if __name__ == "__main__":
    generate_benchmark_tables()
    generate_temperature_tables()
    generate_lodo_tables()
    print("All LaTeX tables generated successfully.")
