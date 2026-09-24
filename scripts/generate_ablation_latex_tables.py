#!/usr/bin/env python3
"""
Publication LaTeX Table Generator for 14-Dataset Combinatorial 5D Ablation Study.

Generates:
1. dataset_tables/table_ablation_5d_macro_progression.tex:
   Cardinality k in 1..5 macro progression for M3-LLaVA and MQT-LLaVA (Decimal Option A).
2. dataset_tables/table_ablation_5d_loo.tex:
   Leave-One-Out (LOO) sensitivity panel showing degradation when dropping x1..x5.
3. dataset_tables/table_ablation_5d_14ds_grid.tex:
   Exhaustive 14-benchmark grid comparing 1D, Best-2D, Best-3D, Best-4D, and Full-5D.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS, FEATURE_NAMES

ROLE_DESCRIPTIONS = {
    "x1": "Base Fine-Scale Log-Odds Anchor ($\\ell_{576/256}$)",
    "x2": "Categorical Scale Consistency",
    "x3": "Predictive Entropy Decay Dynamics",
    "x4": "Argmax Scale Transition Rate",
    "x5": "Directional Monotonicity Trend",
}

DATASET_DISPLAY_NAMES = {
    "ai2d": "AI2D",
    "chartqa": "ChartQA",
    "docvqa": "DocVQA",
    "gqa": "GQA",
    "infographicvqa": "InfoVQA",
    "lego-puzzles": "LegoPuzzles",
    "mmbench": "MMBench",
    "mmmu": "MMMU",
    "pope": "POPE",
    "scienceqa": "ScienceQA",
    "seedbench": "SEED-Bench",
    "textvqa": "TextVQA",
    "vizwiz-vqa": "VizWiz-VQA",
    "vqav2": "VQAv2",
}


def rank_and_format_decimal(
    vals: list[float],
    higher_is_better: bool = False,
    decimals: int = 4,
) -> list[str]:
    """Rank methods using Option A (Bold Rank 1, Italic Rank 2) in pure decimal representation."""
    valid_vals = [round(v, decimals) for v in vals if not np.isnan(v)]
    if not valid_vals:
        return ["-" for _ in vals]

    sorted_unique = sorted(list(set(valid_vals)), reverse=higher_is_better)
    best_val = sorted_unique[0] if len(sorted_unique) > 0 else None
    second_val = sorted_unique[1] if len(sorted_unique) > 1 else None

    formatted: list[str] = []
    for v in vals:
        if np.isnan(v):
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


def rank_and_format_delta(
    vals: list[float],
    higher_degradation_is_worse: bool = True,
    decimals: int = 4,
) -> list[str]:
    """Rank delta sensitivity using Option A (Bold Rank 1, Italic Rank 2) preserving explicit +/- signs."""
    valid_vals = [round(v, decimals) for v in vals if not np.isnan(v)]
    if not valid_vals:
        return ["-" for _ in vals]

    sorted_unique = sorted(list(set(valid_vals)), reverse=higher_degradation_is_worse)
    best_val = sorted_unique[0] if len(sorted_unique) > 0 else None
    second_val = sorted_unique[1] if len(sorted_unique) > 1 else None

    formatted: list[str] = []
    for v in vals:
        if np.isnan(v):
            formatted.append("-")
            continue
        v_round = round(v, decimals)
        s = f"{v:+.{decimals}f}"
        if best_val is not None and abs(v_round - best_val) < 1e-6:
            formatted.append(f"\\textbf{{{s}}}")
        elif second_val is not None and abs(v_round - second_val) < 1e-6:
            formatted.append(f"\\textit{{{s}}}")
        else:
            formatted.append(s)
    return formatted


def format_latex_formula(subset_str: str) -> str:
    """Format canonical subset string into LaTeX math representation."""
    parts = subset_str.split("+")
    math_parts = [f"x_{{{p[1:]}}}" for p in parts]
    return "$" + " + ".join(math_parts) + "$"


def generate_macro_progression_table(
    df_m3: pd.DataFrame,
    df_mqt: pd.DataFrame,
) -> str:
    """Generate table_ablation_5d_macro_progression.tex comparing cardinalities k=1..5."""
    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Combinatorial 5D Trajectory Feature Cardinality Progression ($k \in \{1 \dots 5\}$)}. "
        r"Macro-averaged out-of-fold calibration error and discrimination across all 14 vision-language benchmarks "
        r"for M3-LLaVA (7B) and MQT-LLaVA (7B). Evaluates all $\sum_{k=1}^5 \binom{5}{k} = 31$ feature subsets. "
        r"Reported in pure decimal format. \textbf{Bold}: Rank 1, \textit{Italic}: Rank 2 within each architecture. "
        r"Note: Subsets lacking the base logit anchor $x_1$ achieve artificially low quantile Ada-ECE via class-prior "
        r"probability clustering, but suffer severe AUROC collapse ($\le 0.64$ vs. $0.71+$ for $x_1$-anchored models); "
        r"joint calibration and discrimination require the full 5D representation.}",
        r"\label{tab:ablation_5d_macro_progression}",
        r"\vspace{2mm}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"\textbf{Cardinality ($k$)} & \textbf{Subsets $\binom{5}{k}$} & \textbf{Optimal Formula} & "
        r"\textbf{Macro Ada-ECE (Best) $\downarrow$} & \textbf{Macro Ada-ECE (Mean $\pm$ Std)} & \textbf{Macro AUROC (Best) $\uparrow$} \\",
        r"\midrule",
    ]

    for arch_name, df_arch in [("M3-LLaVA (7B)", df_m3), ("MQT-LLaVA (7B)", df_mqt)]:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textsc{{\\textbf{{{arch_name}}}}}}} \\\\")
        lines.append(r"\midrule")

        ada_bests = df_arch["best_ada_ece"].tolist()
        auroc_bests = df_arch["best_auroc"].tolist()

        ranked_ada = rank_and_format_decimal(ada_bests, higher_is_better=False, decimals=4)
        ranked_auroc = rank_and_format_decimal(auroc_bests, higher_is_better=True, decimals=3)

        for idx, (_, row) in enumerate(df_arch.iterrows()):
            k = int(row["cardinality"])
            n_sub = int(row["n_subsets"])
            formula = format_latex_formula(str(row["best_subset"]))
            ada_best_str = ranked_ada[idx]
            mean_std_str = f"{row['mean_ada_ece']:.4f} $\\pm$ {row['std_ada_ece']:.4f}"
            auroc_best_str = ranked_auroc[idx]

            lines.append(
                f"$k = {k}$ & {n_sub} & {formula} & {ada_best_str} & {mean_std_str} & {auroc_best_str} \\\\"
            )
        lines.append(r"\midrule")

    # Remove trailing \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def generate_loo_table(
    df_loo_m3: pd.DataFrame,
    df_loo_mqt: pd.DataFrame,
) -> str:
    """Generate table_ablation_5d_loo.tex showing Leave-One-Out (LOO) sensitivity."""
    m3_indexed = df_loo_m3.set_index("feature_dropped").loc[CANONICAL_5D_KEYS]
    mqt_indexed = df_loo_mqt.set_index("feature_dropped").loc[CANONICAL_5D_KEYS]

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Leave-One-Out (LOO) Feature Degradation Sensitivity across 14 Benchmarks}. "
        r"Marginal change in calibration error ($\Delta \text{Ada-ECE} = \text{Ada-ECE}_{-j} - \text{Ada-ECE}_{\text{Full 5D}}$) "
        r"and discrimination ($\Delta \text{AUROC} = \text{AUROC}_{-j} - \text{AUROC}_{\text{Full 5D}}$) when dropping "
        r"individual feature $x_j$ from Trajectory Platt (5D). Positive $\Delta \text{Ada-ECE}$ and negative $\Delta \text{AUROC}$ "
        r"indicate feature necessity. \textbf{Bold}: Rank 1, \textit{Italic}: Rank 2 degradation impact.}",
        r"\label{tab:ablation_5d_loo}",
        r"\vspace{2mm}",
        r"\begin{tabular}{clccccl}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dropped Feature}} & \multirow{2}{*}{\textbf{Functional Role}} & "
        r"\multicolumn{2}{c}{\textbf{M3-LLaVA (7B)}} & \multicolumn{2}{c}{\textbf{MQT-LLaVA (7B)}} & \multirow{2}{*}{\textbf{Macro Impact}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"& & \textbf{$\Delta$ Ada-ECE} & \textbf{$\Delta$ AUROC} & \textbf{$\Delta$ Ada-ECE} & \textbf{$\Delta$ AUROC} & \\",
        r"\midrule",
    ]

    macro_impacts = {
        "x1": "\\textbf{Catastrophic AUROC Collapse (Anchor Loss)}",
        "x2": "Scale Invariance Shift",
        "x3": "Entropy Dynamics Degradation",
        "x4": "Argmax Flipping Instability",
        "x5": "Monotonic Decay Disruption",
    }

    m3_d_ada_vals = m3_indexed["macro_delta_ada_ece"].tolist()
    m3_d_auroc_vals = m3_indexed["macro_delta_auroc"].tolist()
    mqt_d_ada_vals = mqt_indexed["macro_delta_ada_ece"].tolist()
    mqt_d_auroc_vals = mqt_indexed["macro_delta_auroc"].tolist()

    m3_ranked_ada = rank_and_format_delta(
        m3_d_ada_vals, higher_degradation_is_worse=True, decimals=4
    )
    m3_ranked_auroc = rank_and_format_delta(
        m3_d_auroc_vals, higher_degradation_is_worse=False, decimals=3
    )
    mqt_ranked_ada = rank_and_format_delta(
        mqt_d_ada_vals, higher_degradation_is_worse=True, decimals=4
    )
    mqt_ranked_auroc = rank_and_format_delta(
        mqt_d_auroc_vals, higher_degradation_is_worse=False, decimals=3
    )

    for i, feat in enumerate(CANONICAL_5D_KEYS):
        f_name = FEATURE_NAMES.get(feat, feat)
        role = ROLE_DESCRIPTIONS.get(feat, "")
        feat_math = f"$x_{{{feat[1:]}}}$ ({f_name})"

        m3_d_ada_str = m3_ranked_ada[i]
        m3_d_auroc_str = m3_ranked_auroc[i]
        mqt_d_ada_str = mqt_ranked_ada[i]
        mqt_d_auroc_str = mqt_ranked_auroc[i]

        impact = macro_impacts.get(feat, "")
        lines.append(
            f"{feat_math} & {role} & {m3_d_ada_str} & {m3_d_auroc_str} & {mqt_d_ada_str} & {mqt_d_auroc_str} & {impact} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def generate_14ds_grid_table(
    df_raw_m3: pd.DataFrame,
    df_raw_mqt: pd.DataFrame,
) -> str:
    """Generate table_ablation_5d_14ds_grid.tex showing 1D, Best-2D, Best-3D, Best-4D, Full-5D."""
    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{\textbf{Exhaustive 14-Benchmark Progression Grid: Adaptive ECE ($\downarrow$)}. "
        r"Per-dataset comparison across 1D Platt ($\{x_1\}$), Optimal 2D, Optimal 3D, Optimal 4D, and Full Trajectory Platt 5D. "
        r"Option A ranking: \textbf{Bold} Rank 1, \textit{Italic} Rank 2 across each benchmark row.}",
        r"\label{tab:ablation_5d_14ds_grid}",
        r"\vspace{2mm}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Benchmark Dataset} & \textbf{1D Platt ($\{x_1\}$)} & \textbf{Best-2D} & \textbf{Best-3D} & \textbf{Best-4D} & \textbf{Full 5D (Ours)} \\",
        r"\midrule",
    ]

    for arch_name, df_raw in [("M3-LLaVA (7B)", df_raw_m3), ("MQT-LLaVA (7B)", df_raw_mqt)]:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textsc{{\\textbf{{{arch_name}}}}}}} \\\\")
        lines.append(r"\midrule")

        datasets = [ds for ds in DATASET_DISPLAY_NAMES if ds in df_raw["dataset"].unique()]

        col_1d_list: list[float] = []
        col_2d_list: list[float] = []
        col_3d_list: list[float] = []
        col_4d_list: list[float] = []
        col_5d_list: list[float] = []

        for ds in datasets:
            ds_sub = df_raw[df_raw["dataset"] == ds]
            ds_disp = DATASET_DISPLAY_NAMES.get(ds, ds)

            val_1d = float(ds_sub[ds_sub["subset"] == "x1"]["ada_ece"].iloc[0])
            val_2d = float(ds_sub[ds_sub["cardinality"] == 2]["ada_ece"].min())
            val_3d = float(ds_sub[ds_sub["cardinality"] == 3]["ada_ece"].min())
            val_4d = float(ds_sub[ds_sub["cardinality"] == 4]["ada_ece"].min())
            val_5d = float(
                ds_sub[ds_sub["subset"] == "+".join(CANONICAL_5D_KEYS)]["ada_ece"].iloc[0]
            )

            col_1d_list.append(val_1d)
            col_2d_list.append(val_2d)
            col_3d_list.append(val_3d)
            col_4d_list.append(val_4d)
            col_5d_list.append(val_5d)

            row_vals = [val_1d, val_2d, val_3d, val_4d, val_5d]
            ranked = rank_and_format_decimal(row_vals, higher_is_better=False, decimals=4)
            lines.append(
                f"{ds_disp} & {ranked[0]} & {ranked[1]} & {ranked[2]} & {ranked[3]} & {ranked[4]} \\\\"
            )

        # Macro Mean row
        m_row = [
            float(np.mean(col_1d_list)),
            float(np.mean(col_2d_list)),
            float(np.mean(col_3d_list)),
            float(np.mean(col_4d_list)),
            float(np.mean(col_5d_list)),
        ]
        m_ranked = rank_and_format_decimal(m_row, higher_is_better=False, decimals=4)
        lines.append(r"\midrule")
        lines.append(
            f"\\textbf{{Macro Mean}} & {m_ranked[0]} & {m_ranked[1]} & {m_ranked[2]} & {m_ranked[3]} & {m_ranked[4]} \\\\"
        )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def generate_single_dataset_progression_table(
    df_m3_raw: pd.DataFrame,
    df_mqt_raw: pd.DataFrame,
    dataset_key: str,
    dataset_display: str,
) -> str:
    """Generate cardinality progression table k=1..5 for a single dataset."""
    label_slug = dataset_key.replace("-", "")
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{\textbf{Combinatorial 5D Trajectory Feature Cardinality Progression on \textsc{"
            + dataset_display
            + r"} ($k \in \{1 \dots 5\}$)}. "
            r"Out-of-fold calibration error and discrimination across feature cardinality levels for M3-LLaVA (7B) and MQT-LLaVA (7B). "
            r"Evaluated across all $\sum_{k=1}^5 \binom{5}{k} = 31$ feature subsets. Reported in pure decimal format. "
            r"\textbf{Bold}: Rank 1, \textit{Italic}: Rank 2 within each architecture. Note: Subsets lacking the base logit anchor $x_1$ "
            r"achieve artificially low quantile Ada-ECE via class-prior probability clustering, but suffer severe AUROC collapse "
            r"and degraded Brier scores; joint calibration and discrimination require the full 5D representation.}"
        ),
        r"\label{tab:ablation_5d_" + label_slug + "}",
        r"\providecommand{\tablestyle}[2]{\setlength{\tabcolsep}{#1}\renewcommand{\arraystretch}{#2}}",
        r"\tablestyle{3.5pt}{1.05}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        (
            r"\textbf{Cardinality ($k$)} & \textbf{Subsets} & \textbf{Optimal Formula} & "
            r"\textbf{Ada-ECE} $\downarrow$ & \textbf{Mean $\pm$ Std} & \textbf{AUROC} $\uparrow$ & \textbf{Brier} $\downarrow$ \\"
        ),
        r"\midrule",
    ]

    for arch_name, df_raw in [("M3-LLaVA (7B)", df_m3_raw), ("MQT-LLaVA (7B)", df_mqt_raw)]:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{arch_name}}}}} \\\\")
        lines.append(r"\midrule")

        ds_raw = df_raw[df_raw["dataset"] == dataset_key]

        k_data: list[dict[str, object]] = []
        for k in range(1, 6):
            k_sub = ds_raw[ds_raw["cardinality"] == k]
            best_row = k_sub.sort_values("ada_ece").iloc[0]
            k_data.append(
                {
                    "cardinality": k,
                    "n_subsets": len(k_sub),
                    "best_subset": str(best_row["subset"]),
                    "best_ada_ece": float(best_row["ada_ece"]),
                    "mean_ada_ece": float(k_sub["ada_ece"].mean()),
                    "std_ada_ece": float(k_sub["ada_ece"].std()) if len(k_sub) > 1 else 0.0,
                    "best_auroc": float(best_row["auroc"]),
                    "best_brier": float(best_row["brier"]),
                }
            )

        ada_bests = [float(d["best_ada_ece"]) for d in k_data]
        auroc_bests = [float(d["best_auroc"]) for d in k_data]
        brier_bests = [float(d["best_brier"]) for d in k_data]

        ranked_ada = rank_and_format_decimal(ada_bests, higher_is_better=False, decimals=4)
        ranked_auroc = rank_and_format_decimal(auroc_bests, higher_is_better=True, decimals=3)
        ranked_brier = rank_and_format_decimal(brier_bests, higher_is_better=False, decimals=4)

        for idx, d in enumerate(k_data):
            k_val = int(d["cardinality"])
            n_sub = int(d["n_subsets"])
            formula = format_latex_formula(str(d["best_subset"]))
            ada_best_str = ranked_ada[idx]
            mean_std_str = f"{float(d['mean_ada_ece']):.4f} $\\pm$ {float(d['std_ada_ece']):.4f}"
            auroc_best_str = ranked_auroc[idx]
            brier_best_str = ranked_brier[idx]

            lines.append(
                f"$k = {k_val}$ & {n_sub} & {formula} & {ada_best_str} & {mean_std_str} & {auroc_best_str} & {brier_best_str} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication LaTeX tables for 5D combinatorial ablation study."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/experiments/ablation_5d",
        help="Directory containing ablation CSV artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_tables",
        help="Directory to save LaTeX tables.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_m3_macro = pd.read_csv(results_dir / "ablation_5d_m3_macro_progression.csv")
    df_m3_loo = pd.read_csv(results_dir / "ablation_5d_m3_loo_sensitivity.csv")
    df_m3_raw = pd.read_csv(results_dir / "ablation_5d_m3_raw_all_combinations.csv")

    df_mqt_macro = pd.read_csv(results_dir / "ablation_5d_mqt_macro_progression.csv")
    df_mqt_loo = pd.read_csv(results_dir / "ablation_5d_mqt_loo_sensitivity.csv")
    df_mqt_raw = pd.read_csv(results_dir / "ablation_5d_mqt_raw_all_combinations.csv")

    # 1. Macro progression table
    tex_macro = generate_macro_progression_table(df_m3_macro, df_mqt_macro)
    (out_dir / "table_ablation_5d_macro_progression.tex").write_text(tex_macro, encoding="utf-8")
    print(f"Generated {out_dir / 'table_ablation_5d_macro_progression.tex'}")

    # 2. LOO sensitivity table
    tex_loo = generate_loo_table(df_m3_loo, df_mqt_loo)
    (out_dir / "table_ablation_5d_loo.tex").write_text(tex_loo, encoding="utf-8")
    print(f"Generated {out_dir / 'table_ablation_5d_loo.tex'}")

    # 3. 14-dataset grid table
    tex_grid = generate_14ds_grid_table(df_m3_raw, df_mqt_raw)
    (out_dir / "table_ablation_5d_14ds_grid.tex").write_text(tex_grid, encoding="utf-8")
    print(f"Generated {out_dir / 'table_ablation_5d_14ds_grid.tex'}")

    # 4. Single-dataset progression tables
    target_datasets = [
        ("textvqa", "TextVQA"),
        ("vqav2", "VQAv2"),
        ("lego-puzzles", "LegoPuzzles"),
    ]
    for ds_key, ds_disp in target_datasets:
        slug = ds_key.replace("-", "")
        tex_ds = generate_single_dataset_progression_table(df_m3_raw, df_mqt_raw, ds_key, ds_disp)
        (out_dir / f"table_ablation_5d_{slug}.tex").write_text(tex_ds, encoding="utf-8")
        print(f"Generated {out_dir / f'table_ablation_5d_{slug}.tex'}")


if __name__ == "__main__":
    main()
