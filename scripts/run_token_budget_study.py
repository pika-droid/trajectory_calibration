#!/usr/bin/env python3
"""
Token Budget vs. Calibration Performance Benchmark Runner.

Evaluates calibration performance scaling across autoregressive visual token budgets:
    M3-LLaVA (7B):   k in {1, 2, 3, 4, 5} -> Single scales: [1, 9, 36, 144, 576]; Cumulative: [1, 10, 46, 190, 766]
    MQT-LLaVA (7B):  k in {1, 2, 3, 4, 5} -> Single scales: [1, 9, 36, 144, 256]; Cumulative: [1, 10, 46, 190, 446]

Evaluates 4 post-hoc calibration methods across 5 stratified train/test seeds (42, 43, 44, 45, 46):
1. Naive Confidence (NC)
2. Temperature Scaling (TS)
3. Platt Scaling (1D)
4. Trajectory Platt (5D)

Across 9 independent datasets (Core 7 + 2 Adversarial/Safety):
Core 7:       ai2d, chartqa, docvqa, scienceqa, textvqa, vizwiz-vqa, vqav2
Adversarial:  avqa, vllm-safety
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.calibrators.baselines import (
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    TemperatureScalingEstimator,
    TrajectoryPlattScaler,
)
from trajectory_calibration.calibrators.residual import evaluate_full_metric_panel
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS
from trajectory_calibration.features.loader import get_stratified_split, load_dataset_features
from trajectory_calibration.utils.helpers import set_seed

CORE_7_DATASETS: list[str] = [
    "ai2d",
    "chartqa",
    "docvqa",
    "scienceqa",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
]

ADVERSARIAL_2_DATASETS: list[str] = [
    "avqa",
    "vllm-safety",
]

ALL_9_DATASETS: list[str] = [*CORE_7_DATASETS, *ADVERSARIAL_2_DATASETS]

M3_SCALES: list[int] = [1, 9, 36, 144, 576]
MQT_SCALES: list[int] = [1, 9, 36, 144, 256]

DEFAULT_SEEDS: list[int] = [42, 43, 44, 45, 46]

METHODS_ORDER: list[str] = [
    "Naive Confidence (NC)",
    "Temperature Scaling (TS)",
    "Platt Scaling (1D)",
    "Trajectory Platt (5D)",
]


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for token budget benchmark."""
    parser = argparse.ArgumentParser(
        description="Token Budget vs. Calibration Performance Experiment."
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features",
        help="Directory containing pre-extracted features.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["m3", "mqt"],
        choices=["m3", "mqt"],
        help="Target model architectures.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_9_DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Train/test stratified random seeds.",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="experiments/token_budget/csv",
        help="Directory to save CSV results.",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="experiments/token_budget/figures",
        help="Directory to save Pareto figures.",
    )
    parser.add_argument(
        "--tables_dir",
        type=str,
        default="dataset_tables/token_budget",
        help="Directory to save LaTeX tables.",
    )
    return parser.parse_args()


def get_scale_levels(arch: str) -> list[dict[str, Any]]:
    """Returns scale hierarchy, single tokens, and cumulative tokens for architecture."""
    scales = M3_SCALES if "m3" in arch.lower() else MQT_SCALES
    levels = []
    for k in range(1, len(scales) + 1):
        prefix = scales[:k]
        single_tokens = scales[k - 1]
        cum_tokens = sum(prefix)
        levels.append(
            {
                "level": k,
                "scale": single_tokens,
                "prefix_scales": prefix,
                "single_tokens": single_tokens,
                "cum_tokens": cum_tokens,
            }
        )
    return levels


def fit_and_eval_level(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Fits NC, TS, 1D Platt, and Trajectory Platt (5D) and evaluates metrics."""
    X_tr_5d = np.asarray(train_df[CANONICAL_5D_KEYS].to_numpy(), dtype=np.float64)
    y_tr = np.asarray(train_df["is_correct"].to_numpy(), dtype=np.int64)

    X_te_5d = np.asarray(test_df[CANONICAL_5D_KEYS].to_numpy(), dtype=np.float64)
    y_te = np.asarray(test_df["is_correct"].to_numpy(), dtype=np.int64)
    c_te = np.asarray(test_df["c_fine"].to_numpy(), dtype=np.float64)

    models: dict[str, Any] = {
        "Naive Confidence (NC)": NaiveConfidenceEstimator(),
        "Temperature Scaling (TS)": TemperatureScalingEstimator(),
        "Platt Scaling (1D)": PlattScalingEstimator(random_state=seed),
        "Trajectory Platt (5D)": TrajectoryPlattScaler(n_features=5),
    }

    results: dict[str, dict[str, Any]] = {}
    for m_name, model in models.items():
        model.fit(X_tr_5d, y_tr)
        probs = model.predict_proba(X_te_5d)
        panel = evaluate_full_metric_panel(probs, y_te, c_te, y_train=y_tr)
        results[m_name] = panel

    return results


def run_dataset_eval(
    features_dir: Path | str,
    ds_name: str,
    arch: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Runs token budget study across all 5 levels and seeds for one dataset."""
    levels = get_scale_levels(arch)
    rows: list[dict[str, Any]] = []

    for lvl_info in levels:
        k = lvl_info["level"]
        prefix = lvl_info["prefix_scales"]
        single_tokens = lvl_info["single_tokens"]
        cum_tokens = lvl_info["cum_tokens"]

        df = load_dataset_features(
            features_dir,
            ds_name=ds_name,
            arch=arch,
            gen_temperature=0.0,
            scales=prefix,
        )

        for seed in seeds:
            tr_idx, te_idx = get_stratified_split(df, test_size=0.2, random_state=seed)
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            test_df = df.iloc[te_idx].reset_index(drop=True)

            evals = fit_and_eval_level(train_df, test_df, seed=seed)

            for m_name in METHODS_ORDER:
                panel = evals[m_name]
                tokens_used = cum_tokens if "5D" in m_name else single_tokens
                row: dict[str, Any] = {
                    "dataset": ds_name,
                    "arch": arch,
                    "level": k,
                    "scale": single_tokens,
                    "single_tokens": single_tokens,
                    "cum_tokens": cum_tokens,
                    "tokens_used": tokens_used,
                    "seed": seed,
                    "method": m_name,
                    "ece_percent": panel["ece_percent"],
                    "adaptive_ece_percent": panel["adaptive_ece_percent"],
                    "brier": panel["brier"],
                    "auroc": panel["auroc"],
                }
                rows.append(row)

    return rows


def run_architecture_benchmark(
    features_dir: Path | str,
    arch: str,
    datasets: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs token budget study across all datasets for a given architecture."""
    all_rows: list[dict[str, Any]] = []
    for ds in datasets:
        print(f"[{arch.upper()}] Running token budget benchmark on {ds}...")
        ds_rows = run_dataset_eval(features_dir, ds, arch, seeds)
        all_rows.extend(ds_rows)
        print(f"[{arch.upper()}] Completed {ds} ({len(ds_rows)} eval rows).")

    raw_df = pd.DataFrame(all_rows)

    metrics = ["ece_percent", "adaptive_ece_percent", "brier", "auroc"]
    agg_dict = {m: ["mean", "std"] for m in metrics}
    grouped = raw_df.groupby(
        [
            "dataset",
            "arch",
            "level",
            "scale",
            "single_tokens",
            "cum_tokens",
            "tokens_used",
            "method",
        ]
    )
    summary_df = grouped.agg(agg_dict).reset_index()
    summary_df.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] else str(c[0]) for c in summary_df.columns
    ]

    return raw_df, summary_df


def compute_core7_macro(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Core 7 Macro-Averaged metrics at each token depth level."""
    core7_df = raw_df[raw_df["dataset"].isin(CORE_7_DATASETS)]

    seed_macro_raw = core7_df.groupby(
        ["arch", "level", "scale", "single_tokens", "cum_tokens", "tokens_used", "seed", "method"]
    )[["ece_percent", "adaptive_ece_percent", "brier", "auroc"]].mean()
    seed_macro = pd.DataFrame(seed_macro_raw).reset_index()

    macro_agg = seed_macro.groupby(
        ["arch", "level", "scale", "single_tokens", "cum_tokens", "tokens_used", "method"]
    ).agg(
        ece_mean=("ece_percent", "mean"),
        ece_std=("ece_percent", "std"),
        ada_ece_mean=("adaptive_ece_percent", "mean"),
        ada_ece_std=("adaptive_ece_percent", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
    )
    macro_df = pd.DataFrame(macro_agg).reset_index()

    macro_df["ece_std"] = macro_df["ece_std"].fillna(0.0)
    macro_df["ada_ece_std"] = macro_df["ada_ece_std"].fillna(0.0)
    macro_df["brier_std"] = macro_df["brier_std"].fillna(0.0)
    macro_df["auroc_std"] = macro_df["auroc_std"].fillna(0.0)

    method_order_map = {m: i for i, m in enumerate(METHODS_ORDER)}
    macro_df["method_order"] = macro_df["method"].map(lambda m: method_order_map.get(m, 999))
    macro_df = macro_df.sort_values(["level", "method_order"]).reset_index(drop=True)
    return macro_df


def rank_and_format_latex(
    vals: list[float | None], higher_is_better: bool = False, decimals: int = 2
) -> list[str]:
    """Option A Universal Formatting: bold rank 1, italic rank 2."""
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


def format_token_budget_latex_subtable(
    df: pd.DataFrame,
    title_caption: str,
    tab_label: str,
    is_macro: bool = False,
) -> str:
    """Renders LaTeX table for an individual dataset or macro evaluation across 5 levels."""
    ece_col = "ece_mean" if is_macro else "ece_percent_mean"
    ada_col = "ada_ece_mean" if is_macro else "adaptive_ece_percent_mean"
    brier_col = "brier_mean"
    auroc_col = "auroc_mean"

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{title_caption} \\textbf{{Bold}}: best; \\textit{{italic}}: second best. $\\downarrow$/$\\uparrow$: lower/higher is better.}}",
        f"\\label{{{tab_label}}}",
        "\\tablestyle{4pt}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{cclcccc}",
        "\\toprule",
        "\\textbf{Depth ($k$)} & \\textbf{Tokens ($T$)} & \\textbf{Calibration Method} & "
        "\\textbf{ECE (\\%)} $\\downarrow$ & \\textbf{Ada-ECE (\\%)} $\\downarrow$ & "
        "\\textbf{Brier} $\\downarrow$ & \\textbf{AUROC} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    levels = sorted(df["level"].unique())
    for lvl_idx, lvl in enumerate(levels):
        sub = cast(pd.DataFrame, df[df["level"] == lvl])
        methods_in_sub = [m for m in METHODS_ORDER if m in list(sub["method"])]

        rows_data = []
        for m in methods_in_sub:
            sub_m = cast(pd.DataFrame, sub[sub["method"] == m])
            t_used = int(np.asarray(sub_m["tokens_used"])[0])
            s_val = int(np.asarray(sub_m["scale"])[0])
            cum_val = int(np.asarray(sub_m["cum_tokens"])[0])
            rows_data.append(
                (
                    m,
                    t_used,
                    s_val,
                    cum_val,
                    float(np.asarray(sub_m[ece_col])[0]),
                    float(np.asarray(sub_m[ada_col])[0]),
                    float(np.asarray(sub_m[brier_col])[0]),
                    float(np.asarray(sub_m[auroc_col])[0]),
                )
            )

        ece_strs = rank_and_format_latex(
            [r[4] for r in rows_data], higher_is_better=False, decimals=2
        )
        ada_strs = rank_and_format_latex(
            [r[5] for r in rows_data], higher_is_better=False, decimals=2
        )
        brier_strs = rank_and_format_latex(
            [r[6] for r in rows_data], higher_is_better=False, decimals=4
        )
        auc_strs = rank_and_format_latex(
            [r[7] for r in rows_data], higher_is_better=True, decimals=3
        )

        for i, (m, t_used, s_val, cum_val, _, _, _, _) in enumerate(rows_data):
            first_col = f"\\multirow{{{len(methods_in_sub)}}}{{*}}{{{lvl}}}" if i == 0 else ""
            is_tp = "5D" in m
            t_str = f"{cum_val}" if is_tp else f"{s_val}"

            if is_tp:
                color_prefix = "\\rowcolor{gray!10} "
                row_str = (
                    f"{color_prefix}{first_col} & {t_str} & \\textbf{{{m}}} & "
                    f"{ece_strs[i]} & {ada_strs[i]} & {brier_strs[i]} & {auc_strs[i]} \\\\"
                )
            else:
                row_str = (
                    f"{first_col} & {t_str} & {m} & "
                    f"{ece_strs[i]} & {ada_strs[i]} & {brier_strs[i]} & {auc_strs[i]} \\\\"
                )
            lines.append(row_str)

        if lvl_idx < len(levels) - 1:
            lines.append("\\midrule")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines)


def generate_all_latex_tables(
    summary_m3: pd.DataFrame,
    summary_mqt: pd.DataFrame,
    macro_m3: pd.DataFrame,
    macro_mqt: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """Generates all 11 LaTeX tables (2 Macro + 9 Individual Datasets)."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Macro Tables
    m3_caption = "\\textbf{Token Budget vs. Macro Calibration Performance Across Core 7 Datasets (M3-LLaVA 7B).}"
    t_macro_m3 = format_token_budget_latex_subtable(
        macro_m3, m3_caption, "tab:token_budget_macro_core7_m3", is_macro=True
    )
    (tables_dir / "token_budget_macro_core7_m3.tex").write_text(t_macro_m3 + "\n", encoding="utf-8")

    mqt_caption = "\\textbf{Token Budget vs. Macro Calibration Performance Across Core 7 Datasets (MQT-LLaVA 7B).}"
    t_macro_mqt = format_token_budget_latex_subtable(
        macro_mqt, mqt_caption, "tab:token_budget_macro_core7_mqt", is_macro=True
    )
    (tables_dir / "token_budget_macro_core7_mqt.tex").write_text(
        t_macro_mqt + "\n", encoding="utf-8"
    )

    # 2. Individual 9 Datasets
    file_map = {
        "ai2d": "ai2d.tex",
        "chartqa": "chartqa.tex",
        "docvqa": "docvqa.tex",
        "scienceqa": "scienceqa.tex",
        "textvqa": "textvqa.tex",
        "vizwiz-vqa": "vizwizvqa.tex",
        "vqav2": "vqav2.tex",
        "avqa": "avqa.tex",
        "vllm-safety": "vllmsafety.tex",
    }

    for ds, fname in file_map.items():
        sub_m3 = cast(pd.DataFrame, summary_m3[summary_m3["dataset"] == ds])
        sub_mqt = cast(pd.DataFrame, summary_mqt[summary_mqt["dataset"] == ds])

        cap_m3 = f"\\textbf{{Token Budget vs. Calibration Performance on \\texttt{{{ds}}} (M3-LLaVA 7B).}}"
        tex_m3 = format_token_budget_latex_subtable(
            sub_m3, cap_m3, f"tab:token_budget_{ds.replace('-', '_')}_m3", is_macro=False
        )

        cap_mqt = f"\\textbf{{Token Budget vs. Calibration Performance on \\texttt{{{ds}}} (MQT-LLaVA 7B).}}"
        tex_mqt = format_token_budget_latex_subtable(
            sub_mqt, cap_mqt, f"tab:token_budget_{ds.replace('-', '_')}_mqt", is_macro=False
        )

        full_ds_tex = tex_m3 + "\n\n" + tex_mqt + "\n"
        (tables_dir / fname).write_text(full_ds_tex, encoding="utf-8")

    print(f"Generated 11 LaTeX tables in: {tables_dir.resolve()}")


def plot_macro_pareto(
    macro_df: pd.DataFrame,
    arch_name: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Plots Pareto calibration efficiency curve (Token Budget vs Macro Ada-ECE)."""
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    method_styles = {
        "Naive Confidence (NC)": {"color": "#7f7f7f", "ls": ":", "marker": "s", "lw": 1.8},
        "Temperature Scaling (TS)": {"color": "#e67e22", "ls": "--", "marker": "o", "lw": 1.8},
        "Platt Scaling (1D)": {"color": "#27ae60", "ls": "-.", "marker": "^", "lw": 2.0},
        "Trajectory Platt (5D)": {"color": "#2980b9", "ls": "-", "marker": "D", "lw": 2.4, "ms": 9},
    }

    for m in METHODS_ORDER:
        sub = cast(pd.DataFrame, macro_df[macro_df["method"] == m])
        if sub.empty:
            continue
        sub = sub.sort_values(by="level")

        x_vals = sub["tokens_used"].to_numpy()
        y_vals = sub["ada_ece_mean"].to_numpy()
        y_stds = sub["ada_ece_std"].to_numpy()

        style = method_styles[m]
        ms = style.get("ms", 7)
        ax.plot(
            x_vals,
            y_vals,
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            label=m,
            linewidth=style["lw"],
            markersize=ms,
        )
        ax.fill_between(
            x_vals,
            np.maximum(0.0, y_vals - y_stds),
            y_vals + y_stds,
            color=style["color"],
            alpha=0.15,
        )

    ax.set_xscale("log")
    tokens_ticks = [1, 10, 46, 190, 766] if "m3" in arch_name.lower() else [1, 10, 46, 190, 446]
    ax.set_xticks(tokens_ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Visual Token Budget ($T$, Log Scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro Adaptive ECE (%) $\\downarrow$", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Visual Token Budget vs. Calibration Efficiency ({arch_name} 7B)\nMacro Average Across Core 7 Vision-Language Benchmarks",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto plot to: {out_png.resolve()}")


def plot_9ds_grid(
    summary_m3: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Plots a 3x3 grid of Token Budget vs Ada-ECE across all 9 datasets."""
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    method_styles = {
        "Naive Confidence (NC)": {"color": "#7f7f7f", "ls": ":", "marker": "s", "lw": 1.5},
        "Temperature Scaling (TS)": {"color": "#e67e22", "ls": "--", "marker": "o", "lw": 1.5},
        "Platt Scaling (1D)": {"color": "#27ae60", "ls": "-.", "marker": "^", "lw": 1.8},
        "Trajectory Platt (5D)": {"color": "#2980b9", "ls": "-", "marker": "D", "lw": 2.2, "ms": 7},
    }

    for idx, ds in enumerate(ALL_9_DATASETS):
        ax = axes[idx]
        sub_ds = cast(pd.DataFrame, summary_m3[summary_m3["dataset"] == ds])

        for m in METHODS_ORDER:
            sub_m = cast(pd.DataFrame, sub_ds[sub_ds["method"] == m])
            if sub_m.empty:
                continue
            sub_m = sub_m.sort_values(by="level")

            x_vals = sub_m["tokens_used"].to_numpy()
            y_vals = sub_m["adaptive_ece_percent_mean"].to_numpy()
            style = method_styles[m]
            ax.plot(
                x_vals,
                y_vals,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                label=m if idx == 0 else "",
                linewidth=style["lw"],
                markersize=style.get("ms", 5),
            )

        ax.set_xscale("log")
        ax.set_title(f"{ds.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tokens ($T$)", fontsize=9)
        ax.set_ylabel("Ada-ECE (%)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Visual Token Budget vs. Calibration Performance across 9 Datasets (M3-LLaVA 7B)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=11
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved 9-dataset grid plot to: {out_png.resolve()}")


def save_csv_results(
    raw_m3: pd.DataFrame,
    summary_m3: pd.DataFrame,
    macro_m3: pd.DataFrame,
    raw_mqt: pd.DataFrame,
    summary_mqt: pd.DataFrame,
    macro_mqt: pd.DataFrame,
    csv_dir: Path,
) -> None:
    """Saves all raw, per-dataset, and macro CSVs."""
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 1. Macro CSVs
    macro_m3.to_csv(csv_dir / "token_budget_macro_core7_m3.csv", index=False)
    macro_mqt.to_csv(csv_dir / "token_budget_macro_core7_mqt.csv", index=False)

    # 2. Combined Raw & Summary
    full_raw = pd.concat([raw_m3, raw_mqt], ignore_index=True)
    full_summary = pd.concat([summary_m3, summary_mqt], ignore_index=True)
    full_raw.to_csv(csv_dir / "token_budget_all_raw.csv", index=False)
    full_summary.to_csv(csv_dir / "token_budget_all_summary.csv", index=False)

    # 3. Individual 9 Dataset CSVs
    for ds in ALL_9_DATASETS:
        sub_ds = full_summary[full_summary["dataset"] == ds]
        safe_ds_name = ds.replace("-", "_")
        sub_ds.to_csv(csv_dir / f"token_budget_{safe_ds_name}.csv", index=False)

    print(f"Saved all CSV results to: {csv_dir.resolve()}")


def main() -> None:
    """Main CLI driver for Token Budget vs Calibration Performance Benchmark."""
    set_seed(42)
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    fig_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)

    print("=" * 80)
    print(" TOKEN BUDGET VS. CALIBRATION PERFORMANCE BENCHMARK (RQ - TOKEN EFFICIENCY)")
    print("=" * 80)

    # 1. Run M3 Benchmark
    print("\n--- Running M3-LLaVA (7B) Benchmark ---")
    raw_m3, summary_m3 = run_architecture_benchmark(
        features_dir=args.features_dir,
        arch="m3",
        datasets=args.datasets,
        seeds=args.seeds,
    )
    macro_m3 = compute_core7_macro(raw_m3)

    # 2. Run MQT Benchmark
    print("\n--- Running MQT-LLaVA (7B) Benchmark ---")
    raw_mqt, summary_mqt = run_architecture_benchmark(
        features_dir=args.features_dir,
        arch="mqt",
        datasets=args.datasets,
        seeds=args.seeds,
    )
    macro_mqt = compute_core7_macro(raw_mqt)

    # 3. Save CSVs
    print("\n--- Saving CSV Summaries ---")
    save_csv_results(raw_m3, summary_m3, macro_m3, raw_mqt, summary_mqt, macro_mqt, csv_dir)

    # 4. Generate LaTeX Tables
    print("\n--- Generating Publication LaTeX Tables ---")
    generate_all_latex_tables(summary_m3, summary_mqt, macro_m3, macro_mqt, tables_dir)

    # 5. Generate Pareto Plots
    print("\n--- Generating Pareto Curves & Figures ---")
    plot_macro_pareto(
        macro_m3,
        "M3-LLaVA",
        fig_dir / "token_budget_m3_core7_pareto.png",
        fig_dir / "token_budget_m3_core7_pareto.pdf",
    )
    plot_macro_pareto(
        macro_mqt,
        "MQT-LLaVA",
        fig_dir / "token_budget_mqt_core7_pareto.png",
        fig_dir / "token_budget_mqt_core7_pareto.pdf",
    )
    plot_9ds_grid(
        summary_m3,
        fig_dir / "token_budget_9ds_grid.png",
        fig_dir / "token_budget_9ds_grid.pdf",
    )

    print("\n" + "=" * 80)
    print(" TOKEN BUDGET BENCHMARK COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
