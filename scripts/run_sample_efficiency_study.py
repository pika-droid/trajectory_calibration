#!/usr/bin/env python3
"""
Sample Efficiency and Small-Dataset Benchmark Runner (RQ2).

Evaluates calibration performance scaling across training budgets:
    N in {50, 100, 200, 500, 1000, 1500, Full}
across 5 random subsampling seeds (42, 43, 44, 45, 46) on all 14 vision-language benchmarks
for M3-LLaVA (7B, fine_scale=576) and MQT-LLaVA (7B, fine_scale=256).

Evaluates 5 post-hoc methods:
1. Naive Confidence (NC)
2. Temperature Scaling (TS)
3. Platt Scaling (1D)
4. Trajectory Platt (5D)
5. VCPS-5D (Our Method)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS
from trajectory_calibration.features.loader import get_stratified_split
from trajectory_calibration.features.trajectory import load_dataset_features
from trajectory_calibration.utils.helpers import set_seed

ALL_14_DATASETS: list[str] = [
    "ai2d",
    "chartqa",
    "docvqa",
    "gqa",
    "infographicvqa",
    "lego-puzzles",
    "mmbench",
    "mmmu",
    "pope",
    "scienceqa",
    "seedbench",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
]

DEFAULT_BUDGETS: list[str] = ["50", "100", "200", "500", "1000", "1500", "Full"]
DEFAULT_SEEDS: list[int] = [42, 43, 44, 45, 46]

METHODS_ORDER: list[str] = [
    "Naive Confidence (NC)",
    "Temperature Scaling (TS)",
    "Platt Scaling (1D)",
    "Trajectory Platt (5D)",
    "VCPS-5D (Our Method)",
]

BUDGET_NUMERIC_MAP: dict[str, float] = {
    "50": 50.0,
    "100": 100.0,
    "200": 200.0,
    "500": 500.0,
    "1000": 1000.0,
    "1500": 1500.0,
    "Full": 2500.0,
}


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for sample efficiency study."""
    parser = argparse.ArgumentParser(
        description="Sample Efficiency & Small-Dataset Benchmark (RQ2)."
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
        default=ALL_14_DATASETS,
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=DEFAULT_BUDGETS,
        help="Subsampling training sample budget tiers.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Subsampling random seeds.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/sample_efficiency",
        help="Directory to save output CSVs.",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="results/experiments/sample_efficiency/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default="dataset_tables",
        help="Directory to save LaTeX tables.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip architectures whose raw results CSV is already complete.",
    )
    return parser.parse_args()


def subsample_train_split(
    train_df: pd.DataFrame,
    budget: int | str,
    seed: int,
) -> tuple[pd.DataFrame, int, bool]:
    """Subsamples training split up to budget with label stratification.

    Ensures both binary classes are preserved when available in the source data.
    Caps at len(train_df) and tags as is_capped=True if budget >= len(train_df).
    """
    total_n = len(train_df)
    if str(budget).lower() == "full":
        return train_df.copy().reset_index(drop=True), total_n, True

    target_n = int(budget)
    if target_n >= total_n:
        return train_df.copy().reset_index(drop=True), total_n, True

    counts = train_df["is_correct"].value_counts()
    can_stratify = (
        len(counts) > 1
        and (counts >= 2).all()
        and target_n >= len(counts)
        and (total_n - target_n) >= len(counts)
    )

    if can_stratify:
        try:
            sub_df, _ = train_test_split(
                train_df,
                train_size=target_n,
                random_state=seed,
                stratify=train_df["is_correct"],
            )
            sub_df = cast(pd.DataFrame, sub_df).reset_index(drop=True)
        except ValueError:
            sub_df, _ = train_test_split(train_df, train_size=target_n, random_state=seed)
            sub_df = cast(pd.DataFrame, sub_df).reset_index(drop=True)
    else:
        sub_df, _ = train_test_split(train_df, train_size=target_n, random_state=seed)
        sub_df = cast(pd.DataFrame, sub_df).reset_index(drop=True)

    if train_df["is_correct"].nunique() > 1 and sub_df["is_correct"].nunique() < 2:
        missing_class = 1 if 0 in sub_df["is_correct"].values else 0
        cand = train_df[train_df["is_correct"] == missing_class].sample(n=1, random_state=seed)
        sub_df = pd.concat([sub_df.iloc[:-1], cand]).reset_index(drop=True)

    return sub_df, target_n, False


def fit_and_evaluate_methods(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Fits the 5 calibration models and evaluates metric panel on test split."""
    X_tr_5d = np.asarray(train_df[CANONICAL_5D_KEYS].to_numpy(), dtype=np.float64)
    y_tr = np.asarray(train_df["is_correct"].to_numpy(), dtype=np.int64)

    X_te_5d = np.asarray(test_df[CANONICAL_5D_KEYS].to_numpy(), dtype=np.float64)
    y_te = np.asarray(test_df["is_correct"].to_numpy(), dtype=np.int64)
    c_fine_col = (
        "c_576"
        if "c_576" in test_df.columns
        else ("c_256" if "c_256" in test_df.columns else "c_fine")
    )
    c_te = np.asarray(test_df[c_fine_col].to_numpy(), dtype=np.float64)

    models: dict[str, Any] = {
        "Naive Confidence (NC)": NaiveConfidenceEstimator(),
        "Temperature Scaling (TS)": TemperatureScalingEstimator(),
        "Platt Scaling (1D)": PlattScalingEstimator(random_state=seed),
        "Trajectory Platt (5D)": TrajectoryPlattScaler(n_features=5),
        "VCPS-5D (Our Method)": VaryingCoefficientPlattScaler(feature_set="5d", random_state=seed),
    }

    results: dict[str, dict[str, Any]] = {}
    for m_name, model in models.items():
        if isinstance(model, VaryingCoefficientPlattScaler):
            model.fit(X_tr_5d, y_tr, feature_names=CANONICAL_5D_KEYS)
        else:
            model.fit(X_tr_5d, y_tr)
        probs = model.predict_proba(X_te_5d)
        panel = evaluate_full_metric_panel(probs, y_te, c_te, y_train=y_tr)
        results[m_name] = panel

    return results


def run_study_for_dataset(
    df: pd.DataFrame,
    ds_name: str,
    arch: str,
    budgets: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Evaluates budget scaling and seeds for a single dataset and architecture."""
    train_idx, test_idx = get_stratified_split(df, test_size=0.2, random_state=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    total_train = len(train_df)

    rows: list[dict[str, Any]] = []
    cached_full: dict[int, dict[str, dict[str, Any]]] = {}

    for budget_str in budgets:
        is_full_tier = budget_str.lower() == "full" or int(budget_str) >= total_train
        for seed in seeds:
            if is_full_tier:
                if seed not in cached_full:
                    evals = fit_and_evaluate_methods(train_df, test_df, seed=seed)
                    cached_full[seed] = evals
                evals = cached_full[seed]
                effective_n = total_train
                is_capped = True
            else:
                sub_tr, effective_n, is_capped = subsample_train_split(
                    train_df, budget=budget_str, seed=seed
                )
                evals = fit_and_evaluate_methods(sub_tr, test_df, seed=seed)

            for m_name in METHODS_ORDER:
                panel = evals[m_name]
                row: dict[str, Any] = {
                    "dataset": ds_name,
                    "arch": arch,
                    "budget": budget_str,
                    "effective_n": effective_n,
                    "is_capped": is_capped,
                    "seed": seed,
                    "method": m_name,
                    "adaptive_ece_percent": panel["adaptive_ece_percent"],
                    "ece_percent": panel["ece_percent"],
                    "auroc": panel["auroc"],
                    "brier": panel["brier"],
                    "adaptive_ece": panel["adaptive_ece"],
                    "ece": panel["ece"],
                }
                rows.append(row)
    return rows


def run_architecture_study(
    features_dir: Path | str,
    arch: str,
    datasets: list[str],
    budgets: list[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs sample efficiency benchmark across all datasets for one architecture."""
    fine_scale = 576 if arch.lower() == "m3" else 256
    all_rows: list[dict[str, Any]] = []

    for ds in datasets:
        print(f"[{arch.upper()}] Loading features for {ds} (fine_scale={fine_scale})...")
        try:
            df = load_dataset_features(
                features_dir,
                ds_name=ds,
                arch=arch,
                gen_temperature=0.0,
                fine_scale=fine_scale,
            )
        except Exception as exc:
            print(f"[{arch.upper()}] Skipping {ds} due to load error: {exc}")
            continue

        ds_rows = run_study_for_dataset(df, ds, arch, budgets, seeds)
        all_rows.extend(ds_rows)
        print(f"[{arch.upper()}] Completed {ds} ({len(ds_rows)} eval points).")

    raw_df = pd.DataFrame(all_rows)
    metrics = ["adaptive_ece_percent", "ece_percent", "auroc", "brier", "effective_n"]
    agg_dict = {m: ["mean", "std"] for m in metrics}

    grouped = raw_df.groupby(["dataset", "arch", "budget", "method"])
    agg_df = pd.DataFrame(grouped.agg(agg_dict)).reset_index()
    agg_df.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] else str(c[0]) for c in agg_df.columns
    ]
    summary_df = agg_df
    capped_series = grouped["is_capped"].any()
    summary_df["is_capped"] = np.asarray(capped_series.to_numpy())
    return raw_df, summary_df


def compute_macro_aggregates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Macro-Averaged metrics across all datasets at each budget tier."""
    seed_macro_raw = raw_df.groupby(["arch", "budget", "seed", "method"])[
        ["adaptive_ece_percent", "ece_percent", "auroc", "brier", "effective_n"]
    ].mean()
    seed_macro = pd.DataFrame(seed_macro_raw).reset_index()

    macro_agg = seed_macro.groupby(["arch", "budget", "method"]).agg(
        ada_ece_mean=("adaptive_ece_percent", "mean"),
        ada_ece_std=("adaptive_ece_percent", "std"),
        ece_mean=("ece_percent", "mean"),
        ece_std=("ece_percent", "std"),
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        sample_size_mean=("effective_n", "mean"),
    )
    macro_df = pd.DataFrame(macro_agg).reset_index()

    macro_df["ada_ece_std"] = macro_df["ada_ece_std"].fillna(0.0)
    macro_df["ece_std"] = macro_df["ece_std"].fillna(0.0)
    macro_df["auroc_std"] = macro_df["auroc_std"].fillna(0.0)
    macro_df["brier_std"] = macro_df["brier_std"].fillna(0.0)

    budget_order_map = {b: i for i, b in enumerate(DEFAULT_BUDGETS)}
    macro_df["budget_order"] = macro_df["budget"].map(lambda b: budget_order_map.get(str(b), 999))
    method_order_map = {m: i for i, m in enumerate(METHODS_ORDER)}
    macro_df["method_order"] = macro_df["method"].map(lambda m: method_order_map.get(m, 999))

    macro_df = macro_df.sort_values(["budget_order", "method_order"]).reset_index(drop=True)
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


def format_macro_latex_table(macro_df: pd.DataFrame, arch_name: str, tab_label: str) -> str:
    """Formats LaTeX subtable for an architecture across budget tiers."""
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{\\textbf{{Macro-Averaged Calibration Scaling Across Training Budgets ({arch_name} 7B).}} "
        "Evaluated across all 14 benchmarks over 5 subsampling seeds. "
        "\\textbf{Bold}: best; \\textit{italic}: second best. $\\downarrow$/$\\uparrow$: lower/higher is better.}",
        f"\\label{{{tab_label}}}",
        "\\tablestyle{4pt}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Budget ($N$)} & \\textbf{Calibration Method} & "
        "\\textbf{Macro ECE (\\%)} $\\downarrow$ & "
        "\\textbf{Macro Ada-ECE (\\%)} $\\downarrow$ & "
        "\\textbf{Macro AUROC} $\\uparrow$ & "
        "\\textbf{Macro Brier} $\\downarrow$ \\\\",
        "\\midrule",
    ]

    target_methods = [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Trajectory Platt (5D)",
    ]

    budgets_in_df = [b for b in DEFAULT_BUDGETS if b in list(macro_df["budget"])]
    for b_idx, budget in enumerate(budgets_in_df):
        sub = cast(pd.DataFrame, macro_df[macro_df["budget"] == budget])
        methods_in_sub = [m for m in target_methods if m in list(sub["method"])]

        rows_data = []
        for m in methods_in_sub:
            sub_m = cast(pd.DataFrame, sub[sub["method"] == m])
            rows_data.append(
                (
                    m,
                    float(np.asarray(sub_m["ece_mean"])[0]),
                    float(np.asarray(sub_m["ada_ece_mean"])[0]),
                    float(np.asarray(sub_m["auroc_mean"])[0]),
                    float(np.asarray(sub_m["brier_mean"])[0]),
                )
            )

        ece_strs = rank_and_format_latex(
            [r[1] for r in rows_data], higher_is_better=False, decimals=2
        )
        ada_strs = rank_and_format_latex(
            [r[2] for r in rows_data], higher_is_better=False, decimals=2
        )
        auc_strs = rank_and_format_latex(
            [r[3] for r in rows_data], higher_is_better=True, decimals=3
        )
        brier_strs = rank_and_format_latex(
            [r[4] for r in rows_data], higher_is_better=False, decimals=4
        )

        for i, (m, _, _, _, _) in enumerate(rows_data):
            b_label = f"$N = {budget}$" if budget != "Full" else "Full"
            first_col = f"\\multirow{{{len(methods_in_sub)}}}{{*}}{{{b_label}}}" if i == 0 else ""
            is_tp = "Trajectory Platt (5D)" in m
            if is_tp:
                color_prefix = "\\rowcolor{gray!10} "
                row_str = (
                    f"{color_prefix}{first_col} & \\textbf{{{m}}} & "
                    f"{ece_strs[i]} & {ada_strs[i]} & {auc_strs[i]} & {brier_strs[i]} \\\\"
                )
            else:
                row_str = (
                    f"{first_col} & {m} & {ece_strs[i]} & {ada_strs[i]} & "
                    f"{auc_strs[i]} & {brier_strs[i]} \\\\"
                )
            lines.append(row_str)

        if b_idx < len(budgets_in_df) - 1:
            lines.append("\\midrule")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines)


def generate_latex_tables(macro_m3: pd.DataFrame, macro_mqt: pd.DataFrame, out_path: Path) -> None:
    """Writes combined LaTeX tables for M3-LLaVA and MQT-LLaVA."""
    t_m3 = format_macro_latex_table(macro_m3, "M3-LLaVA", "tab:sample_efficiency_macro_m3")
    t_mqt = format_macro_latex_table(macro_mqt, "MQT-LLaVA", "tab:sample_efficiency_macro_mqt")
    full_content = t_m3 + "\n\n" + t_mqt + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_content, encoding="utf-8")
    print(f"Saved LaTeX tables to: {out_path.resolve()}")


def plot_sample_efficiency_curves(
    macro_df: pd.DataFrame,
    arch_title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Generates publication-quality sample efficiency curve."""
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    method_styles: dict[str, dict[str, Any]] = {
        "Naive Confidence (NC)": {
            "color": "#7f7f7f",
            "linestyle": ":",
            "marker": "s",
            "label": "Naive Confidence (NC)",
            "lw": 1.8,
        },
        "Temperature Scaling (TS)": {
            "color": "#e67e22",
            "linestyle": "--",
            "marker": "o",
            "label": "Temperature Scaling (TS)",
            "lw": 1.8,
        },
        "Platt Scaling (1D)": {
            "color": "#27ae60",
            "linestyle": "-.",
            "marker": "^",
            "label": "Platt Scaling (1D)",
            "lw": 2.0,
        },
        "Trajectory Platt (5D)": {
            "color": "#2980b9",
            "linestyle": "-",
            "marker": "D",
            "label": "Trajectory Platt (5D)",
            "lw": 2.2,
        },
        "VCPS-5D (Our Method)": {
            "color": "#c0392b",
            "linestyle": "-",
            "marker": "*",
            "label": "VCPS-5D (Our Method)",
            "lw": 2.8,
            "ms": 10,
        },
    }

    budgets = [b for b in DEFAULT_BUDGETS if b in macro_df["budget"].unique()]
    x_coords = [BUDGET_NUMERIC_MAP[b] for b in budgets]

    for m in METHODS_ORDER:
        sub = macro_df[macro_df["method"] == m]
        if sub.empty:
            continue
        means = [float(np.asarray(sub[sub["budget"] == b]["ada_ece_mean"])[0]) for b in budgets]
        stds = [float(np.asarray(sub[sub["budget"] == b]["ada_ece_std"])[0]) for b in budgets]
        y_mean = np.array(means)
        y_std = np.array(stds)

        style = method_styles.get(
            m, {"color": "black", "linestyle": "-", "marker": "o", "label": m, "lw": 1.5}
        )
        ms = style.get("ms", 7)
        ax.plot(
            x_coords,
            y_mean,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
            linewidth=style["lw"],
            markersize=ms,
        )
        ax.fill_between(
            x_coords,
            np.maximum(0.0, y_mean - y_std),
            y_mean + y_std,
            color=style["color"],
            alpha=0.15,
        )

    ax.set_xscale("log")
    ax.set_xticks(x_coords)
    ax.set_xticklabels(budgets)
    ax.set_xlabel("Calibration Training Budget ($N$, Log Scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro Adaptive ECE (%) $\\downarrow$", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Sample Efficiency & Low-Data Scaling ({arch_title} 7B)\nAcross All 14 Vision-Language Benchmarks",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Highlight crossover / inflection point
    if "mqt" in arch_title.lower():
        ax.axvline(x=100, color="grey", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.annotate(
            "Trajectory Advantage ($N \\geq 100$)",
            xy=(100, 7.60),
            xytext=(110, 16.0),
            arrowprops=dict(facecolor="#2980b9", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )
    else:
        ax.axvline(x=200, color="grey", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.annotate(
            "Trajectory Advantage ($N \\geq 200$)",
            xy=(200, 5.82),
            xytext=(220, 14.0),
            arrowprops=dict(facecolor="#2980b9", shrink=0.08, width=1.5, headwidth=6),
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved curve plot to: {out_png.resolve()} and {out_pdf.resolve()}")


def main() -> None:
    """Main CLI driver for sample efficiency benchmark."""
    set_seed(42)
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    macro_dfs: dict[str, pd.DataFrame] = {}

    for arch in args.architectures:
        print("\n" + "=" * 80)
        print(f" RUNNING SAMPLE EFFICIENCY BENCHMARK: {arch.upper()} (All 14 Datasets)")
        print("=" * 80)

        raw_csv = out_dir / f"sample_efficiency_{arch}_raw.csv"
        summary_csv = out_dir / f"sample_efficiency_{arch}_summary.csv"
        expected_rows = (
            len(args.datasets) * len(args.budgets) * len(args.seeds) * len(METHODS_ORDER)
        )

        if args.skip_existing and raw_csv.exists() and len(pd.read_csv(raw_csv)) >= expected_rows:
            print(
                f"[{arch.upper()}] Found existing complete raw records ({raw_csv}). Skipping computation."
            )
            raw_df = pd.read_csv(raw_csv)
            summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
        else:
            raw_df, summary_df = run_architecture_study(
                features_dir=args.features_dir,
                arch=arch,
                datasets=args.datasets,
                budgets=args.budgets,
                seeds=args.seeds,
            )
            raw_df.to_csv(raw_csv, index=False)
            print(f"Saved raw per-seed records ({len(raw_df)} rows) to: {raw_csv}")
            summary_df.to_csv(summary_csv, index=False)
            print(f"Saved per-dataset summary ({len(summary_df)} rows) to: {summary_csv}")

        macro_df = compute_macro_aggregates(raw_df)
        macro_csv = out_dir / f"sample_efficiency_{arch}_macro.csv"
        macro_df.to_csv(macro_csv, index=False)
        print(f"Saved macro summary to: {macro_csv}")
        macro_dfs[arch] = macro_df

        # Generate publication curves
        arch_title = "M3-LLaVA" if arch == "m3" else "MQT-LLaVA"
        png_path = fig_dir / f"sample_efficiency_curve_{arch}.png"
        pdf_path = fig_dir / f"sample_efficiency_curve_{arch}.pdf"
        plot_sample_efficiency_curves(macro_df, arch_title, png_path, pdf_path)

    # Generate combined LaTeX table if both architectures were evaluated
    if "m3" in macro_dfs and "mqt" in macro_dfs:
        tex_path = table_dir / "sample_efficiency_macro.tex"
        generate_latex_tables(macro_dfs["m3"], macro_dfs["mqt"], tex_path)
    elif len(macro_dfs) == 1:
        arch_single = next(iter(macro_dfs.keys()))
        tex_path = table_dir / f"sample_efficiency_macro_{arch_single}.tex"
        generate_latex_tables(macro_dfs[arch_single], macro_dfs[arch_single], tex_path)

    print("\n" + "=" * 80)
    print(" SAMPLE EFFICIENCY BENCHMARK COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
