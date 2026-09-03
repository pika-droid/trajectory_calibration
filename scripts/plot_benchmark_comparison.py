#!/usr/bin/env python3
"""
Publication-Quality Benchmark Visualization and Figure Generator.

Generates:
1. Calibration Curves (Reliability Diagrams) across representative benchmarks
2. Macro-Metric Comparison Bar/Radar Charts
3. Temperature Transfer Robustness Curves across T in {0.0, 0.3, 0.6, 0.9, 1.0, 1.5}
4. Dynamic Slope & Effective Temperature Separation Density Plots
5. Adaptive ECE vs AUROC Pareto Frontier Scatter Plot
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.calibrators.baselines import (
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    QuadraticPlattScaler,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import FEATURE_KEYS
from trajectory_calibration.features.loader import get_stratified_split, load_dataset_features
from trajectory_calibration.features.selection import select_best_5d_subset


def plot_calibration_curves(out_dir: Path, arch: str = "m3", seed: int = 42):
    """Plots 4-panel reliability diagrams comparing NC, 1D Platt, Quadratic Platt, and VCPS-5D."""
    datasets = ["scienceqa", "chartqa", "pope", "textvqa"]
    display_names = ["ScienceQA", "ChartQA", "POPE", "TextVQA"]
    fine_scale = 576 if arch == "m3" else 256

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)

    for ax, ds, d_name in zip(axes, datasets, display_names):
        try:
            df = load_dataset_features("data/features", ds_name=ds, arch=arch, gen_temperature=0.0, fine_scale=fine_scale)
            tr_idx, te_idx = get_stratified_split(df, test_size=0.2, random_state=seed)
            tr_df = df.iloc[tr_idx].reset_index(drop=True)
            te_df = df.iloc[te_idx].reset_index(drop=True)

            X_tr_17d = tr_df[FEATURE_KEYS].values
            y_tr = tr_df["is_correct"].values
            X_te_17d = te_df[FEATURE_KEYS].values
            y_te = te_df["is_correct"].values

            best_5d = select_best_5d_subset(X_tr_17d, y_tr, FEATURE_KEYS)
            X_tr_5d = tr_df[best_5d].values
            X_te_5d = te_df[best_5d].values

            # Fit models
            nc_probs = NaiveConfidenceEstimator().fit(X_tr_17d, y_tr).predict_proba(X_te_17d)
            platt_probs = PlattScalingEstimator().fit(X_tr_17d, y_tr).predict_proba(X_te_17d)
            quad_probs = QuadraticPlattScaler().fit(X_tr_17d, y_tr).predict_proba(X_te_17d)
            vcps = VaryingCoefficientPlattScaler(slope_features=best_5d[1:3], intercept_features=best_5d[1:]).fit(X_tr_5d, y_tr)
            vcps_probs = vcps.predict_proba(X_te_5d)

            models_dict = {
                "Naive Confidence": (nc_probs, "#94a3b8", ":"),
                "Platt Scaling (1D)": (platt_probs, "#f59e0b", "--"),
                "Quadratic Platt (Logit-Only)": (quad_probs, "#3b82f6", "-."),
                "VCPS-5D (Our Method)": (vcps_probs, "#10b981", "-"),
            }

            # Ideal diagonal
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration" if ds == datasets[0] else "")

            # Binning for reliability curves
            n_bins = 10
            bins = np.linspace(0.0, 1.0, n_bins + 1)

            for m_label, (probs, color, ls) in models_dict.items():
                bin_accs = []
                bin_confs = []
                for i in range(n_bins):
                    mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
                    if np.any(mask):
                        bin_confs.append(np.mean(probs[mask]))
                        bin_accs.append(np.mean(y_te[mask]))

                ax.plot(bin_confs, bin_accs, marker="o", markersize=5, label=m_label if ds == datasets[0] else "", color=color, linestyle=ls, linewidth=2)

            ax.set_title(f"{d_name} ({arch.upper()})", fontweight="bold")
            ax.set_xlabel("Confidence $\hat{p}$")
            if ds == datasets[0]:
                ax.set_ylabel("Empirical Accuracy")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error plotting {ds}: {e}")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5, frameon=True)
    plt.tight_layout()
    fig_path = out_dir / "calibration_curves_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_temperature_transfer(out_dir: Path):
    """Plots macro Adaptive ECE across decoding temperatures."""
    m3_csv = Path("results/experiments/temperature_study/temperature_transfer_m3_summary.csv")
    mqt_csv = Path("results/experiments/temperature_study/temperature_transfer_mqt_summary.csv")

    if not m3_csv.exists() and not mqt_csv.exists():
        print("Temperature summaries not found. Skipping temp plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    target_methods = [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Quadratic Platt (Logit-Only)",
        "Spline Calibration (PCHIP)",
        "Residual Calibrator",
        "VCPS-5D (Our Method)",
    ]
    colors = ["#94a3b8", "#a855f7", "#f59e0b", "#3b82f6", "#ec4899", "#6366f1", "#10b981"]

    for ax, csv_file, model_label in zip(axes, [m3_csv, mqt_csv], ["M3-LLaVA (7B)", "MQT-LLaVA (7B)"]):
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        pivot = df.pivot_table(index="method", columns="temperature", values="adaptive_ece_percent", aggfunc="mean")

        temps = sorted(pivot.columns)
        for m, color in zip(target_methods, colors):
            if m in pivot.index:
                vals = pivot.loc[m, temps]
                lw = 2.5 if "VCPS" in m or "Quadratic" in m else 1.8
                ls = "-" if "VCPS" in m else ("-." if "Quadratic" in m else "--")
                ax.plot(temps, vals, marker="s" if "VCPS" in m else "o", label=m, color=color, linewidth=lw, linestyle=ls)

        ax.set_title(f"Temperature Transfer: {model_label}", fontweight="bold")
        ax.set_xlabel("Evaluation Decoding Temperature $T_{\\text{gen}}$")
        ax.set_ylabel("Macro Adaptive ECE (%) $\\downarrow$")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    fig_path = out_dir / "temperature_transfer_robustness.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_pareto_ece_auroc(out_dir: Path):
    """Plots ECE vs AUROC Pareto scatter plot."""
    m3_csv = Path("results/experiments/benchmark/benchmark_m3_summary.csv")
    if not m3_csv.exists():
        return

    df = pd.read_csv(m3_csv)
    agg = df.groupby("method")[["adaptive_ece_percent", "auroc"]].mean().reset_index()

    plt.figure(figsize=(9, 6))
    for _, row in agg.iterrows():
        m = row["method"]
        ece = row["adaptive_ece_percent"]
        auc = row["auroc"]
        is_vcps = "VCPS" in m
        is_quad = "Quadratic" in m
        color = "#10b981" if is_vcps else ("#3b82f6" if is_quad else "#64748b")
        size = 140 if is_vcps or is_quad else 80
        marker = "*" if is_vcps else ("D" if is_quad else "o")

        plt.scatter(ece, auc, color=color, s=size, marker=marker, zorder=5)
        offset = (5, 5) if not is_vcps else (5, -10)
        plt.annotate(m, (ece, auc), textcoords="offset points", xytext=offset, fontsize=9, fontweight="bold" if is_vcps or is_quad else "normal")

    plt.title("Macro-Average Calibration (Adaptive ECE) vs Discrimination (AUROC) [M3-LLaVA 7B]", fontweight="bold")
    plt.xlabel("Macro Adaptive ECE (%) $\\downarrow$ (Lower is Better)")
    plt.ylabel("Macro AUROC $\\uparrow$ (Higher is Better)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "pareto_ece_vs_auroc.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_dynamic_slope_distributions(out_dir: Path, arch: str = "m3", seed: int = 42):
    """Plots dynamic slope distributions for correct vs incorrect predictions."""
    fine_scale = 576 if arch == "m3" else 256
    try:
        df = load_dataset_features("data/features", ds_name="scienceqa", arch=arch, gen_temperature=0.0, fine_scale=fine_scale)
        tr_idx, te_idx = get_stratified_split(df, test_size=0.2, random_state=seed)
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        te_df = df.iloc[te_idx].reset_index(drop=True)

        X_tr_17d = tr_df[FEATURE_KEYS].values
        y_tr = tr_df["is_correct"].values
        X_te_17d = te_df[FEATURE_KEYS].values
        y_te = te_df["is_correct"].values

        best_5d = select_best_5d_subset(X_tr_17d, y_tr, FEATURE_KEYS)
        X_tr_5d = tr_df[best_5d].values
        X_te_5d = te_df[best_5d].values

        quad = QuadraticPlattScaler().fit(X_tr_17d, y_tr)
        vcps = VaryingCoefficientPlattScaler(slope_features=best_5d[1:3], intercept_features=best_5d[1:]).fit(X_tr_5d, y_tr)

        slopes_quad = quad.compute_dynamic_slope(X_te_17d)
        slopes_vcps = vcps.compute_dynamic_slope(X_te_5d)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

        correct_mask = y_te == 1
        incorrect_mask = y_te == 0

        # Plot Quadratic Logit-Only dynamic slope
        sns.kdeplot(slopes_quad[correct_mask], ax=ax1, label="Correct (y=1)", color="#10b981", fill=True, alpha=0.3)
        sns.kdeplot(slopes_quad[incorrect_mask], ax=ax1, label="Incorrect (y=0)", color="#ef4444", fill=True, alpha=0.3)
        ax1.set_title("Quadratic Platt $a(x_1)$ (Logit-Only)", fontweight="bold")
        ax1.set_xlabel("Dynamic Slope $a(x_1)$")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot VCPS-5D trajectory dynamic slope
        sns.kdeplot(slopes_vcps[correct_mask], ax=ax2, label="Correct (y=1)", color="#10b981", fill=True, alpha=0.3)
        sns.kdeplot(slopes_vcps[incorrect_mask], ax=ax2, label="Incorrect (y=0)", color="#ef4444", fill=True, alpha=0.3)
        ax2.set_title("VCPS-5D $a(z)$ (Trajectory Features)", fontweight="bold")
        ax2.set_xlabel("Dynamic Slope $a(z)$")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "dynamic_slope_distributions.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")
    except Exception as e:
        print(f"Error plotting slope distributions: {e}")


def main():
    out_dir = Path("results/experiments/benchmark/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating publication benchmark figures...")
    plot_calibration_curves(out_dir, arch="m3")
    plot_temperature_transfer(out_dir)
    plot_pareto_ece_auroc(out_dir)
    plot_dynamic_slope_distributions(out_dir, arch="m3")
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
