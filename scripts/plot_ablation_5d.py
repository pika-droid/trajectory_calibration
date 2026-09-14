#!/usr/bin/env python3
"""
Publication Figure Generator for 14-Dataset Combinatorial 5D Ablation Study.

Generates:
1. figure_ablation_5d_cardinality_progression.png:
   Dual subplots (M3-LLaVA & MQT-LLaVA) illustrating Ada-ECE variance collapse,
   monotonic progression, and AUROC gains across cardinality k in 1..5.
2. figure_ablation_5d_loo_marginal_utility.png:
   Grouped bar chart showing average performance degradation (Delta Ada-ECE %)
   when dropping each canonical feature x1..x5 across all 14 benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

FEATURE_LABELS = {
    "x1": r"$x_1$: Final Logit" + "\n(Uncal. Anchor)",
    "x2": r"$x_2$: Answer Stability" + "\n(Scale Consistency)",
    "x3": r"$x_3$: Entropy Slope" + "\n(Decay Dynamics)",
    "x4": r"$x_4$: Flip Frequency" + "\n(Argmax Transitions)",
    "x5": r"$x_5$: Monotonicity" + "\n(Directionality)",
}


def load_ablation_artifacts(
    base_dir: Path, arch: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load macro progression, LOO sensitivity, and raw combinations for an architecture."""
    macro_file = base_dir / f"ablation_5d_{arch}_macro_progression.csv"
    loo_file = base_dir / f"ablation_5d_{arch}_loo_sensitivity.csv"
    raw_file = base_dir / f"ablation_5d_{arch}_raw_all_combinations.csv"

    if not macro_file.exists() or not loo_file.exists() or not raw_file.exists():
        raise FileNotFoundError(
            f"Ablation artifacts for {arch} not found in {base_dir}. Please run run_combinatorial_ablation_5d.py first."
        )

    df_macro = pd.read_csv(macro_file)
    df_loo = pd.read_csv(loo_file)
    df_raw = pd.read_csv(raw_file)
    return df_macro, df_loo, df_raw


def plot_cardinality_progression(
    df_macro_m3: pd.DataFrame,
    df_macro_mqt: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate Figure: Cardinality Progression & Variance Collapse (M3 vs MQT).
    Dual panels for M3-LLaVA and MQT-LLaVA showing Ada-ECE (left y-axis, %) and AUROC (right y-axis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=False)

    panels = [
        (axes[0], df_macro_m3, "M3-LLaVA (7B)", "#1f77b4", "#aec7e8", "#ff7f0e"),
        (axes[1], df_macro_mqt, "MQT-LLaVA (7B)", "#2ca02c", "#98df8a", "#d62728"),
    ]

    for ax, df, title, col_pri, col_fill, col_sec in panels:
        k_vals = df["cardinality"].values
        mean_ada = df["mean_ada_ece"].values * 100.0
        min_ada = df["min_ada_ece"].values * 100.0
        max_ada = df["max_ada_ece"].values * 100.0
        best_ada = df["best_ada_ece"].values * 100.0
        best_auroc = df["best_auroc"].values

        # Left Y-axis: Ada-ECE (%)
        ax.fill_between(
            k_vals,
            min_ada,
            max_ada,
            color=col_fill,
            alpha=0.45,
            label=r"Combinatorial Envelope [$\min, \max$]",
        )
        l1 = ax.plot(
            k_vals,
            mean_ada,
            color=col_pri,
            linestyle="--",
            marker="s",
            linewidth=2,
            label=r"Mean of $\binom{5}{k}$ Subsets",
        )
        l2 = ax.plot(
            k_vals,
            best_ada,
            color=col_pri,
            linestyle="-",
            marker="o",
            markersize=8,
            linewidth=2.5,
            label="Optimal Subset (Ada-ECE)",
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(r"Trajectory Feature Cardinality ($k \in \{1 \dots 5\}$)", fontsize=11)
        ax.set_ylabel(r"Macro Adaptive ECE ($\% \downarrow$)", fontsize=11, color=col_pri)
        ax.tick_params(axis="y", labelcolor=col_pri)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"$k={k}$" for k in k_vals], fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Right Y-axis: AUROC
        ax_r = ax.twinx()
        l3 = ax_r.plot(
            k_vals,
            best_auroc,
            color=col_sec,
            linestyle="-.",
            marker="^",
            markersize=8,
            linewidth=2,
            label="Optimal Subset AUROC",
        )
        ax_r.set_ylabel(r"Macro AUROC ($\uparrow$)", fontsize=11, color=col_sec)
        ax_r.tick_params(axis="y", labelcolor=col_sec)
        ax_r.grid(False)

        # Combined Legend
        lines = l2 + l1 + [plt.Line2D([0], [0], color=col_fill, lw=6, alpha=0.6)] + l3
        labels = [
            "Best Formula (Ada-ECE)",
            r"Mean $\binom{5}{k}$ Subsets",
            r"$\binom{5}{k}$ Variance Range",
            "Best Formula AUROC",
        ]
        ax.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Combinatorial 5D Trajectory Feature Ablation: Cardinality Progression & Variance Collapse",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cardinality progression figure to {output_path}")


def plot_loo_marginal_utility(
    df_loo_m3: pd.DataFrame,
    df_loo_mqt: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate Figure: Leave-One-Out (LOO) Marginal Feature Utility across 14 Benchmarks.
    Side-by-side grouped bar chart comparing Delta Ada-ECE (%) for M3-LLaVA and MQT-LLaVA.
    """
    # Align order by canonical features x1..x5
    order = ["x1", "x2", "x3", "x4", "x5"]
    m3_indexed = df_loo_m3.set_index("feature_dropped").loc[order]
    mqt_indexed = df_loo_mqt.set_index("feature_dropped").loc[order]

    x = np.arange(len(order))
    width = 0.35

    m3_deltas = m3_indexed["macro_delta_ada_ece"].values * 100.0
    mqt_deltas = mqt_indexed["macro_delta_ada_ece"].values * 100.0

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    rects1 = ax.bar(
        x - width / 2,
        m3_deltas,
        width,
        label="M3-LLaVA (7B)",
        color="#1f77b4",
        edgecolor="black",
        alpha=0.88,
    )
    rects2 = ax.bar(
        x + width / 2,
        mqt_deltas,
        width,
        label="MQT-LLaVA (7B)",
        color="#2ca02c",
        edgecolor="black",
        alpha=0.88,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_ylabel(
        r"Marginal Degradation $\Delta$ Ada-ECE ($\% \uparrow$, Higher = More Essential)",
        fontsize=11,
    )
    ax.set_title(
        r"Leave-One-Out (LOO) Feature Degradation $\Delta \text{Ada-ECE} = \text{Ada-ECE}_{-j} - \text{Ada-ECE}_{\text{5D}}$ (All 14 Benchmarks)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in order], fontsize=9.5)
    ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate bar values
    def autolabel(rects: list[plt.Rectangle]) -> None:
        for rect in rects:
            h = rect.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.annotate(
                f"+{h:.2f}%" if h >= 0 else f"{h:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -10),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8.5,
                fontweight="bold",
            )

    autolabel(list(rects1))
    autolabel(list(rects2))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved LOO marginal utility figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot publication figures for 5D combinatorial ablation study."
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
        default="results/experiments/ablation_5d/figures",
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_macro_m3, df_loo_m3, _ = load_ablation_artifacts(results_dir, "m3")
    df_macro_mqt, df_loo_mqt, _ = load_ablation_artifacts(results_dir, "mqt")

    plot_cardinality_progression(
        df_macro_m3,
        df_macro_mqt,
        out_dir / "figure_ablation_5d_cardinality_progression.png",
    )
    plot_loo_marginal_utility(
        df_loo_m3,
        df_loo_mqt,
        out_dir / "figure_ablation_5d_loo_marginal_utility.png",
    )


if __name__ == "__main__":
    main()
