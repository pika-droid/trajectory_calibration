#!/usr/bin/env python3
"""
Update README.md Benchmark Tables & Analytics.

Reads the latest experiment outputs:
- results/experiments/benchmark/benchmark_{m3,mqt}_summary.csv
- results/umpire_eval/{m3,mqt}_llava_cumulative_summary.csv
- results/experiments/temperature_study/temperature_transfer_{m3,mqt}_summary.csv
- results/experiments/lodo/lodo_{m3,mqt}_summary.csv

Formats:
- Rank 1 (best): **bold**
- Rank 2 (second best): *italic*
- Win counts: family win (min(VCPS-17D, VCPS-5D) < baseline) AND separate counts for VCPS-17D and VCPS-5D.
Updates README.md cleanly and safely.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("c:/Users/ashmi/OneDrive/Documents/trajectory_calibration")
README_PATH = ROOT / "README.md"

ALL_14_DATASETS = [
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

ADA_ECE_DISPLAY_METHODS = [
    "Naive Confidence (NC)",
    "Temperature Scaling (TS)",
    "Platt Scaling (1D)",
    "Trajectory LR",
    "Trajectory LR (No Bias)",
    "Trajectory Platt (5D)",
    "Trajectory Platt (17D)",
    "Spline Calibration",
    "Adaptive TS (ATS)",
    "Residual Calibrator",
    "VCPS-5D (Our Method)",
    "VCPS-17D (Our Method)",
]


def generate_ada_ece_table_and_bullets(arch: str, arch_label: str) -> str:
    csv_path = ROOT / f"results/experiments/benchmark/benchmark_{arch}_summary.csv"
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="method", columns="dataset", values="adaptive_ece_percent")
    available_methods = [m for m in ADA_ECE_DISPLAY_METHODS if m in pivot.index]
    sub_pivot = pivot.loc[available_methods, ALL_14_DATASETS]

    ranks = {}
    for ds in ALL_14_DATASETS:
        vals = sorted(sub_pivot[ds].unique())
        best = vals[0]
        second = vals[1] if len(vals) > 1 else None
        ranks[ds] = (best, second)

    lines = []
    lines.append(f"###### {arch_label}: Adaptive ECE (%) [Lower is Better]\n")
    lines.append("| Calibration Method | " + " | ".join(ALL_14_DATASETS) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * len(ALL_14_DATASETS)) + " |")

    for m in available_methods:
        row_name = f"**{m}**" if ("VCPS" in m or "Naive" in m) else m
        row = [row_name]
        for ds in ALL_14_DATASETS:
            val = sub_pivot.loc[m, ds]
            best, second = ranks[ds]
            s = f"{val:.2f}%"
            if abs(val - best) < 1e-5:
                row.append(f"**{s}**")
            elif second is not None and abs(val - second) < 1e-5:
                row.append(f"*{s}*")
            else:
                row.append(s)
        lines.append("| " + " | ".join(row) + " |")

    ts_vals = sub_pivot.loc["Temperature Scaling (TS)"]
    vcps5_vals = sub_pivot.loc["VCPS-5D (Our Method)"]
    vcps17_vals = sub_pivot.loc["VCPS-17D (Our Method)"]
    platt_vals = sub_pivot.loc["Platt Scaling (1D)"]

    family_ts_wins = [
        ds for ds in ALL_14_DATASETS if min(vcps5_vals[ds], vcps17_vals[ds]) < ts_vals[ds]
    ]
    family_ts_losses = [ds for ds in ALL_14_DATASETS if ds not in family_ts_wins]
    vcps17_ts_wins = [ds for ds in ALL_14_DATASETS if vcps17_vals[ds] < ts_vals[ds]]
    vcps5_ts_wins = [ds for ds in ALL_14_DATASETS if vcps5_vals[ds] < ts_vals[ds]]
    loss_formatted = ", ".join([f"`{d}`" for d in family_ts_losses])
    ts_loss_str = f" (all except {loss_formatted})" if family_ts_losses else ""

    family_platt_wins = [
        ds for ds in ALL_14_DATASETS if min(vcps5_vals[ds], vcps17_vals[ds]) < platt_vals[ds]
    ]
    vcps17_platt_wins = [ds for ds in ALL_14_DATASETS if vcps17_vals[ds] < platt_vals[ds]]
    vcps5_platt_wins = [ds for ds in ALL_14_DATASETS if vcps5_vals[ds] < platt_vals[ds]]
    platt_formatted = ", ".join([f"`{d}`" for d in family_platt_wins])

    traj_methods = ["VCPS-5D (Our Method)", "VCPS-17D (Our Method)", "Residual Calibrator"]
    num_1_wins = []
    for ds in ALL_14_DATASETS:
        best_val, _ = ranks[ds]
        for tm in traj_methods:
            if abs(sub_pivot.loc[tm, ds] - best_val) < 1e-5:
                short_m = "VCPS-17D" if "17D" in tm else ("VCPS-5D" if "5D" in tm else "Residual")
                num_1_wins.append(f"`{ds}` ({short_m}: **{best_val:.2f}%**)")
                break

    lines.append("")
    lines.append("- **VCPS vs. Global Temperature Scaling (TS)**:")
    lines.append(
        f"  - **Family Win**: VCPS beats TS on **{len(family_ts_wins)} / 14 datasets**{ts_loss_str}."
    )
    lines.append(
        f"  - **Separate Counts**: VCPS-17D alone beats TS on **{len(vcps17_ts_wins)} / 14 datasets**; VCPS-5D alone beats TS on **{len(vcps5_ts_wins)} / 14 datasets**."
    )
    lines.append("- **VCPS vs. 1D Platt Scaling**:")
    lines.append(
        f"  - **Family Win**: VCPS beats 1D Platt Scaling on **{len(family_platt_wins)} / 14 datasets**:"
    )
    lines.append(f"    {platt_formatted}.")
    lines.append(
        f"  - **Separate Counts**: VCPS-17D alone beats 1D Platt on **{len(vcps17_platt_wins)} / 14 datasets**; VCPS-5D alone beats 1D Platt on **{len(vcps5_platt_wins)} / 14 datasets**."
    )
    lines.append(
        f"- **Our Trajectory Methods (VCPS-5D, VCPS-17D, Residual Calibrator)** win the #1 lowest Adaptive ECE on **{len(num_1_wins)} / 14 benchmarks**:"
    )
    lines.append(f"  {', '.join(num_1_wins)}.")

    return "\n".join(lines)


def generate_macro_table() -> str:
    lines = [
        "### Macro-Average Comparison Across 17 Methods on 14 Datasets ($T_{\\text{gen}} = 0.00$, $1\\times$ Compute)\n",
        "| Model | Calibration Method | Paradigm / Regime | Sampling | Macro ECE (%) $\\downarrow$ | Macro Ada-ECE (%) $\\downarrow$ | Macro AUROC $\\uparrow$ |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
    ]

    methods_order = [
        ("Quadratic Platt (Logit-Only)", "Polynomial Logit", "Greedy ($T=0.0, K=1$)"),
        ("VCPS-17D (Our Method)", "Trajectory Calibration", "Greedy ($T=0.0, K=1$)"),
        ("Spline Calibration", "Non-Parametric (Isotonic)", "Greedy ($T=0.0, K=1$)"),
        ("VCPS-5D (Our Method)", "Trajectory Calibration", "Greedy ($T=0.0, K=1$)"),
        ("Trajectory LR", "Linear Trajectory", "Greedy ($T=0.0, K=1$)"),
        ("Platt Scaling (1D)", "Classic Linear Post-Hoc", "Greedy ($T=0.0, K=1$)"),
        ("Residual Calibrator", "Feature-Aided", "Greedy ($T=0.0, K=1$)"),
        ("Trajectory LR (No Bias)", "Linear Trajectory (Zero-Bias)", "Greedy ($T=0.0, K=1$)"),
        ("Trajectory Platt (5D)", "Platt on 5D features", "Greedy ($T=0.0, K=1$)"),
        ("Trajectory Platt (17D)", "Platt on 17D features", "Greedy ($T=0.0, K=1$)"),
        ("Temperature Scaling (TS)", "Classic Post-Hoc", "Greedy ($T=0.0, K=1$)"),
        ("Adaptive TS (ATS)", "Adaptive Calibrator", "Greedy ($T=0.0, K=1$)"),
        ("Naive Confidence (NC)", "Uncalibrated Baseline", "Greedy ($T=0.0, K=1$)"),
        ("umpire", "Multi-Pass Semantic Volume", "Stochastic ($T=0.5, K=10$)"),
        ("eigen_score", "SVD Covariance Dispersion", "Stochastic ($T=0.5, K=10$)"),
        ("ln_entropy", "Predictive Entropy", "Stochastic ($T=0.5, K=10$)"),
        ("semantic_entropy", "DeBERTa NLI Clustering", "Stochastic ($T=0.5, K=10$)"),
    ]

    for arch, arch_label in [("m3", "M3-LLaVA"), ("mqt", "MQT-LLaVA")]:
        bench_df = pd.read_csv(ROOT / f"results/experiments/benchmark/benchmark_{arch}_summary.csv")
        ump_df = pd.read_csv(ROOT / f"results/umpire_eval/{arch}_llava_cumulative_summary.csv")

        rows = []
        for m, paradigm, sampling in methods_order:
            if m in ["umpire", "eigen_score", "ln_entropy", "semantic_entropy"]:
                sub = ump_df[ump_df["method"] == m]
                if sub.empty:
                    continue
                ece = float(sub["cece"].mean() * 100.0)
                ada_ece = None
                auroc = float(sub["auc"].mean())
            else:
                sub = bench_df[bench_df["method"] == m]
                if sub.empty:
                    continue
                ece = float(sub["ece_percent"].mean())
                ada_ece = float(sub["adaptive_ece_percent"].mean())
                auroc = float(sub["auroc"].mean())
            rows.append(
                {
                    "method": m,
                    "paradigm": paradigm,
                    "sampling": sampling,
                    "ece": ece,
                    "ada_ece": ada_ece,
                    "auroc": auroc,
                }
            )

        m_df = pd.DataFrame(rows)
        ece_sorted = sorted(m_df["ece"].dropna().unique())
        ada_sorted = sorted(m_df["ada_ece"].dropna().unique())
        auc_sorted = sorted(m_df["auroc"].dropna().unique(), reverse=True)

        b_ece, s_ece = ece_sorted[0], ece_sorted[1]
        b_ada, s_ada = ada_sorted[0], ada_sorted[1]
        b_auc, s_auc = auc_sorted[0], auc_sorted[1]

        for idx, r in m_df.iterrows():
            ece_s = f"{r['ece']:.2f}%"
            if abs(r["ece"] - b_ece) < 1e-4:
                ece_f = f"**{ece_s}**"
            elif abs(r["ece"] - s_ece) < 1e-4:
                ece_f = f"*{ece_s}*"
            else:
                ece_f = ece_s

            if r["ada_ece"] is not None and not np.isnan(r["ada_ece"]):
                ada_s = f"{r['ada_ece']:.2f}%"
                if abs(r["ada_ece"] - b_ada) < 1e-4:
                    ada_f = f"**{ada_s}**"
                elif abs(r["ada_ece"] - s_ada) < 1e-4:
                    ada_f = f"*{ada_s}*"
                else:
                    ada_f = ada_s
            else:
                ada_f = "-"

            auc_s = f"{r['auroc']:.3f}"
            if abs(r["auroc"] - b_auc) < 1e-4:
                auc_f = f"**{auc_s}**"
            elif abs(r["auroc"] - s_auc) < 1e-4:
                auc_f = f"*{auc_s}*"
            else:
                auc_f = auc_s

            is_ump = r["method"] in ["umpire", "eigen_score", "ln_entropy", "semantic_entropy"]
            m_name = (
                f"`{r['method']}`"
                if is_ump
                else (f"**{r['method']}**" if "VCPS" in r["method"] else r["method"])
            )
            model_col = f"**{arch_label}**" if idx == 0 else ""
            lines.append(
                f"| {model_col} | {m_name} | {r['paradigm']} | {r['sampling']} | {ece_f} | {ada_f} | {auc_f} |"
            )

    return "\n".join(lines)


def generate_temp_table() -> str:
    lines = [
        "#### Macro Transfer ECE (%) Across Temperatures (Trained at $T=0.0$)\n",
        "| Model | Calibration Method | $T=0.0$ | $T=0.3$ | $T=0.6$ | $T=1.0$ | $T=1.5$ | Mean ECE $\\downarrow$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    methods = [
        "Spline Calibration (PCHIP)",
        "Quadratic Platt (Logit-Only)",
        "Platt Scaling (1D)",
        "Residual Calibrator",
        "VCPS-5D (Our Method)",
        "VCPS-17D (Our Method)",
        "Trajectory LR (No Bias)",
        "Trajectory LR",
        "Trajectory Platt (5D)",
        "Trajectory Platt (17D)",
        "Temperature Scaling (TS)",
        "Adaptive TS (ATS)",
        "Naive Confidence (NC)",
    ]

    for arch, arch_label in [("m3", "M3-LLaVA"), ("mqt", "MQT-LLaVA")]:
        t_df = pd.read_csv(
            ROOT / f"results/experiments/temperature_study/temperature_transfer_{arch}_summary.csv"
        )
        t_mean = t_df.groupby(["method", "temperature"])["ece_percent"].mean().unstack()
        t_sub = t_mean.loc[[m for m in methods if m in t_mean.index], [0.0, 0.3, 0.6, 1.0, 1.5]]
        t_sub["mean_ece"] = t_sub.mean(axis=1)
        t_sub = t_sub.sort_values("mean_ece")

        bests = {}
        seconds = {}
        for col in [0.0, 0.3, 0.6, 1.0, 1.5, "mean_ece"]:
            vals = sorted(t_sub[col].unique())
            bests[col] = vals[0]
            seconds[col] = vals[1] if len(vals) > 1 else None

        for idx, (m, row) in enumerate(t_sub.iterrows()):
            cells = []
            for col in [0.0, 0.3, 0.6, 1.0, 1.5, "mean_ece"]:
                v = row[col]
                s = f"{v:.2f}%"
                if abs(v - bests[col]) < 1e-4:
                    cells.append(f"**{s}**")
                elif seconds[col] is not None and abs(v - seconds[col]) < 1e-4:
                    cells.append(f"*{s}*")
                else:
                    cells.append(s)
            m_name = f"**{m}**" if "VCPS" in m else m
            model_col = f"**{arch_label}**" if idx == 0 else ""
            lines.append(f"| {model_col} | {m_name} | {' | '.join(cells)} |")

    return "\n".join(lines)


def main():
    print(f"Reading {README_PATH}...")
    readme_text = README_PATH.read_text(encoding="utf-8")

    # 1. Update M3 Table & Bullets
    m3_section = generate_ada_ece_table_and_bullets("m3", "M3-LLaVA")
    m3_pattern = re.compile(
        r"###### M3-LLaVA: Adaptive ECE \(%\).*?(?=\n---\n\n### MQT-LLaVA)", re.DOTALL
    )
    if not m3_pattern.search(readme_text):
        raise RuntimeError("Could not find M3-LLaVA Adaptive ECE section in README.md")
    readme_text = m3_pattern.sub(lambda m: m3_section, readme_text)
    print("[PASS] Generated and formatted M3-LLaVA Ada-ECE table and analytical bullets.")

    # 2. Update MQT Table & Bullets
    mqt_section = generate_ada_ece_table_and_bullets("mqt", "MQT-LLaVA")
    mqt_section = mqt_section.replace("###### MQT-LLaVA:", "### MQT-LLaVA:")
    mqt_pattern = re.compile(
        r"### MQT-LLaVA: Adaptive ECE \(%\).*?(?=\n---\n\n## Benchmark Logs)", re.DOTALL
    )
    if not mqt_pattern.search(readme_text):
        raise RuntimeError("Could not find MQT-LLaVA Adaptive ECE section in README.md")
    readme_text = mqt_pattern.sub(lambda m: mqt_section, readme_text)
    print("[PASS] Generated and formatted MQT-LLaVA Ada-ECE table and analytical bullets.")

    # 3. Update Macro-Average Comparison Table
    macro_section = generate_macro_table()
    macro_pattern = re.compile(
        r"### Macro-Average Comparison.*?(?=\n---\n\n### Decoding Temperature)", re.DOTALL
    )
    if not macro_pattern.search(readme_text):
        raise RuntimeError("Could not find Macro-Average Comparison section in README.md")
    readme_text = macro_pattern.sub(lambda m: macro_section, readme_text)
    print("[PASS] Generated and formatted Macro-Average Comparison table.")

    # 4. Update Temperature Transfer Table
    temp_section = generate_temp_table()
    temp_pattern = re.compile(
        r"#### Macro Transfer ECE \(%\) Across Temperatures.*?(?=\n---\n\n### Leave-One-Dataset-Out)",
        re.DOTALL,
    )
    if not temp_pattern.search(readme_text):
        raise RuntimeError("Could not find Temperature Transfer section in README.md")
    readme_text = temp_pattern.sub(lambda m: temp_section, readme_text)
    print("[PASS] Generated and formatted Temperature Transfer table.")

    # Write back to README.md
    README_PATH.write_text(readme_text, encoding="utf-8")
    print(f"[SUCCESS] Updated {README_PATH} successfully!")


if __name__ == "__main__":
    main()
