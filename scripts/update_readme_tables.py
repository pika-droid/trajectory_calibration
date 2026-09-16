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
- Win counts: Trajectory Platt (5D) vs. baselines (NC, TS, 1D Platt) across Core 6 datasets.
Updates README.md cleanly and safely.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"

CORE_7_DATASETS = [
    "ai2d",
    "chartqa",
    "docvqa",
    "scienceqa",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
]

CORE_METHODS = [
    "Naive Confidence (NC)",
    "Temperature Scaling (TS)",
    "Platt Scaling (1D)",
    "Trajectory Platt (5D)",
]


def generate_ada_ece_table_and_bullets(arch: str, arch_label: str) -> str:
    """Generates the Core 7 Adaptive ECE breakdown table and analytical bullets."""
    csv_path = ROOT / f"results/experiments/benchmark/benchmark_{arch}_summary.csv"
    df = pd.read_csv(csv_path)
    df["dataset"] = df["dataset"].replace({"vqav2_5scale": "vqav2"})
    sub_df = df[(df["dataset"].isin(CORE_7_DATASETS)) & (df["method"].isin(CORE_METHODS))]

    pivot = sub_df.pivot(index="method", columns="dataset", values="adaptive_ece_percent")
    pivot = pivot.reindex(CORE_METHODS)
    pivot["Macro Mean"] = pivot[CORE_7_DATASETS].mean(axis=1)

    all_cols = CORE_7_DATASETS + ["Macro Mean"]
    ranks: dict[str, tuple[float, float | None]] = {}
    for col in all_cols:
        vals = sorted(pivot[col].unique())
        best = vals[0]
        second = vals[1] if len(vals) > 1 else None
        ranks[col] = (best, second)

    lines = []
    lines.append(
        f"###### {arch_label}: Adaptive ECE (%) Across Core 7 Datasets [Lower is Better]\n"
    )
    lines.append("| Calibration Method | " + " | ".join(all_cols) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * len(all_cols)) + " |")

    for m in CORE_METHODS:
        row_name = f"**{m} [Our Method]**" if "5D" in m else m
        row = [row_name]
        for col in all_cols:
            val = float(pivot.loc[m, col])
            best, second = ranks[col]
            s = f"{val:.2f}%"
            if abs(val - best) < 1e-4:
                row.append(f"**{s}**")
            elif second is not None and abs(val - second) < 1e-4:
                row.append(f"*{s}*")
            else:
                row.append(s)
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(format_analytical_bullets(pivot, ranks))
    return "\n".join(lines)


def format_analytical_bullets(
    pivot: pd.DataFrame, ranks: dict[str, tuple[float, float | None]]
) -> list[str]:
    """Formats analytical win bullets for Trajectory Platt (5D) vs Baselines."""
    lines: list[str] = []
    ts_vals = pivot.loc["Temperature Scaling (TS)"]
    tp5_vals = pivot.loc["Trajectory Platt (5D)"]
    platt_vals = pivot.loc["Platt Scaling (1D)"]

    # 1. TP-5D vs TS
    tp5_ts_wins = [ds for ds in CORE_7_DATASETS if tp5_vals[ds] < ts_vals[ds]]
    tp5_ts_losses = [ds for ds in CORE_7_DATASETS if ds not in tp5_ts_wins]
    ts_loss_str = (
        f" (all except {', '.join([f'`{d}`' for d in tp5_ts_losses])})" if tp5_ts_losses else ""
    )
    lines.append("")
    lines.append("- **Trajectory Platt (5D) vs. Global Temperature Scaling (TS)**:")
    lines.append(
        f"  - Trajectory Platt (5D) beats TS on **{len(tp5_ts_wins)} / 7 Core datasets**{ts_loss_str}."
    )

    # 2. TP-5D vs 1D Platt
    tp5_platt_wins = [ds for ds in CORE_7_DATASETS if tp5_vals[ds] < platt_vals[ds]]
    platt_formatted = ", ".join([f"`{d}`" for d in tp5_platt_wins])
    lines.append("- **Trajectory Platt (5D) vs. 1D Platt Scaling**:")
    lines.append(
        f"  - Trajectory Platt (5D) beats 1D Platt Scaling on **{len(tp5_platt_wins)} / 7 Core datasets**: {platt_formatted}."
    )

    # 3. Top-1 lowest Adaptive ECE
    num_1_wins = []
    for ds in CORE_7_DATASETS:
        best_val, _ = ranks[ds]
        if abs(tp5_vals[ds] - best_val) < 1e-4:
            num_1_wins.append(f"`{ds}` (**{best_val:.2f}%**)")

    lines.append(
        f"- **Trajectory Platt (5D)** achieves the #1 lowest Adaptive ECE on **{len(num_1_wins)} / 7 Core benchmarks**: {', '.join(num_1_wins)}."
    )
    return lines


def generate_macro_table() -> str:
    """Generates the standardized Macro-Average summary table across Core 7 datasets."""
    lines = [
        r"### Macro-Average Calibration Benchmark Across Core 7 Datasets ($T_{\text{gen}} = 0.00$, $1\times$ Compute)"
        + "\n",
        r"| Model | Calibration Method | Regime / Sampling | Macro ECE (%) $\downarrow$ | Macro Ada-ECE (%) $\downarrow$ | Macro Brier $\downarrow$ | Macro AUROC $\uparrow$ |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for arch, arch_label in [("m3", "M3-LLaVA"), ("mqt", "MQT-LLaVA")]:
        bench_df = pd.read_csv(ROOT / f"results/experiments/benchmark/benchmark_{arch}_summary.csv")
        bench_df["dataset"] = bench_df["dataset"].replace({"vqav2_5scale": "vqav2"})
        sub_df = bench_df[
            (bench_df["dataset"].isin(CORE_7_DATASETS)) & (bench_df["method"].isin(CORE_METHODS))
        ]

        rows = []
        for m in CORE_METHODS:
            sub = sub_df[sub_df["method"] == m]
            if sub.empty:
                continue
            ece = float(sub["ece_percent"].mean())
            ada_ece = float(sub["adaptive_ece_percent"].mean())
            brier = float(sub["brier"].mean())
            auroc = float(sub["auroc"].mean())
            rows.append(
                {
                    "method": m,
                    "regime": "Single-Pass ($T = 0.0$)",
                    "ece": ece,
                    "ada_ece": ada_ece,
                    "brier": brier,
                    "auroc": auroc,
                }
            )

        m_df = pd.DataFrame(rows)
        ece_sorted = sorted(m_df["ece"].dropna().unique())
        ada_sorted = sorted(m_df["ada_ece"].dropna().unique())
        brier_sorted = sorted(m_df["brier"].dropna().unique())
        auc_sorted = sorted(m_df["auroc"].dropna().unique(), reverse=True)

        b_ece, s_ece = ece_sorted[0], ece_sorted[1] if len(ece_sorted) > 1 else None
        b_ada, s_ada = ada_sorted[0], ada_sorted[1] if len(ada_sorted) > 1 else None
        b_brier, s_brier = brier_sorted[0], brier_sorted[1] if len(brier_sorted) > 1 else None
        b_auc, s_auc = auc_sorted[0], auc_sorted[1] if len(auc_sorted) > 1 else None

        for idx, r in m_df.iterrows():
            ece_s = f"{r['ece']:.2f}%"
            if abs(r["ece"] - b_ece) < 1e-4:
                ece_f = f"**{ece_s}**"
            elif s_ece is not None and abs(r["ece"] - s_ece) < 1e-4:
                ece_f = f"*{ece_s}*"
            else:
                ece_f = ece_s

            ada_s = f"{r['ada_ece']:.2f}%"
            if abs(r["ada_ece"] - b_ada) < 1e-4:
                ada_f = f"**{ada_s}**"
            elif s_ada is not None and abs(r["ada_ece"] - s_ada) < 1e-4:
                ada_f = f"*{ada_s}*"
            else:
                ada_f = ada_s

            brier_s = f"{r['brier']:.4f}"
            if abs(r["brier"] - b_brier) < 1e-4:
                brier_f = f"**{brier_s}**"
            elif s_brier is not None and abs(r["brier"] - s_brier) < 1e-4:
                brier_f = f"*{brier_s}*"
            else:
                brier_f = brier_s

            auc_s = f"{r['auroc']:.3f}"
            if abs(r["auroc"] - b_auc) < 1e-4:
                auc_f = f"**{auc_s}**"
            elif s_auc is not None and abs(r["auroc"] - s_auc) < 1e-4:
                auc_f = f"*{auc_s}*"
            else:
                auc_f = auc_s

            m_name = f"**{r['method']} [Our Method]**" if "5D" in r["method"] else r["method"]
            model_col = f"**{arch_label}**" if idx == 0 else ""
            lines.append(
                f"| {model_col} | {m_name} | {r['regime']} | {ece_f} | {ada_f} | {brier_f} | {auc_f} |"
            )

    return "\n".join(lines)


def generate_vqav2_table() -> str:
    """Generates dedicated VQAv2 single-pass (1x) vs multi-rollout (10x) comparison table."""
    lines = [
        r"### VQAv2 Benchmark: Single-Pass Trajectory Calibration ($1\times$) vs. Multi-Rollout UMPIRE Suite ($10\times$)"
        + "\n",
        r"| Model | Calibration Method | Paradigm / Regime | Sampling | ECE (%) $\downarrow$ | Ada-ECE (%) $\downarrow$ | Brier $\downarrow$ | AUROC $\uparrow$ |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    single_pass = [
        ("Naive Confidence (NC)", "Uncalibrated Baseline", "Single-Pass ($T=0.0, K=1$)"),
        ("Temperature Scaling (TS)", "Classic Post-Hoc", "Single-Pass ($T=0.0, K=1$)"),
        ("Platt Scaling (1D)", "Classic Linear Post-Hoc", "Single-Pass ($T=0.0, K=1$)"),
        ("Trajectory Platt (5D)", "Trajectory Calibration", "Single-Pass ($T=0.0, K=1$)"),
    ]

    multi_pass = [
        ("ln_entropy", "LN-Entropy", "Predictive Entropy", "Multi-Pass ($T=0.5, K=10$)"),
        (
            "semantic_entropy",
            "Semantic Entropy",
            "DeBERTa NLI Clustering",
            "Multi-Pass ($T=0.5, K=10$)",
        ),
        ("eigen_score", "EigenScore", "SVD Covariance Dispersion", "Multi-Pass ($T=0.5, K=10$)"),
        ("umpire", "UMPIRE", "Multi-Pass Semantic Volume", "Multi-Pass ($T=0.5, K=10$)"),
    ]

    for arch, arch_label in [("m3", "M3-LLaVA"), ("mqt", "MQT-LLaVA")]:
        bench_df = pd.read_csv(ROOT / f"results/experiments/benchmark/benchmark_{arch}_summary.csv")
        bench_df["dataset"] = bench_df["dataset"].replace({"vqav2_5scale": "vqav2"})
        ump_df = pd.read_csv(ROOT / f"results/umpire_eval/{arch}_llava_cumulative_summary.csv")

        rows = []
        for m, paradigm, sampling in single_pass:
            sub = bench_df[(bench_df["dataset"] == "vqav2") & (bench_df["method"] == m)]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(
                    {
                        "method": f"**{m} [Our Method]**" if "5D" in m else m,
                        "paradigm": paradigm,
                        "sampling": sampling,
                        "ece": float(r["ece_percent"]),
                        "ada_ece": float(r["adaptive_ece_percent"]),
                        "brier": float(r["brier"]),
                        "auroc": float(r["auroc"]),
                    }
                )

        for m_key, disp_name, paradigm, sampling in multi_pass:
            sub = ump_df[(ump_df["dataset"] == "vqav2") & (ump_df["method"] == m_key)]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(
                    {
                        "method": f"`{disp_name}`",
                        "paradigm": paradigm,
                        "sampling": sampling,
                        "ece": float(r["cece"] * 100.0),
                        "ada_ece": None,
                        "brier": None,
                        "auroc": float(r["auc"]),
                    }
                )

        m_df = pd.DataFrame(rows)
        ece_sorted = sorted(m_df["ece"].dropna().unique())
        ada_sorted = sorted(m_df["ada_ece"].dropna().unique())
        brier_sorted = sorted(m_df["brier"].dropna().unique())
        auc_sorted = sorted(m_df["auroc"].dropna().unique(), reverse=True)

        b_ece, s_ece = ece_sorted[0], ece_sorted[1] if len(ece_sorted) > 1 else None
        b_ada, s_ada = (
            (ada_sorted[0], ada_sorted[1])
            if len(ada_sorted) > 1
            else (ada_sorted[0], None)
            if ada_sorted
            else (None, None)
        )
        b_brier, s_brier = (
            (brier_sorted[0], brier_sorted[1])
            if len(brier_sorted) > 1
            else (brier_sorted[0], None)
            if brier_sorted
            else (None, None)
        )
        b_auc, s_auc = auc_sorted[0], auc_sorted[1] if len(auc_sorted) > 1 else None

        for idx, r in m_df.iterrows():
            ece_s = f"{r['ece']:.2f}%"
            if abs(r["ece"] - b_ece) < 1e-4:
                ece_f = f"**{ece_s}**"
            elif s_ece is not None and abs(r["ece"] - s_ece) < 1e-4:
                ece_f = f"*{ece_s}*"
            else:
                ece_f = ece_s

            if r["ada_ece"] is not None and not np.isnan(r["ada_ece"]):
                ada_s = f"{r['ada_ece']:.2f}%"
                if b_ada is not None and abs(r["ada_ece"] - b_ada) < 1e-4:
                    ada_f = f"**{ada_s}**"
                elif s_ada is not None and abs(r["ada_ece"] - s_ada) < 1e-4:
                    ada_f = f"*{ada_s}*"
                else:
                    ada_f = ada_s
            else:
                ada_f = "-"

            if r["brier"] is not None and not np.isnan(r["brier"]):
                brier_s = f"{r['brier']:.4f}"
                if b_brier is not None and abs(r["brier"] - b_brier) < 1e-4:
                    brier_f = f"**{brier_s}**"
                elif s_brier is not None and abs(r["brier"] - s_brier) < 1e-4:
                    brier_f = f"*{brier_s}*"
                else:
                    brier_f = brier_s
            else:
                brier_f = "-"

            auc_s = f"{r['auroc']:.3f}"
            if abs(r["auroc"] - b_auc) < 1e-4:
                auc_f = f"**{auc_s}**"
            elif s_auc is not None and abs(r["auroc"] - s_auc) < 1e-4:
                auc_f = f"*{auc_s}*"
            else:
                auc_f = auc_s

            model_col = f"**{arch_label}**" if idx == 0 else ""
            lines.append(
                f"| {model_col} | {r['method']} | {r['paradigm']} | {r['sampling']} | {ece_f} | {ada_f} | {brier_f} | {auc_f} |"
            )

    return "\n".join(lines)


def generate_temp_table() -> str:
    """Generates temperature transfer table across the 4 core calibration methods."""
    lines = [
        "#### Macro Transfer ECE (%) Across Temperatures (Trained at $T=0.0$)\n",
        "| Model | Calibration Method | $T=0.0$ | $T=0.3$ | $T=0.6$ | $T=1.0$ | $T=1.5$ | Mean ECE $\\downarrow$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for arch, arch_label in [("m3", "M3-LLaVA"), ("mqt", "MQT-LLaVA")]:
        t_df = pd.read_csv(
            ROOT / f"results/experiments/temperature_study/temperature_transfer_{arch}_summary.csv"
        )
        t_mean = t_df.pivot_table(
            index="method", columns="temperature", values="ece_percent", aggfunc="mean"
        )
        t_sub = t_mean.loc[
            [m for m in CORE_METHODS if m in t_mean.index], [0.0, 0.3, 0.6, 1.0, 1.5]
        ]
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
                v = float(row[col])
                s = f"{v:.2f}%"
                if abs(v - bests[col]) < 1e-4:
                    cells.append(f"**{s}**")
                elif seconds[col] is not None and abs(v - seconds[col]) < 1e-4:
                    cells.append(f"*{s}*")
                else:
                    cells.append(s)
            m_name = f"**{m} [Our Method]**" if "5D" in str(m) else str(m)
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

    # 3. Update Macro-Average Comparison Table & VQAv2 Dedicated Table
    macro_section = generate_macro_table() + "\n\n" + generate_vqav2_table()
    macro_pattern = re.compile(
        r"### Macro-Average.*?(?=\n---\n\n### Decoding Temperature)", re.DOTALL
    )
    if not macro_pattern.search(readme_text):
        raise RuntimeError("Could not find Macro-Average Comparison section in README.md")
    readme_text = macro_pattern.sub(lambda m: macro_section, readme_text)
    print("[PASS] Generated and formatted Macro-Average & VQAv2 Comparison tables.")

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
