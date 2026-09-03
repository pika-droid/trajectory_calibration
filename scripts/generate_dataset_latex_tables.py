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
TEMP_DIR = TABLES_DIR / "temp_ablation"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Dataset key mapping to filename
DATASET_FILE_MAP = {
    "ai2d": "ai2d.tex",
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
    "vqav2": "vqav2.tex",
}

DATASET_NAME_MAP = {
    "lego-puzzles": "legopuzzles",
    "vizwiz-vqa": "vizwizvqa",
    "vqav2_5scale": "vqav2",
}

TARGET_METHODS_ORDER = [
    ("Naive Confidence (NC)", "Single-Pass ($T = 0.0$)", False),
    ("Temperature Scaling (TS)", "Single-Pass ($T = 0.0$)", False),
    ("Platt Scaling (1D)", "Single-Pass ($T = 0.0$)", False),
    ("Trajectory LR (No Bias)", "Single-Pass ($T = 0.0$)", False),
    ("Quadratic Platt (Logit-Only)", "Single-Pass ($T = 0.0$)", False),
    ("Spline Calibration", "Single-Pass ($T = 0.0$)", False),
    ("Adaptive TS (ATS)", "Single-Pass ($T = 0.0$)", False),
    ("ln_entropy", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("semantic_entropy", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("eigen_score", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("umpire", "Multi-Pass ($T = 0.5, K = 10$)", False),
    ("Residual Calibrator", "Single-Pass ($T = 0.0$)", False),
    ("VCPS-5D (Our Method)", "Single-Pass ($T = 0.0$)", True),
    ("VCPS-17D (Our Method)", "Single-Pass ($T = 0.0$)", True),
]

UMP_DISPLAY_NAMES = {
    "ln_entropy": "LN-Entropy",
    "semantic_entropy": "Semantic Entropy",
    "eigen_score": "EigenScore",
    "umpire": "UMPIRE",
}


def rank_and_format(vals: list[float | None], higher_is_better: bool = False, decimals: int = 2) -> list[str]:
    valid_vals = [v for v in vals if v is not None and not np.isnan(v)]
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
        s = f"{v:.{decimals}f}"
        if abs(v - best_val) < 1e-6:
            formatted.append(f"\\textbf{{{s}}}")
        elif second_val is not None and abs(v - second_val) < 1e-6:
            formatted.append(f"\\underline{{{s}}}")
        else:
            formatted.append(s)
    return formatted


def generate_single_table(df_bench: pd.DataFrame, df_ump: pd.DataFrame, ds: str, arch_label: str, arch_model: str, is_macro: bool = False) -> str:
    caption_name = f"on \\texttt{{{ds}}}" if not is_macro else "Across All 14 Benchmarks"
    tab_label = f"tab:benchmark_{arch_label}_{DATASET_NAME_MAP.get(ds, ds)}" if not is_macro else f"tab:benchmark_macro_{arch_label}"
    macro_prefix = "Macro-Averaged " if is_macro else ""

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{\\textbf{{{macro_prefix}Calibration Benchmark {caption_name} ({arch_model} 7B).}} \\textbf{{Bold}}: best; \\underline{{underline}}: second best. $\\downarrow$/$\\uparrow$: lower/higher is better.}}",
        f"\\label{{{tab_label}}}",
        "\\tablestyle{4pt}{1.05}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f"\\textbf{{Calibration Method}} & \\textbf{{Regime / Sampling}} & \\textbf{{{macro_prefix}ECE (\\%)}} $\\downarrow$ & \\textbf{{{macro_prefix}Ada-ECE (\\%)}} $\\downarrow$ & \\textbf{{{macro_prefix}AUROC}} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    rows = []
    for m_key, regime, is_vcps in TARGET_METHODS_ORDER:
        disp_name = UMP_DISPLAY_NAMES.get(m_key, m_key)
        if m_key in UMP_DISPLAY_NAMES:
            if is_macro:
                sub = df_ump[df_ump["method"] == m_key]
                if not sub.empty:
                    ece = float(sub["cece"].mean() * 100.0)
                    ada_ece = None
                    auroc = float(sub["auc"].mean())
                    rows.append((disp_name, regime, is_vcps, ece, ada_ece, auroc))
            else:
                sub = df_ump[(df_ump["dataset"] == ds) & (df_ump["method"] == m_key)]
                if not sub.empty:
                    r = sub.iloc[0]
                    rows.append((disp_name, regime, is_vcps, float(r["cece"] * 100.0), None, float(r["auc"])))
        else:
            if is_macro:
                sub = df_bench[df_bench["method"] == m_key]
                if not sub.empty:
                    ece = float(sub["ece_percent"].mean())
                    ada_ece = float(sub["adaptive_ece_percent"].mean())
                    auroc = float(sub["auroc"].mean())
                    rows.append((disp_name, regime, is_vcps, ece, ada_ece, auroc))
            else:
                sub = df_bench[(df_bench["dataset"] == ds) & (df_bench["method"] == m_key)]
                if not sub.empty:
                    r = sub.iloc[0]
                    rows.append((disp_name, regime, is_vcps, float(r["ece_percent"]), float(r["adaptive_ece_percent"]), float(r["auroc"])))

    if not rows:
        return ""

    ece_formatted = rank_and_format([r[3] for r in rows], higher_is_better=False, decimals=2)
    ada_ece_formatted = rank_and_format([r[4] for r in rows], higher_is_better=False, decimals=2)
    auroc_formatted = rank_and_format([r[5] for r in rows], higher_is_better=True, decimals=3)

    for i, (disp_name, regime, is_vcps, _, _, _) in enumerate(rows):
        ece_str = ece_formatted[i]
        ada_str = ada_ece_formatted[i]
        auc_str = auroc_formatted[i]
        if is_vcps:
            row_str = f"\\rowcolor{{gray!10}} \\textbf{{{disp_name}}} & {regime} & {ece_str} & {ada_str} & {auc_str} \\\\"
        else:
            row_str = f"{disp_name} & {regime} & {ece_str} & {ada_str} & {auc_str} \\\\"
        lines.append(row_str)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def generate_benchmark_tables():
    df_m3 = pd.read_csv(ROOT / "results/experiments/benchmark/benchmark_m3_summary.csv")
    df_mqt = pd.read_csv(ROOT / "results/experiments/benchmark/benchmark_mqt_summary.csv")
    df_m3["dataset"] = df_m3["dataset"].replace({"vqav2_5scale": "vqav2"})
    df_mqt["dataset"] = df_mqt["dataset"].replace({"vqav2_5scale": "vqav2"})

    ump_m3 = pd.read_csv(ROOT / "results/umpire_eval/m3_llava_cumulative_summary.csv")
    ump_mqt = pd.read_csv(ROOT / "results/umpire_eval/mqt_llava_cumulative_summary.csv")

    # 1. Per-dataset tables
    for ds_key, filename in DATASET_FILE_MAP.items():
        t_m3 = generate_single_table(df_m3, ump_m3, ds_key, "m3", "M3-LLaVA", is_macro=False)
        t_mqt = generate_single_table(df_mqt, ump_mqt, ds_key, "mqt", "MQT-LLaVA", is_macro=False)
        content = t_m3 + "\n\n" + t_mqt + "\n"
        out_file = TABLES_DIR / filename
        out_file.write_text(content, encoding="utf-8")
        print(f"Generated: {out_file}")

    # 2. Macro mean table
    macro_m3 = generate_single_table(df_m3, ump_m3, "macro_mean", "macro_m3", "M3-LLaVA", is_macro=True)
    macro_mqt = generate_single_table(df_mqt, ump_mqt, "macro_mean", "macro_mqt", "MQT-LLaVA", is_macro=True)
    (TABLES_DIR / "macro_mean.tex").write_text(macro_m3 + "\n\n" + macro_mqt + "\n", encoding="utf-8")
    print(f"Generated: {TABLES_DIR / 'macro_mean.tex'}")


def generate_temperature_tables():
    df_m3 = pd.read_csv(ROOT / "results/experiments/temperature_study/temperature_transfer_m3_summary.csv")
    df_mqt = pd.read_csv(ROOT / "results/experiments/temperature_study/temperature_transfer_mqt_summary.csv")
    df_m3["dataset"] = df_m3["dataset"].replace({"vqav2_5scale": "vqav2"})
    df_mqt["dataset"] = df_mqt["dataset"].replace({"vqav2_5scale": "vqav2"})

    temp_methods = [
        "Naive Confidence (NC)",
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "Trajectory LR (No Bias)",
        "Quadratic Platt (Logit-Only)",
        "Spline Calibration (PCHIP)",
        "Adaptive TS (ATS)",
        "Residual Calibrator",
        "VCPS-5D (Our Method)",
        "VCPS-17D (Our Method)",
    ]

    target_temps = [0.0, 0.3, 0.6, 1.0, 1.5]
    temp_datasets = ["pope", "scienceqa", "textvqa", "vizwiz-vqa"]

    def build_temp_table(df: pd.DataFrame, ds_name: str | None, arch_label: str, arch_model: str, is_macro: bool = False) -> str:
        caption_name = f"\\texttt{{{ds_name}}}" if not is_macro else "Macro-Averaged Across 4 Benchmarks"
        tab_label = f"tab:temp_transfer_{arch_label}_{'macro' if is_macro else ds_name.replace('-', '')}"

        lines = [
            "\\begin{table}[t]",
            f"\\caption{{\\textbf{{Temperature Transfer Robustness on {caption_name} ({arch_model} 7B).}} ECE (\\%) $\\downarrow$ across sampling temperatures $T \\in \\{{0.0, 0.3, 0.6, 1.0, 1.5\\}}$ (trained at $T=0.0$). \\textbf{{Bold}}: best; \\underline{{underline}}: second best.}}",
            f"\\label{{{tab_label}}}",
            "\\tablestyle{4pt}{1.05}",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "\\textbf{Calibration Method} & \\textbf{$T=0.0$} & \\textbf{$T=0.3$} & \\textbf{$T=0.6$} & \\textbf{$T=1.0$} & \\textbf{$T=1.5$} & \\textbf{Mean} $\\downarrow$ \\\\",
            "\\midrule",
        ]

        sub = df if is_macro else df[df["dataset"] == ds_name]
        if is_macro:
            sub = sub[sub["dataset"].isin(temp_datasets)]

        rows_data = []
        for m in temp_methods:
            m_sub = sub[sub["method"] == m]
            if m_sub.empty:
                continue
            vals = []
            for t in target_temps:
                t_sub = m_sub[m_sub["temperature"] == t]
                val = t_sub["ece_percent"].mean() if not t_sub.empty else np.nan
                vals.append(val)
            mean_val = np.nanmean(vals)
            rows_data.append((m, vals, mean_val))

        col_formatted = []
        for col_idx in range(len(target_temps)):
            col_vals = [r[1][col_idx] for r in rows_data]
            col_formatted.append(rank_and_format(col_vals, higher_is_better=False, decimals=2))

        mean_formatted = rank_and_format([r[2] for r in rows_data], higher_is_better=False, decimals=2)

        for row_idx, (m, _, _) in enumerate(rows_data):
            c_strs = [col_formatted[col_idx][row_idx] for col_idx in range(len(target_temps))]
            m_str = mean_formatted[row_idx]
            is_vcps = "VCPS" in m
            if is_vcps:
                lines.append(f"\\rowcolor{{gray!10}} \\textbf{{{m}}} & {' & '.join(c_strs)} & {m_str} \\\\")
            else:
                lines.append(f"{m} & {' & '.join(c_strs)} & {m_str} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ])
        return "\n".join(lines)

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


def generate_lodo_tables():
    df_m3 = pd.read_csv(ROOT / "results/experiments/lodo/lodo_m3_summary.csv")
    df_mqt = pd.read_csv(ROOT / "results/experiments/lodo/lodo_mqt_summary.csv")

    def build_lodo_table(df: pd.DataFrame, arch_label: str, arch_model: str) -> str:
        lines = [
            "\\begin{table}[t]",
            f"\\caption{{\\textbf{{Leave-One-Dataset-Out (LODO) Cross-Domain Transfer ({arch_model} 7B).}} Macro-averaged across all 14 held-out target benchmarks. Trained on pooled 13 benchmarks. \\textbf{{Bold}}: best; \\underline{{underline}}: second best.}}",
            f"\\label{{tab:lodo_transfer_{arch_label}}}",
            "\\tablestyle{4pt}{1.05}",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "\\textbf{Calibration Method} & \\textbf{Transfer Protocol} & \\textbf{Macro ECE (\\%)} $\\downarrow$ & \\textbf{Macro Ada-ECE (\\%)} $\\downarrow$ & \\textbf{Macro AUROC} $\\uparrow$ \\\\",
            "\\midrule",
        ]

        methods_order = [
            "Platt Scaling (1D)",
            "Best 5D Trajectory",
            "Two-Stage Residual",
            "VCPS-5D (Our Method)",
            "VCPS-17D (Our Method)",
        ]

        # Aggregate across all 14 held-out datasets
        agg = df.groupby(["method", "transfer_mode"])[["ece_percent", "adaptive_ece_percent", "auroc"]].mean().reset_index()

        rows = []
        for m in methods_order:
            for mode in ["Zero-Shot Base", "Target Adapted (Saerens-EM)"]:
                sub = agg[(agg["method"] == m) & (agg["transfer_mode"] == mode)]
                if not sub.empty:
                    r = sub.iloc[0]
                    rows.append((m, mode, float(r["ece_percent"]), float(r["adaptive_ece_percent"]), float(r["auroc"]), "VCPS" in m))

        ece_formatted = rank_and_format([r[2] for r in rows], higher_is_better=False, decimals=2)
        ada_formatted = rank_and_format([r[3] for r in rows], higher_is_better=False, decimals=2)
        auc_formatted = rank_and_format([r[4] for r in rows], higher_is_better=True, decimals=3)

        for i, (m, mode, _, _, _, is_vcps) in enumerate(rows):
            ece_str = ece_formatted[i]
            ada_str = ada_formatted[i]
            auc_str = auc_formatted[i]
            if is_vcps:
                lines.append(f"\\rowcolor{{gray!10}} \\textbf{{{m}}} & {mode} & {ece_str} & {ada_str} & {auc_str} \\\\")
            else:
                lines.append(f"{m} & {mode} & {ece_str} & {ada_str} & {auc_str} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ])
        return "\n".join(lines)

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
