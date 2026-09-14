#!/usr/bin/env python3
"""Generate Pairwise Zero-Shot Cross-Domain Transfer Analysis (Workbook 3).

Evaluates the pairwise zero-shot cross-domain transfer matrix across the 13
qualifying benchmarks for both M3-LLaVA and MQT-LLaVA without target supervision.
Evaluates the 5 key methods (NC, TS, Platt 1D, Platt 5D, VCPS-5D) across the
4 core metrics: ECE (%), Adaptive ECE (%), AUROC, and Brier Score.

Produces sheets/pairwise_cross_domain_transfer.xlsx containing 17 plain tabs:
1. Matrix_AdaECE: 13x13 zero-shot transfer matrices for Adaptive ECE (%)
2. Matrix_ECE: 13x13 zero-shot transfer matrices for standard ECE (%)
3. Matrix_AUROC: 13x13 zero-shot transfer matrices for AUROC
4. Matrix_Brier: 13x13 zero-shot transfer matrices for Brier Score
5-17. src_{dataset}: 13 source dataset sheets listing all 12 target benchmarks row-by-row
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trajectory_calibration.calibrators.baselines import (
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    TemperatureScalingEstimator,
    TrajectoryPlattScaler,
)
from trajectory_calibration.calibrators.residual import evaluate_full_metric_panel
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.trajectory import (
    CANONICAL_5D_KEYS,
    load_dataset_features,
)
from trajectory_calibration.utils.helpers import set_seed
from trajectory_calibration.utils.math import sigmoid

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data" / "features"
SHEETS_DIR = ROOT / "sheets"
SHEETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = SHEETS_DIR / "pairwise_cross_domain_transfer.xlsx"

# The 13 qualifying benchmarks
ALL_DATASETS: list[str] = [
    "ai2d",
    "chartqa",
    "docvqa",
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

EVAL_METHODS: list[str] = ["NC", "TS", "Platt 1D", "Platt 5D", "VCPS-5D"]
MATRIX_METHODS: list[str] = ["NC", "TS", "Platt 1D", "Platt 5D", "VCPS-5D"]


def load_all_features() -> dict[str, dict[str, pd.DataFrame]]:
    """Pre-load all 13 qualifying dataset features into memory for M3 and MQT."""
    data: dict[str, dict[str, pd.DataFrame]] = {"m3": {}, "mqt": {}}
    for arch in ["m3", "mqt"]:
        fine_scale = 576 if arch == "m3" else 256
        for ds in ALL_DATASETS:
            print(f"Loading {arch.upper()} features for {ds}...", flush=True)
            df = load_dataset_features(
                str(FEATURES_DIR),
                ds_name=ds,
                arch=arch,
                gen_temperature=0.0,
                fine_scale=fine_scale,
            )
            data[arch][ds] = df
    return data


def train_source_models(df_train: pd.DataFrame) -> dict[str, Any]:
    """Fit candidate calibrators (NC, TS, Platt 1D, Platt 5D, VCPS-5D) on source dataset."""
    X_train_5d = np.asarray(df_train[CANONICAL_5D_KEYS].values, dtype=np.float64)
    y_train = np.asarray(df_train["is_correct"].values, dtype=np.float64)

    nc = NaiveConfidenceEstimator().fit(X_train_5d, y_train)
    ts = TemperatureScalingEstimator().fit(X_train_5d, y_train)
    p1 = PlattScalingEstimator().fit(X_train_5d, y_train)
    p5 = TrajectoryPlattScaler(n_features=5).fit(X_train_5d, y_train)
    vcps = VaryingCoefficientPlattScaler(feature_set="5d")
    vcps.fit(X_train_5d, y_train, feature_names=CANONICAL_5D_KEYS)

    return {
        "NC": nc,
        "TS": ts,
        "Platt 1D": p1,
        "Platt 5D": p5,
        "VCPS-5D": vcps,
    }


def evaluate_transfer_pair(
    models: dict[str, Any], df_target: pd.DataFrame, fine_scale: int = 576
) -> dict[str, dict[str, float]]:
    """Evaluate fitted models zero-shot across ECE, Ada-ECE, AUROC, and Brier."""
    X_test_5d = np.asarray(df_target[CANONICAL_5D_KEYS].values, dtype=np.float64)
    y_test = np.asarray(df_target["is_correct"].values, dtype=np.float64)
    if f"c_{fine_scale}" in df_target.columns:
        c_test = np.asarray(df_target[f"c_{fine_scale}"].values, dtype=np.float64)
    elif "c_fine" in df_target.columns:
        c_test = np.asarray(df_target["c_fine"].values, dtype=np.float64)
    elif "c_576" in df_target.columns:
        c_test = np.asarray(df_target["c_576"].values, dtype=np.float64)
    else:
        c_test = np.asarray(sigmoid(X_test_5d[:, 0]), dtype=np.float64)

    results: dict[str, dict[str, float]] = {}
    for m_name, model in models.items():
        probs = np.asarray(model.predict_proba(X_test_5d), dtype=np.float64)
        panel = evaluate_full_metric_panel(probs, y_test, c_test)
        results[m_name] = {
            "ECE (%)": round(float(panel["ece_percent"]), 4),
            "Adaptive ECE (%)": round(float(panel["adaptive_ece_percent"]), 4),
            "AUROC": round(float(panel["auroc"]), 4),
            "Brier Score": round(float(panel["brier"]), 4),
        }
    return results


def run_all_pairwise_evaluations(data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Run all 13x12 = 156 pairwise evaluations per arch (312 total)."""
    records = []
    for arch in ["m3", "mqt"]:
        arch_label = arch.upper()
        fine_scale = 576 if arch == "m3" else 256
        for src in ALL_DATASETS:
            df_train = data[arch][src]
            models = train_source_models(df_train)

            for tgt in ALL_DATASETS:
                if tgt == src:
                    continue
                df_target = data[arch][tgt]
                pair_metrics = evaluate_transfer_pair(models, df_target, fine_scale=fine_scale)
                for m_name, metrics in pair_metrics.items():
                    records.append(
                        {
                            "model": arch_label,
                            "source": src,
                            "target": tgt,
                            "method": m_name,
                            **metrics,
                        }
                    )
    return pd.DataFrame(records)


def build_matrix_dataframe(
    df_eval: pd.DataFrame, arch: str, method: str, metric_col: str
) -> pd.DataFrame:
    """Build a clean 13x13 pairwise matrix DataFrame with row and column marginal means."""
    df_sub = df_eval[(df_eval["model"] == arch) & (df_eval["method"] == method)]
    piv = df_sub.pivot(index="source", columns="target", values=metric_col)

    # Reindex to ensure consistent order of all 13 datasets
    piv = piv.reindex(index=ALL_DATASETS, columns=ALL_DATASETS)

    # Build matrix with '-' on the diagonal
    matrix_dict: dict[str, list[Any]] = {"Source": ALL_DATASETS}
    all_vals: list[float] = []
    row_means: list[float] = []

    for src in ALL_DATASETS:
        vals = []
        for tgt in ALL_DATASETS:
            if src == tgt:
                continue
            val = piv.loc[src, tgt]
            if pd.notna(val):
                vals.append(float(val))
                all_vals.append(float(val))
        row_means.append(round(float(np.mean(vals)), 4) if vals else np.nan)

    for tgt in ALL_DATASETS:
        col_entries = []
        for src in ALL_DATASETS:
            if src == tgt:
                col_entries.append("-")
            else:
                val = piv.loc[src, tgt]
                col_entries.append(round(float(val), 4) if pd.notna(val) else np.nan)
        matrix_dict[tgt] = col_entries

    matrix_dict["Source Mean"] = row_means
    df_mat = pd.DataFrame(matrix_dict)

    # Add Target Mean bottom row
    target_mean_row: dict[str, Any] = {"Source": "Target Mean"}
    for tgt in ALL_DATASETS:
        col_vals = [
            float(piv.loc[src, tgt])
            for src in ALL_DATASETS
            if src != tgt and pd.notna(piv.loc[src, tgt])
        ]
        target_mean_row[tgt] = round(float(np.mean(col_vals)), 4) if col_vals else np.nan
    target_mean_row["Source Mean"] = round(float(np.mean(all_vals)), 4) if all_vals else np.nan

    df_mat = pd.concat([df_mat, pd.DataFrame([target_mean_row])], ignore_index=True)
    return df_mat


def write_matrix_sheet(
    writer: pd.ExcelWriter,
    df_eval: pd.DataFrame,
    sheet_name: str,
    metric_col: str,
) -> None:
    """Write plain 13x13 matrix tables for M3 and MQT across candidate methods."""
    start_row = 0
    for arch in ["M3", "MQT"]:
        for method in MATRIX_METHODS:
            title_df = pd.DataFrame(
                [[f"{arch}-LLaVA (7B) - {method} Zero-Shot Transfer: {metric_col}"]]
            )
            title_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                header=False,
                startrow=start_row,
            )
            start_row += 1

            df_mat = build_matrix_dataframe(
                df_eval, arch=arch, method=method, metric_col=metric_col
            )
            df_mat.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                header=True,
                startrow=start_row,
            )
            start_row += len(df_mat) + 3


def build_source_table(df_eval: pd.DataFrame, arch: str, src_ds: str) -> pd.DataFrame:
    """Build plain target-by-target breakdown table for a given source dataset and architecture."""
    df_sub = df_eval[(df_eval["model"] == arch) & (df_eval["source"] == src_ds)]
    targets = [d for d in ALL_DATASETS if d != src_ds]

    piv_ece = df_sub.pivot(index="target", columns="method", values="ECE (%)")
    piv_ada = df_sub.pivot(index="target", columns="method", values="Adaptive ECE (%)")
    piv_auroc = df_sub.pivot(index="target", columns="method", values="AUROC")
    piv_brier = df_sub.pivot(index="target", columns="method", values="Brier Score")

    rows = []
    for tgt in targets:
        row: dict[str, Any] = {"Target Dataset": tgt}
        for m in EVAL_METHODS:
            row[f"{m} ECE (%)"] = round(float(piv_ece.loc[tgt, m]), 4)
        for m in EVAL_METHODS:
            row[f"{m} Ada-ECE (%)"] = round(float(piv_ada.loc[tgt, m]), 4)
        for m in EVAL_METHODS:
            row[f"{m} AUROC"] = round(float(piv_auroc.loc[tgt, m]), 4)
        for m in EVAL_METHODS:
            row[f"{m} Brier"] = round(float(piv_brier.loc[tgt, m]), 4)
        rows.append(row)

    df_res = pd.DataFrame(rows)

    # Add Macro Mean bottom row
    mean_row: dict[str, Any] = {"Target Dataset": f"Macro Mean ({len(targets)} Targets)"}
    for col in df_res.columns:
        if col != "Target Dataset":
            mean_row[col] = round(float(df_res[col].mean()), 4)

    df_res = pd.concat([df_res, pd.DataFrame([mean_row])], ignore_index=True)
    return df_res


def write_source_sheet(writer: pd.ExcelWriter, df_eval: pd.DataFrame, src_ds: str) -> None:
    """Write plain source dataset sheet containing M3 and MQT target tables."""
    sheet_name = f"src_{src_ds}"
    start_row = 0

    for arch in ["M3", "MQT"]:
        title_df = pd.DataFrame([[f"{arch}-LLaVA (7B) Zero-Shot Transfer from Source: {src_ds}"]])
        title_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            header=False,
            startrow=start_row,
        )
        start_row += 1

        df_table = build_source_table(df_eval, arch=arch, src_ds=src_ds)
        df_table.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            header=True,
            startrow=start_row,
        )
        start_row += len(df_table) + 3


def main() -> None:
    """Main execution entry point."""
    set_seed(42)
    print("Pre-loading features across 13 benchmarks for M3 and MQT...", flush=True)
    data = load_all_features()

    print(
        "Running pairwise cross-domain zero-shot evaluations (156 pairs per arch, 312 total)...",
        flush=True,
    )
    df_eval = run_all_pairwise_evaluations(data)

    print(f"Creating plain Excel workbook with 17 tabs at {OUTPUT_FILE}...", flush=True)
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            # 1. Matrix_AdaECE
            print("Writing Matrix_AdaECE...", flush=True)
            write_matrix_sheet(
                writer, df_eval, sheet_name="Matrix_AdaECE", metric_col="Adaptive ECE (%)"
            )

            # 2. Matrix_ECE
            print("Writing Matrix_ECE...", flush=True)
            write_matrix_sheet(writer, df_eval, sheet_name="Matrix_ECE", metric_col="ECE (%)")

            # 3. Matrix_AUROC
            print("Writing Matrix_AUROC...", flush=True)
            write_matrix_sheet(writer, df_eval, sheet_name="Matrix_AUROC", metric_col="AUROC")

            # 4. Matrix_Brier
            print("Writing Matrix_Brier...", flush=True)
            write_matrix_sheet(writer, df_eval, sheet_name="Matrix_Brier", metric_col="Brier Score")

            # 5-17. src_{dataset}
            for src in ALL_DATASETS:
                print(f"Writing src_{src}...", flush=True)
                write_source_sheet(writer, df_eval, src_ds=src)

        print(f"Successfully generated {OUTPUT_FILE} with 17 plain tabs.")
    except PermissionError:
        fallback_file = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}_updated.xlsx")
        print(
            f"Notice: {OUTPUT_FILE.name} is open in Excel. Saving to fallback: {fallback_file.name}",
            flush=True,
        )
        with pd.ExcelWriter(fallback_file, engine="openpyxl") as writer:
            write_matrix_sheet(
                writer, df_eval, sheet_name="Matrix_AdaECE", metric_col="Adaptive ECE (%)"
            )
            write_matrix_sheet(writer, df_eval, sheet_name="Matrix_ECE", metric_col="ECE (%)")
            write_matrix_sheet(writer, df_eval, sheet_name="Matrix_AUROC", metric_col="AUROC")
            write_matrix_sheet(writer, df_eval, sheet_name="Matrix_Brier", metric_col="Brier Score")
            for src in ALL_DATASETS:
                write_source_sheet(writer, df_eval, src_ds=src)
        print(f"Successfully generated {fallback_file} with 17 plain tabs.")


if __name__ == "__main__":
    main()
