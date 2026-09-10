#!/usr/bin/env python3
"""Generate Zero-Shot Cross-Domain Transfer Excel file (Workbook 2).

Evaluates single-source cross-domain transfer (train on 1, test on remaining qualifying datasets),
using direct zero-shot evaluation without Saerens-EM adaptation.

Produces sheets/zero_shot_cross_domain_transfer.xlsx containing:
1. Platt_5D_Macro_Transfer: Overall macro transfer across 11 qualifying source datasets.
2. VCPS_5D_Macro_Transfer: Overall macro transfer across 11 qualifying source datasets.
3. Individual source dataset sheets showing generalization on target datasets.
"""

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
OUTPUT_FILE = SHEETS_DIR / "zero_shot_cross_domain_transfer.xlsx"

PLATT_5D_QUALIFYING = [
    "chartqa",
    "docvqa",
    "infographicvqa",
    "lego-puzzles",
    "mmbench",
    "mmmu",
    "scienceqa",
    "seedbench",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
]

VCPS_5D_QUALIFYING = [
    "ai2d",
    "chartqa",
    "docvqa",
    "infographicvqa",
    "lego-puzzles",
    "mmbench",
    "mmmu",
    "pope",
    "scienceqa",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
]

ALL_QUALIFYING_SOURCES = sorted(list(set(PLATT_5D_QUALIFYING).union(set(VCPS_5D_QUALIFYING))))


def load_all_features() -> dict[str, dict[str, pd.DataFrame]]:
    """Pre-load all qualifying dataset features into memory for M3 and MQT."""
    data: dict[str, dict[str, pd.DataFrame]] = {"m3": {}, "mqt": {}}
    for arch in ["m3", "mqt"]:
        fine_scale = 576 if arch == "m3" else 256
        for ds in ALL_QUALIFYING_SOURCES:
            df = load_dataset_features(
                str(FEATURES_DIR),
                ds_name=ds,
                arch=arch,
                gen_temperature=0.0,
                fine_scale=fine_scale,
            )
            data[arch][ds] = df
    return data


def train_calibrators_on_source(df_train: pd.DataFrame, include_vcps: bool) -> dict[str, Any]:
    """Fit candidate calibrators on a single source dataset."""
    X_train_5d = np.asarray(df_train[CANONICAL_5D_KEYS].values, dtype=np.float64)
    y_train = np.asarray(df_train["is_correct"].values, dtype=np.float64)

    models: dict[str, Any] = {
        "NC": NaiveConfidenceEstimator().fit(X_train_5d, y_train),
        "Temp": TemperatureScalingEstimator().fit(X_train_5d, y_train),
        "Platt 1D": PlattScalingEstimator().fit(X_train_5d, y_train),
        "Platt 5D": TrajectoryPlattScaler(n_features=5).fit(X_train_5d, y_train),
    }

    if include_vcps:
        vcps = VaryingCoefficientPlattScaler(feature_set="5d")
        vcps.fit(X_train_5d, y_train, feature_names=CANONICAL_5D_KEYS)
        models["VCPS Platt 5D"] = vcps

    return models


def evaluate_transfer_pair(
    models: dict[str, Any], df_target: pd.DataFrame, fine_scale: int = 576
) -> dict[str, dict[str, float]]:
    """Evaluate fitted models zero-shot on a target dataset."""
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


def run_all_cross_domain_evaluations(
    data: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, object]]:
    """Run single-source transfer evaluations across all qualifying source datasets."""
    records = []
    for arch in ["m3", "mqt"]:
        arch_label = arch.upper()
        fine_scale = 576 if arch == "m3" else 256
        for src in ALL_QUALIFYING_SOURCES:
            df_train = data[arch][src]
            is_p5 = src in PLATT_5D_QUALIFYING
            is_vcps = src in VCPS_5D_QUALIFYING
            models = train_calibrators_on_source(df_train, include_vcps=is_vcps)

            target_pool = [ds for ds in ALL_QUALIFYING_SOURCES if ds != src]
            for tgt in target_pool:
                df_target = data[arch][tgt]
                pair_metrics = evaluate_transfer_pair(models, df_target, fine_scale=fine_scale)
                for m_name, metrics in pair_metrics.items():
                    # Filter evaluation validity based on track
                    valid_p5 = (
                        is_p5 and (tgt in PLATT_5D_QUALIFYING) and (m_name != "VCPS Platt 5D")
                    )
                    valid_vcps = is_vcps and (tgt in VCPS_5D_QUALIFYING) and (m_name != "Platt 5D")
                    records.append(
                        {
                            "model": arch_label,
                            "source": src,
                            "target": tgt,
                            "method": m_name,
                            "valid_p5_track": valid_p5,
                            "valid_vcps_track": valid_vcps,
                            **metrics,
                        }
                    )
    return records


def build_macro_transfer_table(
    df_eval: pd.DataFrame, track: str, candidate_name: str
) -> pd.DataFrame:
    """Build overall Grand Macro Transfer table across all qualifying source datasets."""
    methods = ["NC", "Temp", "Platt 1D", candidate_name]
    track_col = "valid_p5_track" if track == "p5" else "valid_vcps_track"
    df_track = df_eval[df_eval[track_col]].copy()

    rows = []
    for arch_label in ["M3", "MQT"]:
        df_arch = df_track[df_track["model"] == arch_label]
        for m in methods:
            df_m = df_arch[df_arch["method"] == m]
            rows.append(
                {
                    "Model": arch_label,
                    "Method": m,
                    "ECE (%)": round(float(df_m["ECE (%)"].mean()), 4),
                    "Adaptive ECE (%)": round(float(df_m["Adaptive ECE (%)"].mean()), 4),
                    "AUROC": round(float(df_m["AUROC"].mean()), 4),
                    "Brier Score": round(float(df_m["Brier Score"].mean()), 4),
                }
            )
    return pd.DataFrame(rows)


def build_source_sheet_table(
    df_eval: pd.DataFrame, src_dataset: str
) -> tuple[pd.DataFrame, list[str]]:
    """Build transfer macro table for a specific source dataset across qualifying target datasets."""
    df_src = df_eval[df_eval["source"] == src_dataset]
    rows = []
    targets = sorted(list(set(str(t) for t in df_src["target"].tolist())))

    for arch_label in ["M3", "MQT"]:
        df_arch = df_src[df_src["model"] == arch_label]
        methods = ["NC", "Temp", "Platt 1D", "Platt 5D", "VCPS Platt 5D"]
        for m in methods:
            if m == "Platt 5D":
                df_m = df_arch[(df_arch["method"] == m) & (df_arch["valid_p5_track"])]
            elif m == "VCPS Platt 5D":
                df_m = df_arch[(df_arch["method"] == m) & (df_arch["valid_vcps_track"])]
            else:
                df_m = df_arch[
                    (df_arch["method"] == m)
                    & (df_arch["valid_p5_track"] | df_arch["valid_vcps_track"])
                ]

            if len(df_m) > 0:
                rows.append(
                    {
                        "Model": arch_label,
                        "Method": m,
                        "ECE (%)": round(float(df_m["ECE (%)"].mean()), 4),
                        "Adaptive ECE (%)": round(float(df_m["Adaptive ECE (%)"].mean()), 4),
                        "AUROC": round(float(df_m["AUROC"].mean()), 4),
                        "Brier Score": round(float(df_m["Brier Score"].mean()), 4),
                    }
                )

    return pd.DataFrame(rows), targets


def write_all_sheets(
    writer: pd.ExcelWriter,
    macro_p5_df: pd.DataFrame,
    macro_vcps_df: pd.DataFrame,
    df_eval: pd.DataFrame,
) -> None:
    """Write all summary and individual source transfer sheets into the Excel writer."""
    # Sheet 1: Platt 5D Macro Transfer
    macro_p5_df.to_excel(writer, sheet_name="Platt_5D_Macro_Transfer", index=False, startrow=0)
    note_p5 = pd.DataFrame(
        [
            {
                "Note": (
                    f"Note: Grand Macro Transfer Mean across {len(PLATT_5D_QUALIFYING)} source datasets "
                    f"trained individually and evaluated zero-shot on the other 10 qualifying datasets: "
                    f"{', '.join(PLATT_5D_QUALIFYING)}."
                )
            }
        ]
    )
    note_p5.to_excel(
        writer, sheet_name="Platt_5D_Macro_Transfer", index=False, startrow=len(macro_p5_df) + 2
    )

    # Sheet 2: VCPS Platt 5D Macro Transfer
    macro_vcps_df.to_excel(writer, sheet_name="VCPS_5D_Macro_Transfer", index=False, startrow=0)
    note_vcps = pd.DataFrame(
        [
            {
                "Note": (
                    f"Note: Grand Macro Transfer Mean across {len(VCPS_5D_QUALIFYING)} source datasets "
                    f"trained individually and evaluated zero-shot on the other {len(VCPS_5D_QUALIFYING) - 1} qualifying datasets: "
                    f"{', '.join(VCPS_5D_QUALIFYING)}."
                )
            }
        ]
    )
    note_vcps.to_excel(
        writer,
        sheet_name="VCPS_5D_Macro_Transfer",
        index=False,
        startrow=len(macro_vcps_df) + 2,
    )

    # Individual source dataset sheets (macro-mean only)
    for src_ds in ALL_QUALIFYING_SOURCES:
        src_sheet_name = f"src_{src_ds}"
        src_table, _ = build_source_sheet_table(df_eval, src_ds)
        src_table.to_excel(writer, sheet_name=src_sheet_name, index=False, startrow=0)

        p5_targets = [d for d in PLATT_5D_QUALIFYING if d != src_ds]
        vcps_targets = [d for d in VCPS_5D_QUALIFYING if d != src_ds]

        notes = []
        if src_ds in PLATT_5D_QUALIFYING:
            notes.append(
                f"Platt 5D Macro Mean evaluated across {len(p5_targets)} qualifying targets: {', '.join(p5_targets)}."
            )
        else:
            notes.append("Platt 5D did not qualify on this source dataset.")

        if src_ds in VCPS_5D_QUALIFYING:
            notes.append(
                f"VCPS Platt 5D Macro Mean evaluated across {len(vcps_targets)} qualifying targets: {', '.join(vcps_targets)}."
            )
        else:
            notes.append("VCPS Platt 5D did not qualify on this source dataset.")

        note_src = pd.DataFrame({"Notes": notes})
        note_src.to_excel(
            writer,
            sheet_name=src_sheet_name,
            index=False,
            startrow=len(src_table) + 2,
        )


def main() -> None:
    """Main execution entry point."""
    set_seed(42)
    print("Loading pre-extracted features for qualifying datasets...")
    data = load_all_features()

    print("Running cross-domain transfer evaluations (single-source zero-shot)...")
    eval_records = run_all_cross_domain_evaluations(data)
    df_eval = pd.DataFrame(eval_records)

    macro_p5_df = build_macro_transfer_table(df_eval, "p5", "Platt 5D")
    macro_vcps_df = build_macro_transfer_table(df_eval, "vcps", "VCPS Platt 5D")

    print(f"Writing Excel workbook to {OUTPUT_FILE}...")
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            write_all_sheets(writer, macro_p5_df, macro_vcps_df, df_eval)
        print(f"Successfully created {OUTPUT_FILE} with {2 + len(ALL_QUALIFYING_SOURCES)} sheets.")
    except PermissionError:
        fallback_file = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}_updated.xlsx")
        print(
            f"Notice: {OUTPUT_FILE.name} is open in Excel. Saving to fallback: {fallback_file.name}"
        )
        with pd.ExcelWriter(fallback_file, engine="openpyxl") as writer:
            write_all_sheets(writer, macro_p5_df, macro_vcps_df, df_eval)
        print(
            f"Successfully created {fallback_file} with {2 + len(ALL_QUALIFYING_SOURCES)} sheets."
        )


if __name__ == "__main__":
    main()
