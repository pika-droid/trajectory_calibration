#!/usr/bin/env python3
"""
Multi-Dataset Decoding Temperature (T_gen) Robustness Study.

Implements:
- Greedy-Trained Transfer Robustness (Trained on T=0.0, evaluated across T in {0.0, 0.3, 0.6, 0.9, 1.0, 1.5})
- Dynamic Slope & Effective Temperature Tracking across temperatures
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    MultiScaleSemanticConsistency,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    QuadraticPlattScaler,
    SplineCalibrator,
    TemperatureScalingEstimator,
    TrajectoryLREstimator,
    TrajectoryPlattScaler,
)
from trajectory_calibration.calibrators.residual import (
    ResidualTrajectoryCalibrator,
    evaluate_full_metric_panel,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS, FEATURE_KEYS
from trajectory_calibration.features.loader import get_stratified_split, load_dataset_features
from trajectory_calibration.utils.helpers import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Decoding Temperature Robustness Study."
    )
    parser.add_argument(
        "--features_dir", type=str, default="data/features", help="Path to features dir."
    )
    parser.add_argument(
        "--arch", type=str, default="m3", choices=["m3", "mqt"], help="Model architecture."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pope", "scienceqa", "textvqa", "vizwiz-vqa"],
        help="Datasets to evaluate across temperatures.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.6, 1.0, 1.5],
        help="Decoding temperatures to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/temperature_study",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    fine_scale = 576 if args.arch == "m3" else 256
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f" GREEDY-TRAINED TEMPERATURE TRANSFER ROBUSTNESS [{args.arch.upper()}]")
    print("=" * 80)

    # 1. Load T=0.0 features as training anchors
    train_models: dict[str, dict[str, Any]] = {}
    train_splits: dict[str, tuple[np.ndarray, list[str]]] = {}

    for ds in args.datasets:
        try:
            df_0 = load_dataset_features(
                args.features_dir,
                ds_name=ds,
                arch=args.arch,
                gen_temperature=0.0,
                fine_scale=fine_scale,
            )
            tr_idx, te_idx = get_stratified_split(df_0, test_size=0.2, random_state=args.seed)
            test_qids = set(df_0.iloc[te_idx]["question_id"].values)
            tr_df = df_0.iloc[tr_idx].reset_index(drop=True)

            X_tr_17d = tr_df[FEATURE_KEYS].values
            y_tr = tr_df["is_correct"].values
            best_5d = CANONICAL_5D_KEYS
            X_tr_5d = tr_df[CANONICAL_5D_KEYS].values

            models = {
                "Naive Confidence (NC)": NaiveConfidenceEstimator().fit(X_tr_17d, y_tr),
                "Temperature Scaling (TS)": TemperatureScalingEstimator().fit(X_tr_17d, y_tr),
                "Platt Scaling (1D)": PlattScalingEstimator().fit(X_tr_17d, y_tr),
                "Trajectory LR": TrajectoryLREstimator(fit_intercept=True).fit(X_tr_5d, y_tr),
                "Trajectory LR (No Bias)": TrajectoryLREstimator(fit_intercept=False).fit(
                    X_tr_5d, y_tr
                ),
                "Trajectory Platt (5D)": TrajectoryPlattScaler(n_features=len(best_5d)).fit(
                    X_tr_5d, y_tr
                ),
                "Trajectory Platt (17D)": TrajectoryPlattScaler(n_features=len(FEATURE_KEYS)).fit(
                    X_tr_17d, y_tr
                ),
                "Quadratic Platt (Logit-Only)": QuadraticPlattScaler().fit(X_tr_17d, y_tr),
                "Spline Calibration (PCHIP)": SplineCalibrator().fit(X_tr_17d, y_tr),
                "Adaptive TS (ATS)": AdaptiveTemperatureScaling().fit(X_tr_5d, y_tr),
                "MSSC (Multi-Scale Proxy)": MultiScaleSemanticConsistency().fit(X_tr_17d, y_tr),
                "Residual Calibrator": ResidualTrajectoryCalibrator().fit(X_tr_5d, y_tr),
                "VCPS-5D (Our Method)": VaryingCoefficientPlattScaler(feature_set="5d").fit(
                    X_tr_5d, y_tr, feature_names=best_5d
                ),
                "VCPS-17D (Our Method)": VaryingCoefficientPlattScaler(feature_set="17d").fit(
                    X_tr_17d, y_tr, feature_names=FEATURE_KEYS
                ),
            }
            train_models[ds] = models
            train_splits[ds] = (y_tr, best_5d, test_qids)
            print(f"Trained greedy anchor models on {ds.upper()} (N_train={len(tr_df)})")
        except Exception as e:
            print(f"Warning: Could not train T=0.0 anchor for {ds}: {e}")

    # 2. Evaluate across all temperatures zero-shot
    transfer_results = []
    tracking_results = []

    for T in args.temperatures:
        print(f"\nEvaluating temperature slice T_gen = {T:.1f}...")
        for ds in args.datasets:
            if ds not in train_models:
                continue
            try:
                df_T = load_dataset_features(
                    args.features_dir,
                    ds_name=ds,
                    arch=args.arch,
                    gen_temperature=T,
                    fine_scale=fine_scale,
                )
                y_tr, best_5d, test_qids = train_splits[ds]

                te_mask = df_T["question_id"].isin(test_qids)
                te_df = df_T[te_mask].reset_index(drop=True)
                tr_df_T = df_T[~te_mask].reset_index(drop=True)

                X_te_17d = te_df[FEATURE_KEYS].values
                y_te = te_df["is_correct"].values
                c_te = te_df["c_576"].values

                X_te_5d = te_df[best_5d].values

                # Evaluate Greedy Transfer
                for m_name, model in train_models[ds].items():
                    X_input = (
                        X_te_5d
                        if (
                            "5D" in m_name
                            or "ATS" in m_name
                            or "Residual" in m_name
                            or "Trajectory LR" in m_name
                        )
                        else X_te_17d
                    )
                    probs = model.predict_proba(X_input)
                    panel = evaluate_full_metric_panel(probs, y_te, c_te, y_train=y_tr)
                    panel["dataset"] = ds
                    panel["temperature"] = T
                    panel["method"] = m_name
                    panel["study"] = "temperature_transfer"
                    transfer_results.append(panel)

                # Dynamic Slope & Parameter Tracking: Independent Refit at Temperature T
                X_tr_T_17d = tr_df_T[FEATURE_KEYS].values
                y_tr_T = tr_df_T["is_correct"].values
                best_5d_T = CANONICAL_5D_KEYS
                X_tr_T_5d = tr_df_T[CANONICAL_5D_KEYS].values
                X_te_T_5d = te_df[CANONICAL_5D_KEYS].values

                vcps_refit = VaryingCoefficientPlattScaler(
                    feature_set="5d",
                ).fit(X_tr_T_5d, y_tr_T, feature_names=best_5d_T)
                vcps_probs = vcps_refit.predict_proba(X_te_T_5d)
                panel_refit = evaluate_full_metric_panel(vcps_probs, y_te, c_te, y_train=y_tr_T)

                slopes = vcps_refit.compute_dynamic_slope(X_te_T_5d)
                teffs = vcps_refit.get_effective_temperature(X_te_T_5d)

                panel_refit["dataset"] = ds
                panel_refit["temperature"] = T
                panel_refit["method"] = "VCPS-5D (Refitted)"
                panel_refit["study"] = "temperature_tracking"
                panel_refit["base_slope_a0"] = vcps_refit.a0
                panel_refit["mean_dynamic_slope"] = float(np.mean(slopes))
                panel_refit["mean_effective_temp"] = float(np.mean(teffs))
                tracking_results.append(panel_refit)

            except Exception as e:
                print(f"  Skipping {ds} at T={T:.1f}: {e}")

    # Export temperature study summaries
    df_transfer = pd.DataFrame(transfer_results)
    df_tracking = pd.DataFrame(tracking_results)

    df_transfer.to_csv(out_dir / f"temperature_transfer_{args.arch}_summary.csv", index=False)
    df_tracking.to_csv(out_dir / f"temperature_tracking_{args.arch}_summary.csv", index=False)
    if out_dir.name in ["m3", "mqt"]:
        df_transfer.to_csv(
            out_dir.parent / f"temperature_transfer_{args.arch}_summary.csv", index=False
        )
        df_tracking.to_csv(
            out_dir.parent / f"temperature_tracking_{args.arch}_summary.csv", index=False
        )
    elif (out_dir / args.arch).exists():
        df_transfer.to_csv(
            out_dir / args.arch / f"temperature_transfer_{args.arch}_summary.csv", index=False
        )
        df_tracking.to_csv(
            out_dir / args.arch / f"temperature_tracking_{args.arch}_summary.csv", index=False
        )

    print("\n" + "=" * 85)
    print(" GREEDY TRANSFER MACRO-MEAN ECE (%) ACROSS TEMPERATURES")
    print("=" * 85)
    pivot_transfer = df_transfer.pivot_table(
        index="method", columns="temperature", values="adaptive_ece_percent", aggfunc="mean"
    )
    print(pivot_transfer.round(2).to_string())

    print("\n" + "=" * 85)
    print(" VCPS PARAMETER TRACKING ACROSS TEMPERATURES")
    print("=" * 85)
    tracking = (
        df_tracking.groupby("temperature")[
            ["adaptive_ece_percent", "base_slope_a0", "mean_dynamic_slope", "mean_effective_temp"]
        ]
        .mean()
        .reset_index()
    )
    print(tracking.round(3).to_string(index=False))
    print("=" * 85)


if __name__ == "__main__":
    main()
