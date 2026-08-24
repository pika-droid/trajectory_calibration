#!/usr/bin/env python3
"""
Multi-Dataset Decoding Temperature (T_gen) Robustness Study (Method 30).

Implements:
- Sub-Study 30a: Greedy-Trained Transfer Robustness (Trained on T=0.0, evaluated across T in {0.0, 0.3, 0.6, 0.9, 1.0, 1.5})
- Sub-Study 30b: Per-Temperature Independent Refitting & Dynamic Slope/Temperature Tracking
"""

import argparse
import sys
from pathlib import Path
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
    SplineCalibrator,
    TemperatureScalingEstimator,
)
from trajectory_calibration.calibrators.residual import (
    ResidualTrajectoryCalibrator,
    evaluate_full_metric_panel,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.trajectory import (
    FEATURE_KEYS,
    get_stratified_split,
    load_dataset_features,
    select_best_5d_subset,
)
from trajectory_calibration.utils.helpers import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Decoding Temperature Robustness Study across Methods.")
    parser.add_argument("--features_dir", type=str, default="data/features", help="Base directory of features.")
    parser.add_argument("--arch", type=str, default="m3", choices=["m3", "mqt"], help="Model architecture.")
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.6, 0.9, 1.0, 1.5],
        help="Temperatures to evaluate."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pope", "scienceqa", "textvqa", "vizwiz-vqa"],
        help="Datasets to include in temperature study."
    )
    parser.add_argument("--output_dir", type=str, default="results/experiments/temperature_study", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    fine_scale = 576 if args.arch == "m3" else 256
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f" SUB-STUDY 30a: GREEDY-TRAINED TRANSFER ROBUSTNESS [{args.arch.upper()}]")
    print("=" * 80)

    # 1. Load T=0.0 features as training anchors
    train_models: dict[str, dict[str, Any]] = {}
    train_splits: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}

    for ds in args.datasets:
        try:
            df_0 = load_dataset_features(args.features_dir, ds_name=ds, arch=args.arch, gen_temperature=0.0, fine_scale=fine_scale)
            tr_idx, te_idx = get_stratified_split(df_0, test_size=0.2, random_state=args.seed)
            tr_df = df_0.iloc[tr_idx].reset_index(drop=True)

            X_tr_17d = tr_df[FEATURE_KEYS].values
            y_tr = tr_df["is_correct"].values
            best_5d = select_best_5d_subset(X_tr_17d, y_tr, FEATURE_KEYS)
            X_tr_5d = tr_df[best_5d].values

            models = {
                "Naive Confidence (NC)": NaiveConfidenceEstimator().fit(X_tr_17d, y_tr),
                "Temperature Scaling (TS)": TemperatureScalingEstimator().fit(X_tr_17d, y_tr),
                "Platt Scaling (1D)": PlattScalingEstimator().fit(X_tr_17d, y_tr),
                "Spline Calibration (PCHIP)": SplineCalibrator().fit(X_tr_17d, y_tr),
                "Adaptive TS (ATS)": AdaptiveTemperatureScaling().fit(X_tr_5d, y_tr),
                "MSSC (Multi-Scale Proxy)": MultiScaleSemanticConsistency().fit(X_tr_17d, y_tr),
                "Best 5D Trajectory": ResidualTrajectoryCalibrator().fit(X_tr_5d, y_tr),
                "Two-Stage Residual": ResidualTrajectoryCalibrator().fit(X_tr_5d, y_tr),
                "VCPS-5D (Our Method)": VaryingCoefficientPlattScaler(slope_features=best_5d[1:3], intercept_features=best_5d[1:]).fit(X_tr_5d, y_tr),
                "VCPS-17D (Our Method)": VaryingCoefficientPlattScaler().fit(X_tr_17d, y_tr),
            }
            train_models[ds] = models
            train_splits[ds] = (y_tr, best_5d)
            print(f"Trained greedy anchor models on {ds.upper()} (N_train={len(tr_df)})")
        except Exception as e:
            print(f"Warning: Could not train T=0.0 anchor for {ds}: {e}")

    # 2. Evaluate across all temperatures zero-shot
    exp30a_results = []
    exp30b_results = []

    for T in args.temperatures:
        print(f"\nEvaluating temperature slice T_gen = {T:.1f}...")
        for ds in args.datasets:
            if ds not in train_models:
                continue
            try:
                df_T = load_dataset_features(args.features_dir, ds_name=ds, arch=args.arch, gen_temperature=T, fine_scale=fine_scale)
                _, te_idx = get_stratified_split(df_T, test_size=0.2, random_state=args.seed)
                te_df = df_T.iloc[te_idx].reset_index(drop=True)

                X_te_17d = te_df[FEATURE_KEYS].values
                y_te = te_df["is_correct"].values
                c_te = te_df["c_576"].values

                y_tr, best_5d = train_splits[ds]
                X_te_5d = te_df[best_5d].values

                # Evaluate Greedy Transfer (30a)
                for m_name, model in train_models[ds].items():
                    X_input = X_te_5d if ("5D" in m_name or "ATS" in m_name or "Residual" in m_name) else X_te_17d
                    probs = model.predict_proba(X_input)
                    panel = evaluate_full_metric_panel(probs, y_te, c_te, y_train=y_tr)
                    panel["dataset"] = ds
                    panel["temperature"] = T
                    panel["method"] = m_name
                    panel["study"] = "30a_greedy_transfer"
                    exp30a_results.append(panel)

                # Sub-Study 30b: Independent Refit at Temperature T
                tr_idx, _ = get_stratified_split(df_T, test_size=0.2, random_state=args.seed)
                tr_df_T = df_T.iloc[tr_idx].reset_index(drop=True)
                X_tr_T_17d = tr_df_T[FEATURE_KEYS].values
                y_tr_T = tr_df_T["is_correct"].values
                best_5d_T = select_best_5d_subset(X_tr_T_17d, y_tr_T, FEATURE_KEYS)
                X_tr_T_5d = tr_df_T[best_5d_T].values
                X_te_T_5d = te_df[best_5d_T].values

                vcps_refit = VaryingCoefficientPlattScaler(slope_features=best_5d_T[1:3], intercept_features=best_5d_T[1:]).fit(X_tr_T_5d, y_tr_T)
                vcps_probs = vcps_refit.predict_proba(X_te_T_5d)
                panel_refit = evaluate_full_metric_panel(vcps_probs, y_te, c_te, y_train=y_tr_T)

                slopes = vcps_refit.compute_dynamic_slope(X_te_T_5d)
                teffs = vcps_refit.get_effective_temperature(X_te_T_5d)

                panel_refit["dataset"] = ds
                panel_refit["temperature"] = T
                panel_refit["method"] = "VCPS-5D (Refitted)"
                panel_refit["study"] = "30b_independent_refit"
                panel_refit["base_slope_a0"] = vcps_refit.a0
                panel_refit["mean_dynamic_slope"] = float(np.mean(slopes))
                panel_refit["mean_effective_temp"] = float(np.mean(teffs))
                exp30b_results.append(panel_refit)

            except Exception as e:
                print(f"  Skipping {ds} at T={T:.1f}: {e}")

    # Export 30a and 30b summaries
    df_30a = pd.DataFrame(exp30a_results)
    df_30b = pd.DataFrame(exp30b_results)

    df_30a.to_csv(out_dir / f"exp30a_{args.arch}_greedy_transfer.csv", index=False)
    df_30b.to_csv(out_dir / f"exp30b_{args.arch}_refitted_tracking.csv", index=False)

    print("\n" + "=" * 85)
    print(f" SUB-STUDY 30a: GREEDY TRANSFER MACRO-MEAN ECE (%) ACROSS TEMPERATURES")
    print("=" * 85)
    pivot_30a = df_30a.pivot_table(index="method", columns="temperature", values="adaptive_ece_percent", aggfunc="mean")
    print(pivot_30a.round(2).to_string())

    print("\n" + "=" * 85)
    print(f" SUB-STUDY 30b: VCPS PARAMETER TRACKING ACROSS TEMPERATURES")
    print("=" * 85)
    tracking = df_30b.groupby("temperature")[["adaptive_ece_percent", "base_slope_a0", "mean_dynamic_slope", "mean_effective_temp"]].mean().reset_index()
    print(tracking.round(3).to_string(index=False))
    print("=" * 85)


if __name__ == "__main__":
    main()
