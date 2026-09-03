#!/usr/bin/env python3
"""
Comprehensive Multi-Dataset Calibration Benchmark Runner.

Fits and evaluates post-hoc calibrators across target datasets,
saving publication-ready markdown and CSV summary tables.
"""

import argparse
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd
from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    MultiScaleEigenVariance,
    MultiScaleSemanticConsistency,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    ProbabilityMarginEstimator,
    QuadraticPlattScaler,
    SplineCalibrator,
    TemperatureScalingEstimator,
    TrajectoryLREstimator,
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
    parser = argparse.ArgumentParser(description="Run comprehensive multi-dataset calibration benchmark.")
    parser.add_argument("--features_dir", type=str, default="data/features", help="Base directory of extracted features.")
    parser.add_argument("--arch", type=str, default="m3", choices=["m3", "mqt"], help="Model architecture.")
    parser.add_argument("--gen_temperature", type=float, default=0.0, help="Decoding temperature.")
    parser.add_argument("--datasets", nargs="+", default=["pope", "scienceqa", "textvqa", "vizwiz-vqa"], help="Datasets to evaluate.")
    parser.add_argument("--output_dir", type=str, default="results/experiments/benchmark", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    fine_scale = 576 if args.arch == "m3" else 256
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ds in args.datasets:
        try:
            df = load_dataset_features(
                args.features_dir,
                ds_name=ds,
                arch=args.arch,
                gen_temperature=args.gen_temperature,
                fine_scale=fine_scale,
            )
        except Exception as e:
            print(f"Skipping '{ds}': {e}")
            continue

        print(f"\nEvaluating dataset: {ds.upper()} ({len(df)} unique samples)...")
        train_idx, test_idx = get_stratified_split(df, test_size=0.2, random_state=args.seed)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        X_train_17d = train_df[FEATURE_KEYS].values
        y_train = train_df["is_correct"].values
        X_test_17d = test_df[FEATURE_KEYS].values
        y_test = test_df["is_correct"].values
        c_test_576 = test_df["c_576"].values

        best_5d_keys = select_best_5d_subset(X_train_17d, y_train, FEATURE_KEYS)
        X_train_5d = train_df[best_5d_keys].values
        X_test_5d = test_df[best_5d_keys].values

        methods = {
            "Naive Confidence (NC)": (NaiveConfidenceEstimator(), X_train_17d, X_test_17d),
            "Temperature Scaling (TS)": (TemperatureScalingEstimator(), X_train_17d, X_test_17d),
            "Platt Scaling (1D)": (PlattScalingEstimator(), X_train_17d, X_test_17d),
            "Spline Calibration": (SplineCalibrator(), X_train_17d, X_test_17d),
            "Adaptive TS (ATS)": (AdaptiveTemperatureScaling(), X_train_5d, X_test_5d),
            "Probability Margin (1D)": (ProbabilityMarginEstimator(), X_train_17d, X_test_17d),
            "MSSC (Multi-Scale Proxy)": (MultiScaleSemanticConsistency(), X_train_17d, X_test_17d),
            "MSE-EIGEN (Multi-Scale)": (MultiScaleEigenVariance(), X_train_17d, X_test_17d),
            "Residual Calibrator": (ResidualTrajectoryCalibrator(), X_train_5d, X_test_5d),
            "Trajectory LR (No Bias)": (TrajectoryLREstimator(fit_intercept=False), X_train_5d, X_test_5d),
            "Quadratic Platt (Logit-Only)": (QuadraticPlattScaler(), X_train_17d, X_test_17d),
            "VCPS-5D (Our Method)": (VaryingCoefficientPlattScaler(slope_features=best_5d_keys[1:3], intercept_features=best_5d_keys[1:]), X_train_5d, X_test_5d),
            "VCPS-17D (Our Method)": (VaryingCoefficientPlattScaler(), X_train_17d, X_test_17d),
        }

        for m_name, (model, X_tr, X_te) in methods.items():
            model.fit(X_tr, y_train)
            probs = model.predict_proba(X_te)
            panel = evaluate_full_metric_panel(probs, y_test, c_test_576, y_train=y_train)
            panel["dataset"] = ds
            panel["method"] = m_name
            all_results.append(panel)

    if not all_results:
        print("No datasets were evaluated.")
        return

    results_df = pd.DataFrame(all_results)
    csv_out = out_dir / f"benchmark_{args.arch}_summary.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"\nSaved full results to {csv_out}")

    pivot_ece = results_df.pivot(index="method", columns="dataset", values="adaptive_ece_percent")
    print("\n" + "=" * 75)
    print(f" BENCHMARK SUMMARY: Adaptive ECE (%) [{args.arch.upper()} T_gen={args.gen_temperature}]")
    print("=" * 75)
    print(pivot_ece.round(2).to_string())
    print("=" * 75)


if __name__ == "__main__":
    main()
