#!/usr/bin/env python3
"""
Multi-Seed Cross-Validation Benchmark Runner for Robust Calibration Evaluation.

Evaluates post-hoc calibrators across multiple stratified random splits (default: 10 seeds)
on all target benchmarks to compute mean +/- standard deviation for:
- Expected Calibration Error (ECE %)
- Adaptive ECE (%)
- AUROC
- Brier Score
"""

import argparse
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    BetaCalibrator,
    MultiScaleEigenVariance,
    MultiScaleSemanticConsistency,
    NaiveConfidenceEstimator,
    PlattScalingEstimator,
    ProbabilityMarginEstimator,
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
from trajectory_calibration.features.trajectory import (
    CANONICAL_5D_KEYS,
    FEATURE_KEYS,
    get_stratified_split,
    load_dataset_features,
)
from trajectory_calibration.utils.helpers import set_seed

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Seed Cross-Validation Calibration Benchmark Runner."
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features",
        help="Base directory of extracted features.",
    )
    parser.add_argument(
        "--arch", type=str, default="m3", choices=["m3", "mqt"], help="Model architecture."
    )
    parser.add_argument("--gen_temperature", type=float, default=0.0, help="Decoding temperature.")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_14_DATASETS, help="Datasets to evaluate."
    )
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds / splits.")
    parser.add_argument("--base_seed", type=int, default=42, help="Starting random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/benchmark_cv",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fine_scale = 576 if args.arch == "m3" else 256
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    raw_results = []

    print("=" * 80)
    print(
        f" MULTI-SEED CV BENCHMARK: {args.arch.upper()} across {len(args.datasets)} datasets ({args.n_seeds} seeds)"
    )
    print("=" * 80)

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

        print(f"\nEvaluating dataset: {ds.upper()} ({len(df)} samples, {args.n_seeds} seeds)...")

        for seed_idx, seed in enumerate(seeds, start=1):
            set_seed(seed)
            train_idx, test_idx = get_stratified_split(df, test_size=0.2, random_state=seed)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            X_train_17d = train_df[FEATURE_KEYS].values
            y_train = train_df["is_correct"].values
            X_test_17d = test_df[FEATURE_KEYS].values
            y_test = test_df["is_correct"].values
            c_test_576 = test_df["c_576"].values

            best_5d_keys = CANONICAL_5D_KEYS
            X_train_5d = train_df[CANONICAL_5D_KEYS].values
            X_test_5d = test_df[CANONICAL_5D_KEYS].values

            methods = {
                "Naive Confidence (NC)": (NaiveConfidenceEstimator(), X_train_17d, X_test_17d),
                "Temperature Scaling (TS)": (
                    TemperatureScalingEstimator(),
                    X_train_17d,
                    X_test_17d,
                ),
                "Platt Scaling (1D)": (PlattScalingEstimator(), X_train_17d, X_test_17d),
                "Quadratic Platt (Logit-Only)": (QuadraticPlattScaler(), X_train_17d, X_test_17d),
                "Beta Calibration": (BetaCalibrator(), X_train_17d, X_test_17d),
                "Spline Calibration": (SplineCalibrator(), X_train_17d, X_test_17d),
                "Adaptive TS (ATS)": (AdaptiveTemperatureScaling(), X_train_5d, X_test_5d),
                "Probability Margin (1D)": (ProbabilityMarginEstimator(), X_train_17d, X_test_17d),
                "MSSC (Multi-Scale Proxy)": (
                    MultiScaleSemanticConsistency(),
                    X_train_17d,
                    X_test_17d,
                ),
                "MSE-EIGEN (Multi-Scale)": (MultiScaleEigenVariance(), X_train_17d, X_test_17d),
                "Residual Calibrator": (ResidualTrajectoryCalibrator(), X_train_5d, X_test_5d),
                "Trajectory LR": (
                    TrajectoryLREstimator(fit_intercept=True, random_state=seed),
                    X_train_5d,
                    X_test_5d,
                ),
                "Trajectory LR (No Bias)": (
                    TrajectoryLREstimator(fit_intercept=False, random_state=seed),
                    X_train_5d,
                    X_test_5d,
                ),
                "Trajectory Platt (5D)": (
                    TrajectoryPlattScaler(n_features=len(best_5d_keys)),
                    X_train_5d,
                    X_test_5d,
                ),
                "Trajectory Platt (17D)": (
                    TrajectoryPlattScaler(n_features=len(FEATURE_KEYS)),
                    X_train_17d,
                    X_test_17d,
                ),
                "VCPS-5D (Our Method)": (
                    VaryingCoefficientPlattScaler(feature_set="5d", random_state=seed),
                    X_train_5d,
                    X_test_5d,
                ),
                "VCPS-17D (Our Method)": (
                    VaryingCoefficientPlattScaler(feature_set="17d", random_state=seed),
                    X_train_17d,
                    X_test_17d,
                ),
            }

            for m_name, (model, X_tr, X_te) in methods.items():
                if isinstance(model, VaryingCoefficientPlattScaler):
                    feat_names = best_5d_keys if "5D" in m_name else FEATURE_KEYS
                    model.fit(X_tr, y_train, feature_names=feat_names)
                else:
                    model.fit(X_tr, y_train)

                probs = model.predict_proba(X_te)
                panel = evaluate_full_metric_panel(probs, y_test, c_test_576, y_train=y_train)
                panel["dataset"] = ds
                panel["method"] = m_name
                panel["seed"] = seed
                raw_results.append(panel)

    if not raw_results:
        print("No results gathered.")
        return

    raw_df = pd.DataFrame(raw_results)
    raw_csv = out_dir / f"benchmark_cv_{args.arch}_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\nSaved raw per-seed results to: {raw_csv}")

    # Compute mean +/- std aggregations
    metrics = ["ece_percent", "adaptive_ece_percent", "auroc", "brier"]
    agg_funcs = {m: ["mean", "std"] for m in metrics}

    summary_df = raw_df.groupby(["dataset", "method"]).agg(agg_funcs).reset_index()
    # Flatten MultiIndex column names
    summary_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary_df.columns]

    summary_csv = out_dir / f"benchmark_cv_{args.arch}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved aggregated summary to: {summary_csv}")

    # Create formatted mean +/- std table for Adaptive ECE (%)
    raw_df["ada_ece_str"] = raw_df.apply(lambda r: f"{r['adaptive_ece_percent']:.2f}", axis=1)
    mean_std = (
        raw_df.groupby(["method", "dataset"])["adaptive_ece_percent"]
        .agg(["mean", "std"])
        .reset_index()
    )
    mean_std["mean_std_str"] = mean_std.apply(
        lambda r: f"{r['mean']:.2f} +/- {r['std']:.2f}", axis=1
    )
    pivot_table = mean_std.pivot(index="method", columns="dataset", values="mean_std_str")

    print("\n" + "=" * 90)
    print(f" MULTI-SEED CV BENCHMARK (Mean +/- Std): Adaptive ECE (%) [{args.arch.upper()}]")
    print("=" * 90)
    print(pivot_table.to_string())
    print("=" * 90)


if __name__ == "__main__":
    main()
