#!/usr/bin/env python3
"""
Leave-One-Dataset-Out (LODO) Zero-Shot Cross-Domain Transfer Benchmark (Exp 29b).

Trains calibrators on 13 pooled benchmarks and evaluates zero-shot calibration
on the held-out 14th benchmark, comparing:
1. Zero-Shot Base (No target supervision)
2. Target Adapted (Unsupervised test-time prior shift adaptation via Saerens-EM & target intercept shift)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.calibrators.adaptation import (
    run_saerens_em_binary,
)
from trajectory_calibration.calibrators.baselines import (
    PlattScalingEstimator,
    TrajectoryLREstimator,
    TrajectoryPlattScaler,
)
from trajectory_calibration.calibrators.residual import (
    ResidualTrajectoryCalibrator,
    evaluate_full_metric_panel,
)
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.trajectory import (
    FEATURE_KEYS,
    load_dataset_features,
    select_best_5d_subset,
)
from trajectory_calibration.utils.helpers import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Leave-One-Dataset-Out (LODO) Cross-Domain Transfer Benchmark.")
    parser.add_argument("--features_dir", type=str, default="data/features", help="Base directory of features.")
    parser.add_argument("--arch", type=str, default="m3", choices=["m3", "mqt"], help="Model architecture.")
    parser.add_argument("--gen_temperature", type=float, default=0.0, help="Decoding temperature.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "ai2d", "chartqa", "docvqa", "gqa", "infographicvqa", "lego-puzzles",
            "mmbench", "mmmu", "pope", "scienceqa", "seedbench", "textvqa",
            "vizwiz-vqa", "vqav2_5scale"
        ],
        help="Datasets to include in LODO evaluation."
    )
    parser.add_argument("--output_dir", type=str, default="results/experiments/lodo", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    fine_scale = 576 if args.arch == "m3" else 256
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading all {len(args.datasets)} datasets for LODO evaluation ({args.arch.upper()})...")
    dataset_dfs: dict[str, pd.DataFrame] = {}
    for ds in args.datasets:
        try:
            df = load_dataset_features(
                args.features_dir,
                ds_name=ds,
                arch=args.arch,
                gen_temperature=args.gen_temperature,
                fine_scale=fine_scale,
            )
            ds_name = "vqav2" if ds == "vqav2_5scale" else ds
            dataset_dfs[ds_name] = df
            print(f"  Loaded {ds_name}: {len(df)} samples")
        except Exception as e:
            print(f"  Warning: Skipping {ds}: {e}")

    loaded_names = list(dataset_dfs.keys())
    if len(loaded_names) < 2:
        print("Error: LODO requires at least 2 datasets.")
        return

    all_lodo_results = []

    print("\nStarting Leave-One-Dataset-Out (LODO) Evaluation...")
    for test_name in loaded_names:
        print(f"\n---> Held-out Target Dataset: {test_name.upper()}")

        # 1. Pool remaining N-1 datasets for training
        train_dfs = [dataset_dfs[n] for n in loaded_names if n != test_name]
        pooled_train_df = pd.concat(train_dfs, ignore_index=True)
        target_test_df = dataset_dfs[test_name]

        X_train_17d = pooled_train_df[FEATURE_KEYS].values
        y_train = pooled_train_df["is_correct"].values
        X_test_17d = target_test_df[FEATURE_KEYS].values
        y_test = target_test_df["is_correct"].values
        c_test = target_test_df["c_576"].values

        # Select pooled best 5D subset
        best_5d_keys = select_best_5d_subset(X_train_17d, y_train, FEATURE_KEYS)
        X_train_5d = pooled_train_df[best_5d_keys].values
        X_test_5d = target_test_df[best_5d_keys].values

        # Models to evaluate
        base_models = {
            "Platt Scaling (1D)": (PlattScalingEstimator(), X_train_17d, X_test_17d),
            "Trajectory LR": (TrajectoryLREstimator(fit_intercept=True), X_train_5d, X_test_5d),
            "Trajectory LR (No Bias)": (TrajectoryLREstimator(fit_intercept=False), X_train_5d, X_test_5d),
            "Trajectory Platt (5D)": (TrajectoryPlattScaler(n_features=len(best_5d_keys)), X_train_5d, X_test_5d),
            "Trajectory Platt (17D)": (TrajectoryPlattScaler(n_features=len(FEATURE_KEYS)), X_train_17d, X_test_17d),
            "Best 5D Trajectory": (ResidualTrajectoryCalibrator(), X_train_5d, X_test_5d),
            "Two-Stage Residual": (ResidualTrajectoryCalibrator(), X_train_5d, X_test_5d),
            "VCPS-5D (Our Method)": (VaryingCoefficientPlattScaler(feature_set="5d"), X_train_5d, X_test_5d),
            "VCPS-17D (Our Method)": (VaryingCoefficientPlattScaler(feature_set="17d"), X_train_17d, X_test_17d),
        }

        for m_name, (model, X_tr, X_te) in base_models.items():
            if isinstance(model, VaryingCoefficientPlattScaler):
                feat_names = best_5d_keys if "5D" in m_name else FEATURE_KEYS
                model.fit(X_tr, y_train, feature_names=feat_names)
            else:
                model.fit(X_tr, y_train)
            base_probs = model.predict_proba(X_te)

            # Mode 1: Zero-Shot Base (No target supervision)
            panel_base = evaluate_full_metric_panel(base_probs, y_test, c_test, y_train=y_train)
            panel_base["test_dataset"] = test_name
            panel_base["method"] = m_name
            panel_base["transfer_mode"] = "Zero-Shot Base"
            all_lodo_results.append(panel_base)

            # Mode 2: Unsupervised Target Adapted (Saerens-EM prior shift)
            em_probs, _ = run_saerens_em_binary(base_probs, pi_source=float(np.mean(y_train)), max_iter=100)
            panel_em = evaluate_full_metric_panel(em_probs, y_test, c_test, y_train=y_train)
            panel_em["test_dataset"] = test_name
            panel_em["method"] = m_name
            panel_em["transfer_mode"] = "Target Adapted (Saerens-EM)"
            all_lodo_results.append(panel_em)

    df_lodo = pd.DataFrame(all_lodo_results)
    csv_path = out_dir / f"lodo_{args.arch}_summary.csv"
    df_lodo.to_csv(csv_path, index=False)
    if out_dir.name in ["m3", "mqt"]:
        df_lodo.to_csv(out_dir.parent / f"lodo_{args.arch}_summary.csv", index=False)
    elif (out_dir / args.arch).exists():
        df_lodo.to_csv(out_dir / args.arch / f"lodo_{args.arch}_summary.csv", index=False)
    print(f"\nSaved full LODO transfer results to {csv_path}")

    # Print Macro-Mean Summary
    macro = df_lodo.groupby(["method", "transfer_mode"])[["adaptive_ece_percent", "ece_percent", "auroc", "brier"]].mean().reset_index()
    print("\n" + "=" * 85)
    print(f" LODO CROSS-DOMAIN TRANSFER: MACRO-MEAN SUMMARY [{args.arch.upper()} T_gen={args.gen_temperature}]")
    print("=" * 85)
    print(macro.to_string(index=False))
    print("=" * 85)


if __name__ == "__main__":
    main()
