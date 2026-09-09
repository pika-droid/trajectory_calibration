#!/usr/bin/env python3
"""
Focused VCPS Analysis and Interpretability Script.

Evaluates Varying-Coefficient Platt Scaling (VCPS) on a target dataset,
computing effective temperature distributions T_eff(z) = 1 / a(z),
slope separation for correct vs. incorrect answers, and feature weights.
"""

import argparse
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from trajectory_calibration.calibrators.residual import evaluate_full_metric_panel
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.trajectory import (
    CANONICAL_5D_KEYS,
    FEATURE_KEYS,
    get_stratified_split,
    load_dataset_features,
)
from trajectory_calibration.utils.helpers import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused VCPS analysis.")
    parser.add_argument(
        "--features_dir", type=str, required=True, help="Path to features dir or .pt file."
    )
    parser.add_argument(
        "--arch", type=str, default="m3", choices=["m3", "mqt"], help="Model architecture."
    )
    parser.add_argument(
        "--feature_set", type=str, default="5d", choices=["5d", "17d"], help="Feature space."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/experiments/vcps_analysis", help="Output dir."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    fine_scale = 576 if args.arch == "m3" else 256
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {args.features_dir} (arch={args.arch})...")
    df = load_dataset_features(args.features_dir, fine_scale=fine_scale)
    print(f"Loaded {len(df)} samples.")

    train_idx, test_idx = get_stratified_split(df, test_size=0.2, random_state=args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train_17d = train_df[FEATURE_KEYS].values
    y_train = train_df["is_correct"].values
    X_test_17d = test_df[FEATURE_KEYS].values
    y_test = test_df["is_correct"].values
    c_test_576 = test_df["c_576"].values

    if args.feature_set == "5d":
        feature_keys = CANONICAL_5D_KEYS
        print(f"Selected Canonical 5-D Features: {feature_keys}")
        X_train = train_df[feature_keys].values
        X_test = test_df[feature_keys].values
        vcps = VaryingCoefficientPlattScaler(
            feature_set="5d",
            random_state=args.seed,
        )
    else:
        feature_keys = FEATURE_KEYS
        X_train = X_train_17d
        X_test = X_test_17d
        vcps = VaryingCoefficientPlattScaler(feature_set="17d", random_state=args.seed)

    print("Fitting VCPS...")
    vcps.fit(X_train, y_train, feature_names=feature_keys)
    probs = vcps.predict_proba(X_test)

    panel = evaluate_full_metric_panel(probs, y_test, c_test_576, y_train=y_train)

    # Dynamic slope and effective temperature analysis
    slopes = vcps.compute_dynamic_slope(X_test)
    t_eff = vcps.get_effective_temperature(X_test)

    correct_mask = y_test == 1
    incorrect_mask = y_test == 0

    print("\n" + "=" * 60)
    print(" VCPS CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  ECE:           {panel['ece_percent']:.2f}%")
    print(f"  Adaptive ECE:  {panel['adaptive_ece_percent']:.2f}%")
    print(f"  KDE ECE:       {panel['kde_ece'] * 100:.2f}%")
    print(f"  AUROC:         {panel['auroc']:.4f}")
    print(f"  Brier Score:   {panel['brier']:.4f}")
    print(f"  Status:        {panel['status']}")
    print("-" * 60)
    print(" DYNAMIC SLOPE & TEMPERATURE INTERPRETABILITY")
    print("-" * 60)
    print(
        f"  Slope a(z) [Correct]:   {np.mean(slopes[correct_mask]):.3f} +/- {np.std(slopes[correct_mask]):.3f}"
    )
    print(
        f"  Slope a(z) [Incorrect]: {np.mean(slopes[incorrect_mask]):.3f} +/- {np.std(slopes[incorrect_mask]):.3f}"
    )
    print(
        f"  T_eff(z)   [Correct]:   {np.mean(t_eff[correct_mask]):.3f} +/- {np.std(t_eff[correct_mask]):.3f}"
    )
    print(
        f"  T_eff(z)   [Incorrect]: {np.mean(t_eff[incorrect_mask]):.3f} +/- {np.std(t_eff[incorrect_mask]):.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
