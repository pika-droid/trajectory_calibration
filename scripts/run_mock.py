#!/usr/bin/env python3
"""
Quick CPU smoke test for Trajectory Calibration.

Runs end-to-end fitting and evaluation across distinct single-pass calibrators
using pilot_features_1k in ~3 seconds with zero GPU requirements.
"""

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

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
    set_seed(42)
    print("=" * 70)
    print(" TRAJECTORY CALIBRATION — CPU SMOKE TEST (pilot_features_1k)")
    print("=" * 70)

    pilot_dir = Path(__file__).resolve().parent.parent / "pilot_features_1k" / "m3" / "mock_vqav2"
    pt_file = pilot_dir / "full_extracted_features.pt"

    if not pt_file.exists():
        print(f"Error: Pilot file not found at {pt_file}")
        sys.exit(1)

    print(f"Loading pilot features from: {pt_file}...")
    df = load_dataset_features(pt_file, fine_scale=576)
    print(f"Loaded {len(df)} samples. Features: {len(FEATURE_KEYS)} trajectory features (1 anchor + 16 signatures).")

    train_idx, test_idx = get_stratified_split(df, test_size=0.2, random_state=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train_17d = train_df[FEATURE_KEYS].values
    y_train = train_df["is_correct"].values
    X_test_17d = test_df[FEATURE_KEYS].values
    y_test = test_df["is_correct"].values
    c_test_576 = test_df["c_576"].values

    # Stepwise 5-D selection
    best_5d_keys = select_best_5d_subset(X_train_17d, y_train, FEATURE_KEYS)
    print(f"Selected 5-D Subset: {best_5d_keys}")
    X_train_5d = train_df[best_5d_keys].values
    X_test_5d = test_df[best_5d_keys].values

    calibrators = {
        "Naive Confidence (NC)": (NaiveConfidenceEstimator(), X_train_17d, X_test_17d),
        "Temperature Scaling (TS)": (TemperatureScalingEstimator(), X_train_17d, X_test_17d),
        "Platt Scaling (1D)": (PlattScalingEstimator(), X_train_17d, X_test_17d),
        "Quadratic Platt (Logit-Only)": (QuadraticPlattScaler(), X_train_17d, X_test_17d),
        "Beta Calibration": (BetaCalibrator(), X_train_17d, X_test_17d),
        "Trajectory LR": (TrajectoryLREstimator(fit_intercept=True), X_train_5d, X_test_5d),
        "Trajectory LR (No Bias)": (TrajectoryLREstimator(fit_intercept=False), X_train_5d, X_test_5d),
        "Spline Calibration": (SplineCalibrator(), X_train_17d, X_test_17d),
        "Adaptive TS (ATS)": (AdaptiveTemperatureScaling(), X_train_5d, X_test_5d),
        "Probability Margin (1D)": (ProbabilityMarginEstimator(), X_train_17d, X_test_17d),
        "MSSC (Multi-Scale Proxy)": (MultiScaleSemanticConsistency(), X_train_17d, X_test_17d),
        "MSE-EIGEN (Multi-Scale)": (MultiScaleEigenVariance(), X_train_17d, X_test_17d),
        "Residual Calibrator": (ResidualTrajectoryCalibrator(), X_train_5d, X_test_5d),
        "VCPS-5D (Our Method)": (VaryingCoefficientPlattScaler(feature_set="5d"), X_train_5d, X_test_5d),
        "VCPS-17D (Our Method)": (VaryingCoefficientPlattScaler(feature_set="17d"), X_train_17d, X_test_17d),
    }

    print("\n" + "-" * 85)
    print(f"{'Method':<28} | {'ECE (%)':<8} | {'Ada-ECE (%)':<11} | {'AUROC':<7} | {'Brier':<7} | {'Status':<9}")
    print("-" * 85)

    for name, (model, X_tr, X_te) in calibrators.items():
        if isinstance(model, VaryingCoefficientPlattScaler):
            feat_names = best_5d_keys if "5D" in name else FEATURE_KEYS
            model.fit(X_tr, y_train, feature_names=feat_names)
        else:
            model.fit(X_tr, y_train)
        probs = model.predict_proba(X_te)
        panel = evaluate_full_metric_panel(probs, y_test, c_test_576, y_train=y_train)
        print(f"{name:<28} | {panel['ece_percent']:<8.2f} | {panel['adaptive_ece_percent']:<11.2f} | {panel['auroc']:<7.3f} | {panel['brier']:<7.4f} | {panel['status']:<9}")

    print("-" * 85)
    print(" SMOKE TEST PASSED! All distinct single-pass methods fit and evaluated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
