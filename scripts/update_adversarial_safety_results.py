#!/usr/bin/env python3
"""
Recomputes results/adversarial_safety_benchmark_results.csv using repaired features.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trajectory_calibration.calibrators.baselines import (
    AdaptiveTemperatureScaling,
    BetaCalibrator,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("update_adversarial_results")

ROOT = Path(__file__).resolve().parent.parent
CSV_OUT = ROOT / "results" / "adversarial_safety_benchmark_results.csv"


def _backup_csv(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not bak.exists():
        shutil.copyfile(path, bak)
        logger.info(f"Backed up {path} to {bak}")


def _evaluate_adversarial_dataset(
    ds: str,
    arch: str,
    fine_scale: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluates all 13 single-pass methods on an adversarial dataset."""
    df = load_dataset_features("data/features", ds_name=ds, arch=arch, fine_scale=fine_scale)
    train_idx, test_idx = get_stratified_split(df, test_size=0.2, random_state=seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train_17d = train_df[FEATURE_KEYS].values
    y_train = train_df["is_correct"].values
    X_test_17d = test_df[FEATURE_KEYS].values
    y_test = test_df["is_correct"].values
    c_test_fine = test_df["c_fine"].values

    X_train_5d = train_df[CANONICAL_5D_KEYS].values
    X_test_5d = test_df[CANONICAL_5D_KEYS].values

    methods: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {
        "Naive Confidence (NC)": (NaiveConfidenceEstimator(), X_train_17d, X_test_17d),
        "Temperature Scaling (TS)": (TemperatureScalingEstimator(), X_train_17d, X_test_17d),
        "Platt Scaling (1D)": (PlattScalingEstimator(), X_train_17d, X_test_17d),
        "Quadratic Platt (1D)": (QuadraticPlattScaler(), X_train_17d, X_test_17d),
        "Beta Calibration (1D)": (BetaCalibrator(), X_train_17d, X_test_17d),
        "Spline Calibration": (SplineCalibrator(), X_train_17d, X_test_17d),
        "Adaptive TS (ATS)": (AdaptiveTemperatureScaling(), X_train_17d, X_test_17d),
        "Trajectory LR": (TrajectoryLREstimator(fit_intercept=True), X_train_17d, X_test_17d),
        "Trajectory Platt (5D)": (
            TrajectoryPlattScaler(n_features=len(CANONICAL_5D_KEYS)),
            X_train_5d,
            X_test_5d,
        ),
        "Trajectory Platt (17D)": (
            TrajectoryPlattScaler(n_features=len(FEATURE_KEYS)),
            X_train_17d,
            X_test_17d,
        ),
        "Residual Calibrator": (
            ResidualTrajectoryCalibrator(random_state=seed),
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

    results = []
    for m_name, (model, X_tr, X_te) in methods.items():
        if isinstance(model, VaryingCoefficientPlattScaler):
            feat_names = CANONICAL_5D_KEYS if "5D" in m_name else FEATURE_KEYS
            model.fit(X_tr, y_train, feature_names=feat_names)
        else:
            model.fit(X_tr, y_train)

        probs = model.predict_proba(X_te)
        panel = evaluate_full_metric_panel(probs, y_test, c_test_fine, y_train=y_train)
        panel["arch"] = arch.upper()
        panel["dataset"] = ds
        panel["method"] = m_name
        results.append(panel)

    return results


def update_adversarial_csv() -> pd.DataFrame:
    """Updates results/adversarial_safety_benchmark_results.csv with repaired metrics."""
    set_seed(42)
    _backup_csv(CSV_OUT)
    all_rows = []

    for arch, fine_scale in [("m3", 576), ("mqt", 256)]:
        for ds in ["avqa", "vllm-safety"]:
            logger.info(f"Evaluating {arch.upper()} on {ds}...")
            rows = _evaluate_adversarial_dataset(ds, arch, fine_scale)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    logger.info(f"Successfully wrote {len(df)} rows to {CSV_OUT}")
    return df


def main() -> None:
    update_adversarial_csv()


if __name__ == "__main__":
    main()
