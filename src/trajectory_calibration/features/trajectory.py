"""
Trajectory Feature Calibration Methods.

Extracts the 17-D multi-scale trajectory signature, converts .pt checkpoints
into clean DataFrames, performs stratified 80/20 train/test splitting,
applies forward stepwise 5-D feature selection with VIF collinearity guards,
and evaluates model diagnostic status (VALID, COLLAPSED, SCRAMBLED).

Supports the standardized dataset organization format:
    <data_dir>/<m3_llava|mqt_llava>/temp_<T>/<dataset>.pt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from trajectory_calibration.metrics.calibration import (
    compute_brier,
    compute_ece,
    compute_mce,
    compute_nll,
    compute_prediction_std,
)
from trajectory_calibration.utils.helpers import clean_text, safe_torch_load

logger = logging.getLogger("trajectory_calibration.features.trajectory")

FEATURE_NAMES: dict[str, str] = {
    "x1": "Final Logit",
    "x3": "Confidence Gain",
    "x4": "Monotonicity Count",
    "x6": "Confidence Variance",
    "x8": "Scale Dip Depth",
    "x9": "Log-Scale Slope",
    "x10": "Logprob Gain",
    "x11": "Logprob Variance",
    "x12": "Logprob Acceleration",
    "x13": "Discrete Answer Stability",
    "x14": "Relative Gain Ratio",
    "x15": "Mid-Fine Gain Contrast",
    "x17": "End-Scale Spike Ratio",
    "x18": "Scale Entropy Slope",
    "x19": "Relative Margin Growth",
    "x20": "Answer Flip Frequency",
    "x21": "Logit Trajectory Convexity",
    "x22": "First-to-Final Jump Ratio",
}

FEATURE_KEYS: list[str] = list(FEATURE_NAMES.keys())


def get_logits(confs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes bounded inverse-sigmoid logits: ln(c / (1 - c))."""
    c = np.clip(confs, eps, 1.0 - eps)
    return np.log(c / (1.0 - c))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def variance_inflation_factor(exog: np.ndarray, exog_idx: int) -> float:
    """Calculates Variance Inflation Factor (VIF) for collinearity detection."""
    try:
        from sklearn.linear_model import LinearRegression

        y = exog[:, exog_idx]
        X = np.delete(exog, exog_idx, axis=1)
        r_squared = LinearRegression().fit(X, y).score(X, y)
        return float(1.0 / max(1.0 - r_squared, 1e-6))
    except Exception:
        return 1.0


def compute_features_from_sample(item: dict[str, Any], fine_scale: int = 576) -> dict[str, float]:
    """
    Extracts all 17 trajectory features from a multi-scale inference sample.
    """
    feats = item.get("features", {})
    scales = [1, 9, 36, 144, fine_scale]

    confs = []
    margins = []
    answers = []
    accuracies = []

    for s in scales:
        s_data = feats.get(s, feats.get(str(s), {}))
        confs.append(float(s_data.get("conf_softmax", 0.5)))
        margins.append(float(s_data.get("margin", 0.0)))
        answers.append(str(s_data.get("answer", "")).strip())
        accuracies.append(float(s_data.get("vqa_accuracy", 0.0)))

    c_arr = np.array(confs, dtype=np.float64)
    m_arr = np.array(margins, dtype=np.float64)
    eps = 1e-7

    lp_arr = np.log(np.clip(c_arr, eps, 1.0))
    scale_log = np.log(np.array(scales, dtype=np.float64))

    # x1: Final Logit
    x1 = float(np.log(np.clip(c_arr[-1], eps, 1.0 - eps) / (1.0 - np.clip(c_arr[-1], eps, 1.0 - eps))))
    # x3: Confidence Gain
    x3 = float(c_arr[-1] - c_arr[1])
    # x4: Monotonicity Count
    x4 = float(np.sum(c_arr[1:] > c_arr[:-1]))
    # x6: Confidence Variance
    x6 = float(np.var(c_arr))
    # x8: Scale Dip Depth
    coarse_max = max(c_arr[0], c_arr[1])
    mid_min = min(c_arr[2], c_arr[3])
    x8 = float(max(0.0, coarse_max - mid_min))

    # x9: Log-Scale Slope
    denom = np.sum((scale_log - np.mean(scale_log)) ** 2)
    x9 = float(np.sum((scale_log - np.mean(scale_log)) * (c_arr - np.mean(c_arr))) / denom) if denom > 0 else 0.0

    # x10: Logprob Gain
    x10 = float(lp_arr[-1] - lp_arr[1])
    # x11: Logprob Variance
    x11 = float(np.var(lp_arr))
    # x12: Logprob Acceleration
    x12 = float((lp_arr[-1] - lp_arr[3]) - (lp_arr[3] - lp_arr[2]))

    # x13: Discrete Answer Stability
    norm_answers = [clean_text(a) for a in answers]
    unique_answers = set(norm_answers)
    x13 = float(1.0 / max(1, len(unique_answers)))

    # x14: Relative Gain Ratio
    x14 = float(c_arr[-1] / (c_arr[1] + eps))
    # x15: Mid-Fine Gain Contrast
    x15 = float((c_arr[-1] - c_arr[3]) - (c_arr[3] - c_arr[1]))
    # x17: End-Scale Spike Ratio
    x17 = float(c_arr[-1] - np.mean(c_arr[:-1]))

    # x18: Scale Entropy Slope
    bin_entropy = -(c_arr * np.log(np.clip(c_arr, eps, 1.0)) + (1.0 - c_arr) * np.log(np.clip(1.0 - c_arr, eps, 1.0)))
    x18 = float(np.sum((scale_log - np.mean(scale_log)) * (bin_entropy - np.mean(bin_entropy))) / denom) if denom > 0 else 0.0

    # x19: Relative Margin Growth
    x19 = float(m_arr[-1] / (m_arr[1] + eps))
    # x20: Answer Flip Frequency
    flips = sum(1 for i in range(len(norm_answers) - 1) if norm_answers[i] != norm_answers[i + 1])
    x20 = float(flips / (len(norm_answers) - 1)) if len(norm_answers) > 1 else 0.0
    # x21: Logit Trajectory Convexity
    x21 = float((c_arr[-1] - c_arr[3]) - (c_arr[3] - c_arr[2]))
    # x22: First-to-Final Jump Ratio
    x22 = float((c_arr[-1] - c_arr[0]) / (c_arr[-1] + eps))

    acc_final = accuracies[-1]
    if "is_correct" in item:
        is_correct = int(item["is_correct"])
    else:
        is_correct = 1 if acc_final >= 0.5 else 0

    return {
        "x1": x1,
        "x3": x3,
        "x4": x4,
        "x6": x6,
        "x8": x8,
        "x9": x9,
        "x10": x10,
        "x11": x11,
        "x12": x12,
        "x13": x13,
        "x14": x14,
        "x15": x15,
        "x17": x17,
        "x18": x18,
        "x19": x19,
        "x20": x20,
        "x21": x21,
        "x22": x22,
        "c_576": float(c_arr[-1]),
        "is_correct": is_correct,
        "vqa_accuracy": float(acc_final),
        "question_id": item.get("question_id", 0),
        "answer_type": item.get("answer_type", "open"),
    }


def find_feature_file(
    base_path: Path | str,
    ds_name: str | None = None,
    arch: str = "m3",
    gen_temperature: float = 0.0,
) -> Path:
    """
    Resolves feature file path matching standard layout:
        1. base / arch_folder / f"temp_{gen_temperature}" / f"{ds_name}.pt"
        2. base / ds_name / "full_extracted_features.pt"
        3. base / f"{ds_name}.pt"
        4. Direct file path if base_path is a file
    """
    path = Path(base_path)
    if path.is_file():
        return path

    arch_folder = "mqt_llava" if "mqt" in arch.lower() else "m3_llava"
    temp_folder = f"temp_{gen_temperature:.1f}" if gen_temperature == int(gen_temperature) else f"temp_{gen_temperature}"

    candidates = []
    if ds_name:
        candidates.extend([
            path / arch_folder / temp_folder / f"{ds_name}.pt",
            path / arch_folder / "temp_0.0" / f"{ds_name}.pt",
            path / arch_folder / f"{ds_name}.pt",
            path / ds_name / "full_extracted_features.pt",
            path / f"{ds_name}.pt",
            path / f"{ds_name}_5scale.pt",
        ])
    else:
        candidates.extend([
            path / "full_extracted_features.pt",
        ])

    for cand in candidates:
        if cand.exists():
            return cand

    # If no candidate exists, return the primary preferred path
    return candidates[0] if candidates else path


def load_dataset_features(
    features_dir_or_file: Path | str,
    ds_name: str | None = None,
    arch: str = "m3",
    gen_temperature: float = 0.0,
    fine_scale: int | None = None,
) -> pd.DataFrame:
    """
    Loads dataset features from .pt file and builds a Pandas DataFrame.
    """
    pt_path = find_feature_file(features_dir_or_file, ds_name=ds_name, arch=arch, gen_temperature=gen_temperature)
    if not pt_path.exists():
        raise FileNotFoundError(f"Feature file could not be located at: {pt_path.resolve()}")

    if fine_scale is None:
        fine_scale = 256 if ("mqt" in arch.lower() or "mqt" in str(pt_path).lower()) else 576

    raw_data = safe_torch_load(pt_path)
    rows = []
    for item in raw_data:
        feat_dict = compute_features_from_sample(item, fine_scale=fine_scale)
        rows.append(feat_dict)

    df = pd.DataFrame(rows)

    # POPE Deduplication Guard
    if "question_id" in df.columns:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
        if len(df) < initial_len:
            logger.info(f"Deduplicated dataset: {initial_len} -> {len(df)} unique question_ids.")

    return df


def get_stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generates an 80/20 train/test stratified split on answer_type x is_correct."""
    indices = np.arange(len(df))

    if "answer_type" in df.columns and len(df["answer_type"].unique()) > 1:
        strat_key = df["answer_type"].astype(str) + "_" + df["is_correct"].astype(str)
        counts = strat_key.value_counts()
        if (counts < 2).any():
            strat_key = df["is_correct"].values
    else:
        strat_key = df["is_correct"].values

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=strat_key
    )
    return train_idx, test_idx


def select_best_5d_subset(
    X_train: np.ndarray, y_train: np.ndarray, feature_keys: list[str]
) -> list[str]:
    """
    Forward stepwise selection algorithm to pick the top 5 features minimizing NLL.
    """
    selected = ["x1"] if "x1" in feature_keys else [feature_keys[0]]
    remaining = [k for k in feature_keys if k not in selected]

    while len(selected) < min(5, len(feature_keys)) and remaining:
        best_candidate = None
        best_nll = float("inf")

        for candidate in remaining:
            trial = selected + [candidate]
            trial_indices = [feature_keys.index(k) for k in trial]
            X_trial = X_train[:, trial_indices]

            # VIF Collinearity filter
            if X_trial.shape[1] > 1:
                vifs = [variance_inflation_factor(X_trial, i) for i in range(X_trial.shape[1])]
                if any(v > 10.0 for v in vifs):
                    continue

            try:
                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
                lr.fit(X_trial, y_train)
                probs = lr.predict_proba(X_trial)[:, 1]
                nll = compute_nll(probs, y_train)
                if nll < best_nll:
                    best_nll = nll
                    best_candidate = candidate
            except Exception:
                continue

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            selected.append(remaining.pop(0))

    return selected


def generate_mock_df(
    ds_name: str = "mock_ds", n_samples: int = 100, seed: int = 42, fine_scale: int = 576
) -> pd.DataFrame:
    """Generates synthetic 17-D feature matrices for offline CPU test execution."""
    rng = np.random.RandomState(seed)
    scales = [1, 9, 36, 144, fine_scale]

    raw_items = []
    for i in range(n_samples):
        is_corr = 1 if rng.rand() > 0.35 else 0
        base_c = rng.uniform(0.65, 0.95) if is_corr else rng.uniform(0.25, 0.65)

        feats = {}
        for s in scales:
            c_val = np.clip(base_c + rng.normal(0.0, 0.05), 0.05, 0.98)
            m_val = rng.uniform(0.1, 0.5)
            ans = "yes" if is_corr else ("no" if rng.rand() > 0.5 else "maybe")
            acc = float(is_corr) if s == fine_scale else float(rng.rand() > 0.4)
            feats[s] = {
                "conf_softmax": float(c_val),
                "margin": float(m_val),
                "answer": ans,
                "vqa_accuracy": acc,
            }

        raw_items.append({
            "question_id": 100000 + i,
            "features": feats,
            "answer_type": "open",
        })

    rows = [compute_features_from_sample(item, fine_scale=fine_scale) for item in raw_items]
    return pd.DataFrame(rows)


def evaluate_model_diagnostics(
    probs_test: np.ndarray,
    y_test: np.ndarray,
    c_test: np.ndarray,
    y_train: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Evaluates ECE, MCE, Brier, Brier Gain, Prediction Std, and Spearman Rank Correlation.
    """
    probs_test = np.asarray(probs_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    c_test = np.asarray(c_test, dtype=np.float64)

    ece = compute_ece(probs_test, y_test)
    mce = compute_mce(probs_test, y_test)
    brier = compute_brier(probs_test, y_test)
    nll = compute_nll(probs_test, y_test)
    prob_std = compute_prediction_std(probs_test)

    base_rate = np.mean(y_train) if y_train is not None else np.mean(y_test)
    base_brier = float(np.mean((base_rate - y_test) ** 2))
    brier_gain = float(base_brier - brier)

    if len(probs_test) > 5 and np.std(probs_test) > 1e-6 and np.std(c_test) > 1e-6:
        rho, _ = stats.spearmanr(probs_test, c_test)
        rho = float(rho) if not np.isnan(rho) else 1.0
    else:
        rho = 1.0

    if prob_std < 0.02 or brier_gain <= 0.001:
        status = "COLLAPSED"
    elif rho < 0.10:
        status = "SCRAMBLED"
    else:
        status = "VALID"

    return {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "brier_gain": brier_gain,
        "nll": nll,
        "prob_std": prob_std,
        "spearman_rho": rho,
        "status": status,
    }


def calculate_vif(X: np.ndarray) -> float:
    """Calculates maximum Variance Inflation Factor across feature columns."""
    if X.shape[1] <= 1:
        return 1.0
    try:
        vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        return float(np.max(vifs))
    except Exception:
        return 1.0
