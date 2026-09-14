#!/usr/bin/env python3
"""
Exhaustive 14-Dataset Combinatorial 5D Trajectory Feature Ablation Suite.

Evaluates all 31 non-empty subsets of CANONICAL_5D_KEYS across all 14 benchmarks
for M3-LLaVA (7B) and MQT-LLaVA (7B) at greedy decoding (T = 0.0, K = 1).
Enforces the Decoupled Scaling Invariant via scikit-learn ColumnTransformer.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS, FEATURE_NAMES
from trajectory_calibration.features.loader import load_dataset_features
from trajectory_calibration.metrics.ece import compute_adaptive_ece, compute_ece
from trajectory_calibration.metrics.scoring import compute_auroc, compute_brier
from trajectory_calibration.utils.helpers import set_seed
from trajectory_calibration.utils.math import safe_clip_probs

ALL_14_DATASETS: list[str] = [
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


def generate_all_5d_subsets() -> list[list[str]]:
    """Generate all 2^5 - 1 = 31 non-empty subsets of CANONICAL_5D_KEYS ordered by cardinality."""
    subsets: list[list[str]] = []
    for k in range(1, len(CANONICAL_5D_KEYS) + 1):
        for combo in combinations(CANONICAL_5D_KEYS, k):
            subsets.append(list(combo))
    return subsets


def build_ablation_pipeline(subset: list[str]) -> Pipeline:
    """
    Construct scikit-learn Pipeline adhering strictly to Decoupled Scaling Invariant:
    - If x1 present: x1 is passed through raw/uncentered; signatures z are standardized.
    - If x1 absent: all features in subset are standardized via StandardScaler.
    """
    if "x1" in subset:
        z_cols = [c for c in subset if c != "x1"]
        if not z_cols:
            preprocessor = ColumnTransformer(
                transformers=[("x1_pass", "passthrough", ["x1"])],
                remainder="drop",
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("x1_pass", "passthrough", ["x1"]),
                    ("z_scale", StandardScaler(with_mean=True, with_std=True), z_cols),
                ],
                remainder="drop",
            )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("z_scale", StandardScaler(with_mean=True, with_std=True), subset),
            ],
            remainder="drop",
        )

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])


def fit_predict_oof(
    df: pd.DataFrame,
    subset: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Run 5-Fold Stratified Out-of-Fold Cross-Validation, pooling predictions across 100% of samples."""
    X = df[subset].copy()
    y = df["is_correct"].values.astype(int)
    n_samples = len(y)
    oof_preds = np.zeros(n_samples, dtype=np.float64)

    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or min(counts) < n_splits:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X)
    else:
        strat_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = strat_splitter.split(X, y)

    for train_idx, val_idx in splits:
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_val = X.iloc[val_idx]

        unique_tr = np.unique(y_tr)
        if len(unique_tr) < 2:
            oof_preds[val_idx] = float(unique_tr[0])
            continue

        pipeline = build_ablation_pipeline(subset)
        pipeline.fit(X_tr, y_tr)
        oof_preds[val_idx] = pipeline.predict_proba(X_val)[:, 1]

    oof_preds = np.nan_to_num(oof_preds, nan=0.5, posinf=1.0 - 1e-7, neginf=1e-7)
    return safe_clip_probs(oof_preds)


def evaluate_single_subset(
    df: pd.DataFrame,
    ds_name: str,
    subset: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate one subset on one dataset returning full metrics panel."""
    y = df["is_correct"].values.astype(int)
    oof_preds = fit_predict_oof(df, subset, n_splits=n_splits, seed=seed)

    ada_ece = float(compute_adaptive_ece(oof_preds, y, n_bins=15))
    ece = float(compute_ece(oof_preds, y, n_bins=15))
    brier = float(compute_brier(oof_preds, y))
    auroc = float(compute_auroc(oof_preds, y))

    return {
        "dataset": ds_name,
        "cardinality": len(subset),
        "subset": "+".join(subset),
        "has_x1": "x1" in subset,
        "ada_ece": ada_ece,
        "ece": ece,
        "brier": brier,
        "auroc": auroc,
        "ada_ece_percent": ada_ece * 100.0,
        "ece_percent": ece * 100.0,
    }


def compute_macro_progression_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all 31 subsets across cardinalities k in 1..5 to compute macro progression."""
    macro_by_subset = df_raw.groupby(["cardinality", "subset", "has_x1"], as_index=False).agg(
        macro_ada_ece=("ada_ece", "mean"),
        macro_ece=("ece", "mean"),
        macro_brier=("brier", "mean"),
        macro_auroc=("auroc", "mean"),
    )

    progression_rows = []
    for k in range(1, 6):
        k_sub = macro_by_subset[macro_by_subset["cardinality"] == k]
        best_row = k_sub.sort_values("macro_ada_ece").iloc[0]
        worst_row = k_sub.sort_values("macro_ada_ece", ascending=False).iloc[0]

        progression_rows.append(
            {
                "cardinality": k,
                "n_subsets": len(k_sub),
                "best_subset": best_row["subset"],
                "best_has_x1": best_row["has_x1"],
                "best_ada_ece": best_row["macro_ada_ece"],
                "best_ece": best_row["macro_ece"],
                "best_brier": best_row["macro_brier"],
                "best_auroc": best_row["macro_auroc"],
                "worst_subset": worst_row["subset"],
                "worst_ada_ece": worst_row["macro_ada_ece"],
                "mean_ada_ece": float(k_sub["macro_ada_ece"].mean()),
                "std_ada_ece": float(k_sub["macro_ada_ece"].std(ddof=0)),
                "min_ada_ece": float(k_sub["macro_ada_ece"].min()),
                "max_ada_ece": float(k_sub["macro_ada_ece"].max()),
                "mean_auroc": float(k_sub["macro_auroc"].mean()),
                "std_auroc": float(k_sub["macro_auroc"].std(ddof=0)),
            }
        )

    return pd.DataFrame(progression_rows)


def compute_loo_sensitivity_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute Leave-One-Out (LOO) marginal sensitivity relative to full 5D model."""
    full_subset_str = "+".join(CANONICAL_5D_KEYS)
    df_full = df_raw[df_raw["subset"] == full_subset_str].set_index("dataset")

    loo_rows = []
    for feat in CANONICAL_5D_KEYS:
        minus_feat = [f for f in CANONICAL_5D_KEYS if f != feat]
        minus_str = "+".join(minus_feat)
        df_minus = df_raw[df_raw["subset"] == minus_str].set_index("dataset")

        delta_ada_ece_list = []
        delta_ece_list = []
        delta_brier_list = []
        delta_auroc_list = []
        minus_ada_ece_list = []
        minus_auroc_list = []

        for ds in df_full.index:
            f_ada = df_full.loc[ds, "ada_ece"]
            f_ece = df_full.loc[ds, "ece"]
            f_brier = df_full.loc[ds, "brier"]
            f_auroc = df_full.loc[ds, "auroc"]

            m_ada = df_minus.loc[ds, "ada_ece"]
            m_ece = df_minus.loc[ds, "ece"]
            m_brier = df_minus.loc[ds, "brier"]
            m_auroc = df_minus.loc[ds, "auroc"]

            delta_ada_ece_list.append(m_ada - f_ada)
            delta_ece_list.append(m_ece - f_ece)
            delta_brier_list.append(m_brier - f_brier)
            delta_auroc_list.append(m_auroc - f_auroc)
            minus_ada_ece_list.append(m_ada)
            minus_auroc_list.append(m_auroc)

        loo_rows.append(
            {
                "feature_dropped": feat,
                "feature_name": FEATURE_NAMES.get(feat, feat),
                "subset_minus_j": minus_str,
                "macro_ada_ece_minus_j": float(np.mean(minus_ada_ece_list)),
                "macro_delta_ada_ece": float(np.mean(delta_ada_ece_list)),
                "macro_delta_ece": float(np.mean(delta_ece_list)),
                "macro_delta_brier": float(np.mean(delta_brier_list)),
                "macro_delta_auroc": float(np.mean(delta_auroc_list)),
                "macro_auroc_minus_j": float(np.mean(minus_auroc_list)),
            }
        )

    return pd.DataFrame(loo_rows).sort_values("macro_delta_ada_ece", ascending=False)


def compute_per_dataset_best_subsets(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Find the best-performing subset for each dataset at each cardinality k in 1..5."""
    rows = []
    datasets = df_raw["dataset"].unique()
    for ds in datasets:
        ds_sub = df_raw[df_raw["dataset"] == ds]
        for k in range(1, 6):
            k_sub = ds_sub[ds_sub["cardinality"] == k]
            best_k = k_sub.sort_values("ada_ece").iloc[0]
            rows.append(
                {
                    "dataset": ds,
                    "cardinality": k,
                    "best_subset": best_k["subset"],
                    "ada_ece": best_k["ada_ece"],
                    "ece": best_k["ece"],
                    "brier": best_k["brier"],
                    "auroc": best_k["auroc"],
                }
            )
    return pd.DataFrame(rows)


def run_combinatorial_ablation(
    arch: str = "m3",
    features_dir: str = "data/features",
    gen_temp: float = 0.0,
    datasets: list[str] | None = None,
    out_dir_path: str = "results/experiments/ablation_5d",
    seed: int = 42,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete combinatorial ablation across datasets, saving structured CSVs."""
    set_seed(seed)
    ds_list = datasets or ALL_14_DATASETS
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fine_scale = 576 if arch == "m3" else 256

    print(f"\n{'=' * 80}")
    print(f" 14-DATASET COMBINATORIAL 5D ABLATION STUDY [{arch.upper()}] (T={gen_temp})")
    print(f" Datasets ({len(ds_list)}): {', '.join(ds_list)}")
    print(f"{'=' * 80}\n", flush=True)

    loaded_dfs: dict[str, pd.DataFrame] = {}
    for ds in ds_list:
        try:
            df = load_dataset_features(
                features_dir,
                ds_name=ds,
                arch=arch,
                gen_temperature=gen_temp,
                fine_scale=fine_scale,
            )
            loaded_dfs[ds] = df
            print(f" Loaded {ds:16s}: {len(df):5d} samples")
        except Exception as exc:
            print(f" [ERROR] Could not load {ds}: {exc}")

    if not loaded_dfs:
        raise RuntimeError(f"No datasets loaded for architecture {arch}.")

    all_subsets = generate_all_5d_subsets()
    total_evals = len(loaded_dfs) * len(all_subsets)
    print(
        f"\nEvaluating {len(all_subsets)} subsets across {len(loaded_dfs)} benchmarks ({total_evals} total evaluations)..."
    )

    results: list[dict[str, object]] = []
    completed = 0
    for ds_name, df in loaded_dfs.items():
        for subset in all_subsets:
            row = evaluate_single_subset(df, ds_name, subset, n_splits=n_splits, seed=seed)
            results.append(row)
            completed += 1
            if completed % 100 == 0 or completed == total_evals:
                print(
                    f" Progress: {completed}/{total_evals} evaluations completed ({completed / total_evals * 100:.1f}%)",
                    flush=True,
                )

    df_raw = pd.DataFrame(results)
    df_raw.to_csv(out_dir / f"ablation_5d_{arch}_raw_all_combinations.csv", index=False)

    df_macro = compute_macro_progression_summary(df_raw)
    df_macro.to_csv(out_dir / f"ablation_5d_{arch}_macro_progression.csv", index=False)

    df_loo = compute_loo_sensitivity_summary(df_raw)
    df_loo.to_csv(out_dir / f"ablation_5d_{arch}_loo_sensitivity.csv", index=False)

    df_best = compute_per_dataset_best_subsets(df_raw)
    df_best.to_csv(out_dir / f"ablation_5d_{arch}_per_dataset_best_subsets.csv", index=False)

    print(f"\nSuccessfully generated and saved ablation artifacts to {out_dir}:")
    print(f" - ablation_5d_{arch}_raw_all_combinations.csv ({len(df_raw)} rows)")
    print(f" - ablation_5d_{arch}_macro_progression.csv ({len(df_macro)} rows)")
    print(f" - ablation_5d_{arch}_loo_sensitivity.csv ({len(df_loo)} rows)")
    print(f" - ablation_5d_{arch}_per_dataset_best_subsets.csv ({len(df_best)} rows)\n")

    return df_raw, df_macro, df_loo, df_best


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for combinatorial ablation study."""
    parser = argparse.ArgumentParser(
        description="Run Exhaustive 14-Dataset Combinatorial 5D Trajectory Feature Ablation Study."
    )
    parser.add_argument(
        "--arch", type=str, default="m3", choices=["m3", "mqt"], help="VLM architecture."
    )
    parser.add_argument(
        "--features_dir", type=str, default="data/features", help="Base directory for features."
    )
    parser.add_argument(
        "--gen_temperature", type=float, default=0.0, help="Generation temperature."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_14_DATASETS, help="Datasets to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/ablation_5d",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cross-validation.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_combinatorial_ablation(
        arch=args.arch,
        features_dir=args.features_dir,
        gen_temp=args.gen_temperature,
        datasets=args.datasets,
        out_dir_path=args.output_dir,
        seed=args.seed,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()
