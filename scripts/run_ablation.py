#!/usr/bin/env python3
"""
Trajectory Feature Sensitivity & Ablation Study Suite (Week 16/17 Experiments).

Implements:
- Experiment A: Univariate Feature Ranking (evaluates calibration power of each of the 17 features in isolation)
- Experiment B: Leave-One-Out (LOO) Feature Degradation Sensitivity (measures Delta ECE when dropping feature j)
- Experiment C: Pareto-Optimal Subset Progression (K in {1, 2, 3, 5, 7, 10, 17} with VIF collinearity tracking)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.calibrators.residual import evaluate_full_metric_panel
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.trajectory import (
    FEATURE_KEYS,
    calculate_vif,
    get_stratified_split,
    load_dataset_features,
    select_best_5d_subset,
    sigmoid,
)
from trajectory_calibration.metrics.calibration import compute_adaptive_ece, compute_auroc
from trajectory_calibration.utils.helpers import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trajectory Feature Sensitivity and Ablation Study.")
    parser.add_argument("--features_dir", type=str, default="data/features", help="Base directory of features.")
    parser.add_argument("--arch", type=str, default="m3", choices=["m3", "mqt"], help="Model architecture.")
    parser.add_argument("--gen_temperature", type=float, default=0.0, help="Decoding temperature.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pope", "scienceqa", "textvqa", "vizwiz-vqa"],
        help="Datasets to evaluate."
    )
    parser.add_argument("--output_dir", type=str, default="results/experiments/ablation", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    fine_scale = 576 if args.arch == "m3" else 256
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets
    dfs = []
    for ds in args.datasets:
        try:
            df = load_dataset_features(args.features_dir, ds_name=ds, arch=args.arch, gen_temperature=args.gen_temperature, fine_scale=fine_scale)
            df["dataset_source"] = ds
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {ds}: {e}")

    if not dfs:
        print("No datasets loaded.")
        return

    pooled_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} datasets for ablation study ({args.arch.upper()}). Total samples: {len(pooled_df)}")

    tr_idx, te_idx = get_stratified_split(pooled_df, test_size=0.2, random_state=args.seed)
    train_df = pooled_df.iloc[tr_idx].reset_index(drop=True)
    test_df = pooled_df.iloc[te_idx].reset_index(drop=True)

    X_train_full = train_df[FEATURE_KEYS].values
    y_train = train_df["is_correct"].values
    X_test_full = test_df[FEATURE_KEYS].values
    y_test = test_df["is_correct"].values
    c_test = test_df["c_576"].values

    # =========================================================================
    # EXPERIMENT A: UNIVARIATE FEATURE RANKING
    # =========================================================================
    print("\n" + "=" * 80)
    print(f" EXPERIMENT A: UNIVARIATE FEATURE RANKING [{args.arch.upper()}]")
    print("=" * 80)

    univariate_results = []
    for j, f_key in enumerate(FEATURE_KEYS):
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(X_train_full[:, [j]], y_train)
        preds = lr.predict_proba(X_test_full[:, [j]])[:, 1]
        ada_ece = compute_adaptive_ece(preds, y_test, n_bins=15) * 100.0
        auroc = compute_auroc(preds, y_test)
        univariate_results.append({
            "feature": f_key,
            "adaptive_ece_percent": ada_ece,
            "auroc": auroc,
            "weight": float(lr.coef_[0][0]),
        })

    df_uni = pd.DataFrame(univariate_results).sort_values("adaptive_ece_percent")
    df_uni.to_csv(out_dir / f"exp_a_{args.arch}_univariate_ranking.csv", index=False)
    print(df_uni.round(3).to_string(index=False))

    # =========================================================================
    # EXPERIMENT B: LEAVE-ONE-OUT (LOO) SENSITIVITY
    # =========================================================================
    print("\n" + "=" * 80)
    print(f" EXPERIMENT B: LEAVE-ONE-OUT (LOO) SENSITIVITY [{args.arch.upper()}]")
    print("=" * 80)

    full_lr = LogisticRegression(C=1.0, max_iter=200).fit(X_train_full, y_train)
    full_preds = full_lr.predict_proba(X_test_full)[:, 1]
    base_ece = compute_adaptive_ece(full_preds, y_test, n_bins=15) * 100.0

    loo_results = []
    for j, f_key in enumerate(FEATURE_KEYS):
        subset_cols = [k for k in range(len(FEATURE_KEYS)) if k != j]
        lr_loo = LogisticRegression(C=1.0, max_iter=200).fit(X_train_full[:, subset_cols], y_train)
        loo_preds = lr_loo.predict_proba(X_test_full[:, subset_cols])[:, 1]
        loo_ece = compute_adaptive_ece(loo_preds, y_test, n_bins=15) * 100.0
        delta_ece = loo_ece - base_ece
        loo_results.append({
            "feature_dropped": f_key,
            "baseline_ece": base_ece,
            "loo_ece": loo_ece,
            "delta_ece": delta_ece,
        })

    df_loo = pd.DataFrame(loo_results).sort_values("delta_ece", ascending=False)
    df_loo.to_csv(out_dir / f"exp_b_{args.arch}_loo_sensitivity.csv", index=False)
    print(df_loo.round(3).to_string(index=False))

    # =========================================================================
    # EXPERIMENT C: PARETO SUBSET PROGRESSION (K in {1, 2, 3, 5, 7, 10, 17})
    # =========================================================================
    print("\n" + "=" * 80)
    print(f" EXPERIMENT C: PARETO SUBSET PROGRESSION [{args.arch.upper()}]")
    print("=" * 80)

    k_progression = [1, 2, 3, 5, 7, 10, 17]
    pareto_results = []
    selected_features = ["x1"]  # always root anchor

    for k in k_progression:
        if k == 1:
            current_subset = ["x1"]
        elif k == 17:
            current_subset = FEATURE_KEYS
        else:
            # Stepwise greedy addition
            while len(selected_features) < k:
                best_cand = None
                best_cand_ece = 1e9
                for cand in FEATURE_KEYS:
                    if cand in selected_features:
                        continue
                    cand_cols = [FEATURE_KEYS.index(f) for f in selected_features + [cand]]
                    lr_cand = LogisticRegression(C=1.0, max_iter=200).fit(X_train_full[:, cand_cols], y_train)
                    c_preds = lr_cand.predict_proba(X_test_full[:, cand_cols])[:, 1]
                    c_ece = compute_adaptive_ece(c_preds, y_test, n_bins=15)
                    if c_ece < best_cand_ece:
                        best_cand_ece = c_ece
                        best_cand = cand
                if best_cand:
                    selected_features.append(best_cand)
            current_subset = selected_features[:k]

        cols = [FEATURE_KEYS.index(f) for f in current_subset]
        lr_k = LogisticRegression(C=1.0, max_iter=200).fit(X_train_full[:, cols], y_train)
        k_preds = lr_k.predict_proba(X_test_full[:, cols])[:, 1]
        k_panel = evaluate_full_metric_panel(k_preds, y_test, c_test, y_train=y_train)

        vif = calculate_vif(X_train_full[:, cols]) if len(cols) > 1 else 1.0
        pareto_results.append({
            "K_features": k,
            "subset": ", ".join(current_subset),
            "adaptive_ece_percent": k_panel["adaptive_ece_percent"],
            "auroc": k_panel["auroc"],
            "brier_score": k_panel.get("brier", k_panel.get("brier_score", 0.0)),
            "max_vif": vif,
        })

    df_pareto = pd.DataFrame(pareto_results)
    df_pareto.to_csv(out_dir / f"exp_c_{args.arch}_pareto_progression.csv", index=False)
    print(df_pareto.round(3).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
