"""
Forward stepwise feature selection and Variance Inflation Factor (VIF) collinearity filters.
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from trajectory_calibration.metrics.scoring import compute_nll

logger = logging.getLogger("trajectory_calibration.features.selection")


def variance_inflation_factor(exog: np.ndarray, exog_idx: int) -> float:
    """
    Calculates Variance Inflation Factor (VIF) for collinearity detection on column exog_idx.
    """
    if exog.ndim != 2 or exog.shape[1] <= 1:
        return 1.0

    try:
        y = exog[:, exog_idx]
        X = np.delete(exog, exog_idx, axis=1)
        r_squared = LinearRegression().fit(X, y).score(X, y)
        return float(1.0 / max(1.0 - r_squared, 1e-6))
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.debug(f"Linear regression failed during VIF calculation: {e}. Defaulting to 1.0")
        return 1.0


def calculate_vif(X: np.ndarray) -> float:
    """Calculates maximum Variance Inflation Factor across all feature columns."""
    if X.ndim != 2 or X.shape[1] <= 1:
        return 1.0
    try:
        vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        return float(np.max(vifs))
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.debug(f"VIF computation failed: {e}. Defaulting to 1.0")
        return 1.0


def select_best_5d_subset(
    X_train: np.ndarray, y_train: np.ndarray, feature_keys: list[str]
) -> list[str]:
    """
    Forward stepwise selection algorithm to pick the top 5 trajectory signatures minimizing NLL.

    Strictly anchors to 'x1' (Final Logit) to prevent prediction collapse, selecting
    5 trajectory signatures (total 6 features: x1 + 5 signatures).
    """
    selected = ["x1"] if "x1" in feature_keys else [feature_keys[0]]
    remaining = [k for k in feature_keys if k not in selected]

    while len(selected) < min(6, len(feature_keys)) and remaining:
        best_candidate = None
        best_nll = float("inf")
        min_vif_candidate = None
        min_vif_val = float("inf")

        for candidate in remaining:
            trial = selected + [candidate]
            trial_indices = [feature_keys.index(k) for k in trial]
            X_trial = X_train[:, trial_indices]

            # VIF Collinearity filter (< 10.0)
            if X_trial.shape[1] > 1:
                vifs = [variance_inflation_factor(X_trial, i) for i in range(X_trial.shape[1])]
                max_vif = float(np.max(vifs))
            else:
                max_vif = 1.0

            if max_vif < min_vif_val:
                min_vif_val = max_vif
                min_vif_candidate = candidate

            if max_vif > 10.0:
                continue

            try:
                scaler = StandardScaler()
                X_trial_scaled = scaler.fit_transform(X_trial)
                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                lr.fit(X_trial_scaled, y_train)
                probs = lr.predict_proba(X_trial_scaled)[:, 1]
                nll = compute_nll(probs, y_train)
                if nll < best_nll:
                    best_nll = nll
                    best_candidate = candidate
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.debug(f"Logistic regression candidate trial failed for {candidate}: {e}")
                continue

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        elif min_vif_candidate is not None:
            selected.append(min_vif_candidate)
            remaining.remove(min_vif_candidate)
        else:
            break

    return selected
