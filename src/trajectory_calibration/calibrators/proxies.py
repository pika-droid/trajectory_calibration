"""
Uncertainty Quantification (UQ) single-pass trajectory proxies.

Provides scikit-learn estimators for:
- Multi-Scale Semantic Consistency (MSSC / single-pass semantic stability on x13)
- Multi-Scale Eigen Variance (MSE-EIGEN / single-pass trajectory dispersion on x6)
- Probability Margin (Single-pass top-1 vs top-2 margin on x19)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _resolve_feature_col(
    X: np.ndarray, feature_key: str, feature_names: list[str] | None, fallback_idx: int
) -> np.ndarray:
    """Helper to safely extract the target feature column by name or index."""
    if feature_names is not None:
        if feature_key in feature_names:
            col_idx = feature_names.index(feature_key)
            return X[:, [col_idx]]
        raise ValueError(
            f"Feature '{feature_key}' not found in provided feature_names: {feature_names}"
        )

    if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
        return X.reshape(-1, 1)

    if X.ndim == 2 and X.shape[1] > fallback_idx:
        return X[:, [fallback_idx]]

    raise ValueError(
        f"Cannot resolve feature '{feature_key}' from input shape {X.shape} "
        f"(fallback index {fallback_idx} exceeds column count {X.shape[1] if X.ndim == 2 else 1}). "
        f"Provide explicit feature_names."
    )


class MultiScaleSemanticConsistency:
    """Single-pass Multi-Scale Semantic Consistency proxy (scaled on x13)."""

    def __init__(self, C: float = 1.0, feature_names: list[str] | None = None) -> None:
        self.C = C
        self.feature_names = feature_names
        self.lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> MultiScaleSemanticConsistency:
        if feature_names is not None:
            self.feature_names = feature_names
        scores = _resolve_feature_col(X_train, "x13", self.feature_names, fallback_idx=9)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = _resolve_feature_col(X_test, "x13", self.feature_names, fallback_idx=9)
        return self.lr.predict_proba(scores)[:, 1]


class MultiScaleEigenVariance:
    """Single-pass trajectory covariance dispersion proxy (scaled on x6)."""

    def __init__(self, C: float = 1.0, feature_names: list[str] | None = None) -> None:
        self.C = C
        self.feature_names = feature_names
        self.lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> MultiScaleEigenVariance:
        if feature_names is not None:
            self.feature_names = feature_names
        scores = _resolve_feature_col(X_train, "x6", self.feature_names, fallback_idx=3)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = _resolve_feature_col(X_test, "x6", self.feature_names, fallback_idx=3)
        return self.lr.predict_proba(scores)[:, 1]


class ProbabilityMarginEstimator:
    """Probability Margin proxy calibrated via Platt scaling (scaled on x19)."""

    def __init__(self, C: float = 1.0, feature_names: list[str] | None = None) -> None:
        self.C = C
        self.feature_names = feature_names
        self.lr = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> ProbabilityMarginEstimator:
        if feature_names is not None:
            self.feature_names = feature_names
        scores = _resolve_feature_col(X_train, "x19", self.feature_names, fallback_idx=14)
        self.lr.fit(scores, y_train)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        scores = _resolve_feature_col(X_test, "x19", self.feature_names, fallback_idx=14)
        return self.lr.predict_proba(scores)[:, 1]
