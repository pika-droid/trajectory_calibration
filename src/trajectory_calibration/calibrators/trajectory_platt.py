"""
Trajectory Platt Scaling Calibrator.

Implements K-dimensional Platt scaling p = σ(a^T z + b) where
a ∈ R^K and b ∈ R are explicit torch.nn.Parameter objects optimized via L-BFGS.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryPlattScaler(nn.Module):
    """Trajectory Platt Scaling: p = σ(a^T z + b).

    Generalizes 1D Platt scaling from scalar logit to K-dimensional
    trajectory features. a ∈ R^K and b ∈ R are nn.Parameters optimized
    via L-BFGS to minimize binary cross-entropy (NLL).
    """

    def __init__(self, n_features: int, C: float | None = 1.0) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if C is not None and C <= 0:
            raise ValueError(f"C must be positive or None, got C={C}")
        self.n_features = n_features
        self.C = C
        self.a = nn.Parameter(torch.zeros(n_features))
        self.b = nn.Parameter(torch.zeros(1))
        self.scaler: StandardScaler | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute calibrated probabilities: σ(a^T z + b)."""
        return torch.sigmoid(z @ self.a + self.b)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TrajectoryPlattScaler:
        """Fit a and b by minimizing NLL via torch.optim.LBFGS."""
        X = np.asarray(X_train, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, self.n_features) if self.n_features > 1 else X.reshape(-1, 1)
        if len(X) == 0:
            raise ValueError("Cannot fit TrajectoryPlattScaler with 0 samples.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        y = np.asarray(y_train, dtype=np.float32).ravel()
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of samples, got len(X)={len(X)} and len(y)={len(y)}")

        # Reset parameters to zeros for idempotent fitting
        with torch.no_grad():
            self.a.zero_()
            self.b.zero_()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        z_t = torch.as_tensor(X_scaled, dtype=self.a.dtype, device=self.a.device)
        y_t = torch.as_tensor(y, dtype=self.a.dtype, device=self.a.device)

        optimizer = torch.optim.LBFGS(
            [self.a, self.b],
            lr=1.0,
            max_iter=100,
            line_search_fn="strong_wolfe",
        )
        n_samples = len(y)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = z_t @ self.a + self.b
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            if self.C is not None and self.C > 0:
                loss = loss + 0.5 * (1.0 / (self.C * n_samples)) * torch.sum(self.a ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities as numpy array."""
        if self.scaler is None:
            raise RuntimeError("TrajectoryPlattScaler must be fitted before predict_proba.")
        X = np.asarray(X_test, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1) if self.n_features > 1 else X.reshape(-1, 1)
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            z_t = torch.as_tensor(X_scaled, dtype=self.a.dtype, device=self.a.device)
            probs = self.forward(z_t).cpu().numpy().ravel()
        return probs
