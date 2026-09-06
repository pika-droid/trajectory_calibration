"""
EigenScore and Matrix Log-Determinant Uncertainty Quantification.

Implements SVD spectral dispersion on generation embeddings (Chen et al., 2024)
and matrix log-determinant volume metrics (UMPIRE).
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import fast_logdet


def normalize_embedding(embeddings: np.ndarray) -> np.ndarray:
    """Normalizes embedding vectors to unit L2 norm."""
    arr = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def compute_eigenscore(embeddings: np.ndarray, jitter: float = 1e-3) -> float:
    """
    Calculates EigenScore via SVD on normalized embedding sample covariance.

    Formula: (1/K) * sum_i log10(s_i)
    Higher value indicates greater spectral dispersion (higher uncertainty).
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0

    normed = normalize_embedding(arr)
    cov_matrix = np.cov(normed)

    # Regularized SVD
    reg_cov = cov_matrix + jitter * np.eye(cov_matrix.shape[0])
    _, s, _ = np.linalg.svd(reg_cov)
    s = np.clip(s, 1e-12, None)
    eigen_score = np.mean(np.log10(s))
    return float(eigen_score)


def compute_eigenscore_gram(embeddings: np.ndarray, jitter: float = 1e-3) -> float:
    """
    Calculates EigenScore from the Gram matrix G = E * E^T.
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0

    normed = normalize_embedding(arr)
    gram = np.matmul(normed, normed.T)
    reg_gram = gram + jitter * np.eye(gram.shape[0])
    _, s, _ = np.linalg.svd(reg_gram)
    s = np.clip(s, 1e-12, None)
    return float(np.mean(np.log10(s)))


def compute_logdet(k_matrix: np.ndarray, alpha: float = 1e-8) -> float:
    """
    Computes log-determinant volume of covariance / kernel matrix: log det(K + alpha * I).
    Guarantees strictly finite values via eigenvalue thresholding (Issue 12).
    """
    arr = np.asarray(k_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return 0.0
    try:
        evals = np.maximum(np.linalg.eigvalsh(arr), 0.0) + alpha
        return float(np.sum(np.log(evals)))
    except Exception:
        reg_k = arr + np.eye(arr.shape[0]) * alpha
        try:
            val = float(fast_logdet(reg_k))
            return val if np.isfinite(val) else 0.0
        except Exception:
            sign, logdet = np.linalg.slogdet(reg_k)
            return float(logdet) if sign > 0 else 0.0


def compute_umpire_metric(
    embeddings: np.ndarray,
    sequence_log_probs: list[float] | np.ndarray,
    alpha_param: float = 1.0,
    jitter: float = 1e-8,
) -> float:
    """
    Computes UMPIRE Uncertainty Metric: 1/(2*k) * LogDet(G) + alpha * 1/k * ||1 - P(seq)||_1.
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    k = len(arr)
    if k == 0:
        return 0.0

    normed = normalize_embedding(arr)
    gram = np.matmul(normed, normed.T)
    v_logdet = compute_logdet(gram, alpha=jitter) / (2.0 * k)

    seq_probs = np.exp(np.asarray(sequence_log_probs, dtype=np.float64))
    prob_penalty = float(np.sum(1.0 - seq_probs)) / k

    return float(v_logdet + alpha_param * prob_penalty)
