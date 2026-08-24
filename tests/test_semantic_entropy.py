"""Tests for Kuhn Semantic Entropy, Chen EigenScore, and UMPIRE metrics."""

import numpy as np
import pytest
from trajectory_calibration.uq.eigenscore import (
    compute_eigenscore,
    compute_eigenscore_gram,
    compute_logdet,
    compute_umpire_metric,
    normalize_embedding,
)
from trajectory_calibration.uq.semantic_entropy import (
    FastStringEntailment,
    cluster_assignment_entropy,
    compute_cluster_assignment_entropy,
    compute_predictive_entropy,
    compute_semantic_entropy,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
)


def test_get_semantic_ids_and_entropy() -> None:
    matcher = FastStringEntailment()
    strings = ["dog", "dog", "cat", "dog.", "bird"]
    ids = get_semantic_ids(strings, model=matcher)
    assert len(ids) == len(strings)
    # "dog" and "dog." should be clustered together
    assert ids[0] == ids[1] == ids[3]
    assert ids[2] != ids[0]  # cat
    assert ids[4] != ids[0]  # bird

    lps = [-0.1, -0.2, -0.5, -0.15, -0.8]
    se = compute_semantic_entropy(ids, lps)
    assert se >= 0.0

    pe = predictive_entropy(lps)
    assert pe > 0.0


def test_cluster_assignment_entropy() -> None:
    # All same cluster -> 0 entropy
    ids_uniform = [0, 0, 0, 0]
    h = compute_cluster_assignment_entropy(ids_uniform)
    assert pytest.approx(h, abs=1e-5) == 0.0

    # Equal split
    ids_split = [0, 1]
    h_split = cluster_assignment_entropy(ids_split)
    assert pytest.approx(h_split, abs=1e-5) == np.log(2.0)


def test_eigenscore_and_umpire() -> None:
    # Identical vectors (rank 1)
    vecs = np.ones((5, 10))
    normed = normalize_embedding(vecs)
    score_cov = compute_eigenscore(normed)
    score_gram = compute_eigenscore_gram(normed)
    assert isinstance(score_cov, float)
    assert isinstance(score_gram, float)

    k_mat = np.eye(4)
    logdet = compute_logdet(k_mat, alpha=1e-8)
    assert pytest.approx(logdet, abs=1e-3) == 0.0

    umpire = compute_umpire_metric(vecs, [-0.1, -0.2, -0.3, -0.4, -0.5])
    assert isinstance(umpire, float)
