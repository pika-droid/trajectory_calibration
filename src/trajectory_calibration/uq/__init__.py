"""Uncertainty quantification modules (UQLM white-box scorers, Semantic Entropy, EigenScore)."""

from trajectory_calibration.uq.eigenscore import (
    compute_eigenscore,
    compute_eigenscore_gram,
    compute_logdet,
    compute_umpire_metric,
    normalize_embedding,
)
from trajectory_calibration.uq.semantic_entropy import (
    EntailmentDeberta,
    FastStringEntailment,
    compute_cluster_assignment_entropy,
    compute_predictive_entropy,
    compute_semantic_entropy,
    get_semantic_ids,
    logsumexp_by_id,
)
from trajectory_calibration.uq.whitebox import WhiteBoxScorers

__all__ = [
    "EntailmentDeberta",
    "FastStringEntailment",
    "WhiteBoxScorers",
    "compute_cluster_assignment_entropy",
    "compute_eigenscore",
    "compute_eigenscore_gram",
    "compute_logdet",
    "compute_predictive_entropy",
    "compute_semantic_entropy",
    "compute_umpire_metric",
    "get_semantic_ids",
    "logsumexp_by_id",
    "normalize_embedding",
]
