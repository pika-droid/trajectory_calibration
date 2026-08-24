"""
Canonical Uncertainty Quantification (UQ) Methods and Scorers.

Includes:
- UQLM White-box token probability scorers (Sequence Prob, Min Prob, Token Entropy, Margin)
- Kuhn et al. / UMPIRE Semantic Entropy (DeBERTa NLI clustering, LogSumExp, Cluster Entropy)
- Chen et al. / UMPIRE EigenScore and Log-Determinant Spectral Dispersion metrics
"""

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
    cluster_assignment_entropy,
    compute_cluster_assignment_entropy,
    compute_predictive_entropy,
    compute_semantic_entropy,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
)
from trajectory_calibration.uq.whitebox import WhiteBoxScorers

__all__ = [
    "WhiteBoxScorers",
    "FastStringEntailment",
    "EntailmentDeberta",
    "get_semantic_ids",
    "logsumexp_by_id",
    "compute_semantic_entropy",
    "cluster_assignment_entropy",
    "compute_cluster_assignment_entropy",
    "predictive_entropy",
    "compute_predictive_entropy",
    "normalize_embedding",
    "compute_eigenscore",
    "compute_eigenscore_gram",
    "compute_logdet",
    "compute_umpire_metric",
]
