"""
Semantic Entropy Implementation.

Implements exact Kuhn et al. (2023) / UMPIRE Semantic Entropy using bidirectional
natural language entailment (NLI) clustering, LogSumExp cluster probability aggregation,
and cluster assignment entropy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from trajectory_calibration.utils.helpers import clean_text

logger = logging.getLogger("trajectory_calibration.uq.semantic_entropy")


class FastStringEntailment:
    """
    High-throughput string equivalence & substring containment entailment matcher.

    Used for zero-dependency CPU execution and single-word/short-phrase VQA evaluation.
    """

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        """
        Returns:
        - 2: Entailment (equivalent / bidirectional match)
        - 1: Neutral
        - 0: Contradiction
        """
        t1 = clean_text(text1)
        t2 = clean_text(text2)

        if not t1 or not t2:
            return 1
        if t1 == t2:
            return 2
        if t1 in t2 or t2 in t1:
            return 2

        # Check binary antonyms
        binary_opposites = {
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("right", "wrong"),
        }
        for op1, op2 in binary_opposites:
            if (t1 == op1 and t2 == op2) or (t1 == op2 and t2 == op1):
                return 0

        return 1


class EntailmentDeberta:
    """
    DeBERTa-v2-xlarge-mnli bidirectional NLI entailment classifier.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # DeBERTa-MNLI classes: 0 -> contradiction, 1 -> neutral, 2 -> entailment
            pred_idx = torch.argmax(F.softmax(logits, dim=-1)).cpu().item()
        return int(pred_idx)


def get_semantic_ids(
    strings_list: list[str],
    model: Any | None = None,
    strict_entailment: bool = False,
    example: dict[str, Any] | None = None,
) -> list[int]:
    """
    Groups a list of generated answer strings into semantic equivalence clusters.

    Args:
        strings_list: List of generation outputs across rollouts or scales.
        model: Entailment model instance (EntailmentDeberta or FastStringEntailment).
        strict_entailment: If True, requires both t1->t2 and t2->t1 to be entailment (2).
                           If False, requires neither to be contradiction (0) and not both neutral (1,1).

    Returns:
        List of integer cluster IDs matching the length of strings_list.
    """
    if model is None:
        model = FastStringEntailment()

    def are_equivalent(text1: str, text2: str) -> bool:
        imp1 = model.check_implication(text1, text2, example=example)
        imp2 = model.check_implication(text2, text1, example=example)

        if strict_entailment:
            return (imp1 == 2) and (imp2 == 2)
        else:
            return (0 not in [imp1, imp2]) and ([1, 1] != [imp1, imp2])

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0

    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    return semantic_set_ids


def logsumexp_by_id(
    semantic_ids: list[int], log_likelihoods: list[float] | np.ndarray
) -> list[float]:
    """
    Aggregates log-likelihoods for identical semantic cluster IDs via LogSumExp.
    """
    log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
    unique_ids = sorted(list(set(semantic_ids)))
    cluster_log_probs = []

    # Total normalizer
    max_lp = np.max(log_likelihoods)
    total_logsumexp = max_lp + np.log(np.sum(np.exp(log_likelihoods - max_lp)))

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_lps = log_likelihoods[id_indices]

        max_c = np.max(id_lps)
        cluster_lse = max_c + np.log(np.sum(np.exp(id_lps - max_c)))

        # Normalized cluster probability in log space
        norm_cluster_lp = cluster_lse - total_logsumexp
        cluster_log_probs.append(float(norm_cluster_lp))

    return cluster_log_probs


def compute_semantic_entropy(
    semantic_ids: list[int], log_likelihoods: list[float] | np.ndarray
) -> float:
    """
    Computes exact Kuhn et al. Semantic Entropy: -sum_k P(C_k) * ln P(C_k).
    """
    if len(semantic_ids) == 0:
        return 0.0

    cluster_log_probs = logsumexp_by_id(semantic_ids, log_likelihoods)
    probs = np.exp(cluster_log_probs)
    probs = probs / np.sum(probs)  # Numerical safety normalization

    se = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
    return float(max(0.0, se))


def compute_cluster_assignment_entropy(semantic_ids: list[int]) -> float:
    """
    Computes cluster assignment entropy from frequency of assigned clusters: -sum_k p_k * ln p_k.
    """
    if len(semantic_ids) == 0:
        return 0.0

    counts = np.bincount(semantic_ids)
    probs = counts / len(semantic_ids)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(max(0.0, entropy))


def compute_predictive_entropy(log_likelihoods: list[float] | np.ndarray) -> float:
    """
    Computes average sequence negative log-likelihood: - (1/N) * sum_i log P(s_i).
    """
    arr = np.asarray(log_likelihoods, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(-np.mean(arr))
