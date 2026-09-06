"""
Semantic Entropy Implementation.

Implements exact Kuhn et al. (2023) / UMPIRE Semantic Entropy using bidirectional
natural language entailment (NLI) clustering, LogSumExp cluster probability aggregation,
and cluster assignment entropy.
"""

from __future__ import annotations

import logging
import re
from typing import Any
import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy

from trajectory_calibration.utils.helpers import clean_text

logger = logging.getLogger("trajectory_calibration.uq.semantic_entropy")


class FastStringEntailment:
    """High-throughput string equivalence & word-boundary entailment matcher."""

    def check_implication(self, text1: str, text2: str, *args: Any, **kwargs: Any) -> int:
        """Returns 2 (Entailment), 1 (Neutral), or 0 (Contradiction)."""
        t1 = clean_text(text1)
        t2 = clean_text(text2)

        if not t1 or not t2:
            return 1
        if t1 == t2:
            return 2

        binary_opposites = {
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("right", "wrong"),
        }
        for op1, op2 in binary_opposites:
            if (t1 == op1 and t2 == op2) or (t1 == op2 and t2 == op1):
                return 0

        # Negation polarity guard: prevent false positive entailments
        negation_words = {"not", "no", "never", "none"}
        words1 = set(re.findall(r"\b\w+\b", t1.lower()))
        words2 = set(re.findall(r"\b\w+\b", t2.lower()))
        if bool(words1 & negation_words) != bool(words2 & negation_words):
            return 0

        # Word boundary matching for phrase containment
        if len(t1.split()) > 1 or len(t2.split()) > 1:
            if bool(re.search(r"\b" + re.escape(t1) + r"\b", t2)) or bool(re.search(r"\b" + re.escape(t2) + r"\b", t1)):
                return 2

        return 1


class EntailmentDeberta:
    """DeBERTa-v2-xlarge-mnli bidirectional NLI entailment classifier."""

    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def check_implication(self, text1: str, text2: str, *args: Any, **kwargs: Any) -> int:
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(text1, text2, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_idx = torch.argmax(F.softmax(outputs.logits, dim=-1)).cpu().item()
        return int(pred_idx)


def get_semantic_ids(
    strings_list: list[str],
    model: Any | None = None,
    strict_entailment: bool = False,
    example: dict[str, Any] | None = None,
) -> list[int]:
    """Groups a list of generated answer strings into semantic equivalence clusters."""
    if model is None:
        model = FastStringEntailment()

    def are_equivalent(text1: str, text2: str) -> bool:
        imp1 = model.check_implication(text1, text2, example=example)
        imp2 = model.check_implication(text2, text1, example=example)
        if strict_entailment:
            return (imp1 == 2) and (imp2 == 2)
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
    semantic_ids: list[int], log_likelihoods: list[float] | np.ndarray, agg: str = "sum_normalized"
) -> list[float]:
    """Aggregates log-likelihoods for identical semantic cluster IDs via LogSumExp."""
    lps = np.asarray(log_likelihoods, dtype=np.float64)
    if len(lps) == 0:
        return []

    unique_ids = sorted(list(set(semantic_ids)))
    cluster_log_probs = []

    total_logsumexp = float(logsumexp(lps))

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_lps = lps[id_indices]
        cluster_lse = float(logsumexp(id_lps))
        norm_cluster_lp = cluster_lse - total_logsumexp
        cluster_log_probs.append(float(norm_cluster_lp))

    return cluster_log_probs


def compute_semantic_entropy(
    semantic_ids: list[int], log_likelihoods: list[float] | np.ndarray
) -> float:
    """Computes exact Kuhn et al. Semantic Entropy: -sum_k P(C_k) * ln P(C_k)."""
    if len(semantic_ids) == 0:
        return 0.0

    cluster_log_probs = logsumexp_by_id(semantic_ids, log_likelihoods)
    if not cluster_log_probs:
        return 0.0
    probs = np.exp(cluster_log_probs)
    probs = probs / np.sum(probs)

    return float(max(0.0, entropy(probs)))


def cluster_assignment_entropy(semantic_ids: list[int]) -> float:
    """Computes cluster assignment entropy: -sum_k p_k * ln p_k."""
    if len(semantic_ids) == 0:
        return 0.0
    counts = np.bincount(semantic_ids)
    probs = counts / len(semantic_ids)
    probs = probs[probs > 0]
    return float(max(0.0, entropy(probs)))


compute_cluster_assignment_entropy = cluster_assignment_entropy


def predictive_entropy(log_probs: list[float] | np.ndarray) -> float:
    """Computes average sequence negative log-likelihood: - (1/N) * sum_i log P(s_i)."""
    arr = np.asarray(log_probs, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(-np.mean(arr))


compute_predictive_entropy = predictive_entropy
