"""
White-Box Token-Probability Uncertainty Scorers.

Implements token-probability confidence scorers following the CVS Health UQLM standard:
- Sequence Probability (Joint / Length-Normalized Geometric Mean)
- Min Probability (Bottleneck Token Confidence)
- Token Entropy & Negentropy
- Probability Margin
"""

from __future__ import annotations

import numpy as np


class WhiteBoxScorers:
    """Collection of white-box confidence estimators from token distributions."""

    @staticmethod
    def sequence_probability(
        token_logprobs: list[float] | np.ndarray, length_normalize: bool = True
    ) -> float:
        """
        Calculates joint sequence probability from per-token log-probabilities.

        - If length_normalize=True: exp( (1/T) * sum_t log p_t ) = Geometric Mean.
        - If length_normalize=False: exp( sum_t log p_t ) = Joint Product.
        """
        arr = np.asarray(token_logprobs, dtype=np.float64)
        if len(arr) == 0:
            return 0.5
        if length_normalize:
            avg_lp = float(np.mean(arr))
            return float(np.exp(np.clip(avg_lp, -35.0, 0.0)))
        else:
            total_lp = float(np.sum(arr))
            return float(np.exp(np.clip(total_lp, -35.0, 0.0)))

    @staticmethod
    def min_probability(token_probs: list[float] | np.ndarray) -> float:
        """
        Calculates minimum token probability across the generated sequence.

        Identifies the highest-uncertainty bottleneck token.
        """
        arr = np.asarray(token_probs, dtype=np.float64)
        if len(arr) == 0:
            return 0.5
        return float(np.min(arr))

    @staticmethod
    def token_entropy(
        token_distribution: np.ndarray, eps: float = 1e-12, base: float = np.e
    ) -> float:
        """
        Calculates Shannon entropy of a single token probability distribution: -sum p * log(p).
        """
        p = np.asarray(token_distribution, dtype=np.float64)
        p = np.clip(p, eps, 1.0)
        p = p / np.sum(p)
        h = -np.sum(p * (np.log(p) / np.log(base)))
        return float(h)

    @staticmethod
    def mean_token_negentropy(
        token_distributions: np.ndarray | list[np.ndarray], num_classes: int = 32000
    ) -> float:
        """
        Calculates average token negentropy bounded in [0, 1].

        Score = 1.0 - (mean(H) / log(num_classes)).
        High score indicates high certainty.
        """
        if len(token_distributions) == 0:
            return 0.5

        entropies = [
            WhiteBoxScorers.token_entropy(dist) for dist in token_distributions
        ]
        mean_h = float(np.mean(entropies))
        max_h = np.log(max(num_classes, 2))
        negentropy = 1.0 - (mean_h / max_h)
        return float(np.clip(negentropy, 0.0, 1.0))

    @staticmethod
    def min_token_negentropy(
        token_distributions: np.ndarray | list[np.ndarray], num_classes: int = 32000
    ) -> float:
        """
        Calculates minimum token negentropy across generated sequence tokens.
        """
        if len(token_distributions) == 0:
            return 0.5

        entropies = [
            WhiteBoxScorers.token_entropy(dist) for dist in token_distributions
        ]
        max_h = np.log(max(num_classes, 2))
        min_negentropy = 1.0 - (max(entropies) / max_h)
        return float(np.clip(min_negentropy, 0.0, 1.0))

    @staticmethod
    def probability_margin(top1_prob: float, top2_prob: float) -> float:
        """
        Calculates top-1 minus top-2 softmax probability margin.
        """
        return float(np.clip(top1_prob - top2_prob, 0.0, 1.0))
