"""
Trajectory feature dictionary definitions and constant keys.

Defines the 17 trajectory signature keys (x1..x17) computed from multi-scale
visual token sweeps (5 scales: 1, 9, 36, 144, 576/256), ordered by empirical
calibration importance.
"""

from __future__ import annotations

FEATURE_NAMES: dict[str, str] = {
    "x1": "Final Logit",
    "x2": "Discrete Answer Stability",
    "x3": "Scale Entropy Slope",
    "x4": "Answer Flip Frequency",
    "x5": "Monotonicity Count",
    "x6": "Logprob Variance",
    "x7": "Mid-Fine Gain Contrast",
    "x8": "Confidence Variance",
    "x9": "End-Scale Spike Ratio",
    "x10": "Scale Dip Depth",
    "x11": "First-to-Final Jump Ratio",
    "x12": "Log-Scale Slope",
    "x13": "Relative Gain Ratio",
    "x14": "Logprob Acceleration",
    "x15": "Confidence Gain",
    "x16": "Relative Margin Growth",
    "x17": "Logprob Gain",
}

FEATURE_KEYS: list[str] = list(FEATURE_NAMES.keys())
CANONICAL_5D_KEYS: list[str] = FEATURE_KEYS[:5]

__all__ = ["CANONICAL_5D_KEYS", "FEATURE_KEYS", "FEATURE_NAMES"]
