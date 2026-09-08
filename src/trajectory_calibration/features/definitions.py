"""
Trajectory feature dictionary definitions and constant keys.

Defines the 17 trajectory signature keys (x1..x17) computed from multi-scale
visual token sweeps (5 scales: 1, 9, 36, 144, 576/256), ordered by empirical
calibration importance.
"""

from __future__ import annotations

FEATURE_NAMES: dict[str, str] = {
    "x1": "Final Logit",
    "x2": "Monotonicity Count",
    "x3": "Discrete Answer Stability",
    "x4": "Scale Entropy Slope",
    "x5": "Logprob Variance",
    "x6": "Scale Dip Depth",
    "x7": "Answer Flip Frequency",
    "x8": "Mid-Fine Gain Contrast",
    "x9": "Log-Scale Slope",
    "x10": "Confidence Gain",
    "x11": "Logprob Acceleration",
    "x12": "First-to-Final Jump Ratio",
    "x13": "Confidence Variance",
    "x14": "Relative Gain Ratio",
    "x15": "End-Scale Spike Ratio",
    "x16": "Logprob Gain",
    "x17": "Relative Margin Growth",
}

FEATURE_KEYS: list[str] = list(FEATURE_NAMES.keys())

__all__ = ["FEATURE_KEYS", "FEATURE_NAMES"]
