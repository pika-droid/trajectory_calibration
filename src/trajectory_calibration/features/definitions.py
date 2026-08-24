"""
Trajectory feature dictionary definitions and constant keys.
"""

from __future__ import annotations

FEATURE_NAMES: dict[str, str] = {
    "x1": "Final Logit",
    "x3": "Confidence Gain",
    "x4": "Monotonicity Count",
    "x6": "Confidence Variance",
    "x8": "Scale Dip Depth",
    "x9": "Log-Scale Slope",
    "x10": "Logprob Gain",
    "x11": "Logprob Variance",
    "x12": "Logprob Acceleration",
    "x13": "Discrete Answer Stability",
    "x14": "Relative Gain Ratio",
    "x15": "Mid-Fine Gain Contrast",
    "x17": "End-Scale Spike Ratio",
    "x18": "Scale Entropy Slope",
    "x19": "Relative Margin Growth",
    "x20": "Answer Flip Frequency",
    "x21": "Logit Trajectory Convexity",
    "x22": "First-to-Final Jump Ratio",
}

FEATURE_KEYS: list[str] = list(FEATURE_NAMES.keys())

__all__ = ["FEATURE_NAMES", "FEATURE_KEYS"]
