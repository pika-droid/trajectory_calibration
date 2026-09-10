"""
Multi-scale trajectory signature feature extraction logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logit

from trajectory_calibration.utils.helpers import clean_text

SCALE_LOG_576 = np.log(np.array([1, 9, 36, 144, 576], dtype=np.float64))
DENOM_576 = float(np.sum((SCALE_LOG_576 - np.mean(SCALE_LOG_576)) ** 2))

SCALE_LOG_256 = np.log(np.array([1, 9, 36, 144, 256], dtype=np.float64))
DENOM_256 = float(np.sum((SCALE_LOG_256 - np.mean(SCALE_LOG_256)) ** 2))


def compute_features_from_sample(
    item: dict[str, Any], fine_scale: int = 576, idx: int = 0
) -> dict[str, float | int | str]:
    """
    Extracts the 17-D trajectory feature vector (1 base anchor x1 + 16 multi-scale trajectory signatures)
    from a multi-scale inference sample.
    """
    feats = item.get("features", {})
    scales = [1, 9, 36, 144, fine_scale]

    confs = []
    margins = []
    answers = []
    accuracies = []

    for s in scales:
        s_data = feats.get(s, feats.get(str(s), {}))
        confs.append(float(s_data.get("conf_softmax", 0.5)))
        margins.append(float(s_data.get("margin", 0.0)))
        answers.append(str(s_data.get("answer", "")).strip())
        accuracies.append(float(s_data.get("vqa_accuracy", 0.0)))

    c_arr = np.array(confs, dtype=np.float64)
    m_arr = np.array(margins, dtype=np.float64)
    eps = 1e-7

    lp_arr = np.log(np.clip(c_arr, eps, 1.0))
    if fine_scale == 576:
        scale_log = SCALE_LOG_576
        denom = DENOM_576
    elif fine_scale == 256:
        scale_log = SCALE_LOG_256
        denom = DENOM_256
    else:
        scale_log = np.log(np.array(scales, dtype=np.float64))
        denom = float(np.sum((scale_log - np.mean(scale_log)) ** 2))

    # x1: Final Logit (inverse sigmoid on c_final)
    c_final_clipped = np.clip(c_arr[-1], eps, 1.0 - eps)
    x1 = float(logit(c_final_clipped))

    # x2: Discrete Answer Stability (clean text unique answers)
    norm_answers = [clean_text(a) for a in answers]
    unique_answers = set(norm_answers)
    x2 = float(1.0 / max(1, len(unique_answers)))

    # x3: Scale Entropy Slope (binary entropy slope)
    bin_entropy = -(
        c_arr * np.log(np.clip(c_arr, eps, 1.0))
        + (1.0 - c_arr) * np.log(np.clip(1.0 - c_arr, eps, 1.0))
    )
    x3 = (
        float(
            np.sum((scale_log - np.mean(scale_log)) * (bin_entropy - np.mean(bin_entropy))) / denom
        )
        if denom > 0
        else 0.0
    )

    # x4: Answer Flip Frequency
    flips = sum(1 for i in range(len(norm_answers) - 1) if norm_answers[i] != norm_answers[i + 1])
    x4 = float(flips / (len(norm_answers) - 1)) if len(norm_answers) > 1 else 0.0

    # x5: Monotonicity Count
    x5 = float(np.sum(c_arr[1:] > c_arr[:-1]))

    # x6: Logprob Variance
    x6 = float(np.var(lp_arr))

    # x7: Mid-Fine Gain Contrast
    x7 = float((c_arr[-1] - c_arr[3]) - (c_arr[3] - c_arr[1]))

    # x8: Confidence Variance
    x8 = float(np.var(c_arr))

    # x9: End-Scale Spike Ratio
    x9 = float(c_arr[-1] - np.mean(c_arr[:-1]))

    # x10: Scale Dip Depth
    coarse_max = max(c_arr[0], c_arr[1])
    mid_min = min(c_arr[2], c_arr[3])
    x10 = float(max(0.0, coarse_max - mid_min))

    # x11: First-to-Final Jump Ratio (clipped to [0.0, 50.0])
    x11 = float(np.clip((c_arr[-1] - c_arr[0]) / (c_arr[-1] + eps), 0.0, 50.0))

    # x12: Log-Scale Slope
    x12 = (
        float(np.sum((scale_log - np.mean(scale_log)) * (c_arr - np.mean(c_arr))) / denom)
        if denom > 0
        else 0.0
    )

    # x13: Relative Gain Ratio (clipped to [0.0, 50.0])
    x13 = float(np.clip(c_arr[-1] / (c_arr[1] + eps), 0.0, 50.0))

    # x14: Logprob Acceleration
    x14 = float((lp_arr[-1] - lp_arr[3]) - (lp_arr[3] - lp_arr[2]))

    # x15: Confidence Gain (c_fine - c_9)
    x15 = float(c_arr[-1] - c_arr[1])

    # x16: Relative Margin Growth (clipped to [0.0, 50.0])
    x16 = float(np.clip(m_arr[-1] / (m_arr[1] + eps), 0.0, 50.0))

    # x17: Logprob Gain
    x17 = float(lp_arr[-1] - lp_arr[1])

    acc_final = accuracies[-1]
    if "is_correct" in item:
        is_correct = int(item["is_correct"])
    else:
        is_correct = 1 if acc_final >= 0.5 else 0

    qid = (
        item.get("question_id")
        or item.get("id")
        or item.get("questionId")
        or item.get("sample_idx")
    )

    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "x6": x6,
        "x7": x7,
        "x8": x8,
        "x9": x9,
        "x10": x10,
        "x11": x11,
        "x12": x12,
        "x13": x13,
        "x14": x14,
        "x15": x15,
        "x16": x16,
        "x17": x17,
        "c_576": float(c_arr[-1]),
        f"c_{fine_scale}": float(c_arr[-1]),
        "c_fine": float(c_arr[-1]),
        "is_correct": is_correct,
        "vqa_accuracy": float(acc_final),
        "question_id": qid if qid is not None else f"sample_{idx}",
        "answer_type": item.get("answer_type", "open"),
    }
