"""
Synthetic mock data generation for fast offline unit testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trajectory_calibration.features.extractor import compute_features_from_sample


def generate_mock_df(
    ds_name: str = "mock_ds", n_samples: int = 100, seed: int = 42, fine_scale: int = 576
) -> pd.DataFrame:
    """Generates synthetic 17-D feature DataFrames for CPU smoke tests."""
    rng = np.random.RandomState(seed)
    scales = [1, 9, 36, 144, fine_scale]

    raw_items = []
    for i in range(n_samples):
        is_corr = 1 if rng.rand() > 0.35 else 0
        base_c = rng.uniform(0.65, 0.95) if is_corr else rng.uniform(0.25, 0.65)

        feats = {}
        for s in scales:
            c_val = np.clip(base_c + rng.normal(0.0, 0.05), 0.05, 0.98)
            m_val = rng.uniform(0.1, 0.5)
            ans = "yes" if is_corr else ("no" if rng.rand() > 0.5 else "maybe")
            acc = float(is_corr) if s == fine_scale else float(rng.rand() > 0.4)
            feats[s] = {
                "conf_softmax": float(c_val),
                "margin": float(m_val),
                "answer": ans,
                "vqa_accuracy": acc,
            }

        raw_items.append({
            "question_id": 100000 + i,
            "features": feats,
            "answer_type": "open",
        })

    rows = [compute_features_from_sample(item, fine_scale=fine_scale) for item in raw_items]
    return pd.DataFrame(rows)
