"""
Dataset loading, feature file discovery, and stratified dataset splitting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trajectory_calibration.features.extractor import compute_features_from_sample
from trajectory_calibration.utils.helpers import safe_torch_load

logger = logging.getLogger("trajectory_calibration.features.loader")


def find_feature_file(
    base_path: Path | str,
    ds_name: str | None = None,
    arch: str = "m3",
    gen_temperature: float = 0.0,
) -> Path:
    """
    Resolves feature file path matching standard directory layouts:
        1. base / arch_folder / f"temp_{gen_temperature}" / f"{ds_name}.pt"
        2. base / ds_name / "full_extracted_features.pt"
        3. base / f"{ds_name}.pt"
        4. Direct file path if base_path is a file
    """
    path = Path(base_path)
    if path.is_file():
        return path

    arch_folder = "mqt_llava" if "mqt" in arch.lower() else "m3_llava"
    temp_folder = f"temp_{gen_temperature:.1f}" if gen_temperature == int(gen_temperature) else f"temp_{gen_temperature}"

    candidates = []
    if ds_name:
        candidates.extend([
            path / arch_folder / temp_folder / f"{ds_name}.pt",
            path / arch_folder / temp_folder / f"{ds_name}_5scale.pt",
            path / arch_folder / "temp_0.0" / f"{ds_name}.pt",
            path / arch_folder / "temp_0.0" / f"{ds_name}_5scale.pt",
            path / arch_folder / f"{ds_name}.pt",
            path / arch_folder / f"{ds_name}_5scale.pt",
            path / ds_name / "full_extracted_features.pt",
            path / f"{ds_name}.pt",
            path / f"{ds_name}_5scale.pt",
        ])
    else:
        candidates.extend([
            path / "full_extracted_features.pt",
        ])

    for cand in candidates:
        if cand.exists():
            return cand

    return candidates[0] if candidates else path


def load_dataset_features(
    features_dir_or_file: Path | str,
    ds_name: str | None = None,
    arch: str = "m3",
    gen_temperature: float = 0.0,
    fine_scale: int | None = None,
) -> pd.DataFrame:
    """
    Loads dataset features from .pt file and builds a Pandas DataFrame.
    """
    pt_path = find_feature_file(features_dir_or_file, ds_name=ds_name, arch=arch, gen_temperature=gen_temperature)
    if not pt_path.exists():
        raise FileNotFoundError(f"Feature file could not be located at: {pt_path.resolve()}")

    if fine_scale is None:
        fine_scale = 256 if ("mqt" in arch.lower() or "mqt" in str(pt_path).lower()) else 576

    raw_data = safe_torch_load(pt_path)
    rows = []
    for idx, item in enumerate(raw_data):
        feat_dict = compute_features_from_sample(item, fine_scale=fine_scale, idx=idx)
        rows.append(feat_dict)

    df = pd.DataFrame(rows)

    # Deduplication Guard: Only drop duplicates if genuine non-trivial duplicate IDs are present (> 1 unique ID)
    if "question_id" in df.columns and len(df) > 1:
        initial_len = len(df)
        n_unique = df["question_id"].nunique()
        if 1 < n_unique < initial_len:
            df = df.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
            if len(df) < initial_len:
                logger.info(f"Deduplicated dataset: {initial_len} -> {len(df)} unique question_ids.")
        elif n_unique == 1 and initial_len > 1:
            logger.warning(
                f"Skipping deduplication: all {initial_len} samples share the same question_id ({df['question_id'].iloc[0]})."
            )

    return df


def get_stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generates an 80/20 train/test stratified split on answer_type x is_correct."""
    indices = np.arange(len(df))

    if "answer_type" in df.columns and len(df["answer_type"].unique()) > 1:
        strat_key = df["answer_type"].astype(str) + "_" + df["is_correct"].astype(str)
        counts = strat_key.value_counts()
        if (counts < 2).any():
            strat_key = df["is_correct"].values
    else:
        strat_key = df["is_correct"].values

    counts = pd.Series(strat_key).value_counts()
    stratify_param = strat_key if (counts >= 2).all() and len(counts) > 1 else None

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    return train_idx, test_idx
