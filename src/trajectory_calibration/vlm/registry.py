"""
14-Benchmark Vision-Language Dataset Registry and HuggingFace Loader.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("trajectory_calibration.vlm.registry")

# Centralized 14-Benchmark Registry with non-withheld ground truth splits
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "infographicvqa": {
        "hf_repo": "lmms-lab/DocVQA",
        "config": "InfographicVQA",
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "ai2d": {
        "hf_repo": "lmms-lab/ai2d",
        "config": None,
        "default_split": "test",
        "answer_type": "mc_index",
    },
    "chartqa": {
        "hf_repo": "lmms-lab/ChartQA",
        "config": None,
        "default_split": "test",
        "answer_type": "open",
    },
    "docvqa": {
        "hf_repo": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "mmbench": {
        "hf_repo": "lmms-lab/MMBench_EN",
        "config": None,
        "default_split": "dev",
        "answer_type": "mc_letter",
    },
    "mmmu": {
        "hf_repo": "lmms-lab/MMMU",
        "config": None,
        "default_split": "validation",
        "answer_type": "mc_letter",
    },
    "seedbench": {
        "hf_repo": "lmms-lab/SEED-Bench",
        "config": None,
        "default_split": "test",
        "answer_type": "mc_letter",
    },
    "vqav2": {
        "hf_repo": "lmms-lab/vqav2",
        "config": None,
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "scienceqa": {
        "hf_repo": "lmms-lab/ScienceQA",
        "config": "ScienceQA-IMG",
        "default_split": "test",
        "answer_type": "mc_index",
    },
    "textvqa": {
        "hf_repo": "lmms-lab/textvqa",
        "config": None,
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "pope": {
        "hf_repo": "lmms-lab/POPE",
        "config": None,
        "default_split": "test",
        "answer_type": "open",
    },
    "gqa": {
        "hf_repo": "lmms-lab/GQA",
        "config": "testdev_balanced_instructions",
        "default_split": "testdev",
        "answer_type": "open",
    },
    "vizwiz-vqa": {
        "hf_repo": "lmms-lab/VizWiz-VQA",
        "config": None,
        "default_split": "val",
        "answer_type": "list_soft",
    },
    "lego-puzzles": {
        "hf_repo": "lmms-lab/LEGO-Puzzles",
        "config": None,
        "default_split": "test",
        "answer_type": "open",
    },
}

ALL_DATASET_KEYS = list(DATASET_REGISTRY.keys())


def load_hf_dataset(dataset_key: str, subset_size: int | None = None) -> Any:
    """Loads benchmark dataset from HuggingFace with custom filtering and merges."""
    from datasets import load_dataset

    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Valid keys: {ALL_DATASET_KEYS}")

    cfg = DATASET_REGISTRY[dataset_key]
    repo = cfg["hf_repo"]
    config_name = cfg["config"]
    target_split = cfg["default_split"]

    logger.info(f"Loading HF dataset '{repo}' (config={config_name}, split={target_split})...")

    if dataset_key == "mmmu":
        try:
            ds = (
                load_dataset(repo, config_name, split=target_split)
                if config_name
                else load_dataset(repo, split=target_split)
            )
        except Exception:
            ds_dict = load_dataset(repo)
            ds = ds_dict[target_split if target_split in ds_dict else next(iter(ds_dict.keys()))]

        if "image_2" in ds.column_names:
            ds = ds.filter(lambda img2: img2 is None, input_columns=["image_2"])
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    elif dataset_key == "seedbench":
        ds = load_dataset(repo, split=target_split)
        if "data_type" in ds.column_names:
            ds = ds.filter(lambda dt: dt == "image", input_columns=["data_type"])
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    elif dataset_key == "gqa":
        inst_ds = load_dataset(repo, "testdev_balanced_instructions", split=target_split)
        img_ds = load_dataset(repo, "testdev_balanced_images", split=target_split)
        img_map = {x["id"]: x["image"] for x in img_ds}
        ds = inst_ds.map(lambda ex: {"image": img_map.get(ex.get("imageId"))})
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    if config_name:
        ds = load_dataset(repo, config_name, split=target_split)
    else:
        ds = load_dataset(repo, split=target_split)

    if subset_size and len(ds) > subset_size:
        ds = ds.select(range(subset_size))
    return ds
