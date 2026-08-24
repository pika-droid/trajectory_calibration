"""
Unified Multi-Dataset Registry and Ground-Truth Evaluation Engine.

Covers all 14 vision-language benchmarks with non-withheld validation splits
and standardized multi-format evaluation (10-annotator soft consensus,
option letter matching, index matching, and open-ended text normalization).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from PIL import Image

from trajectory_calibration.utils.helpers import clean_text

logger = logging.getLogger("trajectory_calibration.vlm.datasets")

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
    "vqav2_5scale": {
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


def evaluate_accuracy(pred_answer: str, sample: dict[str, Any], dataset_key: str) -> float:
    """
    Computes VQA accuracy score against ground-truth labels.
    """
    pred_clean = pred_answer.strip().lower()
    pred_norm = clean_text(pred_clean)
    cfg = DATASET_REGISTRY.get(dataset_key, {"answer_type": "open"})
    ans_type = cfg.get("answer_type", "open")

    if ans_type == "open":
        gt_ans = str(sample.get("answer", sample.get("label", sample.get("ground_truth", "")))).strip()
        if not gt_ans:
            return 0.0
        gt_clean = clean_text(gt_ans)
        if pred_norm == gt_clean or pred_clean == gt_ans.lower():
            return 1.0
        if len(pred_norm) >= 3 and (pred_norm in gt_clean or gt_clean in pred_norm):
            return 1.0
        return 0.0

    elif ans_type == "list_soft":
        gt_answers = sample.get("answers", sample.get("annotations", []))
        if not gt_answers:
            gt_ans = sample.get("answer", sample.get("label"))
            gt_answers = [gt_ans] if gt_ans is not None else []

        match_count = 0
        for gt in gt_answers:
            gt_text = gt.get("answer", "") if isinstance(gt, dict) else str(gt)
            gt_clean = clean_text(gt_text)
            if (
                gt_clean == pred_norm
                or pred_clean == str(gt_text).strip().lower()
                or (len(pred_norm) >= 3 and (pred_norm in gt_clean or gt_clean in pred_norm))
            ):
                match_count += 1
        return min(1.0, match_count / 3.0) if match_count > 0 else 0.0

    elif ans_type == "mc_index":
        correct_idx = sample.get("answer", sample.get("label", sample.get("correct_choice")))
        options = sample.get("choices", sample.get("options", []))
        if correct_idx is not None and str(correct_idx).isdigit():
            idx_int = int(correct_idx)
            if 0 <= idx_int < len(options):
                correct_letter = chr(65 + idx_int).lower()
                correct_option_text = clean_text(str(options[idx_int]))
                if pred_clean.startswith(correct_letter) or pred_clean == correct_letter or pred_norm == correct_option_text:
                    return 1.0
        return 0.0

    elif ans_type == "mc_letter":
        gt_ans = str(sample.get("answer", sample.get("label", ""))).strip().upper()
        if not gt_ans:
            return 0.0

        if pred_clean == gt_ans.lower() or pred_clean.startswith(gt_ans.lower()):
            return 1.0

        options = sample.get("options", sample.get("choices", []))
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = []

        if isinstance(options, list) and len(options) > 0:
            target_idx = ord(gt_ans[0]) - ord("A")
            if 0 <= target_idx < len(options):
                target_option = clean_text(str(options[target_idx]))
                if pred_norm == target_option:
                    return 1.0

        letter_idx_map = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
        if gt_ans in letter_idx_map:
            target_col = letter_idx_map[gt_ans]
            if target_col in sample and sample[target_col]:
                if pred_norm == clean_text(str(sample[target_col])):
                    return 1.0

        return 0.0

    return 1.0 if pred_norm == clean_text(str(sample.get("answer", sample.get("label", "")))) else 0.0


def format_question(sample: dict[str, Any], dataset_key: str) -> str:
    """Formats question text and options consistently for multiple-choice tasks."""
    question = sample.get(
        "question",
        sample.get(
            "problem",
            sample.get("query", sample.get("text", sample.get("prompt", sample.get("user_query", "")))),
        ),
    )
    if isinstance(question, (list, tuple)) and len(question) > 0:
        question = question[0]
    question = str(question).strip()

    if dataset_key == "ai2d":
        options = sample.get("options", [])
        if options and isinstance(options, list):
            opts_str = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            question = f"{question}\n{opts_str}\nAnswer with the option letter."

    elif dataset_key == "mmbench":
        opts = []
        for ltr in ["A", "B", "C", "D"]:
            val = sample.get(ltr)
            if val and not (isinstance(val, float) and str(val) == "nan"):
                opts.append(f"({ltr}) {val}")
        if opts:
            question = f"{question}\n" + "\n".join(opts) + "\nAnswer with the option letter."

    elif dataset_key == "mmmu":
        options = sample.get("options", [])
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = []
        if isinstance(options, list) and len(options) > 0:
            opts_str = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            question = f"{question}\n{opts_str}\nAnswer with the option letter."

    elif dataset_key == "scienceqa":
        choices = sample.get("choices", sample.get("options", []))
        if choices and isinstance(choices, list):
            opts_str = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(choices)])
            question = f"{question}\n{opts_str}\nAnswer with the option letter."

    elif dataset_key == "seedbench":
        opts = []
        for i, col in enumerate(["choice_a", "choice_b", "choice_c", "choice_d"]):
            val = sample.get(col)
            if val and isinstance(val, str) and val.strip():
                opts.append(f"({chr(65 + i)}) {val.strip()}")
        if opts:
            question = f"{question}\n" + "\n".join(opts) + "\nAnswer with the option letter."

    return question


def load_image_from_sample(sample: dict[str, Any]) -> Image.Image | None:
    """Extracts and normalizes PIL Image from heterogeneous sample keys."""
    img_raw = None
    for k in ["images", "image", "image_1", "img", "image_path", "img_path", "picture"]:
        if k in sample and sample[k] is not None:
            img_raw = sample[k]
            break

    if img_raw is None:
        return None

    if isinstance(img_raw, Image.Image):
        return img_raw.convert("RGB")
    elif isinstance(img_raw, (list, tuple)) and len(img_raw) > 0:
        first = img_raw[0]
        if isinstance(first, Image.Image):
            return first.convert("RGB")

    return None


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
            ds = load_dataset(repo, config_name, split=target_split) if config_name else load_dataset(repo, split=target_split)
        except Exception:
            ds_dict = load_dataset(repo)
            ds = ds_dict[target_split if target_split in ds_dict else list(ds_dict.keys())[0]]

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
