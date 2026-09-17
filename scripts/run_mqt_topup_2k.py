#!/usr/bin/env python3
"""
MQT-LLaVA Single-Pass Top-Up to 2,000 Samples (ADR 0006).

Extracts single-pass multi-scale features for MQT-LLaVA (scales: 1, 9, 36, 144, 256)
on samples 1,000..1,999 from data/canonical_manifest_all.json for:
- ai2d
- chartqa
- docvqa
- vqav2_5scale
concatenating them with existing 1,000 samples to achieve exact 2,000-sample parity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path for direct python execution
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Configure PyTorch CUDA memory allocator
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

from trajectory_calibration.utils.helpers import safe_torch_load, set_seed
from trajectory_calibration.vlm.evaluators import evaluate_accuracy
from trajectory_calibration.vlm.formatting import format_question, load_image_from_sample
from trajectory_calibration.vlm.registry import load_hf_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_mqt_topup_2k")

MQT_TOPUP_DATASETS = ["ai2d", "chartqa", "docvqa", "vqav2_5scale"]
MQT_SCALES = [1, 9, 36, 144, 256]


def _generate_mock_mqt_record(
    manifest_item: dict[str, Any],
    dataset_key: str,
    sample_idx: int,
) -> dict[str, Any]:
    """Generates a synthetic MQT single-pass multi-scale feature record."""
    is_corr = random.random() > 0.35
    base_c = random.uniform(0.7, 0.95) if is_corr else random.uniform(0.2, 0.6)
    feats = {}
    for s in MQT_SCALES:
        c = min(0.99, max(0.01, base_c + random.gauss(0.0, 0.04)))
        m = random.uniform(0.1, 0.5)
        ans = manifest_item["answers"][0] if manifest_item["answers"] and is_corr else "unrelated"
        acc = 1.0 if (is_corr and s == 256) else (1.0 if random.random() > 0.5 else 0.0)
        feats[s] = {
            "conf_softmax": float(c),
            "margin": float(m),
            "answer": ans,
            "vqa_accuracy": float(acc),
        }

    return {
        "question_id": manifest_item["question_id"],
        "question": manifest_item["question"],
        "ground_truth": manifest_item["answers"],
        "dataset": dataset_key,
        "sample_idx": sample_idx,
        "features": feats,
    }


def _extract_single_sample_features(
    wrapper: Any,
    manifest_item: dict[str, Any],
    dataset_key: str,
    sample_idx: int,
    gen_temp: float = 0.0,
    hf_dataset: Any = None,
) -> dict[str, Any] | None:
    """Extracts real MQT multi-scale features for a single manifest item."""
    image = load_image_from_sample(manifest_item)
    if image is None and hf_dataset is not None and sample_idx < len(hf_dataset):
        image = load_image_from_sample(hf_dataset[sample_idx])
    if image is None:
        logger.warning(f"Could not load image for {manifest_item.get('question_id')}. Skipping.")
        return None

    question = format_question(manifest_item, dataset_key)
    # MQT Query Transformer requires vt = 256
    scale_results = wrapper.sweep(image, question, gen_temperature=gen_temp)

    feats = {}
    for s, res in scale_results.items():
        acc = evaluate_accuracy(res["answer"], manifest_item, dataset_key)
        feats[s] = {
            "conf_softmax": res["conf_softmax"],
            "margin": res["margin"],
            "answer": res["answer"],
            "vqa_accuracy": acc,
        }

    return {
        "question_id": manifest_item["question_id"],
        "question": question,
        "ground_truth": manifest_item.get("answers", []),
        "dataset": dataset_key,
        "sample_idx": sample_idx,
        "features": feats,
    }


def topup_dataset(
    dataset_name: str,
    manifest_items: list[dict[str, Any]],
    output_dir: Path,
    wrapper: Any = None,
    mock: bool = False,
    gen_temp: float = 0.0,
) -> int:
    """Tops up a single MQT dataset to 2,000 samples."""
    pt_path = output_dir / f"{dataset_name}.pt"
    tmp_path = pt_path.with_suffix(".pt.tmp")

    existing_data: list[dict[str, Any]] = []
    if pt_path.exists():
        try:
            existing_data = safe_torch_load(pt_path)
            logger.info(f"Loaded existing {len(existing_data)} samples for {dataset_name}.")
        except Exception as e:
            logger.warning(f"Failed to load existing {pt_path}: {e}")

    if len(existing_data) >= 2000:
        logger.info(f"{dataset_name} already has {len(existing_data)} samples. No top-up needed.")
        return 0

    needed = 2000 - len(existing_data)
    start_idx = len(existing_data)
    items_to_process = manifest_items[start_idx:2000]

    logger.info(
        f"Topping up {dataset_name}: adding {len(items_to_process)} samples (indices {start_idx}..1999)..."
    )
    records = list(existing_data)

    hf_ds = None
    if not mock:
        try:
            hf_ds_key = "vqav2" if dataset_name == "vqav2_5scale" else dataset_name
            hf_ds = load_hf_dataset(hf_ds_key)
        except Exception as e:
            logger.warning(f"Could not preload HF dataset for {dataset_name}: {e}")

    for i, item in enumerate(tqdm(items_to_process, desc=f"Top-up {dataset_name}")):
        cur_idx = start_idx + i
        if mock:
            rec = _generate_mock_mqt_record(item, dataset_name, cur_idx)
        else:
            rec = _extract_single_sample_features(
                wrapper, item, dataset_name, cur_idx, gen_temp=gen_temp, hf_dataset=hf_ds
            )
            if rec is None:
                continue

        records.append(rec)
        if len(records) % 50 == 0:
            torch.save(records, tmp_path)

    torch.save(records, tmp_path)
    tmp_path.replace(pt_path)
    logger.info(f"Completed {dataset_name}. Final total samples: {len(records)} -> {pt_path}")
    return len(items_to_process)


def run_topup(
    repo_root: Path,
    model_path: str = "gordonhu/MQT-LLaVA-7b",
    datasets: list[str] | None = None,
    mock: bool = False,
    gen_temp: float = 0.0,
) -> None:
    """Orchestrates MQT top-up across specified datasets."""
    manifest_path = repo_root / "data" / "canonical_manifest_all.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Canonical manifest not found: {manifest_path}. Run build_canonical_manifest.py first."
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    out_dir = repo_root / "data" / "features" / "mqt_llava" / "temp_0.0"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ds = datasets or MQT_TOPUP_DATASETS
    wrapper = None

    if not mock:
        from trajectory_calibration.vlm.wrapper import UnifiedVLMWrapper

        logger.info(f"Initializing UnifiedVLMWrapper for MQT ({model_path})...")
        wrapper = UnifiedVLMWrapper(model_path=model_path, arch="mqt", fine_scale=256)

    for ds_name in target_ds:
        manifest_key = "vqav2" if ds_name == "vqav2_5scale" else ds_name
        if manifest_key not in manifest:
            logger.warning(f"Dataset key {manifest_key} not in manifest. Skipping.")
            continue

        items = manifest[manifest_key]
        topup_dataset(ds_name, items, out_dir, wrapper=wrapper, mock=mock, gen_temp=gen_temp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-up MQT-LLaVA Single-Pass Features to 2,000 Samples"
    )
    parser.add_argument("--model_path", type=str, default="gordonhu/MQT-LLaVA-7b")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic mock samples")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset to top up")
    parser.add_argument("--gen_temperature", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(42)
    repo_root = Path(__file__).resolve().parent.parent
    ds_list = [args.dataset] if args.dataset else None
    run_topup(
        repo_root=repo_root,
        model_path=args.model_path,
        datasets=ds_list,
        mock=args.mock,
        gen_temp=args.gen_temperature,
    )


if __name__ == "__main__":
    main()
