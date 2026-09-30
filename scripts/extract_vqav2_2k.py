#!/usr/bin/env python3
"""Dedicated VQAv2 Feature Extraction (2,000 Canonical Samples).

Extracts multi-scale logit trajectory features across scales:
  - M3-LLaVA: [1, 9, 36, 144, 576]
  - MQT-LLaVA: [1, 9, 36, 144, 256]
Matching the exact sample sequence in data/canonical_manifest_all.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from trajectory_calibration.utils.config import ARCH_SCALES
from trajectory_calibration.utils.helpers import safe_torch_load, set_seed
from trajectory_calibration.vlm.evaluators import evaluate_accuracy
from trajectory_calibration.vlm.formatting import format_question
from trajectory_calibration.vlm.wrapper import UnifiedVLMWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_vqav2_2k")


def extract_vqav2_record(
    wrapper: UnifiedVLMWrapper,
    sample: dict[str, Any],
    qid: str,
    gen_temp: float = 0.0,
    scales: list[int] | None = None,
) -> dict[str, Any]:
    """Sweeps trajectory scales and evaluates VQA soft-consensus accuracy."""
    target_scales = scales or wrapper.scales
    raw_img = sample.get("image")
    if raw_img is not None and hasattr(raw_img, "convert"):
        image: Image.Image = raw_img.convert("RGB")
    elif isinstance(raw_img, Image.Image):
        image = raw_img.convert("RGB")
    else:
        image = Image.new("RGB", (336, 336), (0, 0, 0))

    question_text = sample.get("question", "")
    formatted_q = format_question(sample, "vqav2")
    sweep_results = wrapper.sweep(
        image=image, question=formatted_q, scales=target_scales, gen_temperature=gen_temp
    )

    feats: dict[int, dict[str, Any]] = {}
    for s in target_scales:
        res = sweep_results[s]
        acc = evaluate_accuracy(res["answer"], sample, "vqav2")
        conf = float(res["conf_softmax"])
        feats[s] = {
            "answer": str(res["answer"]).strip(),
            "vqa_accuracy": float(acc),
            "conf_softmax": conf,
            "avg_log_prob": float(np.log(max(1e-7, conf))),
            "margin": float(res["margin"]),
        }

    fine_scale = target_scales[-1]
    fine_acc = float(feats[fine_scale]["vqa_accuracy"])
    return {
        "question_id": str(qid),
        "question": question_text,
        "ground_truth": sample.get("answers", []),
        "dataset": "vqav2",
        "temperature": gen_temp,
        "sample": {
            "question_type": sample.get("question_type"),
            "multiple_choice_answer": sample.get("multiple_choice_answer"),
            "answers": sample.get("answers"),
            "image_id": sample.get("image_id"),
            "answer_type": sample.get("answer_type"),
            "question_id": sample.get("question_id"),
            "question": question_text,
        },
        "features": feats,
        "is_correct": fine_acc,
        "vqa_accuracy": fine_acc,
    }


def load_canonical_items(repo_root: Path, target_count: int) -> list[dict[str, Any]]:
    """Loads target_count canonical VQAv2 items from data/canonical_manifest_all.json."""
    manifest_p = repo_root / "data" / "canonical_manifest_all.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Missing canonical manifest at: {manifest_p}")
    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)
    canonical_items = manifest.get("vqav2", [])[:target_count]
    logger.info(f"Loaded {len(canonical_items)} canonical VQAv2 items from manifest.")
    return canonical_items


def resolve_existing_checkpoint(
    out_file: Path, canonical_qids: list[str], target_count: int
) -> list[dict[str, Any]]:
    """Loads existing records if matching canonical prefix, otherwise creates backup."""
    if not out_file.exists():
        return []
    try:
        existing = safe_torch_load(out_file)
        existing_qids = [str(r.get("question_id")) for r in existing]
        expected_prefix = canonical_qids[: len(existing)]
        if existing_qids == expected_prefix:
            logger.info(f"Resuming: found {len(existing)} matching canonical samples.")
            return existing
        bak_file = out_file.with_suffix(".pt.legacy_bak")
        shutil.move(out_file, bak_file)
        logger.warning(
            f"Existing checkpoint QID mismatch against canonical manifest! "
            f"Backed up to {bak_file} and starting fresh."
        )
    except Exception as e:
        logger.warning(f"Could not load checkpoint ({e}). Starting fresh.")
    return []


def run_extraction(
    repo_root: Path,
    arch: str = "m3",
    model_path: str | None = None,
    gen_temp: float = 0.0,
    target_count: int = 2000,
    save_interval: int = 50,
) -> None:
    """Runs extraction for all 2,000 canonical VQAv2 samples with checkpoint/resume."""
    arch = arch.lower()
    scales = ARCH_SCALES[arch]
    default_model = "mucai/llava-v1.5-7b-m3" if arch == "m3" else "gordonhu/MQT-LLaVA-7b"
    resolved_model = model_path or default_model

    canonical_items = load_canonical_items(repo_root, target_count)
    canonical_qids = [str(x["question_id"]) for x in canonical_items]

    out_dir = repo_root / "data" / "features" / f"{arch}_llava" / f"temp_{gen_temp:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "vqav2_5scale.pt"

    existing_records = resolve_existing_checkpoint(out_file, canonical_qids, target_count)
    if len(existing_records) >= target_count:
        logger.info(f"Extraction already complete ({len(existing_records)}/{target_count}). Done!")
        return

    processed_qids = {str(r["question_id"]) for r in existing_records}
    logger.info("Loading VQAv2 validation split from lmms-lab/vqav2...")
    ds_val = load_dataset("lmms-lab/vqav2", split="validation")

    logger.info(f"Initializing UnifiedVLMWrapper ({arch.upper()}) from {resolved_model}...")
    wrapper = UnifiedVLMWrapper(
        model_path=resolved_model, arch=arch, fine_scale=scales[-1], precision="fp16"
    )

    records = list(existing_records)
    for idx, item in enumerate(tqdm(canonical_items, desc=f"Extracting {arch.upper()} VQAv2")):
        qid_str = str(item["question_id"])
        if qid_str in processed_qids:
            continue

        raw_sample = ds_val[idx]
        assert str(raw_sample["question_id"]) == qid_str, (
            f"Dataset index {idx} QID {raw_sample['question_id']} does not match canonical QID {qid_str}!"
        )

        rec = extract_vqav2_record(
            wrapper=wrapper, sample=raw_sample, qid=qid_str, gen_temp=gen_temp, scales=scales
        )
        rec["sample_idx"] = idx
        records.append(rec)
        processed_qids.add(qid_str)

        if len(records) % save_interval == 0:
            torch.save(records, out_file)
            logger.info(f"Checkpoint saved: {len(records)}/{target_count} samples.")

    torch.save(records, out_file)
    logger.info(f"Extraction complete! Saved {len(records)} samples to {out_file}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="VQAv2 2K Canonical Feature Extraction")
    parser.add_argument("--arch", type=str, default="m3", choices=["m3", "mqt"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--gen_temperature", type=float, default=0.0)
    parser.add_argument("--target_count", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=50)
    args = parser.parse_args()

    set_seed(42)
    repo_root = Path(__file__).resolve().parent.parent
    run_extraction(
        repo_root=repo_root,
        arch=args.arch,
        model_path=args.model_path,
        gen_temp=args.gen_temperature,
        target_count=args.target_count,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
