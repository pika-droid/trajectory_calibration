#!/usr/bin/env python3
"""
Dedicated M3-LLaVA Feature Extraction on ScienceQA (2,000 Canonical Samples).

Extracts multi-scale logit trajectory features across scales [1, 9, 36, 144, 576]
for M3-LLaVA (7B) on the official ScienceQA test split, matching the exact
sample sequence in data/canonical_manifest_all.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

from trajectory_calibration.utils.helpers import safe_torch_load, set_seed
from trajectory_calibration.vlm.evaluators import evaluate_accuracy
from trajectory_calibration.vlm.formatting import format_question
from trajectory_calibration.vlm.wrapper import UnifiedVLMWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_m3_scienceqa_2k")

M3_SCALES = [1, 9, 36, 144, 576]


def extract_scienceqa_record(
    wrapper: UnifiedVLMWrapper,
    sample: dict[str, Any],
    qid: str,
    gen_temp: float = 0.0,
) -> dict[str, Any]:
    """Sweeps scales [1, 9, 36, 144, 576] and evaluates accuracy against ground truth."""
    raw_img = sample.get("image")
    if raw_img is not None and hasattr(raw_img, "convert"):
        image: Image.Image = raw_img.convert("RGB")
    elif isinstance(raw_img, Image.Image):
        image = raw_img.convert("RGB")
    else:
        image = Image.new("RGB", (336, 336), (0, 0, 0))

    formatted_q = format_question(sample, "scienceqa")
    sweep_results = wrapper.sweep(
        image=image, question=formatted_q, scales=M3_SCALES, gen_temperature=gen_temp
    )

    feats: dict[int, dict[str, Any]] = {}
    for s in M3_SCALES:
        res = sweep_results[s]
        acc = evaluate_accuracy(res["answer"], sample, "scienceqa")
        conf = float(res["conf_softmax"])
        feats[s] = {
            "answer": str(res["answer"]).strip(),
            "vqa_accuracy": float(acc),
            "conf_softmax": conf,
            "avg_log_prob": float(np.log(max(1e-7, conf))),
            "margin": float(res["margin"]),
        }

    fine_acc = float(feats[576]["vqa_accuracy"])
    return {
        "question_id": str(qid),
        "dataset": "scienceqa",
        "temperature": gen_temp,
        "is_correct": fine_acc,
        "vqa_accuracy": fine_acc,
        "answer_type": "mc_index",
        "sample": {
            "question": sample.get("question", ""),
            "choices": sample.get("choices", []),
            "answer": sample.get("answer", 0),
        },
        "features": feats,
    }


def run_extraction(
    repo_root: Path,
    model_path: str = "mucai/llava-v1.5-7b-m3",
    gen_temp: float = 0.0,
    target_count: int = 2000,
    save_interval: int = 50,
) -> None:
    """Runs extraction for all 2,000 canonical ScienceQA samples with checkpoint/resume."""
    manifest_p = repo_root / "data" / "canonical_manifest_all.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Missing canonical manifest at: {manifest_p}")

    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)

    canonical_items = manifest.get("scienceqa", [])[:target_count]
    logger.info(f"Loaded {len(canonical_items)} canonical ScienceQA items from manifest.")

    out_dir = repo_root / "data" / "features" / "m3_llava" / f"temp_{gen_temp:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "scienceqa.pt"

    existing_records: list[dict[str, Any]] = []
    processed_qids: set[str] = set()
    if out_file.exists():
        try:
            existing_records = safe_torch_load(out_file)
            processed_qids = {str(r["question_id"]) for r in existing_records}
            logger.info(f"Resuming: found {len(existing_records)} already-extracted samples.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}). Starting fresh.")
            existing_records = []

    if len(existing_records) >= target_count:
        logger.info(f"Extraction already complete ({len(existing_records)}/{target_count}). Done!")
        return

    logger.info("Loading ScienceQA test split from lmms-lab/ScienceQA...")
    ds_test = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split="test")

    logger.info(f"Initializing UnifiedVLMWrapper (M3-LLaVA) from {model_path}...")
    wrapper = UnifiedVLMWrapper(model_path=model_path, arch="m3", fine_scale=576, precision="fp16")

    records = list(existing_records)
    for idx, item in enumerate(tqdm(canonical_items, desc="Extracting M3 ScienceQA")):
        qid_str = str(item["question_id"])
        if qid_str in processed_qids:
            continue

        raw_sample = ds_test[int(qid_str)]
        rec = extract_scienceqa_record(
            wrapper=wrapper, sample=raw_sample, qid=qid_str, gen_temp=gen_temp
        )
        records.append(rec)
        processed_qids.add(qid_str)

        if len(records) % save_interval == 0:
            torch.save(records, out_file)
            logger.info(f"Checkpoint saved: {len(records)}/{target_count} samples.")

    torch.save(records, out_file)
    logger.info(f"Extraction complete! Saved {len(records)} samples to {out_file}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M3-LLaVA ScienceQA 2K Canonical Feature Extraction"
    )
    parser.add_argument("--model_path", type=str, default="mucai/llava-v1.5-7b-m3")
    parser.add_argument("--gen_temperature", type=float, default=0.0)
    parser.add_argument("--target_count", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=50)
    args = parser.parse_args()

    set_seed(42)
    repo_root = Path(__file__).resolve().parent.parent
    run_extraction(
        repo_root=repo_root,
        model_path=args.model_path,
        gen_temp=args.gen_temperature,
        target_count=args.target_count,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
