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
import math
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


def validate_feature_records(
    records: list[dict[str, Any]],
    arch: str,
    canonical_qids: list[str] | None = None,
) -> dict[str, Any]:
    """Exhaustively validates feature records for NaNs, Infs, blanks, and schema integrity."""
    if not records:
        raise ValueError("Cannot validate empty records list!")

    expected_scales = ARCH_SCALES[arch.lower()]
    nan_count = 0
    inf_count = 0
    blank_ans_count = 0
    all_confs: list[float] = []
    all_accs: list[float] = []

    for idx, rec in enumerate(records):
        qid = str(rec.get("question_id", ""))
        if not qid:
            raise ValueError(f"Record at index {idx} has missing or empty question_id!")

        if canonical_qids is not None and idx < len(canonical_qids):
            expected_qid = str(canonical_qids[idx])
            if qid != expected_qid:
                raise ValueError(
                    f"Sample {idx} QID mismatch: got {qid}, expected {expected_qid} from canonical manifest!"
                )

        q_text = str(rec.get("question", "")).strip()
        if not q_text:
            raise ValueError(f"Sample {idx} (QID={qid}) has empty question text!")

        gt = rec.get("ground_truth") or rec.get("sample", {}).get("answers")
        if not gt or not isinstance(gt, list) or len(gt) == 0:
            raise ValueError(f"Sample {idx} (QID={qid}) has missing or empty ground truth answers!")

        feats = rec.get("features", {})
        if not isinstance(feats, dict):
            raise ValueError(f"Sample {idx} (QID={qid}) has invalid features dict!")

        for s in expected_scales:
            if s not in feats:
                raise ValueError(
                    f"Sample {idx} (QID={qid}) missing expected scale {s} in features!"
                )
            f_scale = feats[s]
            ans = str(f_scale.get("answer", "")).strip()
            if not ans:
                blank_ans_count += 1

            for metric_key in ("conf_softmax", "margin", "avg_log_prob", "vqa_accuracy"):
                if metric_key not in f_scale:
                    continue
                val = f_scale[metric_key]
                if val is None or not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Sample {idx} (scale {s}) metric '{metric_key}' is invalid/missing: {val}"
                    )
                val_f = float(val)
                if math.isnan(val_f):
                    nan_count += 1
                    raise ValueError(f"Sample {idx} (scale {s}) metric '{metric_key}' is NaN!")
                if math.isinf(val_f):
                    inf_count += 1
                    raise ValueError(f"Sample {idx} (scale {s}) metric '{metric_key}' is Inf!")

            conf = float(f_scale["conf_softmax"])
            if not (0.0 <= conf <= 1.0):
                raise ValueError(
                    f"Sample {idx} (scale {s}) conf_softmax={conf} out of [0, 1] range!"
                )
            all_confs.append(conf)

        top_acc = rec.get("vqa_accuracy")
        if top_acc is not None:
            top_acc_f = float(top_acc)
            if math.isnan(top_acc_f) or math.isinf(top_acc_f):
                raise ValueError(f"Sample {idx} top-level vqa_accuracy is invalid: {top_acc}")
            all_accs.append(top_acc_f)

    fine_scale = expected_scales[-1]
    fine_accs = [
        float(r["features"][fine_scale].get("vqa_accuracy", 0.0))
        for r in records
        if fine_scale in r.get("features", {})
    ]
    summary = {
        "total_records": len(records),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "blank_answers": blank_ans_count,
        "min_conf": min(all_confs) if all_confs else 0.0,
        "max_conf": max(all_confs) if all_confs else 0.0,
        "mean_accuracy": float(np.mean(fine_accs)) if fine_accs else 0.0,
    }
    return summary


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
        logger.info(f"Extraction already complete ({len(existing_records)}/{target_count}).")
        summary = validate_feature_records(
            existing_records, arch=arch, canonical_qids=canonical_qids
        )
        logger.info(
            f"Validation PASSED for {len(existing_records)} samples: "
            f"0 NaNs, 0 Infs, {summary['blank_answers']} blanks, "
            f"mean fine acc={summary['mean_accuracy']:.2%}"
        )
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
    summary = validate_feature_records(records, arch=arch, canonical_qids=canonical_qids)
    logger.info(
        f"Validation PASSED for {len(records)} samples: "
        f"0 NaNs, 0 Infs, {summary['blank_answers']} blanks, "
        f"mean fine acc={summary['mean_accuracy']:.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VQAv2 2K Canonical Feature Extraction")
    parser.add_argument("--arch", type=str, default="m3", choices=["m3", "mqt"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--gen_temperature", type=float, default=0.0)
    parser.add_argument("--target_count", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument(
        "--validate_only", action="store_true", help="Only validate existing .pt file"
    )
    args = parser.parse_args()

    set_seed(42)
    repo_root = Path(__file__).resolve().parent.parent
    if args.validate_only:
        arch = args.arch.lower()
        out_file = (
            repo_root
            / "data"
            / "features"
            / f"{arch}_llava"
            / f"temp_{args.gen_temperature:.1f}"
            / "vqav2_5scale.pt"
        )
        if not out_file.exists():
            raise FileNotFoundError(f"Feature file not found for validation: {out_file}")
        records = safe_torch_load(out_file)
        canonical_items = load_canonical_items(repo_root, args.target_count)
        canonical_qids = [str(x["question_id"]) for x in canonical_items]
        summary = validate_feature_records(records, arch=arch, canonical_qids=canonical_qids)
        logger.info(
            f"Validation PASSED: {summary['total_records']} samples, "
            f"0 NaNs, 0 Infs, {summary['blank_answers']} blanks, "
            f"mean fine acc={summary['mean_accuracy']:.2%}"
        )
        return

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
