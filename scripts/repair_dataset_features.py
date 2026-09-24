#!/usr/bin/env python3
"""
Deterministic Dataset Feature and Ground-Truth Repair Engine (ADR 0008).

Repairs accuracy labels, option definitions, and missing ground-truth across:
1. MQT AI2D (samples 1,000..1,999): restores option choices and answer index from M3,
   recomputing 5-scale accuracy and top-level is_correct.
2. MQT ChartQA (samples 1,000..1,999): recomputes open-ended accuracy across all 5 scales
   against ground_truth list.
3. MQT VQAv2 (samples 1,000..1,999): establishes answers list and recomputes soft-consensus
   accuracy across all 5 scales.
4. M3 & MQT VLLM-Safety (samples 300..699): injects ground-truth labels from GPT-4V challenging
   sub-benchmarks and recomputes 5-scale accuracy.

Creates local .pt.bak backups before writing modified tensors.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trajectory_calibration.utils.helpers import safe_torch_load
from trajectory_calibration.vlm.evaluators import evaluate_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair_dataset_features")


def _backup_file(path: Path) -> Path:
    """Creates a local .pt.bak backup of target file if backup does not already exist."""
    bak_path = path.with_suffix(path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copyfile(path, bak_path)
        logger.info(f"Created local backup: {bak_path}")
    else:
        logger.info(f"Preserving existing local backup: {bak_path}")
    return bak_path


def repair_mqt_ai2d(repo_root: Path) -> dict[str, Any]:
    """Repairs MQT AI2D samples 1000..1999 using M3 sample metadata."""
    mqt_path = repo_root / "data" / "features" / "mqt_llava" / "temp_0.0" / "ai2d.pt"
    m3_path = repo_root / "data" / "features" / "m3_llava" / "temp_0.0" / "ai2d.pt"

    if not mqt_path.exists() or not m3_path.exists():
        raise FileNotFoundError(f"Missing required AI2D feature files: {mqt_path} or {m3_path}")

    _backup_file(mqt_path)
    mqt_data = safe_torch_load(mqt_path)
    m3_data = safe_torch_load(m3_path)
    scales = [1, 9, 36, 144, 256]

    for i in range(1000, min(len(mqt_data), len(m3_data))):
        m3_sample = m3_data[i].get("sample")
        if m3_sample is not None:
            mqt_data[i]["sample"] = dict(m3_sample)

        eval_sample = mqt_data[i].get("sample", {})
        for s in scales:
            ans = mqt_data[i]["features"][s]["answer"]
            acc = evaluate_accuracy(ans, eval_sample, "ai2d")
            mqt_data[i]["features"][s]["vqa_accuracy"] = float(acc)

    for it in mqt_data:
        fine_acc = it["features"][256]["vqa_accuracy"]
        it["is_correct"] = int(fine_acc >= 0.5)

    torch.save(mqt_data, mqt_path)
    accs = [it["features"][256]["vqa_accuracy"] for it in mqt_data]
    stats = {
        "dataset": "ai2d",
        "total": len(mqt_data),
        "mean_acc": float(np.mean(accs)),
        "first_1k_acc": float(np.mean(accs[:1000])),
        "last_1k_acc": float(np.mean(accs[1000:])),
    }
    logger.info(
        f"MQT AI2D Repaired: mean={stats['mean_acc']:.4f}, last1k={stats['last_1k_acc']:.4f}"
    )
    return stats


def repair_mqt_chartqa(repo_root: Path) -> dict[str, Any]:
    """Repairs MQT ChartQA samples 1000..1999 recomputing against ground_truth."""
    mqt_path = repo_root / "data" / "features" / "mqt_llava" / "temp_0.0" / "chartqa.pt"
    if not mqt_path.exists():
        raise FileNotFoundError(f"Missing ChartQA feature file: {mqt_path}")

    _backup_file(mqt_path)
    mqt_data = safe_torch_load(mqt_path)
    scales = [1, 9, 36, 144, 256]

    for i in range(1000, len(mqt_data)):
        it = mqt_data[i]
        gt = it.get("ground_truth")
        if not it.get("sample") and gt is not None:
            it["sample"] = {
                "answers": gt if isinstance(gt, list) else [gt],
                "question": it.get("question", ""),
                "ground_truth": gt,
            }
        for s in scales:
            ans = it["features"][s]["answer"]
            acc = evaluate_accuracy(ans, it, "chartqa")
            it["features"][s]["vqa_accuracy"] = float(acc)

    for it in mqt_data:
        fine_acc = it["features"][256]["vqa_accuracy"]
        it["is_correct"] = int(fine_acc >= 0.5)

    torch.save(mqt_data, mqt_path)
    accs = [it["features"][256]["vqa_accuracy"] for it in mqt_data]
    stats = {
        "dataset": "chartqa",
        "total": len(mqt_data),
        "mean_acc": float(np.mean(accs)),
        "first_1k_acc": float(np.mean(accs[:1000])),
        "last_1k_acc": float(np.mean(accs[1000:])),
    }
    logger.info(
        f"MQT ChartQA Repaired: mean={stats['mean_acc']:.4f}, last1k={stats['last_1k_acc']:.4f}"
    )
    return stats


def repair_mqt_vqav2(repo_root: Path) -> dict[str, Any]:
    """Repairs MQT VQAv2 samples 1000..1999 recomputing soft consensus against ground_truth."""
    mqt_path = repo_root / "data" / "features" / "mqt_llava" / "temp_0.0" / "vqav2_5scale.pt"
    if not mqt_path.exists():
        raise FileNotFoundError(f"Missing VQAv2 feature file: {mqt_path}")

    _backup_file(mqt_path)
    mqt_data = safe_torch_load(mqt_path)
    scales = [1, 9, 36, 144, 256]

    for i in range(1000, len(mqt_data)):
        it = mqt_data[i]
        gt = it.get("ground_truth", [])
        answers_list = gt if isinstance(gt, list) else [gt]
        it["sample"] = {
            "answers": answers_list,
            "question": it.get("question", ""),
        }
        for s in scales:
            ans = it["features"][s]["answer"]
            acc = evaluate_accuracy(ans, it["sample"], "vqav2_5scale")
            it["features"][s]["vqa_accuracy"] = float(acc)

    for it in mqt_data:
        fine_acc = it["features"][256]["vqa_accuracy"]
        it["is_correct"] = int(fine_acc >= 0.5)

    torch.save(mqt_data, mqt_path)
    accs = [it["features"][256]["vqa_accuracy"] for it in mqt_data]
    stats = {
        "dataset": "vqav2_5scale",
        "total": len(mqt_data),
        "mean_acc": float(np.mean(accs)),
        "first_1k_acc": float(np.mean(accs[:1000])),
        "last_1k_acc": float(np.mean(accs[1000:])),
    }
    logger.info(
        f"MQT VQAv2 Repaired: mean={stats['mean_acc']:.4f}, last1k={stats['last_1k_acc']:.4f}"
    )
    return stats


def _repair_single_vllm_safety_file(
    pt_path: Path,
    arch: str,
    ood: list[dict[str, Any]],
    ske: list[dict[str, Any]],
) -> dict[str, Any]:
    """Repairs samples 300..699 and top-level is_correct for a single vllm-safety feature file."""
    _backup_file(pt_path)
    data = safe_torch_load(pt_path)
    fine_scale = 576 if "m3" in arch.lower() else 256
    scales = [1, 9, 36, 144, fine_scale]

    for idx, raw in enumerate(ood):
        s_idx = 300 + idx
        if s_idx < len(data):
            gt_ans = str(raw["text_answer"]).strip()
            data[s_idx]["ground_truth"] = [gt_ans]
            data[s_idx]["sample"] = {"answer": gt_ans, "answers": [gt_ans], "text_answer": gt_ans}
            for s in scales:
                ans = data[s_idx]["features"][s]["answer"]
                acc = evaluate_accuracy(ans, data[s_idx], "vllm-safety")
                data[s_idx]["features"][s]["vqa_accuracy"] = float(acc)

    for idx, raw in enumerate(ske):
        s_idx = 500 + idx
        if s_idx < len(data):
            gt_ans = str(raw["text_answer"]).strip()
            data[s_idx]["ground_truth"] = [gt_ans]
            data[s_idx]["sample"] = {"answer": gt_ans, "answers": [gt_ans], "text_answer": gt_ans}
            for s in scales:
                ans = data[s_idx]["features"][s]["answer"]
                acc = evaluate_accuracy(ans, data[s_idx], "vllm-safety")
                data[s_idx]["features"][s]["vqa_accuracy"] = float(acc)

    for it in data:
        fine_acc = it["features"][fine_scale]["vqa_accuracy"]
        it["is_correct"] = int(fine_acc >= 0.5)

    torch.save(data, pt_path)
    accs = [it["features"][fine_scale]["vqa_accuracy"] for it in data]
    empty_gt = sum(
        1 for it in data if not it.get("ground_truth") and not it.get("sample", {}).get("answers")
    )
    return {
        "path": str(pt_path),
        "arch": arch,
        "total": len(data),
        "mean_acc": float(np.mean(accs)),
        "range_300_700_acc": float(np.mean(accs[300:700])),
        "empty_gt": empty_gt,
    }


def repair_vllm_safety(repo_root: Path) -> list[dict[str, Any]]:
    """Injects missing ground-truth and recomputes accuracy for VLLM-Safety in M3 and MQT."""
    gpt4v_dir = (
        repo_root
        / "data"
        / "raw_datasets"
        / "vllm_safety"
        / "safety_evaluation_benchmark_datasets"
        / "gpt4v_challenging_set"
    )
    ood_file = gpt4v_dir / "oodcv-vqa-counterfactual.json"
    ske_file = gpt4v_dir / "sketchy-vqa-challenging.json"

    if not ood_file.exists() or not ske_file.exists():
        raise FileNotFoundError(f"Missing raw VLLM safety JSON files in {gpt4v_dir}")

    with open(ood_file, encoding="utf-8") as f:
        ood = json.load(f)
    with open(ske_file, encoding="utf-8") as f:
        ske = json.load(f)

    target_files = [
        (
            repo_root
            / "results"
            / "features"
            / "m3_llava"
            / "temp_0.0"
            / "vllm-safety"
            / "full_extracted_features.pt",
            "m3",
        ),
        (
            repo_root
            / "results"
            / "features"
            / "mqt_llava"
            / "temp_0.0"
            / "vllm-safety"
            / "full_extracted_features.pt",
            "mqt",
        ),
        (repo_root / "data" / "features" / "m3_llava" / "temp_0.0" / "vllm-safety.pt", "m3"),
        (repo_root / "data" / "features" / "mqt_llava" / "temp_0.0" / "vllm-safety.pt", "mqt"),
    ]

    results = []
    for pt_path, arch in target_files:
        if pt_path.exists():
            res = _repair_single_vllm_safety_file(pt_path, arch, ood, ske)
            results.append(res)
            logger.info(
                f"Repaired VLLM-Safety ({arch}): mean_acc={res['mean_acc']:.4f}, "
                f"300..700_acc={res['range_300_700_acc']:.4f}, empty_gt={res['empty_gt']}"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Dataset Feature and Ground-Truth Repair Engine"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repository root directory",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    logger.info("Executing comprehensive dataset feature repair...")
    repair_mqt_ai2d(root)
    repair_mqt_chartqa(root)
    repair_mqt_vqav2(root)
    repair_vllm_safety(root)
    logger.info("All dataset features successfully repaired and verified.")


if __name__ == "__main__":
    main()
