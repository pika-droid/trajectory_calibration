#!/usr/bin/env python3
"""
Build Canonical Manifest for Strict Sample Parity (ADR 0006).

Extracts the exact ordered sequence of question IDs, questions, answers, and image pointers
across all 7 Core VLM benchmarks (2,000 samples each) and 2 Robustness benchmarks
(AVQA: 2,000 samples, VLLM Safety: 1,900 samples), creating data/canonical_manifest_all.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_canonical_manifest")

DATASET_TARGETS: dict[str, int] = {
    "ai2d": 2000,
    "chartqa": 2000,
    "docvqa": 2000,
    "scienceqa": 2000,
    "textvqa": 2000,
    "vizwiz-vqa": 2000,
    "vqav2": 2000,
    "avqa": 2000,
    "vllm-safety": 1900,
}


def _normalize_answers(raw_gt: Any) -> list[str]:
    """Normalizes raw ground truth into a standardized list of string answers."""
    if raw_gt is None:
        return []
    if isinstance(raw_gt, list):
        normed = []
        for a in raw_gt:
            if isinstance(a, dict):
                normed.append(str(a.get("answer", "")))
            else:
                normed.append(str(a))
        return [ans.strip() for ans in normed if ans.strip()]
    if isinstance(raw_gt, dict):
        ans = raw_gt.get("answer", raw_gt.get("answers", []))
        return _normalize_answers(ans)
    return [str(raw_gt).strip()]


def _extract_record_info(record: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    """Extracts standardized question_id, question, answers, and image pointer from record."""
    qid = str(record.get("question_id", ""))
    question = record.get("question", "")
    sample = record.get("sample", {})

    if not question and isinstance(sample, dict):
        question = sample.get("question", "")

    # Ground truth answers extraction
    answers: list[str] = []
    if record.get("text_answer"):
        answers = _normalize_answers(record["text_answer"])
    elif isinstance(sample, dict) and sample.get("text_answer"):
        answers = _normalize_answers(sample["text_answer"])
    elif record.get("labels"):
        answers = _normalize_answers(record["labels"])
    elif isinstance(sample, dict) and sample.get("labels"):
        answers = _normalize_answers(sample["labels"])
    elif record.get("ground_truth"):
        answers = _normalize_answers(record["ground_truth"])
    elif isinstance(sample, dict) and "answers" in sample:
        answers = _normalize_answers(sample["answers"])
    elif isinstance(sample, dict) and "answer" in sample:
        ans_raw = sample["answer"]
        if dataset_key in ("ai2d", "scienceqa") and "options" in sample:
            try:
                idx = int(ans_raw)
                options = sample["options"]
                if 0 <= idx < len(options):
                    answers = [str(options[idx])]
                else:
                    answers = [str(ans_raw)]
            except (ValueError, TypeError):
                answers = [str(ans_raw)]
        elif dataset_key == "scienceqa" and "choices" in sample:
            try:
                idx = int(ans_raw)
                choices = sample["choices"]
                if 0 <= idx < len(choices):
                    answers = [str(choices[idx])]
                else:
                    answers = [str(ans_raw)]
            except (ValueError, TypeError):
                answers = [str(ans_raw)]
        else:
            answers = [str(ans_raw)]

    # Image pointer extraction
    image_str = ""
    if record.get("image"):
        image_str = str(record["image"])
    elif isinstance(sample, dict) and "image" in sample:
        image_str = str(sample["image"])
    elif isinstance(sample, dict) and "image_id" in sample:
        img_id = sample["image_id"]
        if dataset_key == "vqav2":
            try:
                image_str = f"COCO_val2014_{int(img_id):012d}.jpg"
            except (ValueError, TypeError):
                image_str = f"{img_id}.jpg"
        elif dataset_key == "textvqa":
            image_str = f"{img_id}.jpg"
        else:
            image_str = str(img_id)
    elif dataset_key == "vizwiz-vqa":
        image_str = f"{qid}.jpg"

    info: dict[str, Any] = {
        "question_id": qid,
        "question": question,
        "answers": answers,
        "image": image_str,
    }
    if dataset_key == "ai2d":
        if isinstance(sample, dict) and "options" in sample:
            info["options"] = sample["options"]
        elif "options" in record:
            info["options"] = record["options"]

        if isinstance(sample, dict) and "answer" in sample:
            info["answer"] = sample["answer"]
        elif "answer" in record:
            info["answer"] = record["answer"]

    return info


def _load_core_dataset_records(
    repo_root: Path, dataset_key: str, target_n: int
) -> list[dict[str, Any]]:
    """Loads records for a core dataset up to target_n from single-pass feature files."""
    feature_file = (
        repo_root / "data" / "features" / "mqt_llava" / "temp_0.0" / f"{dataset_key}.pt"
        if dataset_key in ("scienceqa", "textvqa", "vizwiz-vqa")
        else repo_root
        / "data"
        / "features"
        / "m3_llava"
        / "temp_0.0"
        / ("vqav2_5scale.pt" if dataset_key == "vqav2" else f"{dataset_key}.pt")
    )

    if not feature_file.exists():
        fallback = repo_root / "data" / "features" / "m3_llava" / "temp_0.0" / f"{dataset_key}.pt"
        if fallback.exists():
            feature_file = fallback

    logger.info(f"Reading {dataset_key} from {feature_file}...")
    data = torch.load(feature_file, map_location="cpu", weights_only=False)
    records = []
    for item in data[:target_n]:
        rec = _extract_record_info(item, dataset_key)
        records.append(rec)
    return records


def _load_robustness_records(
    repo_root: Path, dataset_key: str, target_n: int
) -> list[dict[str, Any]]:
    """Loads records for robustness datasets (avqa or vllm-safety)."""
    pt_path = (
        repo_root
        / "results"
        / "features"
        / "m3_llava"
        / "temp_0.0"
        / dataset_key
        / "full_extracted_features.pt"
    )
    if pt_path.exists():
        logger.info(f"Reading {dataset_key} from {pt_path}...")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        return [_extract_record_info(item, dataset_key) for item in data[:target_n]]

    # Local loader fallback
    from trajectory_calibration.vlm.registry import _load_local_avqa, _load_local_vllm_safety

    logger.info(f"Loading {dataset_key} via registry local fallback...")
    raw_list = (
        _load_local_avqa(auto_download=False)
        if dataset_key == "avqa"
        else _load_local_vllm_safety(auto_download=False)
    )
    return [_extract_record_info(item, dataset_key) for item in raw_list[:target_n]]


def build_canonical_manifest(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Extracts canonical samples across all 9 benchmark datasets."""
    manifest: dict[str, list[dict[str, Any]]] = {}

    for dataset_key, target_n in DATASET_TARGETS.items():
        if dataset_key in ("avqa", "vllm-safety"):
            records = _load_robustness_records(repo_root, dataset_key, target_n)
        else:
            records = _load_core_dataset_records(repo_root, dataset_key, target_n)

        if len(records) < target_n:
            raise ValueError(
                f"Dataset '{dataset_key}' yielded only {len(records)} samples, required {target_n}!"
            )
        manifest[dataset_key] = records[:target_n]
        logger.info(
            f"Manifest entry for '{dataset_key}': exactly {len(manifest[dataset_key])} samples verified."
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Canonical Manifest for Strict Sample Parity"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/canonical_manifest_all.json",
        help="Target output path for canonical manifest JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest = build_canonical_manifest(repo_root)

    out_file = repo_root / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_samples = sum(len(v) for v in manifest.values())
    logger.info(
        f"Successfully wrote canonical manifest to {out_file} ({total_samples} total questions)."
    )


if __name__ == "__main__":
    main()
