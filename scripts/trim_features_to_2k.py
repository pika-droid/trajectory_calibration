#!/usr/bin/env python3
"""
Trim Single-Pass Feature Files to Exactly 2,000 Samples (ADR 0006).

Processes all .pt feature files in data/features/m3_llava/ and data/features/mqt_llava/,
aligning question IDs to data/canonical_manifest_all.json and trimming sample count to exactly 2,000.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any

import torch

from trajectory_calibration.features.definitions import FEATURE_KEYS
from trajectory_calibration.features.extractor import compute_features_from_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trim_features_to_2k")


def _load_manifest_qids(manifest_path: Path) -> dict[str, list[str]]:
    """Loads dataset -> list of canonical question IDs from manifest."""
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}. Trimming will use slice [:2000].")
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: [str(item["question_id"]) for item in v] for k, v in data.items()}


def _trim_and_align_records(
    records: list[dict[str, Any]],
    canonical_qids: list[str] | None,
    target_count: int = 2000,
) -> list[dict[str, Any]]:
    """Aligns records to canonical question IDs or slices to target_count."""
    if not canonical_qids:
        return records[:target_count]

    # Map existing records by question ID
    qid_map: dict[str, dict[str, Any]] = {}
    for r in records:
        qid = str(r.get("question_id", ""))
        if qid:
            qid_map[qid] = r

    aligned = []
    for qid in canonical_qids[:target_count]:
        if qid in qid_map:
            aligned.append(qid_map[qid])

    # If canonical QIDs did not match all records (e.g. index-based or different format), fallback
    if len(aligned) == target_count:
        return aligned
    logger.warning(
        f"Canonical QID alignment yielded {len(aligned)}/{target_count} matches. Slicing prefix [:target_count]."
    )
    return records[:target_count]


def _validate_trimmed_record_features(record: dict[str, Any], fine_scale: int) -> bool:
    """Validates that record contains valid multi-scale features."""
    if "features" not in record or not isinstance(record["features"], dict):
        return False
    try:
        feats = compute_features_from_sample(record, fine_scale=fine_scale)
        return len(feats) == len(FEATURE_KEYS)
    except Exception:
        return False


def trim_file(
    file_path: Path,
    manifest_qids: dict[str, list[str]],
    backup: bool = True,
) -> bool:
    """Trims a single .pt file if its sample count exceeds 2,000."""
    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return False

    if not isinstance(data, list) or len(data) <= 2000:
        return False

    dataset_name = file_path.stem
    if dataset_name == "vqav2_5scale":
        dataset_key = "vqav2"
    else:
        dataset_key = dataset_name

    canonical_qids = manifest_qids.get(dataset_key)
    logger.info(f"Trimming {file_path.name} from {len(data)} to 2000 samples...")

    trimmed = _trim_and_align_records(data, canonical_qids, target_count=2000)

    if backup:
        bak_path = file_path.with_suffix(".pt.bak")
        if not bak_path.exists():
            file_path.rename(bak_path)

    torch.save(trimmed, file_path)
    logger.info(f"Successfully saved {file_path} (length={len(trimmed)}).")
    return True


def process_all_feature_files(repo_root: Path, backup: bool = False) -> int:
    """Processes all candidate feature files in data/features."""
    manifest_path = repo_root / "data" / "canonical_manifest_all.json"
    manifest_qids = _load_manifest_qids(manifest_path)

    search_patterns = [
        str(repo_root / "data" / "features" / "m3_llava" / "**" / "*.pt"),
        str(repo_root / "data" / "features" / "mqt_llava" / "**" / "*.pt"),
    ]

    total_trimmed = 0
    for pattern in search_patterns:
        for f_str in glob.glob(pattern, recursive=True):
            f_path = Path(f_str)
            if f_path.name.endswith(".bak") or f_path.name.endswith(".tmp") or "30k" in f_path.name:
                continue
            if trim_file(f_path, manifest_qids, backup=backup):
                total_trimmed += 1

    return total_trimmed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim Single-Pass Features to Exactly 2,000 Samples"
    )
    parser.add_argument("--backup", action="store_true", help="Keep .bak backup files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    count = process_all_feature_files(repo_root, backup=args.backup)
    logger.info(f"Feature trimming completed. Total files trimmed: {count}")


if __name__ == "__main__":
    main()
