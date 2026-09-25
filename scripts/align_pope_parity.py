#!/usr/bin/env python3
"""Aligns all 10 POPE .pt feature files to 100% sample parity.

Standardizes on the 1,595 unique samples shared between M3 and MQT,
ordered exactly according to the canonical first-occurrence sequence.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("align_pope_parity")

TEMPS = ["temp_0.0", "temp_0.3", "temp_0.6", "temp_1.0", "temp_1.5"]
ARCHS = ["m3_llava", "mqt_llava"]
PARITY_COUNT = 1595


def align_pope_files(repo_root: Path) -> None:
    """Aligns all 10 POPE feature files to the 1,595 shared canonical samples."""
    features_dir = repo_root / "data" / "features"
    ref_path = features_dir / "mqt_llava" / "temp_0.0" / "pope.pt"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference MQT POPE file: {ref_path}")

    ref_data: list[dict[str, Any]] = torch.load(ref_path, map_location="cpu")
    canonical_qids = [str(x["question_id"]) for x in ref_data[:PARITY_COUNT]]
    canonical_qid_set = set(canonical_qids)

    logger.info(f"Established POPE canonical reference sequence of {len(canonical_qids)} samples.")

    for arch in ARCHS:
        for temp in TEMPS:
            fpath = features_dir / arch / temp / "pope.pt"
            bak_path = features_dir / arch / temp / "pope.pt.bak"

            if not fpath.exists():
                logger.warning(f"File not found: {fpath}, skipping.")
                continue

            # 1. Create disaster recovery backup
            if not bak_path.exists():
                shutil.copyfile(fpath, bak_path)
                logger.info(f"Created backup: {bak_path}")

            data: list[dict[str, Any]] = torch.load(fpath, map_location="cpu")

            # 2. Index unique records by question_id (keep first occurrence)
            qid_to_rec: dict[str, dict[str, Any]] = {}
            for rec in data:
                qid = str(rec["question_id"])
                if qid in canonical_qid_set and qid not in qid_to_rec:
                    qid_to_rec[qid] = rec

            # 3. Order strictly according to canonical sequence
            aligned_data = [qid_to_rec[qid] for qid in canonical_qids if qid in qid_to_rec]
            if len(aligned_data) != PARITY_COUNT:
                raise ValueError(
                    f"Alignment count mismatch for {arch} {temp}: got {len(aligned_data)}, "
                    f"expected {PARITY_COUNT}."
                )

            # 4. Save aligned tensor
            torch.save(aligned_data, fpath)
            logger.info(f"Aligned {arch} {temp}/pope.pt -> exactly {len(aligned_data)} samples.")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    align_pope_files(repo_root)
    logger.info("All 10 POPE files successfully aligned with 100% sample parity!")


if __name__ == "__main__":
    main()
