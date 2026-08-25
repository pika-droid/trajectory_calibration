#!/usr/bin/env python3
"""
Rollout Quality Audit and Repair Utility.

Inspects multi-pass feature files (.pt) in target directory, identifies samples with
fewer than target rollouts (e.g. 5 instead of 10), and repairs/cleans files:
  1. Deletes .pt files where ALL samples have 5 rollouts (forcing a clean re-extraction with 10 rollouts).
  2. Filters out initial 5-rollout samples from hybrid files (like chartqa, docvqa, mmbench),
     leaving only clean 10-rollout samples so extraction can complete to exactly 2000 samples.
"""

import argparse
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch
from trajectory_calibration.utils.helpers import safe_torch_load


def audit_and_clean(
    features_dir: Path,
    target_rollouts: int = 10,
    fix: bool = False,
) -> None:
    if not features_dir.exists():
        print(f"Error: Directory not found: {features_dir}")
        return

    pt_files = sorted(features_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {features_dir}")
        return

    print(f"\n{'Dataset':<20} | {'Total':<8} | {'10-Rollout':<11} | {'5-Rollout':<10} | {'Status / Action'}")
    print("-" * 75)

    for pt_path in pt_files:
        dataset_name = pt_path.stem
        if dataset_name.endswith(".tmp"):
            continue

        try:
            data = safe_torch_load(pt_path)
        except Exception as e:
            print(f"{dataset_name:<20} | ERROR loading file: {e}")
            continue

        if not isinstance(data, list):
            print(f"{dataset_name:<20} | INVALID payload format")
            continue

        total = len(data)
        count_target = 0
        count_other = 0

        valid_samples = []
        for sample in data:
            r_texts = sample.get("rollout_texts", [])
            rollouts = len(r_texts)
            if rollouts == target_rollouts:
                count_target += 1
                valid_samples.append(sample)
            else:
                count_other += 1

        status = ""
        if count_other == 0:
            status = "OK (All 10 rollouts)"
        elif count_target == 0:
            status = "ALL 5-ROLLOUTS"
            if fix:
                pt_path.unlink(missing_ok=True)
                tmp_path = pt_path.with_suffix(".pt.tmp")
                tmp_path.unlink(missing_ok=True)
                status += " -> DELETED (will re-extract clean 10 rollouts)"
        else:
            status = f"MIXED ({count_other} invalid samples)"
            if fix:
                torch.save(valid_samples, pt_path)
                tmp_path = pt_path.with_suffix(".pt.tmp")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                status += f" -> FIXED (retained {len(valid_samples)} valid 10-rollout samples)"

        print(f"{dataset_name:<20} | {total:<8} | {count_target:<11} | {count_other:<10} | {status}")

    print("-" * 75)
    if not fix:
        print("\n[NOTE] Audit mode only. Run with --fix to repair files & delete 5-rollout checkpoints.\n")


def main():
    parser = argparse.ArgumentParser(description="Audit and Repair Rollout Counts in Multi-Pass Feature Checkpoints.")
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features_multipass/m3_llava/temp_0.5",
        help="Path to feature directory containing .pt files",
    )
    parser.add_argument("--target_rollouts", type=int, default=10, help="Expected rollout count per sample (default: 10)")
    parser.add_argument("--fix", action="store_true", help="Apply cleanup: delete 5-rollout files & strip 5-rollout samples")
    args = parser.parse_args()

    audit_and_clean(Path(args.features_dir), target_rollouts=args.target_rollouts, fix=args.fix)


if __name__ == "__main__":
    main()
