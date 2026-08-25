#!/usr/bin/env python3
"""
Multi-Pass & Multi-Rollout VLM Feature Extraction Script.

Extracts primary greedy calibration outputs alongside M stochastic rollouts,
token-level log-probabilities, and pooled hidden states for Kuhn Semantic Entropy,
Chen EigenScore / UMPIRE, and UQLM White-Box baselines.
"""

import argparse
import gc
import logging
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch
from trajectory_calibration.utils.helpers import safe_torch_load, set_seed
from trajectory_calibration.vlm.datasets import ALL_DATASET_KEYS, load_hf_dataset
from trajectory_calibration.vlm.multipass import (
    extract_multipass_record,
    generate_mock_multipass_sample,
)
from trajectory_calibration.vlm.wrapper import UnifiedVLMWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_features_multipass")


def process_dataset(dataset_key: str, wrapper: UnifiedVLMWrapper | None, args: argparse.Namespace) -> None:
    arch_folder = "mqt_llava" if "mqt" in args.arch.lower() else "m3_llava"
    temp_folder = f"temp_{args.gen_temperature:.1f}" if args.gen_temperature == int(args.gen_temperature) else f"temp_{args.gen_temperature}"
    out_dir = Path(args.output_dir) / arch_folder / temp_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{dataset_key}.pt"
    tmp_path = pt_path.with_suffix(".pt.tmp")

    logger.info(f"Extracting multi-pass features for '{dataset_key}' -> {pt_path}")

    existing_data = []
    processed_qids = set()

    if args.clean:
        pt_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)
    elif pt_path.exists():
        try:
            existing_data = safe_torch_load(pt_path)
            processed_qids = {str(d.get("question_id")) for d in existing_data}
            logger.info(f"Loaded existing checkpoint with {len(existing_data)} samples.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
    elif tmp_path.exists():
        try:
            existing_data = safe_torch_load(tmp_path)
            processed_qids = {str(d.get("question_id")) for d in existing_data}
            logger.info(f"Resuming from temporary checkpoint with {len(existing_data)} samples.")
        except Exception as e:
            logger.warning(f"Could not load temp checkpoint: {e}.")

    limit = args.limit or args.subset_size
    if args.mock:
        logger.info(f"Mock mode enabled: generating synthetic multi-pass samples for '{dataset_key}'...")
        n_samples = limit or 50
        mock_records = [
            generate_mock_multipass_sample(i, dataset_key=dataset_key, num_rollouts=args.num_rollouts)
            for i in range(n_samples)
        ]
        torch.save(mock_records, tmp_path)
        tmp_path.replace(pt_path)
        logger.info(f"Saved {len(mock_records)} mock records to {pt_path}")
        return

    ds = load_hf_dataset(dataset_key, subset_size=limit)
    extracted = list(existing_data)

    for idx, sample in enumerate(tqdm(ds, desc=f"Multi-Pass {dataset_key}")):
        q_id = str(sample.get("question_id", sample.get("id", idx)))
        if q_id in processed_qids:
            continue

        record = extract_multipass_record(
            wrapper=wrapper,
            sample=sample,
            dataset_key=dataset_key,
            num_rollouts=args.num_rollouts,
            gen_temperature=args.gen_temperature,
            top_p=args.top_p,
            max_new_tokens=32,
        )
        if record is None:
            continue

        extracted.append(record)
        processed_qids.add(q_id)

        if len(extracted) % 100 == 0:
            torch.save(extracted, tmp_path)

    torch.save(extracted, tmp_path)
    tmp_path.replace(pt_path)
    logger.info(f"Completed '{dataset_key}'. Total saved: {len(extracted)} -> {pt_path}")

    # Free memory and purge temporary unpacked Arrow cache to prevent disk buildup
    del ds
    del extracted
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Automatically purge uncompressed Arrow cache files (preserves downloaded model weights & parquets)
    for cache_path in [
        os.path.expanduser("~/.cache/huggingface/datasets"),
        "/workspace/.cache/huggingface/datasets",
        "/workspace/.cache/huggingface/downloads/extracted",
        os.path.join(os.environ.get("HF_HOME", ""), "datasets") if os.environ.get("HF_HOME") else None,
        "/tmp/huggingface",
    ]:
        if cache_path and os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Pass & Multi-Rollout VLM Feature Extraction.")
    parser.add_argument("--model_path", type=str, default=None, help="VLM checkpoint / HF path (defaults to arch standard).")
    parser.add_argument("--arch", type=str, default="m3", choices=["m3", "mqt"], help="Architecture.")
    parser.add_argument("--datasets", nargs="+", default=["pope"], help="Datasets to extract (or 'all').")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of sampling rollouts.")
    parser.add_argument("--gen_temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling threshold.")
    parser.add_argument("--output_dir", type=str, default="data/features_multipass", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Sample cap for smoke testing.")
    parser.add_argument("--subset_size", type=int, default=None, help="Alias for --limit.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision.")
    parser.add_argument("--clean", action="store_true", help="Purge existing checkpoint and start fresh.")
    parser.add_argument("--mock", action="store_true", help="Run synthetic mock extraction on CPU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.model_path is None:
        args.model_path = "gordonhu/MQT-LLaVA-7b" if args.arch == "mqt" else "mucai/llava-v1.5-7b-m3"

    dataset_keys = ALL_DATASET_KEYS if (args.datasets == ["all"] or "all" in args.datasets) else args.datasets

    wrapper = None
    if not args.mock:
        logger.info(f"Initializing UnifiedVLMWrapper for {args.arch.upper()} from '{args.model_path}'...")
        wrapper = UnifiedVLMWrapper(model_path=args.model_path, precision=args.precision, arch=args.arch)

    for ds in dataset_keys:
        process_dataset(ds, wrapper, args)


if __name__ == "__main__":
    main()
