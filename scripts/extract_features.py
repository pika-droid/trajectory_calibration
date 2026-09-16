#!/usr/bin/env python3
"""
Fast Multi-Scale VLM Feature Extraction Script (M3-LLaVA & MQT-LLaVA).

Extracts per-scale confidences, margins, logprobs, and answers across visual token scales
(576 for M3, 256 for MQT) with checkpoint/resume, POPE deduplication, and FlashAttention speed.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

from tqdm import tqdm

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch

from trajectory_calibration.utils.helpers import safe_torch_load, set_seed
from trajectory_calibration.vlm.datasets import (
    evaluate_accuracy,
    format_question,
    load_hf_dataset,
    load_image_from_sample,
)
from trajectory_calibration.vlm.wrapper import UnifiedVLMWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_features")


def generate_mock_extraction(
    dataset_key: str, num_samples: int = 10, arch: str = "m3"
) -> list[dict]:
    scales = [1, 9, 36, 144, 576] if arch == "m3" else [1, 9, 36, 144, 256]
    fine_scale = scales[-1]
    mock_data = []
    for i in range(num_samples):
        is_corr = random.random() > 0.35
        base_c = random.uniform(0.7, 0.95) if is_corr else random.uniform(0.2, 0.6)
        feats = {}
        for s in scales:
            c = min(0.99, max(0.01, base_c + random.gauss(0.0, 0.04)))
            m = random.uniform(0.1, 0.5)
            if dataset_key == "vllm-safety":
                ans = "unicorn" if is_corr else "dragon"
            elif dataset_key == "avqa":
                ans = "dog" if is_corr else "elephant"
            else:
                ans = "dog" if is_corr else "cat"
            acc = 1.0 if (is_corr and s == fine_scale) else (1.0 if random.random() > 0.5 else 0.0)
            feats[s] = {
                "conf_softmax": float(c),
                "margin": float(m),
                "answer": ans,
                "vqa_accuracy": float(acc),
            }

        sample_dict: dict = {
            "question_id": 200000 + i,
            "question": f"Mock question {i}?",
            "ground_truth": "dog",
            "answer": "dog",
            "dataset": dataset_key,
            "sample_idx": i,
            "features": feats,
        }
        if dataset_key == "avqa":
            sample_dict["question"] = f"Mock AVQA adversarial question {i}?"
            sample_dict["answers"] = [{"answer": "dog"} for _ in range(8)] + [
                {"answer": "cat"} for _ in range(2)
            ]
            sample_dict["answer"] = "dog"
            sample_dict["answer_type"] = "list_soft"
        elif dataset_key == "vllm-safety":
            sample_dict["question"] = f"Mock VLLM safety question {i}?"
            sample_dict["ground_truth"] = "unicorn"
            sample_dict["answer"] = "unicorn"
            sample_dict["answer_type"] = "open"

        mock_data.append(sample_dict)
    return mock_data


def process_dataset(
    dataset_key: str, wrapper: UnifiedVLMWrapper | None, args: argparse.Namespace
) -> None:
    out_dir = Path(args.output_dir) / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "full_extracted_features.pt"

    logger.info(f"Extracting features for '{dataset_key}' -> {pt_path}")

    existing_data = []
    processed_qids = set()

    if args.clean and pt_path.exists():
        pt_path.unlink(missing_ok=True)
    elif pt_path.exists():
        try:
            existing_data = safe_torch_load(pt_path)
            processed_qids = {str(d.get("question_id")) for d in existing_data}
            logger.info(f"Resuming from checkpoint with {len(existing_data)} samples.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    if args.mock:
        logger.info("Mock flag enabled: Generating synthetic mock extraction data.")
        mock_data = generate_mock_extraction(
            dataset_key, num_samples=args.subset_size or 50, arch=args.arch
        )
        torch.save(mock_data, pt_path)
        logger.info(f"Saved {len(mock_data)} mock samples to {pt_path}")
        return

    ds = load_hf_dataset(dataset_key, subset_size=args.subset_size)
    extracted = list(existing_data)

    for idx, sample in enumerate(tqdm(ds, desc=f"Extracting {dataset_key}")):
        q_id = str(sample.get("question_id", sample.get("id", idx)))
        if q_id in processed_qids:
            continue

        image = load_image_from_sample(sample)
        if image is None:
            continue

        question = format_question(sample, dataset_key)
        scale_results = wrapper.sweep(image, question, gen_temperature=args.gen_temperature)

        feats = {}
        for s, res in scale_results.items():
            acc = evaluate_accuracy(res["answer"], sample, dataset_key)
            feats[s] = {
                "conf_softmax": res["conf_softmax"],
                "margin": res["margin"],
                "answer": res["answer"],
                "vqa_accuracy": acc,
            }

        extracted.append(
            {
                "question_id": q_id,
                "question": question,
                "dataset": dataset_key,
                "sample_idx": idx,
                "features": feats,
            }
        )
        processed_qids.add(q_id)

        if len(extracted) % 100 == 0:
            torch.save(extracted, pt_path)

    torch.save(extracted, pt_path)
    logger.info(
        f"Extraction complete for '{dataset_key}'. Saved {len(extracted)} samples to {pt_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scale VLM feature extraction.")
    parser.add_argument(
        "--model_path", type=str, default="mucai/llava-v1.5-7b-m3", help="VLM checkpoint path."
    )
    parser.add_argument(
        "--arch", type=str, default="m3", choices=["m3", "mqt"], help="Architecture."
    )
    parser.add_argument("--datasets", nargs="+", default=["vqav2"], help="Datasets to extract.")
    parser.add_argument(
        "--output_dir", type=str, default="results/features", help="Output directory."
    )
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size limit.")
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision."
    )
    parser.add_argument("--gen_temperature", type=float, default=0.0, help="Decoding temperature.")
    parser.add_argument("--clean", action="store_true", help="Purge existing checkpoint.")
    parser.add_argument("--mock", action="store_true", help="CPU mock extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()

    set_seed(args.seed)

    if not args.mock:
        logger.info(f"Initializing VLM wrapper from '{args.model_path}'...")
        wrapper = UnifiedVLMWrapper(model_path=args.model_path, precision=args.precision)
    else:
        wrapper = None

    for ds in args.datasets:
        process_dataset(ds, wrapper, args)


if __name__ == "__main__":
    main()
