"""Analyze prediction and ground-truth word length distributions across benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

DATASET_MAP: dict[str, str] = {
    "docvqa": "docvqa",
    "textvqa": "textvqa",
    "chartqa": "chartqa",
    "scienceqa": "scienceqa",
    "ai2d": "ai2d",
    "vizwiz-vqa": "vizwiz-vqa",
    "vqav2_5scale": "vqav2",
    "avqa": "avqa",
    "vllm-safety": "vllm-safety",
}


def get_ground_truth_string(
    file_stem: str,
    sample_pt: dict[str, Any],
    manifest_item: dict[str, Any] | None,
) -> str:
    """Extract standard ground-truth string for a sample."""
    if file_stem == "scienceqa":
        idx = sample_pt.get("sample", {}).get("answer")
        if isinstance(idx, int):
            return chr(ord("A") + idx)
        if manifest_item and manifest_item.get("answers"):
            return str(manifest_item["answers"][0])
        return ""

    if file_stem == "ai2d":
        # Target is multiple-choice option letter ('A', 'B', 'C', 'D')
        return "A"

    if manifest_item and manifest_item.get("answers"):
        return str(manifest_item["answers"][0]).strip()

    return ""


def analyze_model_features(
    model: str = "m3_llava",
    temp: float = 0.0,
    manifest_path: str = "data/canonical_manifest_all.json",
) -> None:
    """Analyze word lengths for model predictions and ground truths."""
    manifest_file = Path(manifest_path)
    manifest: dict[str, list[dict[str, Any]]] = {}
    if manifest_file.exists():
        with manifest_file.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    final_scale = 576 if "m3" in model.lower() else 256

    print(
        f"| {'Benchmark':<14} | {'Total N':<7} | {'Pred 1-Word (%)':<15} | "
        f"{'Pred Mean Words':<15} | {'GT 1-Word (%)':<13} | {'GT Mean Words':<13} |"
    )
    print("|:---|:---:|:---:|:---:|:---:|:---:|")

    for file_stem, mkey in DATASET_MAP.items():
        feature_path = Path(f"data/features/{model}/temp_{temp:.1f}/{file_stem}.pt")
        if not feature_path.exists():
            continue

        data = torch.load(feature_path, map_location="cpu")
        m_items = manifest.get(mkey, [])

        pred_lens: list[int] = []
        gt_lens: list[int] = []

        for i, s in enumerate(data):
            # Prediction
            pred = s.get("features", {}).get(final_scale, {}).get("answer", "")
            pred_lens.append(len(str(pred).strip().split()))

            # Ground Truth
            m_item = m_items[i] if i < len(m_items) else None
            gt_str = get_ground_truth_string(file_stem, s, m_item)
            gt_lens.append(len(gt_str.split()))

        total = len(data)
        p1 = sum(1 for w_len in pred_lens if w_len == 1) / total * 100
        p_mean = sum(pred_lens) / total
        g1 = sum(1 for w_len in gt_lens if w_len == 1) / total * 100
        g_mean = sum(gt_lens) / total

        d_label = "vqav2" if file_stem == "vqav2_5scale" else file_stem
        print(
            f"| {d_label:<14} | {total:<7} | {p1:>14.1f}% | "
            f"{p_mean:>15.2f} | {g1:>12.1f}% | {g_mean:>13.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VQA dataset word lengths.")
    parser.add_argument("--model", type=str, default="m3_llava", help="Model architecture")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    analyze_model_features(model=args.model, temp=args.temp)


if __name__ == "__main__":
    main()
