#!/usr/bin/env python3
"""
Deep Inspection CLI for Multi-Pass Feature Files.

Inspects extracted .pt payloads sample-by-sample to verify generated texts,
ground truths, greedy completions, stochastic rollouts, logit distributions,
and embedding tensor integrity.
"""

import argparse
from pathlib import Path
import numpy as np
import torch


def inspect_directory(features_dir: str, num_samples_to_show: int = 2) -> None:
    feat_path = Path(features_dir)
    if not feat_path.exists():
        print(f"[ERROR] Directory '{features_dir}' does not exist.")
        return

    pt_files = sorted(list(feat_path.glob("*.pt")))
    if not pt_files:
        print(f"[ERROR] No .pt files found in '{features_dir}'.")
        return

    print("=" * 100)
    print(f" DEEP FEATURE INSPECTOR: {feat_path.resolve()}")
    print("=" * 100)

    summary_rows = []

    for pt_file in pt_files:
        dataset_key = pt_file.stem
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ERROR] Failed to load {pt_file.name}: {e}")
            continue

        if not isinstance(data, list) or len(data) == 0:
            print(f"[WARN] {pt_file.name} is empty or not a list.")
            continue

        print(f"\n{'-' * 80}")
        print(f" DATASET: {dataset_key.upper()} ({len(data)} Total Extracted Samples)")
        print(f"{'-' * 80}")

        non_blank_greedy = 0
        non_blank_rollouts = 0
        total_rollouts = 0
        unique_rollout_counts = []
        logit_maxes = []
        emb_norms = []

        # Print detailed inspection of the first few samples
        for idx in range(min(num_samples_to_show, len(data))):
            sample = data[idx]
            q = sample.get("question", "N/A")
            gt = sample.get("ground_truth", "N/A")
            ans = sample.get("greedy_answer", "N/A")
            conf = sample.get("conf_softmax", 0.0)
            acc = sample.get("vqa_accuracy", 0.0)
            r_texts = sample.get("rollout_texts", [])
            r_seq_lps = sample.get("rollout_sequence_logprobs", [])
            embs = np.asarray(sample.get("rollout_embeddings", []))
            logits = np.asarray(sample.get("first_token_logits", []))

            print(f"\n  [Sample {idx + 1} / {len(data)}] QID: {sample.get('question_id', idx)}")
            print(f"    Question      : {q[:120]}{'...' if len(q) > 120 else ''}")
            print(f"    Ground Truth  : {gt}")
            print(f"    Greedy Answer : '{ans}'  (Acc: {acc:.2f}, Softmax Conf: {conf:.4f})")
            print(f"    Rollouts ({len(r_texts)} total):")
            for r_i, r_text in enumerate(r_texts):
                lp = r_seq_lps[r_i] if r_i < len(r_seq_lps) else 0.0
                print(f"      Rollout [{r_i + 1}]: '{r_text}'  (Seq LogProb: {lp:.3f})")

            print(f"    Logits Shape  : {logits.shape}  | Min: {logits.min():.2f}, Max: {logits.max():.2f}, Has NaN: {np.isnan(logits).any()}")
            print(f"    Embeddings    : Shape {embs.shape} | Norms: {[round(float(np.linalg.norm(v)), 2) for v in embs]}")

        # Compute aggregate metrics across all samples in this dataset
        for sample in data:
            ans = sample.get("greedy_answer", "").strip()
            if ans:
                non_blank_greedy += 1
            r_texts = sample.get("rollout_texts", [])
            total_rollouts += len(r_texts)
            for rt in r_texts:
                if rt.strip():
                    non_blank_rollouts += 1
            unique_rollout_counts.append(len(set(r_texts)))

            logits = np.asarray(sample.get("first_token_logits", []))
            if logits.size > 0 and not np.isnan(logits).any():
                logit_maxes.append(float(np.max(logits)))

            embs = np.asarray(sample.get("rollout_embeddings", []))
            if embs.size > 0 and not np.isnan(embs).any():
                emb_norms.extend([float(np.linalg.norm(v)) for v in embs])

        pct_non_blank = (non_blank_greedy / len(data)) * 100.0
        avg_unique = float(np.mean(unique_rollout_counts)) if unique_rollout_counts else 0.0
        avg_logit_max = float(np.mean(logit_maxes)) if logit_maxes else 0.0
        avg_emb_norm = float(np.mean(emb_norms)) if emb_norms else 0.0

        summary_rows.append({
            "dataset": dataset_key,
            "samples": len(data),
            "non_blank": f"{pct_non_blank:.0f}%",
            "unique_rollouts": f"{avg_unique:.1f}/5",
            "avg_logit_max": f"{avg_logit_max:.2f}",
            "avg_emb_norm": f"{avg_emb_norm:.2f}",
            "status": "[VALID]" if (pct_non_blank > 80 and not np.isnan(avg_emb_norm)) else "[CHECK]",
        })

    # Summary Table
    print("\n" + "=" * 100)
    print(" AGGREGATE PAYLOAD INTEGRITY SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Dataset':<18} | {'Samples':<8} | {'Non-Blank':<10} | {'Unique Rollouts':<16} | {'Avg Max Logit':<14} | {'Avg Emb Norm':<13} | {'Status':<8}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['dataset']:<18} | {r['samples']:<8} | {r['non_blank']:<10} | {r['unique_rollouts']:<16} | {r['avg_logit_max']:<14} | {r['avg_emb_norm']:<13} | {r['status']:<8}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Inspector for Multi-Pass Feature Payloads.")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to features directory (e.g. data/features_multipass/m3_llava/temp_0.5).")
    parser.add_argument("--show", type=int, default=2, help="Number of detailed sample records to print per dataset.")
    args = parser.parse_args()

    inspect_directory(args.features_dir, num_samples_to_show=args.show)


if __name__ == "__main__":
    main()
