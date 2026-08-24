#!/usr/bin/env python3
"""
Verification Script for Multi-Pass & Multi-Rollout Extracted Feature Files.

Inspects .pt feature files for corruption, checks tensor dimensions, validates
numerical invariants (no NaNs/Infs), and tests downstream UQ / VCPS integration.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.uq.eigenscore import compute_eigenscore
from trajectory_calibration.uq.semantic_entropy import FastStringEntailment, compute_semantic_entropy, get_semantic_ids
from trajectory_calibration.uq.whitebox import WhiteBoxScorers
from trajectory_calibration.utils.helpers import safe_torch_load


def verify_feature_file(pt_path: Path, expected_rollouts: int = 5) -> dict[str, any]:
    if not pt_path.exists():
        return {"status": "FAIL", "reason": f"File not found: {pt_path}"}

    try:
        data = safe_torch_load(pt_path)
    except Exception as e:
        return {"status": "FAIL", "reason": f"Load error: {e}"}

    if not isinstance(data, list) or len(data) == 0:
        return {"status": "FAIL", "reason": "Empty or invalid dataset payload"}

    req_keys = [
        "question_id", "dataset", "question", "ground_truth", "answer_type",
        "greedy_answer", "is_correct", "vqa_accuracy", "conf_softmax", "first_token_logits",
        "rollout_texts", "rollout_token_logprobs", "rollout_sequence_logprobs", "rollout_embeddings",
    ]

    confs, accs, ses, ess, X_rows, y_rows = [], [], [], [], [], []
    entail_model = FastStringEntailment()

    for idx, sample in enumerate(data):
        for k in req_keys:
            if k not in sample:
                return {"status": "FAIL", "reason": f"Sample {idx} missing key: {k}"}

        c = sample["conf_softmax"]
        if not (0.0 <= c <= 1.0) or np.isnan(c):
            return {"status": "FAIL", "reason": f"Sample {idx} invalid conf_softmax: {c}"}
        confs.append(c)
        accs.append(float(sample["vqa_accuracy"]))

        # Check logits
        logits = np.asarray(sample["first_token_logits"])
        if logits.ndim != 1 or np.isnan(logits).any():
            return {"status": "FAIL", "reason": f"Sample {idx} invalid first_token_logits"}

        # Check rollouts
        texts = sample["rollout_texts"]
        tok_lps = sample["rollout_token_logprobs"]
        seq_lps = sample["rollout_sequence_logprobs"]
        embs = np.asarray(sample["rollout_embeddings"])

        if len(texts) != expected_rollouts or len(tok_lps) != expected_rollouts or len(seq_lps) != expected_rollouts:
            return {"status": "FAIL", "reason": f"Sample {idx} rollout count mismatch"}

        if embs.ndim != 2 or embs.shape[0] != expected_rollouts or np.isnan(embs).any():
            return {"status": "FAIL", "reason": f"Sample {idx} invalid rollout_embeddings shape {embs.shape}"}

        # UQ Baseline Computation Check
        sem_ids = get_semantic_ids(texts, model=entail_model)
        se = compute_semantic_entropy(sem_ids, seq_lps)
        es = compute_eigenscore(embs)
        seq_p = WhiteBoxScorers.sequence_probability(tok_lps[0])

        ses.append(se)
        ess.append(es)

        x1 = float(np.log(np.clip(c, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(c, 1e-6, 1.0 - 1e-6))))
        X_rows.append([x1, se, es, seq_p])
        y_rows.append(1 if sample["is_correct"] else 0)

    # Test VCPS calibration fit if enough samples
    if len(X_rows) >= 5 and len(set(y_rows)) > 1:
        try:
            vcps = VaryingCoefficientPlattScaler(mode="full")
            vcps.fit(np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64))
            _ = vcps.predict_proba(np.array(X_rows, dtype=np.float64))
        except Exception as e:
            return {"status": "FAIL", "reason": f"VCPS calibration failed: {e}"}

    return {
        "status": "PASS",
        "num_samples": len(data),
        "mean_conf": float(np.mean(confs)),
        "mean_acc": float(np.mean(accs)),
        "mean_se": float(np.mean(ses)),
        "mean_eigenscore": float(np.mean(ess)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify extracted multi-pass features.")
    parser.add_argument("--features_dir", type=str, default="data/features_multipass/m3_llava/temp_0.5")
    parser.add_argument("--expected_rollouts", type=int, default=5)
    args = parser.parse_args()

    dir_path = Path(args.features_dir)
    if not dir_path.exists():
        print(f"[ERROR] Directory does not exist: {dir_path}")
        sys.exit(1)

    pt_files = sorted(list(dir_path.glob("*.pt")))
    if not pt_files:
        print(f"[ERROR] No .pt files found in {dir_path}")
        sys.exit(1)

    print(f"\n{'Dataset':<16} | {'Samples':<7} | {'MeanConf':<8} | {'VQA Acc':<8} | {'Mean SE':<8} | {'EigenScore':<10} | {'Status'}")
    print("-" * 80)

    all_passed = True
    for pt in pt_files:
        ds_name = pt.stem
        res = verify_feature_file(pt, expected_rollouts=args.expected_rollouts)
        if res["status"] == "PASS":
            print(f"{ds_name:<16} | {res['num_samples']:<7} | {res['mean_conf']:<8.3f} | {res['mean_acc']:<8.3f} | {res['mean_se']:<8.3f} | {res['mean_eigenscore']:<10.3f} | [PASS]")
        else:
            print(f"{ds_name:<16} | {'N/A':<7} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10} | [FAIL: {res.get('reason')}]")
            all_passed = False

    print("-" * 80)
    if all_passed:
        print("[SUCCESS] All dataset feature payloads are complete, intact, and ready for calibration!")
        sys.exit(0)
    else:
        print("[FAILED] One or more datasets failed integrity checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
