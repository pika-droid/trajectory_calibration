"""
Unit and Integration Tests for Multi-Pass & Multi-Rollout GPU Feature Extraction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.uq.eigenscore import compute_eigenscore, compute_umpire_metric
from trajectory_calibration.uq.semantic_entropy import (
    FastStringEntailment,
    compute_predictive_entropy,
    compute_semantic_entropy,
    get_semantic_ids,
)
from trajectory_calibration.uq.whitebox import WhiteBoxScorers
from trajectory_calibration.utils.helpers import safe_torch_load
from trajectory_calibration.vlm.multipass import generate_mock_multipass_sample


def test_multipass_record_schema():
    rec = generate_mock_multipass_sample(idx=1, dataset_key="pope", num_rollouts=5, hidden_dim=64)

    required_keys = [
        "question_id", "dataset", "question", "ground_truth", "answer_type",
        "greedy_answer", "is_correct", "vqa_accuracy", "conf_softmax", "first_token_logits",
        "rollout_texts", "rollout_token_logprobs", "rollout_sequence_logprobs", "rollout_embeddings",
    ]
    for k in required_keys:
        assert k in rec, f"Missing key: {k}"

    assert isinstance(rec["rollout_texts"], list) and len(rec["rollout_texts"]) == 5
    assert isinstance(rec["rollout_token_logprobs"], list) and len(rec["rollout_token_logprobs"]) == 5
    assert isinstance(rec["rollout_sequence_logprobs"], list) and len(rec["rollout_sequence_logprobs"]) == 5
    assert isinstance(rec["rollout_embeddings"], np.ndarray)
    assert rec["rollout_embeddings"].shape == (5, 64)
    assert isinstance(rec["first_token_logits"], np.ndarray)
    assert rec["first_token_logits"].shape == (32000,)
    assert isinstance(rec["conf_softmax"], float)
    assert isinstance(rec["is_correct"], bool)


def test_multipass_semantic_entropy_integration():
    rec = generate_mock_multipass_sample(idx=2, dataset_key="scienceqa", num_rollouts=5)
    model = FastStringEntailment()
    sem_ids = get_semantic_ids(rec["rollout_texts"], model=model)
    se = compute_semantic_entropy(sem_ids, rec["rollout_sequence_logprobs"])
    pe = compute_predictive_entropy(rec["rollout_sequence_logprobs"])

    assert isinstance(se, float) and se >= 0.0
    assert isinstance(pe, float) and pe >= 0.0


def test_multipass_eigenscore_integration():
    rec = generate_mock_multipass_sample(idx=3, dataset_key="textvqa", num_rollouts=5, hidden_dim=128)
    eigenscore = compute_eigenscore(rec["rollout_embeddings"])
    umpire = compute_umpire_metric(rec["rollout_embeddings"], rec["rollout_sequence_logprobs"])

    assert isinstance(eigenscore, float)
    assert isinstance(umpire, float)


def test_multipass_whitebox_integration():
    rec = generate_mock_multipass_sample(idx=4, dataset_key="vizwiz-vqa", num_rollouts=5)
    seq_prob = WhiteBoxScorers.sequence_probability(rec["rollout_token_logprobs"][0])
    min_prob = WhiteBoxScorers.min_probability(np.exp(rec["rollout_token_logprobs"][0]))

    assert 0.0 <= seq_prob <= 1.0
    assert 0.0 <= min_prob <= 1.0


def test_multipass_vcps_calibration_integration():
    records = [generate_mock_multipass_sample(idx=i, dataset_key="pope", num_rollouts=5) for i in range(40)]

    X_rows, y_rows = [], []
    for r in records:
        c = r["conf_softmax"]
        x1 = float(np.log(np.clip(c, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(c, 1e-6, 1.0 - 1e-6))))
        sem_ids = get_semantic_ids(r["rollout_texts"])
        se = compute_semantic_entropy(sem_ids, r["rollout_sequence_logprobs"])
        es = compute_eigenscore(r["rollout_embeddings"])
        seq_p = WhiteBoxScorers.sequence_probability(r["rollout_token_logprobs"][0])

        X_rows.append([x1, se, es, seq_p])
        y_rows.append(1 if r["is_correct"] else 0)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)

    vcps = VaryingCoefficientPlattScaler(mode="full")
    vcps.fit(X, y)
    probs = vcps.predict_proba(X)

    assert len(probs) == len(records)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
