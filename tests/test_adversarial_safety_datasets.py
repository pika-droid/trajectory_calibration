"""
Comprehensive Unit and Integration Tests for Adversarial & Safety Robustness Datasets.

Covers:
1. DATASET_REGISTRY entries and schema for 'avqa' and 'vllm-safety'.
2. Diverse question and prompt key formatting.
3. PIL image loading across raw objects, byte streams, and path references.
4. Accuracy scoring: soft consensus for AVQA and word-boundary matching for VLLM Safety.
5. Multi-scale mock feature extraction across M3 (576) and MQT (256) architectures.
6. 17-D and canonical 5-D feature extraction completeness and validity.
7. Calibrator fitting & evaluation (VCPS-17D, VCPS-5D, TP-5D, 1D Platt, TS).
8. Local disk fallback loaders with JSON fixtures.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.extract_features import generate_mock_extraction
from trajectory_calibration.calibrators.baselines import (
    PlattScalingEstimator,
    TemperatureScalingEstimator,
    TrajectoryPlattScaler,
)
from trajectory_calibration.calibrators.residual import evaluate_full_metric_panel
from trajectory_calibration.calibrators.vcps import VaryingCoefficientPlattScaler
from trajectory_calibration.features.definitions import CANONICAL_5D_KEYS, FEATURE_KEYS
from trajectory_calibration.features.extractor import compute_features_from_sample
from trajectory_calibration.features.loader import get_stratified_split
from trajectory_calibration.vlm.evaluators import evaluate_accuracy
from trajectory_calibration.vlm.formatting import format_question, load_image_from_sample
from trajectory_calibration.vlm.registry import (
    ALL_DATASET_KEYS,
    DATASET_REGISTRY,
    _load_local_avqa,
    _load_local_vllm_safety,
)


def test_dataset_registry_adversarial_and_safety_entries() -> None:
    """Verify avqa and vllm-safety are properly registered with non-withheld splits and answer types."""
    assert "avqa" in DATASET_REGISTRY
    assert "vllm-safety" in DATASET_REGISTRY
    assert "avqa" in ALL_DATASET_KEYS
    assert "vllm-safety" in ALL_DATASET_KEYS

    avqa_cfg = DATASET_REGISTRY["avqa"]
    assert avqa_cfg["hf_repo"] == "lmms-lab/avqa"
    assert avqa_cfg["config"] is None
    assert avqa_cfg["default_split"] == "val"
    assert avqa_cfg["answer_type"] == "list_soft"

    safety_cfg = DATASET_REGISTRY["vllm-safety"]
    assert safety_cfg["hf_repo"] == "PahaII/vllm_safety_evaluation"
    assert safety_cfg["config"] is None
    assert safety_cfg["default_split"] == "test"
    assert safety_cfg["answer_type"] == "open"


def test_format_question_diverse_keys() -> None:
    """Verify format_question extracts and cleans text across diverse sample key conventions."""
    test_cases = [
        ({"question": "Is the dog brown?"}, "Is the dog brown?"),
        ({"prompt": "Describe the main object."}, "Describe the main object."),
        ({"text": "How many cars are parked?"}, "How many cars are parked?"),
        ({"user_query": "What color is the stop sign?"}, "What color is the stop sign?"),
        ({"query": "Find the unicorn in the image."}, "Find the unicorn in the image."),
        ({"problem": "Calculate the ratio."}, "Calculate the ratio."),
        ({"instruction": "Point to the red apple."}, "Point to the red apple."),
        ({"input": "What is shown here?"}, "What is shown here?"),
        ({"question": {"text": "Nested question text"}}, "Nested question text"),
        ({"question": ["List question item"]}, "List question item"),
        ({"question": "  Leading and trailing whitespace  "}, "Leading and trailing whitespace"),
    ]
    for sample, expected in test_cases:
        formatted = format_question(sample, "avqa")
        assert formatted == expected, (
            f"Failed for {sample}: got '{formatted}', expected '{expected}'"
        )


def test_load_image_from_sample_formats(tmp_path: Path) -> None:
    """Verify load_image_from_sample handles PIL images, bytes, dicts, and path references."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))

    # 1. Direct PIL Image
    assert load_image_from_sample({"image": img}) is not None

    # 2. Raw PNG Bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    loaded_bytes = load_image_from_sample({"image": img_bytes})
    assert loaded_bytes is not None
    assert loaded_bytes.size == (64, 64)

    # 3. Dict with 'bytes' (HuggingFace dataset format)
    loaded_dict_bytes = load_image_from_sample({"image": {"bytes": img_bytes, "path": None}})
    assert loaded_dict_bytes is not None
    assert loaded_dict_bytes.size == (64, 64)

    # 4. Local File Path
    img_file = tmp_path / "sample_img.png"
    img.save(img_file)
    loaded_path_str = load_image_from_sample({"image_path": str(img_file)})
    assert loaded_path_str is not None
    assert loaded_path_str.size == (64, 64)

    loaded_path_obj = load_image_from_sample({"img_path": img_file})
    assert loaded_path_obj is not None

    # 5. List of images
    loaded_list = load_image_from_sample({"images": [img]})
    assert loaded_list is not None

    # 6. Missing or invalid image returns None
    assert load_image_from_sample({}) is None
    assert load_image_from_sample({"image": None}) is None
    assert load_image_from_sample({"image_path": "non_existent_file.png"}) is None


def test_evaluate_accuracy_avqa_soft_consensus() -> None:
    """Verify evaluate_accuracy computes official VQA soft consensus scoring min(1.0, matches / 3.0)."""
    # 1. Consensus with 3+ matching annotations -> 1.0
    sample_full_match = {
        "answers": [{"answer": "dog"}] * 8 + [{"answer": "cat"}] * 2,
    }
    assert evaluate_accuracy("dog", sample_full_match, "avqa") == 1.0

    # 2. Partial consensus with 2 matching annotations -> 2/3
    sample_two_matches = {
        "annotations": [
            {"answer": "dog"},
            {"answer": "dog"},
            {"answer": "wolf"},
            {"answer": "fox"},
        ],
    }
    score_two = evaluate_accuracy("dog", sample_two_matches, "avqa")
    assert np.isclose(score_two, 2.0 / 3.0, atol=1e-5)

    # 3. Partial consensus with 1 matching annotation -> 1/3
    sample_one_match = {
        "answers": ["dog", "wolf", "fox", "bear"],
    }
    score_one = evaluate_accuracy("dog", sample_one_match, "avqa")
    assert np.isclose(score_one, 1.0 / 3.0, atol=1e-5)

    # 4. Zero matches -> 0.0
    assert evaluate_accuracy("tiger", sample_full_match, "avqa") == 0.0

    # 5. Word boundary containment
    sample_phrase = {"answers": [{"answer": "a brown dog"}] * 5}
    assert evaluate_accuracy("there is a brown dog here", sample_phrase, "avqa") == 1.0

    # 6. Fallback to multiple_choice_answer or answer if annotations list empty
    assert evaluate_accuracy("yes", {"multiple_choice_answer": "yes"}, "avqa") == 1.0 / 3.0
    assert evaluate_accuracy("no", {"answer": "no"}, "avqa") == 1.0 / 3.0
    assert evaluate_accuracy("blue", {"answers": []}, "avqa") == 0.0


def test_evaluate_accuracy_vllm_safety_open_ended() -> None:
    """Verify evaluate_accuracy performs word-boundary open-ended scoring for VLLM Safety Benchmark."""
    sample_unicorn = {"answer": "unicorn"}

    # 1. Exact match
    assert evaluate_accuracy("unicorn", sample_unicorn, "vllm-safety") == 1.0

    # 2. Case and whitespace normalization
    assert evaluate_accuracy("  UNICORN  ", sample_unicorn, "vllm-safety") == 1.0

    # 3. Word-boundary containment
    assert (
        evaluate_accuracy("there is a unicorn in this photo", sample_unicorn, "vllm-safety") == 1.0
    )

    # 4. Partial substring contradiction rejection
    sample_negation = {"answer": "not a unicorn"}
    assert evaluate_accuracy("unicorn", sample_negation, "vllm-safety") == 0.0
    assert evaluate_accuracy("not a unicorn", sample_negation, "vllm-safety") == 1.0

    # 5. Total mismatch
    assert evaluate_accuracy("horse", sample_unicorn, "vllm-safety") == 0.0

    # 6. Alternative ground truth keys
    assert evaluate_accuracy("yes", {"label": "yes"}, "vllm-safety") == 1.0
    assert evaluate_accuracy("safe", {"ground_truth": "safe"}, "vllm-safety") == 1.0
    assert evaluate_accuracy("red", {"gt_answer": "red"}, "vllm-safety") == 1.0
    assert evaluate_accuracy("car", {"target": "car"}, "vllm-safety") == 1.0


def test_evaluate_accuracy_short_answers_and_digits() -> None:
    """Verify evaluate_accuracy accurately handles single-digit counts and short 2-letter answers."""
    # 1. Digit count questions (e.g. "0" unicorns)
    sample_zero = {"answer": "0"}
    assert evaluate_accuracy("0", sample_zero, "vllm-safety") == 1.0
    assert evaluate_accuracy("there are 0 unicorns", sample_zero, "vllm-safety") == 1.0
    assert evaluate_accuracy("count is 0", sample_zero, "vllm-safety") == 1.0
    # Digit word boundary: "100" or "20" should not match "0"
    assert evaluate_accuracy("there are 100 unicorns", sample_zero, "vllm-safety") == 0.0
    assert evaluate_accuracy("20", sample_zero, "vllm-safety") == 0.0

    # 2. 2-character binary answers ("no", "on")
    sample_no = {"answer": "no"}
    assert evaluate_accuracy("no", sample_no, "vllm-safety") == 1.0
    assert evaluate_accuracy("no there is none", sample_no, "vllm-safety") == 1.0
    # Word boundary rejection for subwords
    assert evaluate_accuracy("there is snow", sample_no, "vllm-safety") == 0.0
    assert evaluate_accuracy("i know", sample_no, "vllm-safety") == 0.0
    assert evaluate_accuracy("annoy", sample_no, "vllm-safety") == 0.0


def test_format_question_conversations_and_cleanup() -> None:
    """Verify format_question extracts prompt from LLaVA conversations and strips <image> tags."""
    sample_conv = {
        "conversations": [
            {"from": "human", "value": "<image>\nIs this image generated by AI?"},
            {"from": "gpt", "value": "No."},
        ]
    }
    assert format_question(sample_conv, "vllm-safety") == "Is this image generated by AI?"

    sample_msg = {
        "messages": [
            {"role": "user", "content": "What objects are in <image>?"},
        ]
    }
    assert format_question(sample_msg, "avqa") == "What objects are in ?"


def test_load_image_from_sample_base64_and_bytearray(tmp_path: Path) -> None:
    """Verify load_image_from_sample handles base64 data URIs, bytearrays, and nested lists."""
    import base64

    img = Image.new("RGB", (32, 32), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    # 1. Base64 encoded string
    b64_str = "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")
    loaded_b64 = load_image_from_sample({"image": b64_str})
    assert loaded_b64 is not None
    assert loaded_b64.size == (32, 32)

    # 2. Bytearray
    loaded_barr = load_image_from_sample({"image_bytes": bytearray(raw_bytes)})
    assert loaded_barr is not None

    # 3. List with None followed by valid Image
    loaded_list_sparse = load_image_from_sample({"images": [None, img]})
    assert loaded_list_sparse is not None

    # 4. Nested dict with image key
    loaded_nested = load_image_from_sample({"image": {"image": img}})
    assert loaded_nested is not None


def test_mqt_architecture_calibration_pipeline() -> None:
    """Verify complete feature extraction and calibrator fitting on MQT-LLaVA (scale 256)."""
    np.random.seed(42)
    mock_samples = generate_mock_extraction("avqa", num_samples=60, arch="mqt")
    rows = [compute_features_from_sample(s, fine_scale=256) for s in mock_samples]
    df = pd.DataFrame(rows)

    assert "c_256" in df.columns
    assert "c_fine" in df.columns
    assert np.all(df["c_256"] == df["c_fine"])

    X_17d = df[FEATURE_KEYS].values
    X_5d = df[CANONICAL_5D_KEYS].values
    y = df["is_correct"].values

    # Fit VCPS-17D on MQT
    vcps_17d = VaryingCoefficientPlattScaler(feature_set="17d", random_state=42)
    vcps_17d.fit(X_17d, y, feature_names=FEATURE_KEYS)
    preds = vcps_17d.predict_proba(X_17d)
    assert len(preds) == len(df)
    assert np.all((preds >= 0.0) & (preds <= 1.0))

    # Fit VCPS-5D on MQT
    vcps_5d = VaryingCoefficientPlattScaler(feature_set="5d", random_state=42)
    vcps_5d.fit(X_5d, y, feature_names=CANONICAL_5D_KEYS)
    preds_5d = vcps_5d.predict_proba(X_5d)
    assert len(preds_5d) == len(df)
    assert np.all((preds_5d >= 0.0) & (preds_5d <= 1.0))


def test_generate_mock_extraction_multiscale() -> None:
    """Verify generate_mock_extraction generates valid 5-scale feature records for M3 and MQT."""
    for arch, expected_fine in [("m3", 576), ("mqt", 256)]:
        expected_scales = [1, 9, 36, 144, expected_fine]

        # Test AVQA mock extraction
        avqa_data = generate_mock_extraction("avqa", num_samples=5, arch=arch)
        assert len(avqa_data) == 5
        for item in avqa_data:
            assert item["dataset"] == "avqa"
            assert item["answer_type"] == "list_soft"
            assert isinstance(item["answers"], list)
            assert len(item["answers"]) == 10
            feats = item["features"]
            assert sorted(feats.keys()) == expected_scales
            for s in expected_scales:
                assert 0.0 < feats[s]["conf_softmax"] < 1.0
                assert feats[s]["margin"] >= 0.0
                assert feats[s]["vqa_accuracy"] in [0.0, 1.0]

        # Test VLLM Safety mock extraction
        safety_data = generate_mock_extraction("vllm-safety", num_samples=5, arch=arch)
        assert len(safety_data) == 5
        for item in safety_data:
            assert item["dataset"] == "vllm-safety"
            assert item["answer_type"] == "open"
            assert item["ground_truth"] == "unicorn"
            feats = item["features"]
            assert sorted(feats.keys()) == expected_scales
            for s in expected_scales:
                assert 0.0 < feats[s]["conf_softmax"] < 1.0


def test_17d_and_5d_feature_extraction_from_mock() -> None:
    """Verify compute_features_from_sample extracts full 17-D and canonical 5-D vectors with valid math."""
    mock_samples = generate_mock_extraction("avqa", num_samples=10, arch="m3")
    rows = [compute_features_from_sample(s, fine_scale=576) for s in mock_samples]
    df = pd.DataFrame(rows)

    # 1. Contiguous 17-D keys
    for k in FEATURE_KEYS:
        assert k in df.columns, f"Missing feature key {k}"
        assert not df[k].isna().any(), f"NaN in feature {k}"
        assert not np.isinf(df[k]).any(), f"Inf in feature {k}"

    # 2. Canonical 5D Keys
    assert CANONICAL_5D_KEYS == ["x1", "x2", "x3", "x4", "x5"]
    for k in CANONICAL_5D_KEYS:
        assert k in df.columns

    # 3. Probabilistic & Ground Truth fields
    assert "c_576" in df.columns
    assert "c_fine" in df.columns
    assert "is_correct" in df.columns
    assert "vqa_accuracy" in df.columns
    assert np.all((df["c_fine"] >= 0.0) & (df["c_fine"] <= 1.0))
    assert np.all(df["is_correct"].isin([0, 1]))


def test_calibrators_fit_predict_on_adversarial_mock() -> None:
    """Verify single-pass and VCPS calibrators successfully fit and predict on adversarial mock features."""
    import random

    random.seed(42)
    np.random.seed(42)

    mock_samples = generate_mock_extraction("vllm-safety", num_samples=100, arch="m3")
    rows = [compute_features_from_sample(s, fine_scale=576) for s in mock_samples]
    df = pd.DataFrame(rows)

    train_idx, test_idx = get_stratified_split(df, test_size=0.3, random_state=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train_17d = train_df[FEATURE_KEYS].values
    y_train = train_df["is_correct"].values
    X_test_17d = test_df[FEATURE_KEYS].values
    y_test = test_df["is_correct"].values
    c_test_576 = test_df["c_576"].values

    X_train_5d = train_df[CANONICAL_5D_KEYS].values
    X_test_5d = test_df[CANONICAL_5D_KEYS].values

    # 1. VCPS-17D
    vcps_17d = VaryingCoefficientPlattScaler(feature_set="17d", random_state=42)
    vcps_17d.fit(X_train_17d, y_train, feature_names=FEATURE_KEYS)
    p_vcps_17d = vcps_17d.predict_proba(X_test_17d)
    assert len(p_vcps_17d) == len(test_df)
    assert np.all((p_vcps_17d >= 0.0) & (p_vcps_17d <= 1.0))

    # 2. VCPS-5D
    vcps_5d = VaryingCoefficientPlattScaler(feature_set="5d", random_state=42)
    vcps_5d.fit(X_train_5d, y_train, feature_names=CANONICAL_5D_KEYS)
    p_vcps_5d = vcps_5d.predict_proba(X_test_5d)
    assert len(p_vcps_5d) == len(test_df)
    assert np.all((p_vcps_5d >= 0.0) & (p_vcps_5d <= 1.0))

    # 3. Trajectory Platt 5D
    tp_5d = TrajectoryPlattScaler(n_features=len(CANONICAL_5D_KEYS))
    tp_5d.fit(X_train_5d, y_train)
    p_tp_5d = tp_5d.predict_proba(X_test_5d)
    assert len(p_tp_5d) == len(test_df)
    assert np.all((p_tp_5d >= 0.0) & (p_tp_5d <= 1.0))

    # 4. Platt Scaling (1D)
    platt_1d = PlattScalingEstimator()
    platt_1d.fit(X_train_17d, y_train)
    p_platt = platt_1d.predict_proba(X_test_17d)
    assert len(p_platt) == len(test_df)
    assert np.all((p_platt >= 0.0) & (p_platt <= 1.0))

    # 5. Temperature Scaling
    ts = TemperatureScalingEstimator()
    ts.fit(X_train_17d, y_train)
    p_ts = ts.predict_proba(X_test_17d)
    assert len(p_ts) == len(test_df)
    assert np.all((p_ts >= 0.0) & (p_ts <= 1.0))

    # 6. Verify full metric panel
    panel = evaluate_full_metric_panel(p_vcps_17d, y_test, c_test_576, y_train=y_train)
    assert "adaptive_ece_percent" in panel
    assert "ece_percent" in panel
    assert "brier" in panel
    assert "auroc" in panel
    assert np.isfinite(panel["adaptive_ece_percent"])
    assert np.isfinite(panel["brier"])
    assert np.isfinite(panel["auroc"])

    # Murphy decomposition identity: Brier = Rel - Res + Unc + Within
    from trajectory_calibration.metrics.murphy import compute_murphy_brier_decomposition

    decomp = compute_murphy_brier_decomposition(p_vcps_17d, y_test)
    murphy_reconstructed = (
        decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"] + decomp["within"]
    )
    assert abs(decomp["brier"] - murphy_reconstructed) < 1e-5


def test_local_fallback_loaders(tmp_path: Path) -> None:
    """Verify _load_local_avqa and _load_local_vllm_safety parse local synthetic JSON files correctly."""
    # 1. AVQA Local Fallback (list format)
    avqa_dir = tmp_path / "avqa"
    avqa_dir.mkdir(parents=True, exist_ok=True)
    q_file = avqa_dir / "v1_avqa_r1+r2+r3_val_questions.json"
    ann_file = avqa_dir / "v1_avqa_r1+r2+r3_val_annotations.json"

    q_data = {
        "questions": [
            {"question_id": 1001, "question": "Is the light on?", "image_name": "img1.jpg"},
            {"question_id": 1002, "question": "What is the dog doing?", "image_name": "img2.jpg"},
        ]
    }
    ann_data = {
        "annotations": [
            {"question_id": 1001, "answers": [{"answer": "yes"} for _ in range(10)]},
            {"question_id": 1002, "answers": [{"answer": "running"} for _ in range(10)]},
        ]
    }
    q_file.write_text(json.dumps(q_data), encoding="utf-8")
    ann_file.write_text(json.dumps(ann_data), encoding="utf-8")

    avqa_samples = _load_local_avqa(avqa_dir)
    assert len(avqa_samples) == 2
    assert avqa_samples[0]["question_id"] == "1001"
    assert avqa_samples[0]["question"] == "Is the light on?"
    assert len(avqa_samples[0]["answers"]) == 10
    assert avqa_samples[1]["question_id"] == "1002"

    # 2. AVQA Local Fallback (dict of questions format)
    avqa_dict_dir = tmp_path / "avqa_dict"
    avqa_dict_dir.mkdir(parents=True, exist_ok=True)
    q_dict_file = avqa_dict_dir / "questions.json"
    ann_dict_file = avqa_dict_dir / "annotations.json"

    q_dict_data = {
        "2001": {"question": "Is it day or night?", "image_name": "day.jpg"},
    }
    ann_dict_data = {
        "annotations": [
            {"question_id": 2001, "answers": ["day"] * 10},
        ]
    }
    q_dict_file.write_text(json.dumps(q_dict_data), encoding="utf-8")
    ann_dict_file.write_text(json.dumps(ann_dict_data), encoding="utf-8")
    avqa_dict_samples = _load_local_avqa(avqa_dict_dir)
    assert len(avqa_dict_samples) == 1
    assert avqa_dict_samples[0]["question_id"] == "2001"

    # 3. VLLM Safety Local Fallback
    vllm_dir = tmp_path / "vllm_safety"
    misleading_dir = vllm_dir / "redteaming" / "misleading_attack"
    misleading_dir.mkdir(parents=True, exist_ok=True)
    vllm_ann_file = misleading_dir / "annotation.json"

    vllm_data = [
        {"id": 501, "question": "How many unicorns?", "answer": "0", "image": "unicorn_0.png"},
        {"id": 502, "question": "Is there a dragon?", "answer": "no", "image": "dragon_0.png"},
    ]
    vllm_ann_file.write_text(json.dumps(vllm_data), encoding="utf-8")

    safety_samples = _load_local_vllm_safety(vllm_dir)
    assert len(safety_samples) == 2
    assert safety_samples[0]["id"] == 501
    assert safety_samples[0]["question"] == "How many unicorns?"
    assert safety_samples[0]["answer"] == "0"
    assert safety_samples[1]["id"] == 502

    # 4. Graceful handling of non-existent or malformed files
    assert _load_local_avqa(tmp_path / "non_existent", auto_download=False) == []
    assert _load_local_vllm_safety(tmp_path / "non_existent", auto_download=False) == []

    bad_dir = tmp_path / "bad_json"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "questions.json").write_text("INVALID JSON", encoding="utf-8")
    (bad_dir / "annotations.json").write_text("INVALID JSON", encoding="utf-8")
    assert _load_local_avqa(bad_dir) == []
    (bad_dir / "annotation.json").write_text("INVALID JSON", encoding="utf-8")
    assert _load_local_vllm_safety(bad_dir) == []
