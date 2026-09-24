"""
Unit tests for strict sample parity, manifest integrity, rollout slicing, and numerical safeguards (ADR 0006).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts.build_canonical_manifest import DATASET_TARGETS
from scripts.run_mqt_topup_2k import _generate_mock_mqt_record, topup_dataset
from trajectory_calibration.vlm.patches import apply_transformers_compatibility_patches


def test_canonical_manifest_structure_and_counts() -> None:
    """Verify data/canonical_manifest_all.json contains exact sample counts and required fields."""
    manifest_path = Path("data/canonical_manifest_all.json")
    assert manifest_path.exists(), "data/canonical_manifest_all.json must exist"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for ds_name, target_n in DATASET_TARGETS.items():
        assert ds_name in manifest, f"Dataset {ds_name} must be in manifest"
        samples = manifest[ds_name]
        assert len(samples) == target_n, (
            f"{ds_name} must have exactly {target_n} samples, got {len(samples)}"
        )

        # Check required keys in samples
        for s in samples[:10]:
            assert "question_id" in s and s["question_id"] is not None
            assert "question" in s and isinstance(s["question"], str)
            assert "answers" in s and isinstance(s["answers"], list)
            assert "image" in s

    # Total sample count must equal 17,900
    total = sum(len(v) for v in manifest.values())
    assert total == 17900, f"Total manifest samples must be 17900, got {total}"


def test_trimmed_feature_file_lengths_and_parity() -> None:
    """Verify all trimmed core feature files in temp_0.0 have exactly 2,000 samples."""
    m3_dir = Path("data/features/m3_llava/temp_0.0")
    core_datasets = [
        "ai2d",
        "chartqa",
        "docvqa",
        "scienceqa",
        "textvqa",
        "vizwiz-vqa",
        "vqav2_5scale",
    ]

    manifest_path = Path("data/canonical_manifest_all.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for ds in core_datasets:
        pt_path = m3_dir / f"{ds}.pt"
        assert pt_path.exists(), f"{pt_path} must exist"
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert len(data) == 2000, f"{ds} must have exactly 2000 samples, got {len(data)}"

        # Parity check: question_id in data must match manifest
        manifest_key = "vqav2" if ds == "vqav2_5scale" else ds
        expected_ids = [str(x["question_id"]) for x in manifest[manifest_key][:2000]]
        actual_ids = [str(x["question_id"]) for x in data[:2000]]
        assert actual_ids == expected_ids, f"Question IDs for {ds} must match manifest 1-to-1"


def test_transformers_compatibility_patches() -> None:
    """Verify apply_transformers_compatibility_patches intercepts and strips modern kwargs."""

    class DummyModel:
        def __init__(self) -> None:
            self.kwargs_received: dict[str, Any] = {}

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            self.kwargs_received = dict(kwargs)
            return type("Output", (), {"logits": torch.tensor([1.0, 2.0])})()

        def prepare_inputs_for_generation(
            self, input_ids: Any, past_key_values: Any = None, **kwargs: Any
        ) -> dict[str, Any]:
            res = dict(kwargs)
            res["input_ids"] = input_ids
            return res

        def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> dict[str, Any]:
            return model_kwargs

    model = DummyModel()
    apply_transformers_compatibility_patches(model)

    # 1. Forward kwarg stripping
    model.forward(
        torch.tensor([1]),
        cache_position=torch.tensor([0]),
        num_logits_to_keep=1,
        attention_mask=torch.tensor([1]),
    )
    assert "cache_position" not in model.kwargs_received
    assert "num_logits_to_keep" not in model.kwargs_received
    assert "attention_mask" in model.kwargs_received

    # 2. prepare_inputs_for_generation kwarg stripping
    prep_out = model.prepare_inputs_for_generation(
        torch.tensor([1]),
        cache_position=torch.tensor([0]),
        num_logits_to_keep=1,
    )
    assert "cache_position" not in prep_out
    assert "num_logits_to_keep" not in prep_out

    # 3. _validate_model_kwargs stripping
    val_out = model._validate_model_kwargs({"matryoshka_vis_token_scale": 1, "use_cache": True})
    assert "matryoshka_vis_token_scale" not in val_out
    assert val_out.get("use_cache") is True


def test_rotary_embedding_device_alignment() -> None:
    """Verify patch_llama_rotary_embedding dynamically moves inv_freq buffer to input tensor device."""
    from trajectory_calibration.vlm.patches import _make_safe_rotary_forward

    class DummyRotaryEmbedding:
        def __init__(self) -> None:
            self.inv_freq = torch.tensor([1.0, 2.0, 3.0], device="cpu")

        def forward(
            self, x: torch.Tensor, position_ids: torch.Tensor | None = None
        ) -> torch.Tensor:
            return self.inv_freq * x

    rotary = DummyRotaryEmbedding()
    rotary.forward = _make_safe_rotary_forward(rotary.forward)

    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    out = rotary.forward(input_tensor)
    assert out.device == input_tensor.device
    assert rotary.inv_freq.device == input_tensor.device


def test_spectral_logdet_numerical_stability() -> None:
    """Verify spectral logdet eliminates -inf collapse and NaN on collinear and zero matrices."""
    import sys

    umpire_root = Path(__file__).resolve().parent.parent.parent / "umpire_testing"
    if str(umpire_root) not in sys.path:
        sys.path.insert(0, str(umpire_root))

    from modules.logdet_utils import compute_logdet, slice_rollouts

    # 1. Collinear vectors (Gram matrix with repeated rows)
    collinear = np.ones((10, 64))
    K_collinear = np.dot(collinear, collinear.T)
    val = compute_logdet(K_collinear, jitter=1e-6)
    assert np.isfinite(val), f"compute_logdet must be finite, got {val}"
    assert val >= 10.0 * np.log(1e-6) - 1e-5

    # 2. Zero Gram matrix
    K_zero = np.zeros((10, 10))
    val_zero = compute_logdet(K_zero, jitter=1e-6)
    assert np.isfinite(val_zero)
    # logdet(0 + 1e-6 * I) = 10 * ln(1e-6) ~ 10 * (-13.8155) = -138.155
    assert pytest.approx(val_zero, rel=1e-3) == 10.0 * np.log(1e-6)

    # 3. Identity Gram matrix
    K_eye = np.eye(10)
    val_eye = compute_logdet(K_eye, jitter=1e-6)
    assert np.isfinite(val_eye)
    assert pytest.approx(val_eye, abs=1e-4) == 0.0

    # 4. Slice rollouts helper with internal_embedding and embedding
    sample = {
        "generations_text": [f"ans_{i}" for i in range(50)],
        "generations_log_likelihood": [[-0.1] for _ in range(50)],
        "norm_embedding": np.ones((50, 32)),
        "internal_embedding": np.ones((50, 64)),
        "embedding": np.ones((50, 64)),
        "cluster_ids": list(range(50)),
    }
    sliced_10 = slice_rollouts(sample, 10)
    assert len(sliced_10["generations_text"]) == 10
    assert len(sliced_10["generations_log_likelihood"]) == 10
    assert len(sliced_10["norm_embedding"]) == 10
    assert len(sliced_10["internal_embedding"]) == 10
    assert len(sliced_10["embedding"]) == 10
    assert len(sliced_10["cluster_ids"]) == 10


def test_batched_stopping_criteria_sub() -> None:
    """Verify StoppingCriteriaSub does not prematurely halt when sequence 0 hits stop word."""
    import sys

    umpire_root = Path(__file__).resolve().parent.parent.parent / "umpire_testing"
    if str(umpire_root) not in sys.path:
        sys.path.insert(0, str(umpire_root))

    from modules.models.llava_models import StoppingCriteriaSub

    class DummyTokenizer:
        eos_token = "</s>"

        def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
            # Map token 99 to </s>
            tokens = [str(t.item() if hasattr(t, "item") else t) for t in token_ids]
            if "99" in tokens:
                return "answer </s>"
            return "answer continuing"

    sc = StoppingCriteriaSub(
        stops=["</s>"], tokenizer=DummyTokenizer(), match_on="text", initial_length=1
    )

    # Batch of 2 sequences: seq 0 completed (contains 99), seq 1 still generating (contains 10)
    input_ids = torch.tensor(
        [
            [1, 99],  # seq 0
            [1, 10],  # seq 1
        ]
    )
    scores = torch.zeros((2, 100))

    # Must NOT stop since seq 1 is not finished
    assert sc(input_ids, scores) is False

    # Batch of 2 sequences: both completed
    input_ids_all_done = torch.tensor(
        [
            [1, 99],
            [1, 99],
        ]
    )
    assert sc(input_ids_all_done, scores) is True


def test_deberta_fast_path_equivalence_grouping() -> None:
    """Verify semantic entropy fast path groups identical strings in O(K) without extra calls."""
    import sys

    umpire_root = Path(__file__).resolve().parent.parent.parent / "umpire_testing"
    if str(umpire_root) not in sys.path:
        sys.path.insert(0, str(umpire_root))

    from pipeline.compute_umpire_and_evaluate import compute_semantic_entropy_from_scratch

    class MockEntailment:
        def __init__(self) -> None:
            self.call_count = 0

        def check_implication(self, text1: str, text2: str, **kwargs: Any) -> int:
            self.call_count += 1
            return 2 if text1 == text2 else 0

    # 50 rollouts: 40 "Paris", 10 "London"
    texts = ["Paris"] * 40 + ["London"] * 10
    sample = {
        "generations_text": texts,
        "generations_log_likelihood": [[-0.5]] * 50,
    }
    model = MockEntailment()
    se = compute_semantic_entropy_from_scratch(sample, model)
    assert np.isfinite(se)
    # Fast path should only call model on unique pairs (Paris, London), NOT 50x50 pairs!
    assert model.call_count <= 4


def test_mqt_topup_mock_flow(tmp_path: Path) -> None:
    """Verify topup_dataset appends missing records up to 2,000 samples."""
    manifest_items = [
        {"question_id": f"q_{i}", "question": f"Question {i}", "answers": ["ans"]}
        for i in range(2000)
    ]
    # Create initial checkpoint with 1,000 samples
    init_records = [_generate_mock_mqt_record(manifest_items[i], "test_ds", i) for i in range(1000)]
    pt_path = tmp_path / "test_ds.pt"
    torch.save(init_records, pt_path)

    # Top-up remaining 1,000 samples
    added = topup_dataset("test_ds", manifest_items, tmp_path, mock=True)
    assert added == 1000

    # Verify completed file has exactly 2,000 samples
    final_data = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert len(final_data) == 2000
    assert final_data[0]["question_id"] == "q_0"
    assert final_data[1999]["question_id"] == "q_1999"


def test_mqt_repaired_features_non_zero_accuracy() -> None:
    """Verify MQT ai2d, chartqa, and vqav2_5scale have non-zero accuracy in samples 1000..1999."""
    mqt_dir = Path("data/features/mqt_llava/temp_0.0")
    for ds, min_expected_acc in [("ai2d", 0.40), ("chartqa", 0.03), ("vqav2_5scale", 0.60)]:
        pt_path = mqt_dir / f"{ds}.pt"
        assert pt_path.exists(), f"{pt_path} must exist"
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert len(data) == 2000
        last_1k_accs = [it["features"][256]["vqa_accuracy"] for it in data[1000:]]
        mean_last_1k = float(np.mean(last_1k_accs))
        assert mean_last_1k >= min_expected_acc, (
            f"{ds} samples 1000..1999 expected mean acc >= {min_expected_acc}, got {mean_last_1k}"
        )
        assert all("is_correct" in it for it in data)


def test_new_scripts_under_200_loc() -> None:
    """Verify all functions in new scripts are strictly under 200 LOC."""
    new_script_paths = [
        Path("scripts/build_canonical_manifest.py"),
        Path("scripts/trim_features_to_2k.py"),
        Path("scripts/run_mqt_topup_2k.py"),
        Path("scripts/repair_dataset_features.py"),
    ]
    violations = []
    for sp in new_script_paths:
        assert sp.exists(), f"{sp} must exist"
        tree = ast.parse(sp.read_text(encoding="utf-8"), filename=str(sp))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                loc = end_lineno - node.lineno + 1
                if loc >= 200:
                    violations.append(f"{sp.name}::{node.name} ({loc} LOC)")

    assert not violations, f"Functions exceeding 200 LOC: {violations}"
