"""
Unit tests for temperature study feature alignment and sample parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch


@pytest.fixture(scope="module")
def canonical_manifest() -> dict[str, list[dict[str, Any]]]:
    manifest_path = Path("data/canonical_manifest_all.json")
    assert manifest_path.exists(), "Manifest must exist"
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def test_textvqa_and_vizwiz_manifest_alignment(
    canonical_manifest: dict[str, list[dict[str, Any]]],
) -> None:
    """Verify textvqa and vizwiz-vqa match canonical sequence exactly across all 10 files."""
    for ds_name in ["textvqa", "vizwiz-vqa"]:
        canonical_qids = [str(x["question_id"]) for x in canonical_manifest[ds_name]]
        assert len(canonical_qids) == 2000

        for arch in ["m3_llava", "mqt_llava"]:
            for temp in ["temp_0.0", "temp_0.3", "temp_0.6", "temp_1.0", "temp_1.5"]:
                p = Path(f"data/features/{arch}/{temp}/{ds_name}.pt")
                assert p.exists(), f"Missing file: {p}"
                data: list[dict[str, Any]] = torch.load(p, map_location="cpu")
                assert len(data) == 2000
                qids = [str(x["question_id"]) for x in data]
                assert qids == canonical_qids, f"QID sequence mismatch in {p}"


def test_mqt_scienceqa_test_split_integrity(
    canonical_manifest: dict[str, list[dict[str, Any]]],
) -> None:
    """Verify MQT ScienceQA is on the canonical test split across all 5 temperatures."""
    canonical_qids = [str(x["question_id"]) for x in canonical_manifest["scienceqa"]]
    assert len(canonical_qids) == 2000

    for temp in ["temp_0.0", "temp_0.3", "temp_0.6", "temp_1.0", "temp_1.5"]:
        p = Path(f"data/features/mqt_llava/{temp}/scienceqa.pt")
        assert p.exists(), f"Missing file: {p}"
        data: list[dict[str, Any]] = torch.load(p, map_location="cpu")
        assert len(data) == 2000
        qids = [str(x["question_id"]) for x in data]
        assert qids == canonical_qids, f"QID mismatch in {p}"

        # Confirm question text matches canonical test split sample 0
        s0 = data[0].get("sample", {})
        q0 = s0.get("question", "")
        assert "Which property matches this object" in q0, (
            f"Expected canonical test split sample, got '{q0}'"
        )


def test_m3_scienceqa_t0_split_integrity(
    canonical_manifest: dict[str, list[dict[str, Any]]],
) -> None:
    """Verify M3 ScienceQA at T=0.0 is on canonical test split."""
    canonical_qids = [str(x["question_id"]) for x in canonical_manifest["scienceqa"]]
    p = Path("data/features/m3_llava/temp_0.0/scienceqa.pt")
    assert p.exists(), f"Missing file: {p}"
    data: list[dict[str, Any]] = torch.load(p, map_location="cpu")
    assert len(data) == 2000
    qids = [str(x["question_id"]) for x in data]
    assert qids == canonical_qids
    s0 = data[0].get("sample", {})
    q0 = s0.get("question", "")
    assert "Which property matches this object" in q0


def test_pope_10_file_sample_parity() -> None:
    """Verify all 10 POPE files have 100% sample parity (exact same 1,595 QIDs in order)."""
    ref_qids: list[str] | None = None
    for arch in ["m3_llava", "mqt_llava"]:
        for temp in ["temp_0.0", "temp_0.3", "temp_0.6", "temp_1.0", "temp_1.5"]:
            p = Path(f"data/features/{arch}/{temp}/pope.pt")
            assert p.exists(), f"Missing file: {p}"
            data: list[dict[str, Any]] = torch.load(p, map_location="cpu")
            assert len(data) == 1595, f"{p} has {len(data)} samples, expected 1595"
            qids = [str(x["question_id"]) for x in data]
            assert len(set(qids)) == 1595, f"{p} contains duplicate QIDs!"

            if ref_qids is None:
                ref_qids = qids
            else:
                assert qids == ref_qids, f"QID sequence mismatch in {p} against reference!"
