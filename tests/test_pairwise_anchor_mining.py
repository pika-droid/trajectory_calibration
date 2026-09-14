"""Unit tests for pairwise cross-domain anchor mining tool and statistics."""

from __future__ import annotations

import numpy as np
import pytest

from sheets.scripts.mine_pairwise_anchors import (
    LATEX_PATH,
    REPORT_PATH,
    WORKBOOK_PATH,
    compute_head_to_head_summary,
    compute_source_anchor_leaderboard,
    compute_winning_target_panel,
    format_int_rankings_latex,
    format_rankings_latex,
    load_pairwise_data,
)


@pytest.fixture(scope="module")
def pairwise_df():
    """Load pairwise evaluation dataframe from Excel workbook."""
    if not WORKBOOK_PATH.exists():
        pytest.skip(f"Workbook not found at {WORKBOOK_PATH}")
    return load_pairwise_data(WORKBOOK_PATH)


def test_pairwise_data_loading(pairwise_df) -> None:
    """Verify loading correctly parses all 312 records across both architectures."""
    assert len(pairwise_df) == 312
    assert set(pairwise_df["arch"].unique()) == {"M3", "MQT"}
    assert len(pairwise_df[pairwise_df["arch"] == "M3"]) == 156
    assert len(pairwise_df[pairwise_df["arch"] == "MQT"]) == 156

    expected_cols = [
        "arch",
        "source",
        "target",
        "nc_ece",
        "ts_ece",
        "p1_ece",
        "p5_ece",
        "vc_ece",
        "nc_ada",
        "ts_ada",
        "p1_ada",
        "p5_ada",
        "vc_ada",
        "nc_auroc",
        "ts_auroc",
        "p1_auroc",
        "p5_auroc",
        "vc_auroc",
        "nc_brier",
        "ts_brier",
        "p1_brier",
        "p5_brier",
        "vc_brier",
    ]
    for col in expected_cols:
        assert col in pairwise_df.columns
        assert pairwise_df[col].isnull().sum() == 0


def test_source_anchor_leaderboard(pairwise_df) -> None:
    """Verify source anchor leaderboard rankings and metric reductions."""
    m3_lead, mqt_lead, comb_lead = compute_source_anchor_leaderboard(pairwise_df)

    assert len(comb_lead) == 13
    assert len(m3_lead) == 13
    assert len(mqt_lead) == 13

    # VQAv2 is the undisputed #1 universal anchor
    top_anchor = comb_lead.iloc[0]
    assert top_anchor["source"] == "vqav2"
    assert top_anchor["win_p1_ece"] == 22
    assert top_anchor["win_p1_ada"] == 22
    assert top_anchor["ece_reduction"] > 6.0
    assert top_anchor["ada_reduction"] > 6.0

    # POPE case: strong on M3 (9/12) but fragile on MQT (1/12)
    pope_m3 = m3_lead[m3_lead["source"] == "pope"].iloc[0]
    pope_mqt = mqt_lead[mqt_lead["source"] == "pope"].iloc[0]
    assert pope_m3["win_p1_ece"] == 9
    assert pope_mqt["win_p1_ece"] == 1


def test_winning_target_panel_vqav2(pairwise_df) -> None:
    """Verify winning target benchmarks when trained on VQAv2."""
    panel = compute_winning_target_panel(pairwise_df, source="vqav2")
    assert len(panel) == 12

    # Top target gains should be DocVQA and ChartQA
    top_target = panel.iloc[0]["target"]
    assert top_target in ["docvqa", "chartqa"]
    assert panel.iloc[0]["avg_delta_ece"] > 12.0

    # MMMU strictly wins on all 4 metrics across both architectures
    mmmu_row = panel[panel["target"] == "mmmu"].iloc[0]
    assert bool(mmmu_row["m3_p5_wins_all4"]) is True
    assert bool(mmmu_row["mqt_p5_wins_all4"]) is True
    assert mmmu_row["avg_delta_auroc"] > 0.05
    assert mmmu_row["avg_delta_brier"] > 0.05


def test_head_to_head_vcps_vs_platt5d(pairwise_df) -> None:
    """Verify head-to-head comparison between Platt 5D (6 params) and VCPS-5D (10 params)."""
    h2h = compute_head_to_head_summary(pairwise_df)

    comb = h2h["Combined"]
    assert comb["n_pairs"] == 312
    assert comb["p5_beats_vc_ece"] == 181
    assert np.isclose(comb["p5_winrate_vc_ece"], 58.01, atol=0.1)
    assert comb["p5_beats_vc_ada"] == 177
    assert comb["median_diff_ece"] < 0.0  # Platt 5D has lower median error

    # Top 5 universal anchors subset (N=120)
    top5 = h2h["top5_universal"]
    assert top5["n_pairs"] == 120
    assert top5["p5_beats_vc_ece"] == 91
    assert np.isclose(top5["p5_winrate_vc_ece"], 75.83, atol=0.1)
    assert top5["mean_p5_ece"] < top5["mean_vc_ece"]  # P5 strictly lower macro mean ECE


def test_option_a_ranking_formatters() -> None:
    """Verify Option A universal LaTeX ranking formatters."""
    # Lower is better (ECE)
    vals = [41.252, 41.014, 41.246]
    formatted = format_rankings_latex(vals, lower_is_better=True, decimals=2)
    assert formatted[1] == "\\textbf{41.01}"
    # Both 41.252 and 41.246 round to 41.25, so both are italicized as Rank 2
    assert formatted[0] == "\\textit{41.25}"
    assert formatted[2] == "\\textit{41.25}"

    # Integer ranking formatter
    int_vals = [22, 19, 18, 18, 10]
    int_fmt = format_int_rankings_latex(int_vals, total=24)
    assert int_fmt[0] == "\\textbf{22/24}"
    assert int_fmt[1] == "\\textit{19/24}"
    assert int_fmt[2] == "18/24"
    assert int_fmt[4] == "10/24"


def test_generated_reports_exist() -> None:
    """Verify output LaTeX table and Markdown report exist and have content."""
    assert LATEX_PATH.exists()
    assert REPORT_PATH.exists()

    tex_content = LATEX_PATH.read_text(encoding="utf-8")
    assert "\\begin{table*}" in tex_content
    assert "VQAv2" in tex_content
    assert "\\textbf{22/24}" in tex_content

    md_content = REPORT_PATH.read_text(encoding="utf-8")
    assert "# Pairwise Zero-Shot Cross-Domain Anchor Mining" in md_content
    assert "DocVQA" in md_content
    assert "181/312" in md_content
