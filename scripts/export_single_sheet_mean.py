"""Export single-sheet mean_platt5d Excel workbook.

Contains exactly ONE sheet with the 8-dataset means for M3-LLaVA and MQT-LLaVA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
from openpyxl.cell.cell import Cell
from openpyxl.worksheet.worksheet import Worksheet

TARGET_METHODS: list[tuple[str, str, str]] = [
    ("NC", "Naive Confidence (NC)", "bench"),
    ("TS", "Temperature Scaling (TS)", "bench"),
    ("Platt 1d", "Platt Scaling (1D)", "bench"),
    ("Platt (5d)", "Trajectory Platt (5D)", "bench"),
    ("Platt (17D)", "Trajectory Platt (17D)", "bench"),
    ("VCPS (5D)", "VCPS-5D (Our Method)", "bench"),
    ("VCPS (17D)", "VCPS-17D (Our Method)", "bench"),
    ("Umpire", "umpire", "umpire"),
    ("Eigen score", "eigen_score", "umpire"),
    ("LN entropy", "ln_entropy", "umpire"),
    ("Semantic entropy", "semantic_entropy", "umpire"),
]

PLATT5D_DATASETS: list[str] = [
    "chartqa",
    "infographicvqa",
    "lego-puzzles",
    "mmbench",
    "scienceqa",
    "seedbench",
    "textvqa",
    "vizwiz-vqa",
]


def get_cell(ws: Worksheet, row: int, col: int) -> Cell:
    """Safely retrieves a strongly-typed Cell instance from a worksheet."""
    cell = ws.cell(row=row, column=col)
    assert isinstance(cell, Cell)
    return cell


def write_mean_model_rows(
    ws: Worksheet,
    start_row: int,
    model_tag: str,
    bench_df: pd.DataFrame,
    ump_df: pd.DataFrame,
) -> int:
    """Computes and writes the mean metrics across the 8 datasets for a given model."""
    for idx, (disp_m, raw_m, src) in enumerate(TARGET_METHODS):
        cur_row = start_row + idx
        get_cell(ws, cur_row, 1).value = model_tag
        get_cell(ws, cur_row, 2).value = disp_m

        c_ece = get_cell(ws, cur_row, 3)
        c_aece = get_cell(ws, cur_row, 4)
        c_auc = get_cell(ws, cur_row, 5)
        c_brier = get_cell(ws, cur_row, 6)

        if src == "bench":
            sub: Any = bench_df[
                (bench_df["dataset"].isin(PLATT5D_DATASETS)) & (bench_df["method"] == raw_m)
            ]
            c_ece.value = float(sub["ece"].mean())
            c_ece.number_format = "0.00%"
            c_aece.value = float(sub["adaptive_ece"].mean())
            c_aece.number_format = "0.00%"
            c_auc.value = float(sub["auroc"].mean())
            c_auc.number_format = "0.000"
            c_brier.value = float(sub["brier"].mean())
            c_brier.number_format = "0.0000"
        else:
            sub = ump_df[(ump_df["dataset"].isin(PLATT5D_DATASETS)) & (ump_df["method"] == raw_m)]
            c_ece.value = float((sub["cece"]).mean())
            c_ece.number_format = "0.00%"
            c_aece.value = "-"
            c_auc.value = float(sub["auc"].mean())
            c_auc.number_format = "0.000"
            c_brier.value = "-"

    return start_row + len(TARGET_METHODS)


def generate_single_sheet_mean_file(repo_root: Path, output_path: Path) -> None:
    """Generates an Excel file containing only one sheet with the mean results."""
    m3_bench = pd.read_csv(repo_root / "results/experiments/benchmark/benchmark_m3_summary.csv")
    m3_ump = pd.read_csv(repo_root / "results/umpire_eval/m3_llava_cumulative_summary.csv")
    mqt_bench = pd.read_csv(repo_root / "results/experiments/benchmark/benchmark_mqt_summary.csv")
    mqt_ump = pd.read_csv(repo_root / "results/umpire_eval/mqt_llava_cumulative_summary.csv")

    wb = openpyxl.Workbook()
    ws: Any = wb.active
    assert ws is not None
    ws.title = "mean_platt5d"

    headers = ["Model", "Method", "ECE (%)", "Ada-ECE (%)", "AUROC", "Brier"]
    for col_idx, h in enumerate(headers, start=1):
        get_cell(ws, 1, col_idx).value = h

    # M3-LLaVA: rows 2 to 12
    end_m3 = write_mean_model_rows(ws, 2, "M3-LLaVA", m3_bench, m3_ump)
    # Blank separator at row 13
    # MQT-LLaVA: rows 14 to 24
    _ = write_mean_model_rows(ws, end_m3 + 1, "MQT-LLaVA", mqt_bench, mqt_ump)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"Generated single-sheet workbook: {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_file = repo_root / "results/mean_platt5d.xlsx"
    generate_single_sheet_mean_file(repo_root, out_file)


if __name__ == "__main__":
    main()
