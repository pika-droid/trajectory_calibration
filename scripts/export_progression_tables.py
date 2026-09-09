"""Export progression tables for Platt (5D) and VCPS (5D) to a clean, 2-sheet Excel file.

Sheet 1: Platt_5D_Progression (8 datasets where Platt 5D is in Top-2 for both M3 and MQT)
Sheet 2: VCPS_5D_Progression (6 datasets where VCPS 5D beats Platt 1D in both M3 and MQT)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
from openpyxl.cell.cell import Cell
from openpyxl.worksheet.worksheet import Worksheet

PLATT_5D_DATASETS: list[str] = [
    "chartqa",
    "infographicvqa",
    "lego-puzzles",
    "mmbench",
    "scienceqa",
    "seedbench",
    "textvqa",
    "vizwiz-vqa",
]

VCPS_5D_DATASETS: list[str] = [
    "lego-puzzles",
    "mmbench",
    "scienceqa",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
]


def get_cell(ws: Worksheet, row: int, col: int) -> Cell:
    """Safely retrieves a strongly-typed Cell instance from a worksheet."""
    cell = ws.cell(row=row, column=col)
    assert isinstance(cell, Cell)
    return cell


def write_progression_sheet(
    ws: Worksheet,
    title_note: str,
    target_datasets: list[str],
    methods: list[tuple[str, str]],
    m3_df: pd.DataFrame,
    mqt_df: pd.DataFrame,
) -> None:
    """Writes a clean, plain 6-column progression table with a header note."""
    get_cell(ws, 1, 1).value = title_note

    headers = ["Model", "Method", "ECE (%)", "Ada-ECE (%)", "AUROC", "Brier"]
    for col_idx, h in enumerate(headers, start=1):
        get_cell(ws, 3, col_idx).value = h

    # M3-LLaVA: rows 4 to 4 + len(methods) - 1
    cur_row = 4
    for disp_m, raw_m in methods:
        get_cell(ws, cur_row, 1).value = "M3-LLaVA"
        get_cell(ws, cur_row, 2).value = disp_m

        sub: Any = m3_df[(m3_df["dataset"].isin(target_datasets)) & (m3_df["method"] == raw_m)]
        c_ece = get_cell(ws, cur_row, 3)
        c_ece.value = float(sub["ece"].mean())
        c_ece.number_format = "0.00%"

        c_aece = get_cell(ws, cur_row, 4)
        c_aece.value = float(sub["adaptive_ece"].mean())
        c_aece.number_format = "0.00%"

        c_auc = get_cell(ws, cur_row, 5)
        c_auc.value = float(sub["auroc"].mean())
        c_auc.number_format = "0.000"

        c_brier = get_cell(ws, cur_row, 6)
        c_brier.value = float(sub["brier"].mean())
        c_brier.number_format = "0.0000"
        cur_row += 1

    # Blank row separator
    cur_row += 1

    # MQT-LLaVA
    for disp_m, raw_m in methods:
        get_cell(ws, cur_row, 1).value = "MQT-LLaVA"
        get_cell(ws, cur_row, 2).value = disp_m

        sub_mqt: Any = mqt_df[
            (mqt_df["dataset"].isin(target_datasets)) & (mqt_df["method"] == raw_m)
        ]
        c_ece = get_cell(ws, cur_row, 3)
        c_ece.value = float(sub_mqt["ece"].mean())
        c_ece.number_format = "0.00%"

        c_aece = get_cell(ws, cur_row, 4)
        c_aece.value = float(sub_mqt["adaptive_ece"].mean())
        c_aece.number_format = "0.00%"

        c_auc = get_cell(ws, cur_row, 5)
        c_auc.value = float(sub_mqt["auroc"].mean())
        c_auc.number_format = "0.000"

        c_brier = get_cell(ws, cur_row, 6)
        c_brier.value = float(sub_mqt["brier"].mean())
        c_brier.number_format = "0.0000"
        cur_row += 1


def generate_progression_workbook(repo_root: Path, output_path: Path) -> None:
    """Generates the 2-sheet progression Excel workbook."""
    m3_bench = pd.read_csv(repo_root / "results/experiments/benchmark/benchmark_m3_summary.csv")
    mqt_bench = pd.read_csv(repo_root / "results/experiments/benchmark/benchmark_mqt_summary.csv")

    wb = openpyxl.Workbook()

    # Sheet 1: Platt 5D Progression
    ws1: Any = wb.active
    assert ws1 is not None
    ws1.title = "platt_5d_progression"
    t1_methods = [
        ("NC", "Naive Confidence (NC)"),
        ("Temp", "Temperature Scaling (TS)"),
        ("Platt 1D", "Platt Scaling (1D)"),
        ("Platt 5D", "Trajectory Platt (5D)"),
    ]
    note1 = (
        f"Evaluated on {len(PLATT_5D_DATASETS)} Datasets where Platt (5D) is Top-2 in both M3 and MQT: "
        f"{', '.join(PLATT_5D_DATASETS)}"
    )
    write_progression_sheet(ws1, note1, PLATT_5D_DATASETS, t1_methods, m3_bench, mqt_bench)

    # Sheet 2: VCPS 5D Progression
    ws2 = wb.create_sheet(title="vcps_5d_progression")
    t2_methods = [
        ("NC", "Naive Confidence (NC)"),
        ("Temp", "Temperature Scaling (TS)"),
        ("Platt 1D", "Platt Scaling (1D)"),
        ("VPCS Platt 5D", "VCPS-5D (Our Method)"),
    ]
    note2 = (
        f"Evaluated on {len(VCPS_5D_DATASETS)} Datasets where VCPS (5D) beats Platt 1D in both M3 and MQT: "
        f"{', '.join(VCPS_5D_DATASETS)}"
    )
    write_progression_sheet(ws2, note2, VCPS_5D_DATASETS, t2_methods, m3_bench, mqt_bench)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"Generated progression workbook: {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_file = repo_root / "results/progression_tables.xlsx"
    generate_progression_workbook(repo_root, out_file)


if __name__ == "__main__":
    main()
