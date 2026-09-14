#!/usr/bin/env python3
"""
Generate Paper Table 2 (M3-LLaVA 7B) and Table 3 (MQT-LLaVA 7B) for Cross-Domain Transfer.

Reads zero-shot transfer metrics from sheets/pairwise_cross_domain_transfer.xlsx and
in-domain DirectTrain ceilings from benchmark summary CSVs, formatting them into
publication-ready LaTeX tables under dataset_tables/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = ROOT / "sheets" / "pairwise_cross_domain_transfer.xlsx"
BENCHMARK_M3_PATH = ROOT / "results" / "experiments" / "benchmark" / "benchmark_m3_summary.csv"
BENCHMARK_MQT_PATH = ROOT / "results" / "experiments" / "benchmark" / "benchmark_mqt_summary.csv"
OUTPUT_DIR = ROOT / "dataset_tables"
TABLE2_OUTPUT = OUTPUT_DIR / "table2_m3_transfer.tex"
TABLE3_OUTPUT = OUTPUT_DIR / "table3_mqt_transfer.tex"

TRANSFER_METHODS: list[tuple[str, str]] = [
    ("NC", "Uncalibrated (NC)"),
    ("TS", "Temperature Scaling (TS)"),
    ("Platt 1D", "Platt Scaling (1D)"),
    ("Platt 5D", r"\textbf{Platt 5D (Ours)}"),
]
DIRECT_TRAIN_DISPLAY = r"\textsc{In-Domain Calibration (Platt 5D)}"

SOURCE_BLOCKS: list[tuple[str, str, str, list[tuple[str, str, str]]]] = [
    (
        "vqav2",
        "Source: VQAv2",
        "General Scene VQA",
        [
            ("docvqa", "DocVQA", "Document Text"),
            ("chartqa", "ChartQA", "Diagrams"),
            ("textvqa", "TextVQA", "Scene Text"),
        ],
    ),
    (
        "vizwiz-vqa",
        "Source: VizWiz-VQA",
        "Assistive / Blurry",
        [
            ("chartqa", "ChartQA", "Diagrams"),
            ("docvqa", "DocVQA", "Document Text"),
            ("textvqa", "TextVQA", "Scene Text"),
        ],
    ),
    (
        "textvqa",
        "Source: TextVQA",
        "Scene Text",
        [
            ("chartqa", "ChartQA", "Diagrams"),
            ("docvqa", "DocVQA", "Document Text"),
            ("vizwiz-vqa", "VizWiz-VQA", "Assistive"),
        ],
    ),
]


def load_pairwise_transfer_data(
    excel_path: Path,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    """Parse pairwise transfer sheets for M3 and MQT across requested sources and targets."""
    data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {
        "m3": {},
        "mqt": {},
    }

    for src_key, _, _, _ in SOURCE_BLOCKS:
        sheet_name = f"src_{src_key}"
        df_sheet = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

        for arch_key, title_prefix in [("m3", "M3-LLaVA"), ("mqt", "MQT-LLaVA")]:
            if src_key not in data[arch_key]:
                data[arch_key][src_key] = {}

            title_row_idx = -1
            for r in range(len(df_sheet)):
                val = df_sheet.iloc[r, 0]
                if pd.notna(val) and str(val).strip().startswith(title_prefix):
                    title_row_idx = r
                    break

            if title_row_idx < 0:
                raise ValueError(f"Could not locate {title_prefix} section in {sheet_name}")

            header_row_idx = title_row_idx + 1
            raw_header = [
                str(x).strip() if pd.notna(x) else "" for x in df_sheet.iloc[header_row_idx]
            ]

            curr_r = header_row_idx + 1
            while curr_r < len(df_sheet):
                row_label = str(df_sheet.iloc[curr_r, 0]).strip()
                if not row_label or row_label.startswith("Macro Mean") or "LLaVA" in row_label:
                    break

                tgt = row_label
                data[arch_key][src_key][tgt] = {}

                for m_code, _ in TRANSFER_METHODS:
                    ece_col = raw_header.index(f"{m_code} ECE (%)")
                    ada_col = raw_header.index(f"{m_code} Ada-ECE (%)")
                    auroc_col = raw_header.index(f"{m_code} AUROC")
                    brier_col = raw_header.index(f"{m_code} Brier")

                    ece_percent = float(df_sheet.iloc[curr_r, ece_col])
                    ada_percent = float(df_sheet.iloc[curr_r, ada_col])
                    auroc_val = float(df_sheet.iloc[curr_r, auroc_col])
                    brier_val = float(df_sheet.iloc[curr_r, brier_col])

                    data[arch_key][src_key][tgt][m_code] = {
                        "ece": ece_percent / 100.0,
                        "ada_ece": ada_percent / 100.0,
                        "auroc": auroc_val,
                        "brier": brier_val,
                    }
                curr_r += 1

    return data


def load_direct_train_data(csv_path: Path) -> dict[str, dict[str, float]]:
    """Load in-domain calibration ceilings from benchmark summary CSV for Trajectory Platt (5D)."""
    df = pd.read_csv(csv_path)
    sub = df[df["method"] == "Trajectory Platt (5D)"]
    direct_data: dict[str, dict[str, float]] = {}

    for _, row in sub.iterrows():
        ds = str(row["dataset"]).strip()
        direct_data[ds] = {
            "ece": float(row["ece"]),
            "ada_ece": float(row["adaptive_ece"]),
            "auroc": float(row["auroc"]),
            "brier": float(row["brier"]),
        }

    return direct_data


def rank_and_format_transfer(
    vals: list[float],
    higher_is_better: bool = False,
    decimals: int = 4,
) -> list[str]:
    """Rank transfer methods using Option A (Bold Rank 1, Italic Rank 2)."""
    valid_vals = [round(v, decimals) for v in vals if not np.isnan(v)]
    if not valid_vals:
        return ["-" for _ in vals]

    sorted_unique = sorted(list(set(valid_vals)), reverse=higher_is_better)
    best_val = sorted_unique[0] if len(sorted_unique) > 0 else None
    second_val = sorted_unique[1] if len(sorted_unique) > 1 else None

    formatted: list[str] = []
    for v in vals:
        if np.isnan(v):
            formatted.append("-")
            continue
        v_round = round(v, decimals)
        s = f"{v:.{decimals}f}"
        if best_val is not None and abs(v_round - best_val) < 1e-6:
            formatted.append(f"\\textbf{{{s}}}")
        elif second_val is not None and abs(v_round - second_val) < 1e-6:
            formatted.append(f"\\textit{{{s}}}")
        else:
            formatted.append(s)

    return formatted


def render_source_block(
    src_title: str,
    src_context: str,
    targets: list[tuple[str, str, str]],
    src_key: str,
    arch_key: str,
    transfer_data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
    direct_train_data: dict[str, dict[str, float]],
    is_first_block: bool = False,
) -> list[str]:
    """Render a single source domain block (header, column names, DirectTrain, and 4 transfer rows)."""
    lines: list[str] = []
    if not is_first_block:
        lines.append(r"\midrule")
    lines.extend(
        [
            rf"\multicolumn{{13}}{{l}}{{\textbf{{{src_title}}} ({src_context})}} \\",
            r"\midrule",
        ]
    )

    target_headers: list[str] = []
    for _, tgt_disp, tgt_ctx in targets:
        target_headers.append(rf"\multicolumn{{4}}{{c}}{{\textbf{{{tgt_disp}}} ({tgt_ctx})}}")
    lines.append(" & " + " & ".join(target_headers) + r" \\")

    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}")
    subcol_unit = r"\textbf{ECE} $\downarrow$ & \textbf{Ada-ECE} $\downarrow$ & \textbf{AUROC} $\uparrow$ & \textbf{Brier} $\downarrow$"
    lines.append(rf"\textbf{{Method}} & {subcol_unit} & {subcol_unit} & {subcol_unit} \\")
    lines.append(r"\midrule")

    dt_cells: list[str] = []
    for tgt_key, _, _ in targets:
        dt_metrics = direct_train_data[tgt_key]
        dt_cells.append(f"{dt_metrics['ece']:.4f}")
        dt_cells.append(f"{dt_metrics['ada_ece']:.4f}")
        dt_cells.append(f"{dt_metrics['auroc']:.3f}")
        dt_cells.append(f"{dt_metrics['brier']:.4f}")
    lines.append(f"{DIRECT_TRAIN_DISPLAY} & " + " & ".join(dt_cells) + r" \\")

    tgt_formatted: dict[str, dict[str, list[str]]] = {}
    for tgt_key, _, _ in targets:
        ece_vals = [
            transfer_data[arch_key][src_key][tgt_key][m]["ece"] for m, _ in TRANSFER_METHODS
        ]
        ada_vals = [
            transfer_data[arch_key][src_key][tgt_key][m]["ada_ece"] for m, _ in TRANSFER_METHODS
        ]
        auroc_vals = [
            transfer_data[arch_key][src_key][tgt_key][m]["auroc"] for m, _ in TRANSFER_METHODS
        ]
        brier_vals = [
            transfer_data[arch_key][src_key][tgt_key][m]["brier"] for m, _ in TRANSFER_METHODS
        ]

        tgt_formatted[tgt_key] = {
            "ece": rank_and_format_transfer(ece_vals, higher_is_better=False, decimals=4),
            "ada_ece": rank_and_format_transfer(ada_vals, higher_is_better=False, decimals=4),
            "auroc": rank_and_format_transfer(auroc_vals, higher_is_better=True, decimals=3),
            "brier": rank_and_format_transfer(brier_vals, higher_is_better=False, decimals=4),
        }

    for idx, (_, m_display) in enumerate(TRANSFER_METHODS):
        row_cells: list[str] = []
        for tgt_key, _, _ in targets:
            row_cells.append(tgt_formatted[tgt_key]["ece"][idx])
            row_cells.append(tgt_formatted[tgt_key]["ada_ece"][idx])
            row_cells.append(tgt_formatted[tgt_key]["auroc"][idx])
            row_cells.append(tgt_formatted[tgt_key]["brier"][idx])
        lines.append(f"{m_display} & " + " & ".join(row_cells) + r" \\")

    return lines


def generate_paper_table(
    arch_key: str,
    arch_display: str,
    table_label: str,
    transfer_data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
    direct_train_data: dict[str, dict[str, float]],
) -> str:
    """Generate complete publication-ready LaTeX table for one architecture."""
    caption = (
        rf"\textbf{{Zero-Shot Cross-Domain Calibration Transfer ({arch_display} 7B).}} "
        r"Pairwise zero-shot calibration transfer from source datasets across held-out target "
        r"vision-language benchmarks without target domain supervision. "
        r"Evaluated across Expected Calibration Error (\textbf{ECE} $\downarrow$), "
        r"Adaptive ECE (\textbf{Ada-ECE} $\downarrow$), AUROC ($\uparrow$), and Brier score ($\downarrow$). "
        r"\textbf{Bold}: Rank 1; \textit{italic}: Rank 2 among transfer methods."
    )

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{table_label}}}",
        r"\tablestyle{3pt}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l cccc cccc cccc}",
        r"\toprule",
    ]

    for idx, (src_key, src_title, src_context, targets) in enumerate(SOURCE_BLOCKS):
        block_lines = render_source_block(
            src_title=src_title,
            src_context=src_context,
            targets=targets,
            src_key=src_key,
            arch_key=arch_key,
            transfer_data=transfer_data,
            direct_train_data=direct_train_data,
            is_first_block=(idx == 0),
        )
        lines.extend(block_lines)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )

    return "\n".join(lines)


def generate_both_tables() -> tuple[str, str]:
    """Load data and generate both Table 2 (M3) and Table 3 (MQT) LaTeX strings."""
    transfer_data = load_pairwise_transfer_data(EXCEL_PATH)
    dt_m3 = load_direct_train_data(BENCHMARK_M3_PATH)
    dt_mqt = load_direct_train_data(BENCHMARK_MQT_PATH)

    table2_m3 = generate_paper_table(
        arch_key="m3",
        arch_display="M3-LLaVA",
        table_label="tab:table2_m3_transfer",
        transfer_data=transfer_data,
        direct_train_data=dt_m3,
    )

    table3_mqt = generate_paper_table(
        arch_key="mqt",
        arch_display="MQT-LLaVA",
        table_label="tab:table3_mqt_transfer",
        transfer_data=transfer_data,
        direct_train_data=dt_mqt,
    )

    return table2_m3, table3_mqt


def main() -> None:
    """Generate and write Paper Table 2 and Table 3 LaTeX files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table2_m3, table3_mqt = generate_both_tables()

    TABLE2_OUTPUT.write_text(table2_m3, encoding="utf-8")
    print(f"Generated Table 2 (M3-LLaVA 7B) at {TABLE2_OUTPUT}", flush=True)

    TABLE3_OUTPUT.write_text(table3_mqt, encoding="utf-8")
    print(f"Generated Table 3 (MQT-LLaVA 7B) at {TABLE3_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
