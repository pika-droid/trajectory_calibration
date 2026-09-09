#!/usr/bin/env python3
"""Generate within-domain calibration benchmark Excel file (Workbook 1).

Produces sheets/calibration_benchmark_results.xlsx containing:
1. Platt_5D_Macro: Macro mean across 11 dual-architecture top-2 qualifying datasets.
2. VCPS_5D_Macro: Macro mean across 11 dual-architecture top-2 qualifying datasets.
3. 14 Individual dataset sheets with detailed results across M3 and MQT.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "results" / "experiments" / "benchmark"
SHEETS_DIR = ROOT / "sheets"
SHEETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = SHEETS_DIR / "calibration_benchmark_results.xlsx"

ALL_DATASETS = [
    "ai2d",
    "chartqa",
    "docvqa",
    "gqa",
    "infographicvqa",
    "lego-puzzles",
    "mmbench",
    "mmmu",
    "pope",
    "scienceqa",
    "seedbench",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
]

METHOD_MAP = {
    "Naive Confidence (NC)": "NC",
    "Temperature Scaling (TS)": "Temp",
    "Platt Scaling (1D)": "Platt 1D",
    "Trajectory Platt (5D)": "Platt 5D",
    "VCPS-5D (Our Method)": "VCPS Platt 5D",
}


def load_and_preprocess_summaries() -> dict[str, pd.DataFrame]:
    """Load benchmark summary CSVs for M3 and MQT and normalize columns."""
    dfs: dict[str, pd.DataFrame] = {}
    for arch in ["m3", "mqt"]:
        csv_path = BENCHMARK_DIR / f"benchmark_{arch}_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing summary file: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[df["method"].isin(METHOD_MAP.keys())].copy()
        df["method_display"] = df["method"].map(METHOD_MAP)
        # Select and rename metrics
        df["ece_percent"] = df["ece_percent"].round(4)
        df["adaptive_ece_percent"] = df["adaptive_ece_percent"].round(4)
        df["auroc"] = df["auroc"].round(4)
        df["brier"] = df["brier"].round(4)
        dfs[arch] = df
    return dfs


def find_qualifying_datasets(dfs: dict[str, pd.DataFrame], candidate: str) -> list[str]:
    """Identify datasets where candidate is strictly top-2 in Ada-ECE on BOTH M3 and MQT."""
    passing_sets: dict[str, set[str]] = {"m3": set(), "mqt": set()}
    pool = ["NC", "Temp", "Platt 1D", candidate]

    for arch in ["m3", "mqt"]:
        df = dfs[arch]
        df_pool = df[df["method_display"].isin(pool)]
        for ds, grp in df_pool.groupby("dataset"):
            sorted_grp = pd.DataFrame(grp).sort_values(by="adaptive_ece")
            ranked_methods = list(sorted_grp["method_display"])
            if candidate in ranked_methods and ranked_methods.index(candidate) < 2:
                passing_sets[arch].add(str(ds))

    strict_both = sorted(list(passing_sets["m3"].intersection(passing_sets["mqt"])))
    return strict_both


def build_macro_table(
    dfs: dict[str, pd.DataFrame], candidate: str, qualifying_ds: list[str]
) -> pd.DataFrame:
    """Compute simple macro table across qualifying datasets for M3 and MQT."""
    methods = ["NC", "Temp", "Platt 1D", candidate]
    rows = []

    for arch, arch_label in [("m3", "M3"), ("mqt", "MQT")]:
        df = dfs[arch]
        df_qual = df[(df["dataset"].isin(qualifying_ds)) & (df["method_display"].isin(methods))]
        macro = (
            df_qual.groupby("method_display")[
                ["ece_percent", "adaptive_ece_percent", "auroc", "brier"]
            ]
            .mean()
            .reset_index()
        )

        for m in methods:
            m_row = pd.DataFrame(macro[macro["method_display"] == m])
            if len(m_row) > 0:
                rec = m_row.to_dict(orient="records")[0]
                rows.append(
                    {
                        "Model": arch_label,
                        "Method": m,
                        "ECE (%)": round(float(rec["ece_percent"]), 4),
                        "Adaptive ECE (%)": round(float(rec["adaptive_ece_percent"]), 4),
                        "AUROC": round(float(rec["auroc"]), 4),
                        "Brier Score": round(float(rec["brier"]), 4),
                    }
                )

    return pd.DataFrame(rows)


def build_dataset_table(dfs: dict[str, pd.DataFrame], dataset_name: str) -> pd.DataFrame:
    """Build detailed performance table for an individual dataset."""
    methods = ["NC", "Temp", "Platt 1D", "Platt 5D", "VCPS Platt 5D"]
    rows = []

    for arch, arch_label in [("m3", "M3"), ("mqt", "MQT")]:
        df = dfs[arch]
        df_ds = pd.DataFrame(df[df["dataset"] == dataset_name])
        for m in methods:
            m_row = pd.DataFrame(df_ds[df_ds["method_display"] == m])
            if len(m_row) > 0:
                rec = m_row.to_dict(orient="records")[0]
                rows.append(
                    {
                        "Model": arch_label,
                        "Method": m,
                        "ECE (%)": round(float(rec["ece_percent"]), 4),
                        "Adaptive ECE (%)": round(float(rec["adaptive_ece_percent"]), 4),
                        "AUROC": round(float(rec["auroc"]), 4),
                        "Brier Score": round(float(rec["brier"]), 4),
                    }
                )

    return pd.DataFrame(rows)


def write_macro_sheet_with_note(
    writer: pd.ExcelWriter,
    sheet_name: str,
    table_df: pd.DataFrame,
    qualifying_ds: list[str],
    candidate_name: str,
) -> None:
    """Write table and explanatory footnote to an Excel sheet."""
    table_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    note_text = (
        f"Note: Macro-Mean calculated across {len(qualifying_ds)} datasets where "
        f"{candidate_name} is in top-2 (Ada-ECE) on both M3 and MQT: "
        f"{', '.join(qualifying_ds)}."
    )
    note_df = pd.DataFrame([{"Note": note_text}])
    note_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(table_df) + 2)


def write_dataset_sheet_with_status(
    writer: pd.ExcelWriter,
    sheet_name: str,
    table_df: pd.DataFrame,
    p5_qual: bool,
    vcps_qual: bool,
) -> None:
    """Write dataset results table with qualification notes."""
    table_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    notes = [
        f"Platt 5D Qualified (Top-2 Dual Arch): {'Yes' if p5_qual else 'No'}",
        f"VCPS Platt 5D Qualified (Top-2 Dual Arch): {'Yes' if vcps_qual else 'No'}",
    ]
    status_df = pd.DataFrame({"Qualification Status": notes})
    status_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(table_df) + 2)


def main() -> None:
    """Main execution entry point."""
    print("Loading benchmark summaries...")
    dfs = load_and_preprocess_summaries()

    platt_5d_qual = find_qualifying_datasets(dfs, "Platt 5D")
    vcps_5d_qual = find_qualifying_datasets(dfs, "VCPS Platt 5D")

    print(f"Platt 5D Qualifying Datasets ({len(platt_5d_qual)}): {platt_5d_qual}")
    print(f"VCPS Platt 5D Qualifying Datasets ({len(vcps_5d_qual)}): {vcps_5d_qual}")

    macro_p5_df = build_macro_table(dfs, "Platt 5D", platt_5d_qual)
    macro_vcps_df = build_macro_table(dfs, "VCPS Platt 5D", vcps_5d_qual)

    print(f"Writing Excel workbook to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # Sheet 1: Platt 5D Macro
        write_macro_sheet_with_note(
            writer, "Platt_5D_Macro", macro_p5_df, platt_5d_qual, "Platt 5D"
        )
        # Sheet 2: VCPS Platt 5D Macro
        write_macro_sheet_with_note(
            writer, "VCPS_5D_Macro", macro_vcps_df, vcps_5d_qual, "VCPS Platt 5D"
        )
        # Individual dataset sheets
        for ds in ALL_DATASETS:
            ds_df = build_dataset_table(dfs, ds)
            write_dataset_sheet_with_status(
                writer,
                ds,
                ds_df,
                ds in platt_5d_qual,
                ds in vcps_5d_qual,
            )

    print(f"Successfully created {OUTPUT_FILE} with 16 sheets.")


if __name__ == "__main__":
    main()
