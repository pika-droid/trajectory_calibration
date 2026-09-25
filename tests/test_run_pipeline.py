"""
Unit test suite for the unified pipeline runner and reporting extensions.
Covers:
1. Pipeline step composition across architectures, temperatures, and skip flags.
2. Dry-run execution verifying command lines without launching long-running processes.
3. Analytical bullet formatting with Trajectory Platt 5D/17D win counts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.run_pipeline import (
    build_calibration_steps,
    build_full_pipeline,
)
from scripts.update_readme_tables import (
    CORE_7_DATASETS,
    format_analytical_bullets,
)


def test_build_calibration_steps() -> None:
    py_exec = sys.executable
    steps = build_calibration_steps(py_exec, arch="m3", temperatures=[0.0, 0.5, 1.0])
    assert len(steps) >= 5
    step_names = [s.name for s in steps]
    assert "benchmark_m3" in step_names
    assert "benchmark_cv_m3" in step_names
    assert "temperature_study_m3" in step_names
    assert "lodo_m3" in step_names
    assert "ablation_m3" in step_names


def test_build_full_pipeline_combinations() -> None:
    # Default: 1 smoke + 2 * (calibration steps) + 5 reporting
    full = build_full_pipeline(architectures=["m3", "mqt"], temperatures=[0.0, 1.0])
    assert len(full) >= 15
    assert full[0].name == "smoke_test"

    # Skip mock
    no_mock = build_full_pipeline(architectures=["m3", "mqt"], temperatures=[0.0], skip_mock=True)
    assert not any(s.name == "smoke_test" for s in no_mock)

    # Skip calibration (reporting only, with ablation reporting and compendium compilation by default)
    reporting_only = build_full_pipeline(
        architectures=["m3", "mqt"], temperatures=[0.0], skip_calibration=True
    )
    assert len(reporting_only) == 10
    assert all(s.category == "reporting" for s in reporting_only)

    # Skip calibration and ablation reporting
    reporting_no_ablation = build_full_pipeline(
        architectures=["m3", "mqt"],
        temperatures=[0.0],
        skip_calibration=True,
        skip_ablation=True,
    )
    assert len(reporting_no_ablation) == 8
    assert all(s.category == "reporting" for s in reporting_no_ablation)

    # Skip calibration, ablation, and compendium compilation
    reporting_minimal = build_full_pipeline(
        architectures=["m3", "mqt"],
        temperatures=[0.0],
        skip_calibration=True,
        skip_ablation=True,
        skip_compendium=True,
    )
    assert len(reporting_minimal) == 7
    assert all(s.category == "reporting" for s in reporting_minimal)

    # Skip tables (calibration only)
    calibration_only = build_full_pipeline(
        architectures=["m3"], temperatures=[0.0], skip_tables=True, skip_mock=True
    )
    assert all(s.category == "calibration" for s in calibration_only)


def test_pipeline_dry_run() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    res = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--dry-run"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "DRY RUN MODE" in res.stdout
    assert "benchmark_m3" in res.stdout
    assert "benchmark_mqt" in res.stdout
    assert "sample_efficiency_study" in res.stdout
    assert "token_budget_study" in res.stdout
    assert "generate_paper_table2" in res.stdout
    assert "mine_pairwise_anchors" in res.stdout
    assert "generate_token_budget_markdown_log" in res.stdout
    assert "compile_compendium" in res.stdout
    assert "update_readme_tables" in res.stdout
    assert "ablation_5d_m3" in res.stdout
    assert "plot_ablation_5d" in res.stdout
    assert "generate_ablation_latex_tables" in res.stdout


def test_format_analytical_bullets_with_trajectory_platt() -> None:
    # Synthetic data where Trajectory Platt (5D) beats TS and 1D Platt on all Core 7 datasets
    data = {
        "Trajectory Platt (5D)": [3.0] * len(CORE_7_DATASETS),
        "Platt Scaling (1D)": [5.0] * len(CORE_7_DATASETS),
        "Temperature Scaling (TS)": [10.0] * len(CORE_7_DATASETS),
    }

    df = pd.DataFrame(data, index=CORE_7_DATASETS).T
    ranks = {ds: (3.0, 5.0) for ds in CORE_7_DATASETS}

    bullets = format_analytical_bullets(df, ranks)
    joined = "\n".join(bullets)

    assert "Trajectory Platt (5D) vs. Global Temperature Scaling (TS)" in joined
    assert "Trajectory Platt (5D) vs. 1D Platt Scaling" in joined
    assert "Trajectory Platt (5D) beats TS on **7 / 7 Core datasets**" in joined
    assert "Trajectory Platt (5D) beats 1D Platt Scaling on **7 / 7 Core datasets**" in joined
