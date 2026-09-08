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
    ALL_14_DATASETS,
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
    # Default: 1 smoke + 2 * (calibration steps) + 4 reporting
    full = build_full_pipeline(architectures=["m3", "mqt"], temperatures=[0.0, 1.0])
    assert len(full) >= 15
    assert full[0].name == "smoke_test"

    # Skip mock
    no_mock = build_full_pipeline(architectures=["m3", "mqt"], temperatures=[0.0], skip_mock=True)
    assert not any(s.name == "smoke_test" for s in no_mock)

    # Skip calibration (reporting only)
    reporting_only = build_full_pipeline(
        architectures=["m3", "mqt"], temperatures=[0.0], skip_calibration=True
    )
    assert len(reporting_only) == 4
    assert all(s.category == "reporting" for s in reporting_only)

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
    assert "update_readme_tables" in res.stdout


def test_format_analytical_bullets_with_trajectory_platt() -> None:
    methods = [
        "Temperature Scaling (TS)",
        "Platt Scaling (1D)",
        "VCPS-5D (Our Method)",
        "VCPS-17D (Our Method)",
        "Trajectory Platt (5D)",
        "Trajectory Platt (17D)",
        "Residual Calibrator",
    ]
    # Synthetic data where Trajectory Platt (17D) beats TS and 1D Platt on all datasets
    data = {}
    for m in methods:
        if "17D" in m:
            data[m] = [2.0] * len(ALL_14_DATASETS)
        elif "5D" in m:
            data[m] = [3.0] * len(ALL_14_DATASETS)
        elif "1D" in m:
            data[m] = [5.0] * len(ALL_14_DATASETS)
        elif "TS" in m:
            data[m] = [10.0] * len(ALL_14_DATASETS)
        else:
            data[m] = [4.0] * len(ALL_14_DATASETS)

    df = pd.DataFrame(data, index=ALL_14_DATASETS).T
    ranks = {ds: (2.0, 3.0) for ds in ALL_14_DATASETS}

    bullets = format_analytical_bullets(df, ranks)
    joined = "\n".join(bullets)

    assert "Trajectory Platt vs. Global Temperature Scaling (TS)" in joined
    assert "Trajectory Platt vs. 1D Platt Scaling" in joined
    assert "Trajectory Platt beats TS on **14 / 14 datasets**" in joined
    assert "Trajectory Platt beats 1D Platt Scaling on **14 / 14 datasets**" in joined
