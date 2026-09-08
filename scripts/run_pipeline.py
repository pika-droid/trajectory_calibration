#!/usr/bin/env python3
"""
Unified Calibration Benchmark and Documentation Pipeline Runner.

Orchestrates all Section 2 calibration benchmarks and Section 3 table,
markdown, and figure generation scripts across architectures (M3-LLaVA,
MQT-LLaVA) and decoding temperatures.

Features:
- Real-time unbuffered terminal stdout/stderr streaming with flush=True.
- Structured execution stages with timing metrics.
- Flags for selective phase execution (--skip-mock, --skip-calibration, --skip-tables).
- --dry-run flag for command sequence inspection without execution.
- Programmatic synchronization of README.md benchmark tables and win counts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"


@dataclass
class PipelineStep:
    name: str
    description: str
    command: list[str]
    category: str  # 'smoke', 'calibration', or 'reporting'


def run_command_stream(cmd: Sequence[str], description: str) -> int:
    """Run a subprocess command with real-time flushed stdout/stderr streaming."""
    cmd_str = " ".join(cmd)
    print(f"\n{'=' * 80}")
    print(f" [RUNNING] {description}")
    print(f" Command : {cmd_str}")
    print(f" Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n", flush=True)

    start_time = time.perf_counter()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        if process.stdout is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

        retcode = process.wait()
    except KeyboardInterrupt:
        print(f"\n[ABORT] Step interrupted by user: {description}", flush=True)
        return 130
    except Exception as exc:
        print(f"\n[ERROR] Failed to execute step '{description}': {exc}", flush=True)
        return 1

    elapsed = time.perf_counter() - start_time
    status = "SUCCESS" if retcode == 0 else f"FAILED (code {retcode})"
    print(f"\n{'-' * 80}")
    print(f" [{status}] {description} (took {elapsed:.2f}s)")
    print(f"{'-' * 80}\n", flush=True)

    return retcode


def build_calibration_steps(
    py_exec: str,
    arch: str,
    temperatures: list[float],
) -> list[PipelineStep]:
    """Assemble Section 2 calibration benchmark steps for a given architecture."""
    steps: list[PipelineStep] = []
    arch_upper = arch.upper()

    # 1. Comprehensive Benchmark (all 14 datasets at T=0.0)
    steps.append(
        PipelineStep(
            name=f"benchmark_{arch}",
            description=f"{arch_upper} Comprehensive 14-Dataset Calibration Benchmark (T=0.0)",
            command=[
                py_exec,
                str(SCRIPTS_DIR / "run_benchmark.py"),
                "--arch",
                arch,
                "--gen_temperature",
                "0.0",
            ],
            category="calibration",
        )
    )

    # 2. Multi-Seed Cross-Validation
    steps.append(
        PipelineStep(
            name=f"benchmark_cv_{arch}",
            description=f"{arch_upper} Multi-Seed Cross-Validation Benchmark (10 Seeds)",
            command=[
                py_exec,
                str(SCRIPTS_DIR / "run_benchmark_cv.py"),
                "--arch",
                arch,
            ],
            category="calibration",
        )
    )

    # 3. Decoding Temperature Robustness Study
    temp_strs = [str(t) for t in temperatures]
    steps.append(
        PipelineStep(
            name=f"temperature_study_{arch}",
            description=f"{arch_upper} Decoding Temperature Study (T in {temp_strs})",
            command=[
                py_exec,
                str(SCRIPTS_DIR / "run_temperature_study.py"),
                "--arch",
                arch,
                "--temperatures",
                *temp_strs,
            ],
            category="calibration",
        )
    )

    # 4. Leave-One-Dataset-Out (LODO) Transfer
    steps.append(
        PipelineStep(
            name=f"lodo_{arch}",
            description=f"{arch_upper} Leave-One-Dataset-Out (LODO) Zero-Shot Cross-Domain Transfer",
            command=[
                py_exec,
                str(SCRIPTS_DIR / "run_lodo.py"),
                "--arch",
                arch,
            ],
            category="calibration",
        )
    )

    # 5. Trajectory Feature Sensitivity & Ablation
    steps.append(
        PipelineStep(
            name=f"ablation_{arch}",
            description=f"{arch_upper} Trajectory Feature Sensitivity & Ablation Study",
            command=[
                py_exec,
                str(SCRIPTS_DIR / "run_ablation.py"),
                "--arch",
                arch,
            ],
            category="calibration",
        )
    )

    # 6. Focused VCPS Analysis (5D & 17D)
    anchor_pt = ROOT_DIR / "data" / "features" / f"{arch}_llava" / "temp_0.0" / "pope.pt"
    if anchor_pt.exists():
        steps.append(
            PipelineStep(
                name=f"vcps_5d_{arch}",
                description=f"{arch_upper} Focused VCPS-5D Dynamic Slope Analysis (POPE)",
                command=[
                    py_exec,
                    str(SCRIPTS_DIR / "run_vcps.py"),
                    "--arch",
                    arch,
                    "--features_dir",
                    str(anchor_pt),
                    "--feature_set",
                    "5d",
                ],
                category="calibration",
            )
        )
        steps.append(
            PipelineStep(
                name=f"vcps_17d_{arch}",
                description=f"{arch_upper} Focused VCPS-17D Dynamic Slope Analysis (POPE)",
                command=[
                    py_exec,
                    str(SCRIPTS_DIR / "run_vcps.py"),
                    "--arch",
                    arch,
                    "--features_dir",
                    str(anchor_pt),
                    "--feature_set",
                    "17d",
                ],
                category="calibration",
            )
        )

    return steps


def build_reporting_steps(py_exec: str) -> list[PipelineStep]:
    """Assemble Section 3 table, log, figure, and README generation steps."""
    return [
        PipelineStep(
            name="generate_markdown_logs",
            description="Generate UMPIRE Baselines and VCPS vs Baselines Markdown Logs",
            command=[py_exec, str(SCRIPTS_DIR / "generate_markdown_logs.py")],
            category="reporting",
        ),
        PipelineStep(
            name="generate_dataset_latex_tables",
            description="Generate Publication-Ready LaTeX Dataset Tables & Temperature Tables",
            command=[py_exec, str(SCRIPTS_DIR / "generate_dataset_latex_tables.py")],
            category="reporting",
        ),
        PipelineStep(
            name="plot_benchmark_comparison",
            description="Generate Benchmark Figures, Calibration Curves & Pareto Frontiers",
            command=[py_exec, str(SCRIPTS_DIR / "plot_benchmark_comparison.py")],
            category="reporting",
        ),
        PipelineStep(
            name="update_readme_tables",
            description="Programmatically Update README.md Benchmark Tables & Win Counts",
            command=[py_exec, str(SCRIPTS_DIR / "update_readme_tables.py")],
            category="reporting",
        ),
    ]


def build_full_pipeline(
    architectures: list[str],
    temperatures: list[float],
    skip_mock: bool = False,
    skip_calibration: bool = False,
    skip_tables: bool = False,
) -> list[PipelineStep]:
    """Construct the complete sequence of pipeline steps according to configuration flags."""
    py_exec = sys.executable
    steps: list[PipelineStep] = []

    # Stage 0: Mock Smoke Test
    if not skip_mock and not skip_calibration:
        steps.append(
            PipelineStep(
                name="smoke_test",
                description="Fast CPU Smoke Test across All Single-Pass Calibrators",
                command=[py_exec, str(SCRIPTS_DIR / "run_mock.py")],
                category="smoke",
            )
        )

    # Stage 1-6: Section 2 Calibration
    if not skip_calibration:
        for arch in architectures:
            steps.extend(build_calibration_steps(py_exec, arch, temperatures))

    # Stage 7-10: Section 3 Reporting, Figures & README Synchronization
    if not skip_tables:
        steps.extend(build_reporting_steps(py_exec))

    return steps


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the unified pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run all calibration benchmarks and report generators in proper sequence."
    )
    parser.add_argument(
        "--architectures",
        "--arch",
        nargs="+",
        default=["m3", "mqt"],
        choices=["m3", "mqt"],
        help="Architectures to evaluate (default: m3 mqt).",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.6, 1.0, 1.5],
        help="Decoding temperatures to evaluate in temperature study (default: 0.0 0.3 0.6 1.0 1.5).",
    )
    parser.add_argument(
        "--skip-mock",
        action="store_true",
        help="Skip the initial fast CPU smoke test.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip Section 2 calibration benchmarks and execute only Section 3 report generators.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip Section 3 table, log, and figure generation; run only Section 2 benchmarks.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining steps even if an intermediate step fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned execution sequence and command lines without executing.",
    )
    return parser.parse_args()


def execute_pipeline(steps: list[PipelineStep], continue_on_error: bool) -> int:
    """Execute pipeline steps sequentially with real-time output and timing metrics."""
    total_steps = len(steps)
    print(f"\n{'#' * 80}")
    print(f" STARTING TRAJECTORY CALIBRATION PIPELINE ({total_steps} total steps)")
    print(f" Working Directory: {ROOT_DIR}")
    print(f" Started At       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n", flush=True)

    suite_start = time.perf_counter()
    failed_steps: list[tuple[str, int]] = []

    for idx, step in enumerate(steps, start=1):
        banner = f"STAGE {idx}/{total_steps} [{step.category.upper()}]: {step.description}"
        retcode = run_command_stream(step.command, banner)
        if retcode != 0:
            failed_steps.append((step.name, retcode))
            if not continue_on_error:
                print(
                    f"\n[FATAL] Pipeline halted due to failure in '{step.name}' (exit code {retcode}).",
                    flush=True,
                )
                return retcode

    total_elapsed = time.perf_counter() - suite_start
    print(f"\n{'#' * 80}")
    print(" PIPELINE EXECUTION SUMMARY")
    print(f" Total Elapsed Time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print(f" Total Steps       : {total_steps}")
    print(f" Passed Steps      : {total_steps - len(failed_steps)}")
    print(f" Failed Steps      : {len(failed_steps)}")
    if failed_steps:
        print("\n Failed Steps Detail:")
        for name, code in failed_steps:
            print(f"  - {name}: exit code {code}")
        print(f"{'#' * 80}\n", flush=True)
        return 1

    print("\n [SUCCESS] All pipeline steps completed successfully!")
    print(f"{'#' * 80}\n", flush=True)
    return 0


def main() -> None:
    args = parse_args()
    steps = build_full_pipeline(
        architectures=args.architectures,
        temperatures=args.temperatures,
        skip_mock=args.skip_mock,
        skip_calibration=args.skip_calibration,
        skip_tables=args.skip_tables,
    )

    if args.dry_run:
        print(f"\n{'=' * 80}")
        print(f" DRY RUN MODE: {len(steps)} steps planned")
        print(f"{'=' * 80}")
        for idx, step in enumerate(steps, start=1):
            print(f"[{idx:02d}/{len(steps):02d}] ({step.category.upper()}) {step.name}")
            print(f"     Description : {step.description}")
            print(f"     Command     : {' '.join(step.command)}\n")
        return

    sys.exit(execute_pipeline(steps, continue_on_error=args.continue_on_error))


if __name__ == "__main__":
    main()
