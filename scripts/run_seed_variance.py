#!/usr/bin/env python3
"""
Seed Variance & Cross-Validation Benchmark Runner.
Alias / entrypoint for run_benchmark_cv.py.
"""
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

SCRIPTS_PATH = Path(__file__).resolve().parent
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from run_benchmark_cv import main

if __name__ == "__main__":
    main()
