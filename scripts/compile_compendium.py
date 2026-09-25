#!/usr/bin/env python3
"""
Publication LaTeX Compendium Compiler.

Compiles dataset_tables/all_tables_compendium.tex into all_tables_compendium.pdf
using pdflatex if available, and cleans up temporary auxiliary build artifacts.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT_DIR / "dataset_tables"
TEX_PATH = TABLES_DIR / "all_tables_compendium.tex"
PDF_PATH = TABLES_DIR / "all_tables_compendium.pdf"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compile_compendium")

AUX_EXTENSIONS = [".aux", ".log", ".out", ".toc"]


def cleanup_aux_files() -> None:
    """Remove temporary LaTeX compilation artifacts to keep working tree clean."""
    stem = TEX_PATH.stem
    for ext in AUX_EXTENSIONS:
        aux_file = TABLES_DIR / f"{stem}{ext}"
        if aux_file.exists():
            try:
                aux_file.unlink()
            except OSError as err:
                logger.warning(f"Failed to remove {aux_file}: {err}")


def compile_compendium() -> int:
    """Compile all_tables_compendium.tex into all_tables_compendium.pdf."""
    if not TEX_PATH.exists():
        logger.error(f"Compendium TeX source not found: {TEX_PATH}")
        return 1

    pdflatex_bin = shutil.which("pdflatex")
    if not pdflatex_bin:
        logger.warning("pdflatex is not installed or not in PATH. Skipping PDF compilation.")
        return 0

    cmd = [
        pdflatex_bin,
        "-interaction=nonstopmode",
        f"-output-directory={TABLES_DIR}",
        str(TEX_PATH),
    ]

    logger.info(f"Compiling LaTeX compendium via pdflatex: {TEX_PATH.name}")

    # Pass 1
    res1 = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    # Pass 2 for resolving references
    subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)

    cleanup_aux_files()

    if PDF_PATH.exists() and os.path.getsize(PDF_PATH) > 0:
        logger.info(
            f"Successfully generated compendium PDF: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)"
        )
        return 0

    logger.error(f"Failed to generate compendium PDF. Output:\n{res1.stdout}\n{res1.stderr}")
    return 1


def main() -> None:
    sys.exit(compile_compendium())


if __name__ == "__main__":
    main()
