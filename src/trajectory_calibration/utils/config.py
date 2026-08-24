"""
Central configuration dataclass and architectural constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ARCH_SCALES: dict[str, list[int]] = {
    "m3": [1, 9, 36, 144, 576],
    "mqt": [1, 9, 36, 144, 256],
}


@dataclass
class Config:
    """Centralized configuration for feature extraction and calibration benchmarks."""

    model_path: str = "mucai/llava-v1.5-7b-m3"
    arch: str = "m3"  # "m3" or "mqt", auto-resolved in __post_init__
    precision: str = "fp16"
    features_dir: Path = Path("data/features")
    output_dir: Path = Path("results/experiments")
    seed: int = 42
    max_new_tokens: int = 16
    subset_size: int | None = None
    token_scales: list[int] = field(default_factory=lambda: [1, 9, 36, 144, 576])
    fine_scale: int = 576

    def __post_init__(self) -> None:
        arch_norm = self.arch.lower()
        if "mqt" in self.model_path.lower() or "mqt" in arch_norm:
            self.arch = "mqt"
            self.token_scales = list(ARCH_SCALES["mqt"])
        else:
            self.arch = "m3"
            self.token_scales = list(ARCH_SCALES["m3"])
        self.fine_scale = self.token_scales[-1]
        self.features_dir = Path(self.features_dir)
        self.output_dir = Path(self.output_dir)
