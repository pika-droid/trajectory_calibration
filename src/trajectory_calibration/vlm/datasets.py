"""
Vision-Language Datasets and Evaluators Facade Module.

Re-exports the 14-benchmark dataset registry, formatters, PIL image loaders,
and ground-truth accuracy evaluators.
"""

from trajectory_calibration.vlm.evaluators import evaluate_accuracy
from trajectory_calibration.vlm.formatting import format_question, load_image_from_sample
from trajectory_calibration.vlm.registry import (
    ALL_DATASET_KEYS,
    DATASET_REGISTRY,
    load_hf_dataset,
)

__all__ = [
    "ALL_DATASET_KEYS",
    "DATASET_REGISTRY",
    "evaluate_accuracy",
    "format_question",
    "load_hf_dataset",
    "load_image_from_sample",
]
