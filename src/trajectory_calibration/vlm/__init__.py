"""
Vision-Language Model (VLM) and Multi-Dataset Evaluation Module.
"""

from trajectory_calibration.vlm.datasets import (
    ALL_DATASET_KEYS,
    DATASET_REGISTRY,
    evaluate_accuracy,
    format_question,
    load_hf_dataset,
    load_image_from_sample,
)
from trajectory_calibration.vlm.llava_compat import load_llava_modules
from trajectory_calibration.vlm.multipass import (
    extract_multipass_record,
    generate_mock_multipass_sample,
)
from trajectory_calibration.vlm.wrapper import UnifiedVLMWrapper

__all__ = [
    "ALL_DATASET_KEYS",
    "DATASET_REGISTRY",
    "UnifiedVLMWrapper",
    "evaluate_accuracy",
    "extract_multipass_record",
    "format_question",
    "generate_mock_multipass_sample",
    "load_hf_dataset",
    "load_image_from_sample",
    "load_llava_modules",
]
