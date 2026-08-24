"""
Question prompt formatting and image extraction helpers.
"""

from __future__ import annotations

import json
from typing import Any
from PIL import Image


def format_question(sample: dict[str, Any], dataset_key: str) -> str:
    """Formats question text and options consistently for multiple-choice tasks."""
    question = sample.get(
        "question",
        sample.get(
            "problem",
            sample.get("query", sample.get("text", sample.get("prompt", sample.get("user_query", "")))),
        ),
    )
    if isinstance(question, (list, tuple)) and len(question) > 0:
        question = question[0]
    question = str(question).strip()

    if dataset_key == "ai2d":
        options = sample.get("options", [])
        if options and isinstance(options, list):
            opts_str = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            question = f"{question}\n{opts_str}\nAnswer with the option letter."

    elif dataset_key == "mmbench":
        opts = []
        for ltr in ["A", "B", "C", "D"]:
            val = sample.get(ltr)
            if val and not (isinstance(val, float) and str(val) == "nan"):
                opts.append(f"({ltr}) {val}")
        if opts:
            question = f"{question}\n" + "\n".join(opts) + "\nAnswer with the option letter."

    elif dataset_key == "mmmu":
        options = sample.get("options", [])
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = []
        if isinstance(options, list) and len(options) > 0:
            opts_str = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            question = f"{question}\n{opts_str}\nAnswer with the option letter."

    elif dataset_key == "scienceqa":
        choices = sample.get("choices", sample.get("options", []))
        if choices and isinstance(choices, list):
            opts_str = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(choices)])
            question = f"{question}\n{opts_str}\nAnswer with the option letter."

    elif dataset_key == "seedbench":
        opts = []
        for i, col in enumerate(["choice_a", "choice_b", "choice_c", "choice_d"]):
            val = sample.get(col)
            if val and isinstance(val, str) and val.strip():
                opts.append(f"({chr(65 + i)}) {val.strip()}")
        if opts:
            question = f"{question}\n" + "\n".join(opts) + "\nAnswer with the option letter."

    return question


def load_image_from_sample(sample: dict[str, Any]) -> Image.Image | None:
    """Extracts and normalizes PIL Image from heterogeneous sample keys."""
    img_raw = None
    for k in ["images", "image", "image_1", "img", "image_path", "img_path", "picture"]:
        if k in sample and sample[k] is not None:
            img_raw = sample[k]
            break

    if img_raw is None:
        return None

    if isinstance(img_raw, Image.Image):
        return img_raw.convert("RGB")
    elif isinstance(img_raw, (list, tuple)) and len(img_raw) > 0:
        first = img_raw[0]
        if isinstance(first, Image.Image):
            return first.convert("RGB")

    return None
