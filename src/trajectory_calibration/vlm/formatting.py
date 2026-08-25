"""
Question prompt formatting and image extraction helpers.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from PIL import Image


def _parse_options_list(options_raw: Any) -> list[str]:
    """Safely parse multiple-choice options from lists, Python strings, or JSON."""
    if isinstance(options_raw, list):
        return [str(opt) for opt in options_raw]
    if isinstance(options_raw, str):
        options_raw = options_raw.strip()
        if not options_raw:
            return []
        try:
            parsed = ast.literal_eval(options_raw)
            if isinstance(parsed, list):
                return [str(opt) for opt in parsed]
        except Exception:
            pass
        try:
            parsed = json.loads(options_raw)
            if isinstance(parsed, list):
                return [str(opt) for opt in parsed]
        except Exception:
            pass
    return []


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
        options = _parse_options_list(sample.get("options", []))
        if options:
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
        options = _parse_options_list(sample.get("options", []))
        if options:
            opts_str = "\n".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            question = f"{question}\n{opts_str}\nAnswer with the option letter."

    elif dataset_key == "scienceqa":
        choices = _parse_options_list(sample.get("choices", sample.get("options", [])))
        if choices:
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
    elif isinstance(img_raw, (str, Path)):
        p = Path(img_raw)
        if p.exists() and p.is_file():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                return None
    elif isinstance(img_raw, (list, tuple)) and len(img_raw) > 0:
        first = img_raw[0]
        if isinstance(first, Image.Image):
            return first.convert("RGB")
        elif isinstance(first, (str, Path)):
            p = Path(first)
            if p.exists() and p.is_file():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    return None

    return None
