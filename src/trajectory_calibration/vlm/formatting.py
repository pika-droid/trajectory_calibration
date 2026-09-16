"""
Question prompt formatting and image extraction helpers.
"""

from __future__ import annotations

import ast
import io
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
    """Formats question text and options consistently across benchmarks."""
    question = sample.get(
        "question",
        sample.get(
            "problem",
            sample.get(
                "query",
                sample.get(
                    "text",
                    sample.get(
                        "prompt",
                        sample.get(
                            "user_query",
                            sample.get(
                                "instruction",
                                sample.get("input", ""),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    if not question:
        conv = sample.get("conversations", sample.get("messages", []))
        if isinstance(conv, list) and len(conv) > 0:
            for turn in conv:
                if isinstance(turn, dict) and (
                    turn.get("from") in ["human", "user"] or turn.get("role") in ["human", "user"]
                ):
                    question = turn.get("value", turn.get("content", ""))
                    break

    if isinstance(question, dict):
        question = question.get(
            "text", question.get("question", question.get("prompt", str(question)))
        )
    elif isinstance(question, (list, tuple)) and len(question) > 0:
        question = question[0]
    question = str(question or "").replace("<image>", "").replace("<IMAGE>", "").strip()

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


def _to_pil_image(val: Any) -> Image.Image | None:
    """Converts a raw value (Image, Path/str, bytes, dict, base64, URL) to RGB PIL Image."""
    try:
        if val is None:
            return None
        if isinstance(val, Image.Image):
            return val.convert("RGB")
        if isinstance(val, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(val)).convert("RGB")
            except Exception:
                return None
        if isinstance(val, io.BytesIO):
            try:
                return Image.open(val).convert("RGB")
            except Exception:
                return None
        if isinstance(val, (str, Path)):
            val_str = str(val)
            if val_str.startswith(("http://", "https://")):
                try:
                    import urllib.request

                    req = urllib.request.Request(val_str, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        img_bytes = resp.read()
                    # Cache locally if in avqa directory
                    if "images.cocodataset.org" in val_str:
                        fname = val_str.split("/")[-1]
                        cache_dir = Path("data/raw_datasets/avqa/images")
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        cache_file = cache_dir / fname
                        if not cache_file.exists():
                            with open(cache_file, "wb") as f:
                                f.write(img_bytes)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    return None
            p = Path(val)
            if p.is_file():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    return None
            if isinstance(val, str) and (val.startswith("data:image") or len(val) > 100):
                try:
                    import base64

                    b64_str = val.split(",", 1)[1] if "," in val else val
                    img_bytes = base64.b64decode(b64_str)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    pass
        if isinstance(val, dict):
            if "bytes" in val and val["bytes"] is not None:
                try:
                    return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
                except Exception:
                    return None
            if val.get("path"):
                p = Path(val["path"])
                if p.is_file():
                    try:
                        return Image.open(p).convert("RGB")
                    except Exception:
                        return None
            if "image" in val and val["image"] is not None:
                return _to_pil_image(val["image"])
        if isinstance(val, (list, tuple)):
            for item in val:
                img = _to_pil_image(item)
                if img is not None:
                    return img
        return None
    except Exception:
        return None


def load_image_from_sample(sample: dict[str, Any]) -> Image.Image | None:
    """Extracts and normalizes PIL Image from heterogeneous sample keys."""
    for k in [
        "images",
        "image",
        "image_1",
        "img",
        "image_path",
        "img_path",
        "image_name",
        "image_url",
        "coco_url",
        "url",
        "picture",
        "file_name",
        "filename",
        "image_file",
        "image_bytes",
    ]:
        if k in sample and sample[k] is not None:
            img = _to_pil_image(sample[k])
            if img is not None:
                return img

    img_id = sample.get("image_id")
    if img_id is not None and (
        isinstance(img_id, int) or (isinstance(img_id, str) and img_id.isdigit())
    ):
        coco_url = f"http://images.cocodataset.org/val2017/{int(img_id):012d}.jpg"
        img = _to_pil_image(coco_url)
        if img is not None:
            return img

    return None


__all__ = [
    "_parse_options_list",
    "_to_pil_image",
    "format_question",
    "load_image_from_sample",
]
