"""
Architecture-Aware LLaVA Import Router and Namespace Isolation.

Guarantees clean isolation between LLaVA-M3 (2D spatial pooling) and
MQT-LLaVA (Query Transformer with query_abstractor.bin) by fully purging
cached `llava.*` modules from `sys.modules` before every import.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("trajectory_calibration.vlm.llava_compat")


def load_llava_modules(
    arch: str = "m3", custom_search_paths: list[Path] | None = None
) -> dict[str, Any]:
    """
    Purges cached `llava.*` modules and dynamically loads the correct architecture fork.

    Args:
        arch: "m3" or "mqt".
        custom_search_paths: Optional custom directory paths to search first.

    Returns:
        Dictionary containing loaded LLaVA functions and classes.
    """
    arch = arch.lower()
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # 1. Full purge of all cached llava modules from sys.modules
    for mod_name in list(sys.modules.keys()):
        if mod_name == "llava" or mod_name.startswith("llava."):
            del sys.modules[mod_name]

    # 2. Remove any old llava paths from sys.path
    for p_str in list(sys.path):
        if "llava" in p_str.lower() or "matryoshka" in p_str.lower():
            sys.path.remove(p_str)

    # 3. Determine search paths for the target architecture
    search_paths: list[Path] = []
    if custom_search_paths:
        search_paths.extend(custom_search_paths)

    if arch == "mqt":
        mqt_candidates = [
            Path("/workspace/MQT-LLaVA"),
            project_root / "MQT-LLaVA",
            Path("/workspace/trajectory_calibration/MQT-LLaVA"),
            Path("/workspace/mqt-llava"),
            project_root / "mqt-llava",
        ]
        mqt_dir = (
            Path("/workspace/MQT-LLaVA")
            if Path("/workspace").exists()
            else (project_root / "MQT-LLaVA")
        )
        found = any(c.exists() for c in mqt_candidates)
        if not found:
            import subprocess

            logger.info(
                f"Cloning official MQT-LLaVA repository (gordonhu608/MQT-LLaVA) to {mqt_dir}..."
            )
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/gordonhu608/MQT-LLaVA.git",
                        str(mqt_dir),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"MQT-LLaVA repository cloned successfully to {mqt_dir}.")
            except Exception as e:
                logger.warning(f"Could not auto-clone MQT-LLaVA: {e}")

        mqt_env = os.environ.get("MQT_LLAVA_PATH")
        if mqt_env:
            search_paths.append(Path(mqt_env))
        search_paths.extend(mqt_candidates)
    else:
        m3_candidates = [
            Path("/workspace/trajectory_calibration/llava_src"),
            project_root / "llava_src",
            Path("/workspace/llava_src"),
            Path("/workspace/matryoshka-mm"),
            project_root / "src",
        ]
        m3_env = os.environ.get("M3_LLAVA_PATH")
        if m3_env:
            search_paths.append(Path(m3_env))
        search_paths.extend(m3_candidates)

    resolved_path = None
    for p in reversed(search_paths):
        if p.exists():
            p_str = str(p.resolve())
            if p_str in sys.path:
                sys.path.remove(p_str)
            sys.path.insert(0, p_str)
            resolved_path = p

    try:
        import llava  # pyright: ignore[reportMissingImports]
        from llava.constants import (  # pyright: ignore[reportMissingImports]
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_PLACEHOLDER,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates  # pyright: ignore[reportMissingImports]
        from llava.mm_utils import (  # pyright: ignore[reportMissingImports]
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import (  # pyright: ignore[reportMissingImports]
            load_pretrained_model,
        )
        from llava.utils import disable_torch_init  # pyright: ignore[reportMissingImports]

        loaded_from = getattr(llava, "__file__", "unknown")
        logger.info(f"LLaVA ({arch.upper()}) loaded successfully from: {loaded_from}")

        return {
            "load_pretrained_model": load_pretrained_model,
            "disable_torch_init": disable_torch_init,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
            "get_model_name_from_path": get_model_name_from_path,
            "conv_templates": conv_templates,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
            "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
            "IMAGE_PLACEHOLDER": IMAGE_PLACEHOLDER,
            "resolved_path": resolved_path,
            "loaded_file": loaded_from,
        }
    except (ModuleNotFoundError, ImportError) as e:
        logger.debug(f"LLaVA modules not found for {arch}: {e}. Returning safe fallback.")
        return {
            "load_pretrained_model": None,
            "disable_torch_init": lambda: None,
            "process_images": None,
            "tokenizer_image_token": None,
            "get_model_name_from_path": lambda p: Path(p).name,
            "conv_templates": {"llava_v1": None},
            "IMAGE_TOKEN_INDEX": -200,
            "DEFAULT_IMAGE_TOKEN": "<image>",
            "DEFAULT_IM_START_TOKEN": "<img>",
            "DEFAULT_IM_END_TOKEN": "</img>",
            "IMAGE_PLACEHOLDER": "<image-placeholder>",
            "resolved_path": None,
            "loaded_file": None,
        }
