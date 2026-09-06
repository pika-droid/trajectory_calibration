"""
Unified VLM Wrapper for M3-LLaVA and MQT-LLaVA Inference.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any
from PIL import Image
import torch
import transformers

from trajectory_calibration.utils.config import ARCH_SCALES
from trajectory_calibration.vlm.llava_compat import load_llava_modules
from trajectory_calibration.vlm.patches import apply_transformers_compatibility_patches

logger = logging.getLogger("trajectory_calibration.vlm.wrapper")


class UnifiedVLMWrapper:
    """Unified inference wrapper supporting both M3-LLaVA and MQT-LLaVA architectures."""

    def __init__(
        self,
        model_path: str = "mucai/llava-v1.5-7b-m3",
        precision: str = "fp16",
        arch: str | None = None,
    ) -> None:
        self.arch = arch.lower() if arch else ("mqt" if "mqt" in model_path.lower() else "m3")
        self.model_path = model_path
        self.scales = ARCH_SCALES[self.arch]
        self.fine_scale = self.scales[-1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

        logger.info(f"Initializing UnifiedVLMWrapper for {self.arch.upper()} on {self.device} ({precision}).")

        self._llava = load_llava_modules(self.arch)
        load_pretrained_model = self._llava["load_pretrained_model"]
        disable_torch_init = self._llava["disable_torch_init"]
        get_model_name_from_path = self._llava["get_model_name_from_path"]

        if load_pretrained_model is None:
            raise ImportError(f"LLaVA library could not be loaded for architecture: {self.arch}")

        disable_torch_init()
        actual_path = model_path if os.path.exists(model_path) else ("gordonhu/MQT-LLaVA-7b" if self.arch == "mqt" else "mucai/llava-v1.5-7b-m3")
        self.model_name = get_model_name_from_path(actual_path)

        old_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=actual_path,
                model_base=None,
                model_name=self.model_name,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                torch_dtype=self.dtype,
            )
        transformers.logging.set_verbosity(old_verbosity)

        if torch.cuda.is_available():
            model = model.to(device=self.device, dtype=self.dtype)
            if hasattr(model, "get_vision_tower") and model.get_vision_tower() is not None:
                model.get_vision_tower().to(device=self.device, dtype=self.dtype)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        apply_transformers_compatibility_patches(self.model)
        self.conv_mode = self._resolve_conv_mode()

    def _resolve_conv_mode(self) -> str:
        name = self.model_name.lower()
        if "llama-2" in name:
            return "llava_llama_2"
        if "mistral" in name:
            return "mistral_instruct"
        if "v1.6" in name:
            return "chatml_direct"
        if any(k in name for k in ["v1", "mqt", "m3", "matryoshka"]):
            return "llava_v1"
        return "mpt" if "mpt" in name else "llava_v1"

    def preprocess_image(self, image: Image.Image) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        process_images = self._llava["process_images"]
        image_tensor = process_images([image], self.image_processor, self.model.config).to(self.device, dtype=self.dtype)
        return image_tensor, [image.size]

    def format_prompt(self, question: str) -> str:
        if getattr(self.model.config, "mm_use_im_start_end", False):
            img_tok = (
                self._llava["DEFAULT_IM_START_TOKEN"]
                + self._llava["DEFAULT_IMAGE_TOKEN"]
                + self._llava["DEFAULT_IM_END_TOKEN"]
            )
        else:
            img_tok = self._llava["DEFAULT_IMAGE_TOKEN"]

        if self._llava["IMAGE_PLACEHOLDER"] in question:
            qs = re.sub(self._llava["IMAGE_PLACEHOLDER"], img_tok, question)
        else:
            qs = f"{img_tok}\n{question}"
        qs += "\nAnswer the question using a single word or phrase."
        conv = self._llava["conv_templates"][self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.inference_mode()
    def forward_fast(
        self,
        image: Image.Image,
        question: str,
        num_visual_tokens: int = 256,
        gen_temperature: float = 0.0,
        max_new_tokens: int = 16,
    ) -> dict[str, Any]:
        """Fast forward pass with output_attentions=False to enable FlashAttention."""
        prompt = self.format_prompt(question)
        input_ids = self._llava["tokenizer_image_token"](prompt, self.tokenizer, self._llava["IMAGE_TOKEN_INDEX"], return_tensors="pt").unsqueeze(0).to(self.device)
        image_tensor, image_sizes = self.preprocess_image(image)

        vt = None if (num_visual_tokens is not None and num_visual_tokens >= 576) else num_visual_tokens
        if hasattr(self.model, "config"):
            setattr(self.model.config, "num_visual_tokens", vt)
        if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
            setattr(self.model.model.config, "num_visual_tokens", vt)

        gen_kwargs: dict[str, Any] = {
            "inputs": input_ids,
            "images": image_tensor,
            "image_sizes": image_sizes,
            "do_sample": gen_temperature > 0.0,
            "temperature": gen_temperature if gen_temperature > 0.0 else None,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if vt is not None:
            gen_kwargs["matryoshka_vis_token_scale"] = vt

        autocast_device = self.device.type if hasattr(self.device, "type") else "cuda"
        with torch.amp.autocast(autocast_device, dtype=self.dtype):
            outputs = self.model.generate(**gen_kwargs)

        gen_sequence = outputs.sequences[0]
        num_gen = len(outputs.scores) if (outputs.scores is not None and len(outputs.scores) > 0) else (len(gen_sequence) - input_ids.shape[1])
        pred_answer = self.tokenizer.decode(gen_sequence[-num_gen:].tolist(), skip_special_tokens=True).strip()

        first_token_logits = torch.nan_to_num(outputs.scores[0][0].detach().float(), nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(first_token_logits, dim=-1)
        top2_vals, _ = torch.topk(probs, 2, dim=-1)

        return {
            "answer": pred_answer,
            "conf_softmax": float(top2_vals[0].item()),
            "margin": float((top2_vals[0] - top2_vals[1]).item()),
        }

    def sweep(
        self, image: Image.Image, question: str, scales: list[int] | None = None, gen_temperature: float = 0.0
    ) -> dict[int, dict[str, Any]]:
        """Sweeps generation across configured visual scales."""
        target_scales = scales or self.scales
        results = {}
        for m in target_scales:
            if m > self.fine_scale:
                raise ValueError(f"Scale m={m} exceeds maximum of {self.fine_scale}.")
            results[m] = self.forward_fast(image=image, question=question, num_visual_tokens=m, gen_temperature=gen_temperature)
        return results
