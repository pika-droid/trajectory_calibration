"""
Unified VLM Wrapper for M3-LLaVA and MQT-LLaVA.

Provides fast multi-scale feature extraction with:
- Architecture detection and LLaVA namespace isolation
- Automatic scale resolution (576 for M3, 256 for MQT)
- Monkey-patches for transformers >= 4.38 compatibility
- FlashAttention-enabled fast inference (output_attentions=False)
- First-token logit NaN/Inf clamping
"""

from __future__ import annotations

import functools
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any
from PIL import Image

import torch

from trajectory_calibration.vlm.llava_compat import load_llava_modules

logger = logging.getLogger("trajectory_calibration.vlm.wrapper")

ARCH_SCALES: dict[str, list[int]] = {
    "m3": [1, 9, 36, 144, 576],
    "mqt": [1, 9, 36, 144, 256],
}


class UnifiedVLMWrapper:
    """
    Unified inference wrapper supporting both M3-LLaVA and MQT-LLaVA architectures.
    """

    def __init__(self, model_path: str = "mucai/llava-v1.5-7b-m3", precision: str = "fp16") -> None:
        self.model_path = model_path
        self.arch = "mqt" if "mqt" in model_path.lower() else "m3"
        self.scales = ARCH_SCALES[self.arch]
        self.fine_scale = self.scales[-1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if precision == "bf16":
            self.dtype = torch.bfloat16
        elif precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info(f"Initializing UnifiedVLMWrapper for {self.arch.upper()} on {self.device} ({precision}).")

        # 1. Load correct LLaVA package with namespace isolation
        self._llava = load_llava_modules(self.arch)
        load_pretrained_model = self._llava["load_pretrained_model"]
        disable_torch_init = self._llava["disable_torch_init"]
        get_model_name_from_path = self._llava["get_model_name_from_path"]

        if load_pretrained_model is None:
            raise ImportError(f"LLaVA library could not be loaded for architecture: {self.arch}")

        disable_torch_init()

        # 2. Resolve model path / HF Hub fallback
        actual_path = model_path
        if not os.path.exists(actual_path):
            if self.arch == "mqt":
                actual_path = "gordonhu/MQT-LLaVA-7b"
            else:
                actual_path = "mucai/llava-v1.5-7b-m3"
            logger.info(f"Local path not found. Falling back to HF Hub: '{actual_path}'")

        self.model_name = get_model_name_from_path(actual_path)

        # 3. Load model weights
        import transformers

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
            if hasattr(model, "get_vision_tower"):
                vt = model.get_vision_tower()
                if vt is not None and hasattr(vt, "to"):
                    vt.to(device=self.device, dtype=self.dtype)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        # 4. Apply 3 protected monkey-patches for transformers >= 4.38 compatibility
        self._apply_compatibility_patches()

        # 5. Resolve hardened conversation template (llava_v1 for Vicuna-1.5)
        self.conv_mode = self._resolve_conv_mode()
        logger.info(f"Model loaded successfully. Conversation template: '{self.conv_mode}'")

    def _resolve_conv_mode(self) -> str:
        name = self.model_name.lower()
        if "llama-2" in name:
            return "llava_llama_2"
        elif "mistral" in name:
            return "mistral_instruct"
        elif "v1.6" in name:
            return "chatml_direct"
        elif any(k in name for k in ["v1", "mqt", "m3", "matryoshka"]):
            return "llava_v1"
        elif "mpt" in name:
            return "mpt"
        return "llava_v1"

    def _apply_compatibility_patches(self) -> None:
        """Applies 3 essential monkey-patches to prevent transformers >= 4.38 crashes."""
        orig_forward = self.model.forward

        @functools.wraps(orig_forward)
        def patched_forward(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("cache_position", None)
            kwargs.pop("num_logits_to_keep", None)
            return orig_forward(*args, **kwargs)

        self.model.forward = patched_forward

        orig_prep = self.model.prepare_inputs_for_generation

        @functools.wraps(orig_prep)
        def patched_prep(input_ids: Any, past_key_values: Any = None, **kwargs: Any) -> Any:
            model_inputs = orig_prep(input_ids, past_key_values=past_key_values, **kwargs)
            model_inputs.pop("cache_position", None)
            model_inputs.pop("num_logits_to_keep", None)
            return model_inputs

        self.model.prepare_inputs_for_generation = patched_prep

        if hasattr(self.model, "_validate_model_kwargs"):
            orig_validate = self.model._validate_model_kwargs

            @functools.wraps(orig_validate)
            def patched_validate(model_kwargs: dict[str, Any]) -> Any:
                kw_copy = model_kwargs.copy()
                kw_copy.pop("matryoshka_vis_token_scale", None)
                return orig_validate(kw_copy)

            self.model._validate_model_kwargs = patched_validate

    def preprocess_image(self, image: Image.Image) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        process_images = self._llava["process_images"]
        image_sizes = [image.size]
        image_tensor = process_images([image], self.image_processor, self.model.config).to(
            self.device, dtype=self.dtype
        )
        return image_tensor, image_sizes

    def format_prompt(self, question: str) -> str:
        DEFAULT_IMAGE_TOKEN = self._llava["DEFAULT_IMAGE_TOKEN"]
        DEFAULT_IM_START_TOKEN = self._llava["DEFAULT_IM_START_TOKEN"]
        DEFAULT_IM_END_TOKEN = self._llava["DEFAULT_IM_END_TOKEN"]
        IMAGE_PLACEHOLDER = self._llava["IMAGE_PLACEHOLDER"]
        conv_templates = self._llava["conv_templates"]

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in question:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
        else:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = image_token_se + "\n" + question
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + question

        qs = qs + "\nAnswer the question using a single word or phrase."

        conv = conv_templates[self.conv_mode].copy()
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
        """
        Fast forward pass with output_attentions=False to re-enable FlashAttention/SDPA.
        """
        IMAGE_TOKEN_INDEX = self._llava["IMAGE_TOKEN_INDEX"]
        tokenizer_image_token = self._llava["tokenizer_image_token"]

        prompt = self.format_prompt(question)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
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
        num_generated = len(outputs.scores) if (outputs.scores is not None and len(outputs.scores) > 0) else (len(gen_sequence) - input_ids.shape[1])
        gen_tokens = gen_sequence[-num_generated:].tolist()
        pred_answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        first_token_logits = outputs.scores[0][0].detach().float()
        first_token_logits = torch.nan_to_num(first_token_logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(first_token_logits, dim=-1)
        top2_vals, _ = torch.topk(probs, 2, dim=-1)
        top1_prob = float(top2_vals[0].item())
        margin = float((top2_vals[0] - top2_vals[1]).item())

        return {
            "answer": pred_answer,
            "conf_softmax": top1_prob,
            "margin": margin,
        }

    def sweep(
        self,
        image: Image.Image,
        question: str,
        scales: list[int] | None = None,
        gen_temperature: float = 0.0,
    ) -> dict[int, dict[str, Any]]:
        """Sweeps generation across all configured visual scales."""
        target_scales = scales or self.scales
        results = {}
        for m in target_scales:
            if m > self.fine_scale:
                raise ValueError(
                    f"Scale m={m} exceeds {self.arch.upper()} maximum of {self.fine_scale}. "
                    f"Configured scales: {self.scales}"
                )
            res = self.forward_fast(
                image=image,
                question=question,
                num_visual_tokens=m,
                gen_temperature=gen_temperature,
            )
            results[m] = res
        return results
