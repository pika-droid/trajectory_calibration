"""
Multi-Pass and Multi-Rollout VLM Feature Extraction Engine.

Extracts greedy outputs, top-token logits, multi-rollout sampling texts,
per-token log-probabilities, and pooled hidden state embeddings for UQ baselines.
"""

from __future__ import annotations

import logging
import random
from typing import Any
import numpy as np
import torch

from trajectory_calibration.vlm.evaluators import evaluate_accuracy
from trajectory_calibration.vlm.formatting import format_question, load_image_from_sample
from trajectory_calibration.vlm.registry import DATASET_REGISTRY

logger = logging.getLogger("trajectory_calibration.vlm.multipass")


def generate_mock_multipass_sample(
    idx: int,
    dataset_key: str = "pope",
    num_rollouts: int = 5,
    sample: dict[str, Any] | None = None,
    hidden_dim: int = 128,
) -> dict[str, Any]:
    """Generates synthetic multi-pass / multi-rollout record for smoke testing."""
    cfg = DATASET_REGISTRY.get(dataset_key, {"answer_type": "open"})
    ans_type = cfg.get("answer_type", "open")
    qid = str(sample.get("question_id", sample.get("id", 100000 + idx))) if sample else str(100000 + idx)
    question = format_question(sample, dataset_key) if sample else f"Mock question {idx}?"
    gt = sample.get("answer", sample.get("label", "yes")) if sample else "yes"

    is_corr = random.random() > 0.35
    vqa_acc = 1.0 if is_corr else 0.0
    c_fine = random.uniform(0.7, 0.98) if is_corr else random.uniform(0.2, 0.6)

    first_logits = np.random.randn(32000).astype(np.float32)
    top_tok = 100 if is_corr else 200
    first_logits[top_tok] = float(np.log(c_fine / (1.0 - c_fine + 1e-7)) + 5.0)

    base_ans = str(gt) if is_corr else "unrelated answer"
    roll_texts = [base_ans if random.random() > 0.3 else f"maybe {base_ans}" for _ in range(num_rollouts)]

    roll_token_lps = []
    roll_seq_lps = []
    for _ in range(num_rollouts):
        tok_lps = [float(-random.uniform(0.05, 0.6)) for _ in range(random.randint(2, 5))]
        roll_token_lps.append(tok_lps)
        roll_seq_lps.append(float(sum(tok_lps)))

    base_emb = np.random.randn(hidden_dim).astype(np.float32)
    roll_embs = np.stack(
        [base_emb + np.random.randn(hidden_dim).astype(np.float32) * 0.05 for _ in range(num_rollouts)],
        axis=0,
    )

    return {
        "question_id": qid,
        "dataset": dataset_key,
        "question": question,
        "ground_truth": gt,
        "answer_type": ans_type,
        "greedy_answer": base_ans,
        "is_correct": bool(is_corr),
        "vqa_accuracy": float(vqa_acc),
        "conf_softmax": float(c_fine),
        "first_token_logits": first_logits,
        "rollout_texts": roll_texts,
        "rollout_token_logprobs": roll_token_lps,
        "rollout_sequence_logprobs": roll_seq_lps,
        "rollout_embeddings": roll_embs,
    }


@torch.inference_mode()
def extract_multipass_record(
    wrapper: Any,
    sample: dict[str, Any],
    dataset_key: str,
    num_rollouts: int = 5,
    gen_temperature: float = 0.5,
    top_p: float = 0.9,
    max_new_tokens: int = 32,
) -> dict[str, Any] | None:
    """Performs greedy + multi-rollout inference and packages payload for UQ."""
    image = load_image_from_sample(sample)
    if image is None:
        return None

    try:
        cfg = DATASET_REGISTRY.get(dataset_key, {"answer_type": "open"})
        ans_type = cfg.get("answer_type", "open")
        qid = str(sample.get("question_id", sample.get("id", sample.get("sample_idx", 0))))
        question = format_question(sample, dataset_key)
        gt = sample.get("answer", sample.get("label", sample.get("ground_truth", "")))

        prompt = wrapper.format_prompt(question)
        input_ids = wrapper._llava["tokenizer_image_token"](
            prompt, wrapper.tokenizer, wrapper._llava["IMAGE_TOKEN_INDEX"], return_tensors="pt"
        ).unsqueeze(0).to(wrapper.device)
        image_tensor, image_sizes = wrapper.preprocess_image(image)
        input_len = input_ids.shape[1]
        autocast_dev = wrapper.device.type if hasattr(wrapper.device, "type") else "cuda"

        vt = None  # Multi-pass rollouts always evaluate at native fine scale (576 for M3, 256 for MQT)
        if hasattr(wrapper.model, "config"):
            setattr(wrapper.model.config, "num_visual_tokens", vt)

        # 1. Primary Greedy / Argmax Pass
        g_kwargs: dict[str, Any] = {
            "inputs": input_ids, "images": image_tensor, "image_sizes": image_sizes,
            "do_sample": False, "max_new_tokens": max_new_tokens, "use_cache": True,
            "output_attentions": False, "output_hidden_states": False,
            "output_scores": True, "return_dict_in_generate": True,
        }
        if vt is not None:
            g_kwargs["matryoshka_vis_token_scale"] = vt

        g_out = wrapper.model.generate(**g_kwargs)

        g_seq = g_out.sequences[0]
        n_gen = len(g_out.scores) if (g_out.scores is not None and len(g_out.scores) > 0) else (len(g_seq) - input_len)
        greedy_ans = wrapper.tokenizer.decode(g_seq[-n_gen:].tolist(), skip_special_tokens=True).strip()

        if g_out.scores is not None and len(g_out.scores) > 0:
            first_logits_t = g_out.scores[0][0].detach().float()
            first_logits_np = torch.nan_to_num(first_logits_t, nan=-1e4, posinf=1e4, neginf=-1e4).cpu().numpy().astype(np.float32)
            conf_softmax = float(torch.max(torch.softmax(first_logits_t, dim=-1)).item())
        else:
            h_vocab = getattr(wrapper.model.config, "vocab_size", 32000)
            first_logits_np = np.zeros(h_vocab, dtype=np.float32)
            conf_softmax = 0.5
        vqa_acc = float(evaluate_accuracy(greedy_ans, sample, dataset_key))

        # 2. Multi-Rollout Sampling Pass (Parallel Batched Generation: ~1.2s/sample)
        s_kwargs: dict[str, Any] = {
            "inputs": input_ids, "images": image_tensor, "image_sizes": image_sizes,
            "do_sample": True, "temperature": float(gen_temperature), "top_p": float(top_p),
            "renormalize_logits": True, "num_return_sequences": int(num_rollouts),
            "max_new_tokens": max_new_tokens, "use_cache": True,
            "output_attentions": False, "output_hidden_states": True,
            "output_scores": True, "return_dict_in_generate": True,
        }
        if vt is not None:
            s_kwargs["matryoshka_vis_token_scale"] = vt

        s_out = wrapper.model.generate(**s_kwargs)

        n_sc = len(s_out.scores) if s_out.scores is not None else 0
        n_hs = len(s_out.hidden_states) if s_out.hidden_states is not None else 0
        eos_id = getattr(wrapper.tokenizer, "eos_token_id", None)
        roll_texts, roll_tok_lps, roll_seq_lps, roll_embs = [], [], [], []

        for k in range(num_rollouts):
            full_seq = s_out.sequences[k]
            gen_tokens = full_seq[-n_sc:].tolist() if n_sc > 0 else (full_seq[input_len:].tolist() if len(full_seq) > input_len else full_seq.tolist())
            act_len = (gen_tokens.index(eos_id) + 1) if (eos_id is not None and eos_id in gen_tokens) else len(gen_tokens)
            tok_ids = gen_tokens[:act_len]
            roll_texts.append(wrapper.tokenizer.decode(tok_ids, skip_special_tokens=True).strip())

            t_lps = []
            for t in range(min(act_len, n_sc)):
                logits_t = s_out.scores[t][k].detach().float()
                if tok_ids[t] < logits_t.shape[-1]:
                    t_lps.append(float(torch.log_softmax(logits_t, dim=-1)[tok_ids[t]].item()))
            t_lps = t_lps if t_lps else [0.0]
            roll_tok_lps.append(t_lps)
            roll_seq_lps.append(float(sum(t_lps)))

            t_vecs = []
            for t in range(min(act_len, n_hs)):
                t_vecs.append(s_out.hidden_states[t][-1][k, -1, :].detach().float().cpu().numpy())
            h_dim = getattr(wrapper.model.config, "hidden_size", 4096)
            roll_embs.append(np.mean(t_vecs, axis=0).astype(np.float32) if t_vecs else np.zeros(h_dim, dtype=np.float32))

        return {
            "question_id": qid, "dataset": dataset_key, "question": question,
            "ground_truth": gt, "answer_type": ans_type,
            "greedy_answer": greedy_ans, "is_correct": bool(vqa_acc >= 0.5),
            "vqa_accuracy": vqa_acc, "conf_softmax": conf_softmax,
            "first_token_logits": first_logits_np, "rollout_texts": roll_texts,
            "rollout_token_logprobs": roll_tok_lps, "rollout_sequence_logprobs": roll_seq_lps,
            "rollout_embeddings": np.stack(roll_embs, axis=0),
        }
    except Exception as e:
        logger.warning(f"Error extracting record for sample {sample.get('question_id', 'unknown')}: {e}")
        return None
