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
def _extract_single_rollout(
    wrapper: Any,
    sequences: torch.Tensor,
    scores: list[torch.Tensor] | None,
    hidden_states: list[tuple[torch.Tensor, ...]] | None,
    input_len: int,
    k_idx: int = 0,
) -> tuple[str, list[float], float, np.ndarray]:
    """Extracts text, token logprobs, sequence logprob, and embedding for a single sequence index."""
    n_sc = len(scores) if scores is not None else 0
    n_hs = len(hidden_states) if hidden_states is not None else 0
    eos_id = getattr(wrapper.tokenizer, "eos_token_id", None)
    h_dim = getattr(wrapper.model.config, "hidden_size", 4096)

    if n_sc > 0:
        gen_tokens = full_seq[-n_sc:].tolist()
    elif len(full_seq) > input_len:
        gen_tokens = full_seq[input_len:].tolist()
    else:
        gen_tokens = full_seq.tolist()

    if eos_id is not None and eos_id in gen_tokens:
        act_len = gen_tokens.index(eos_id) + 1
    else:
        act_len = len(gen_tokens)
    tok_ids = gen_tokens[:act_len]
    text = wrapper.tokenizer.decode(tok_ids, skip_special_tokens=True).strip()

    t_lps = []
    for t in range(min(act_len, n_sc)):
        logits_t = scores[t][k_idx].detach().float()
        if tok_ids[t] < logits_t.shape[-1]:
            t_lps.append(float(torch.log_softmax(logits_t, dim=-1)[tok_ids[t]].item()))
    t_lps = t_lps if t_lps else [0.0]
    seq_lp = float(sum(t_lps))

    t_vecs = []
    for t in range(min(act_len, n_hs)):
        t_vecs.append(hidden_states[t][-1][k_idx, -1, :].detach().float().cpu().numpy())
    emb = np.mean(t_vecs, axis=0).astype(np.float32) if t_vecs else np.zeros(h_dim, dtype=np.float32)

    return text, t_lps, seq_lp, emb


def _generate_rollouts_with_oom_defense(
    wrapper: Any,
    s_kwargs: dict[str, Any],
    num_rollouts: int,
    input_len: int,
) -> tuple[list[str], list[list[float]], list[float], list[np.ndarray]]:
    """Generates rollouts in parallel, with automatic sequential fallback if CUDA OOM occurs."""
    try:
        batched_kwargs = dict(s_kwargs)
        batched_kwargs["num_return_sequences"] = int(num_rollouts)
        s_out = wrapper.model.generate(**batched_kwargs)

        roll_texts, roll_tok_lps, roll_seq_lps, roll_embs = [], [], [], []
        for k in range(num_rollouts):
            text, t_lps, seq_lp, emb = _extract_single_rollout(
                wrapper=wrapper,
                sequences=s_out.sequences,
                scores=s_out.scores,
                hidden_states=s_out.hidden_states,
                input_len=input_len,
                k_idx=k,
            )
            roll_texts.append(text)
            roll_tok_lps.append(t_lps)
            roll_seq_lps.append(seq_lp)
            roll_embs.append(emb)

        del s_out
        return roll_texts, roll_tok_lps, roll_seq_lps, roll_embs

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg or isinstance(e, torch.cuda.OutOfMemoryError):
            logger.warning("CUDA OOM encountered during parallel rollouts. Falling back to sequential execution...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            seq_kwargs = dict(s_kwargs)
            seq_kwargs["num_return_sequences"] = 1

            roll_texts, roll_tok_lps, roll_seq_lps, roll_embs = [], [], [], []
            for _ in range(num_rollouts):
                s_out_single = wrapper.model.generate(**seq_kwargs)
                text, t_lps, seq_lp, emb = _extract_single_rollout(
                    wrapper=wrapper,
                    sequences=s_out_single.sequences,
                    scores=s_out_single.scores,
                    hidden_states=s_out_single.hidden_states,
                    input_len=input_len,
                    k_idx=0,
                )
                roll_texts.append(text)
                roll_tok_lps.append(t_lps)
                roll_seq_lps.append(seq_lp)
                roll_embs.append(emb)
                del s_out_single

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return roll_texts, roll_tok_lps, roll_seq_lps, roll_embs

        raise e


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
    """
    Executes 1 greedy pass and M sampling rollouts, returning full UQ trajectory payload.
    """
    image = load_image_from_sample(sample)
    if image is None:
        return None

    try:
        cfg = DATASET_REGISTRY.get(dataset_key, {"answer_type": "open"})
        ans_type = cfg.get("answer_type", "open")
        qid = str(sample.get("question_id", sample.get("questionId", sample.get("id", sample.get("sample_idx", sample.get("image_id", 0))))))
        # Extract ground truth cleanly across all benchmark formats
        raw_gt = sample.get("answer", sample.get("label", sample.get("ground_truth", None)))
        if raw_gt is None or raw_gt == "":
            raw_answers = sample.get("answers", sample.get("annotations", []))
            if isinstance(raw_answers, list) and len(raw_answers) > 0:
                if isinstance(raw_answers[0], dict):
                    raw_gt = [a.get("answer", "") for a in raw_answers if isinstance(a, dict)]
                else:
                    raw_gt = [str(a) for a in raw_answers]
            else:
                raw_gt = ""
        gt = raw_gt
        question = format_question(sample, dataset_key)
        prompt = wrapper.format_prompt(question)
        input_ids = wrapper._llava["tokenizer_image_token"](
            prompt, wrapper.tokenizer, wrapper._llava["IMAGE_TOKEN_INDEX"], return_tensors="pt"
        ).unsqueeze(0).to(wrapper.device)
        image_tensor, image_sizes = wrapper.preprocess_image(image)
        input_len = input_ids.shape[1]
        autocast_dev = wrapper.device.type if hasattr(wrapper.device, "type") else "cuda"

        vt = 256 if getattr(wrapper, "arch", "m3") == "mqt" else None
        if hasattr(wrapper.model, "config"):
            setattr(wrapper.model.config, "num_visual_tokens", vt)
        if hasattr(wrapper.model, "model") and hasattr(wrapper.model.model, "config"):
            setattr(wrapper.model.model.config, "num_visual_tokens", vt)

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

        # 2. Multi-Rollout Sampling Pass (Parallel Batched Generation with Sequential OOM Fallback)
        s_kwargs: dict[str, Any] = {
            "inputs": input_ids, "images": image_tensor, "image_sizes": image_sizes,
            "do_sample": True, "temperature": float(gen_temperature), "top_p": float(top_p),
            "renormalize_logits": True,
            "max_new_tokens": max_new_tokens, "use_cache": True,
            "output_attentions": False, "output_hidden_states": True,
            "output_scores": True, "return_dict_in_generate": True,
        }
        if vt is not None:
            s_kwargs["matryoshka_vis_token_scale"] = vt

        roll_texts, roll_tok_lps, roll_seq_lps, roll_embs = _generate_rollouts_with_oom_defense(
            wrapper=wrapper,
            s_kwargs=s_kwargs,
            num_rollouts=num_rollouts,
            input_len=input_len,
        )

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
