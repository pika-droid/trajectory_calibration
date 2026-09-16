"""
Multi-benchmark ground-truth evaluation and scoring engine.
"""

from __future__ import annotations

import re
from typing import Any

from trajectory_calibration.utils.helpers import clean_text
from trajectory_calibration.vlm.formatting import _parse_options_list
from trajectory_calibration.vlm.registry import DATASET_REGISTRY


def _is_word_match(candidate: str, target: str) -> bool:
    """Checks if candidate matches target either exactly or as a whole word boundary."""
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    pattern = r"\b" + re.escape(candidate) + r"\b"
    return bool(re.search(pattern, target))


def evaluate_accuracy(pred_answer: str, sample: dict[str, Any], dataset_key: str) -> float:
    """
    Computes VQA accuracy score against ground-truth labels across multi-format benchmarks.
    """
    pred_clean = pred_answer.strip().lower()
    pred_norm = clean_text(pred_clean)
    cfg = DATASET_REGISTRY.get(dataset_key, {"answer_type": "open"})
    ans_type = cfg.get("answer_type", "open")

    if ans_type == "open":
        gt_ans = str(
            sample.get(
                "answer",
                sample.get(
                    "label",
                    sample.get(
                        "ground_truth",
                        sample.get("gt_answer", sample.get("target", "")),
                    ),
                ),
            )
        ).strip()
        if not gt_ans:
            return 0.0
        gt_clean = clean_text(gt_ans)
        if pred_norm == gt_clean or pred_clean == gt_ans.lower():
            return 1.0

        # Require ground-truth containment inside the prediction
        if (len(gt_clean) >= 2 or gt_clean.isdigit()) and _is_word_match(gt_clean, pred_norm):
            return 1.0
        return 0.0

    elif ans_type == "list_soft":
        gt_answers = sample.get(
            "answers", sample.get("annotations", sample.get("answers_list", []))
        )
        if isinstance(gt_answers, (str, int, float)):
            gt_answers = [gt_answers]
        if not gt_answers:
            gt_ans = sample.get("answer", sample.get("label", sample.get("multiple_choice_answer")))
            gt_answers = [gt_ans] if gt_ans is not None else []

        match_count = 0
        for gt in gt_answers:
            gt_text = (
                gt.get("answer", gt.get("text", gt.get("raw_answer", "")))
                if isinstance(gt, dict)
                else str(gt)
            )
            gt_clean = clean_text(gt_text)
            if not gt_clean:
                continue
            if (
                gt_clean == pred_norm
                or pred_clean == str(gt_text).strip().lower()
                or (
                    (len(gt_clean) >= 2 or gt_clean.isdigit())
                    and _is_word_match(gt_clean, pred_norm)
                )
            ):
                match_count += 1
        return min(1.0, match_count / 3.0) if match_count > 0 else 0.0

    elif ans_type == "mc_index":
        correct_idx = sample.get("answer", sample.get("label", sample.get("correct_choice")))
        options = _parse_options_list(sample.get("choices", sample.get("options", [])))
        if correct_idx is not None and str(correct_idx).isdigit():
            idx_int = int(correct_idx)
            if 0 <= idx_int < len(options):
                correct_letter = chr(65 + idx_int).lower()
                correct_option_text = clean_text(str(options[idx_int]))
                if (
                    pred_clean.startswith(correct_letter)
                    or pred_clean == correct_letter
                    or pred_norm == correct_option_text
                ):
                    return 1.0
        return 0.0

    elif ans_type == "mc_letter":
        gt_ans = str(sample.get("answer", sample.get("label", ""))).strip().upper()
        if not gt_ans:
            return 0.0

        if pred_clean == gt_ans.lower() or pred_clean.startswith(gt_ans.lower()):
            return 1.0

        options = _parse_options_list(sample.get("options", sample.get("choices", [])))
        if options and len(gt_ans) == 1 and "A" <= gt_ans <= "Z":
            target_idx = ord(gt_ans) - ord("A")
            if 0 <= target_idx < len(options):
                target_option = clean_text(str(options[target_idx]))
                if pred_norm == target_option:
                    return 1.0

        letter_idx_map = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
        if gt_ans in letter_idx_map:
            target_col = letter_idx_map[gt_ans]
            if sample.get(target_col) and pred_norm == clean_text(str(sample[target_col])):
                return 1.0

        return 0.0

    return (
        1.0 if pred_norm == clean_text(str(sample.get("answer", sample.get("label", "")))) else 0.0
    )


__all__ = [
    "_is_word_match",
    "evaluate_accuracy",
]
