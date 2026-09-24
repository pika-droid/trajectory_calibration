"""
14-Benchmark Vision-Language Dataset Registry and HuggingFace Loader.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("trajectory_calibration.vlm.registry")

# Centralized 14-Benchmark Registry with non-withheld ground truth splits
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "infographicvqa": {
        "hf_repo": "lmms-lab/DocVQA",
        "config": "InfographicVQA",
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "ai2d": {
        "hf_repo": "lmms-lab/ai2d",
        "config": None,
        "default_split": "test",
        "answer_type": "mc_index",
    },
    "chartqa": {
        "hf_repo": "lmms-lab/ChartQA",
        "config": None,
        "default_split": "test",
        "answer_type": "open",
    },
    "docvqa": {
        "hf_repo": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "mmbench": {
        "hf_repo": "lmms-lab/MMBench_EN",
        "config": None,
        "default_split": "dev",
        "answer_type": "mc_letter",
    },
    "mmmu": {
        "hf_repo": "lmms-lab/MMMU",
        "config": None,
        "default_split": "validation",
        "answer_type": "mc_letter",
    },
    "seedbench": {
        "hf_repo": "lmms-lab/SEED-Bench",
        "config": None,
        "default_split": "test",
        "answer_type": "mc_letter",
    },
    "vqav2": {
        "hf_repo": "lmms-lab/vqav2",
        "config": None,
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "scienceqa": {
        "hf_repo": "lmms-lab/ScienceQA",
        "config": "ScienceQA-IMG",
        "default_split": "test",
        "answer_type": "mc_index",
    },
    "textvqa": {
        "hf_repo": "lmms-lab/textvqa",
        "config": None,
        "default_split": "validation",
        "answer_type": "list_soft",
    },
    "pope": {
        "hf_repo": "lmms-lab/POPE",
        "config": None,
        "default_split": "test",
        "answer_type": "open",
    },
    "gqa": {
        "hf_repo": "lmms-lab/GQA",
        "config": "testdev_balanced_instructions",
        "default_split": "testdev",
        "answer_type": "open",
    },
    "vizwiz-vqa": {
        "hf_repo": "lmms-lab/VizWiz-VQA",
        "config": None,
        "default_split": "val",
        "answer_type": "list_soft",
    },
    "lego-puzzles": {
        "hf_repo": "lmms-lab/LEGO-Puzzles",
        "config": None,
        "default_split": "test",
        "answer_type": "open",
    },
    "avqa": {
        "hf_repo": "lmms-lab/avqa",
        "config": None,
        "default_split": "val",
        "answer_type": "list_soft",
    },
    "vllm-safety": {
        "hf_repo": "PahaII/vllm_safety_evaluation",
        "config": None,
        "default_split": "test",
        "answer_type": "open",
    },
    "vqav2_5scale": {
        "hf_repo": "lmms-lab/vqav2",
        "config": None,
        "default_split": "validation",
        "answer_type": "list_soft",
    },
}

ALL_DATASET_KEYS = list(DATASET_REGISTRY.keys())


def _download_avqa_files(avqa_dir: Path) -> None:
    """Auto-downloads official AdVQA validation questions and annotations if not present."""
    import urllib.request

    avqa_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        "v1_OpenEnded_mscoco_val2017_advqa_questions.json": (
            "https://dl.fbaipublicfiles.com/advqa/v1_OpenEnded_mscoco_val2017_advqa_questions.json"
        ),
        "v1_mscoco_val2017_advqa_annotations.json": (
            "https://dl.fbaipublicfiles.com/advqa/v1_mscoco_val2017_advqa_annotations.json"
        ),
    }
    for fname, url in urls.items():
        out_p = avqa_dir / fname
        if not out_p.exists():
            try:
                logger.info(f"Downloading {fname} from {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(out_p, "wb") as f:
                    f.write(resp.read())
            except Exception as e:
                logger.warning(f"Failed to auto-download {fname}: {e}")


def _load_local_avqa(
    base_path: Path | str = "data/raw_datasets/avqa", auto_download: bool = True
) -> list[dict[str, Any]]:
    """Loads AVQA questions and annotations from local raw dataset directory if present."""
    avqa_dir = Path(base_path)
    q_candidates = [
        avqa_dir / "v1_OpenEnded_mscoco_val2017_advqa_questions.json",
        avqa_dir / "v1_avqa_r1+r2+r3_val_questions.json",
        avqa_dir / "avqa_val_questions.json",
        avqa_dir / "questions.json",
        avqa_dir / "val_questions.json",
    ]
    ann_candidates = [
        avqa_dir / "v1_mscoco_val2017_advqa_annotations.json",
        avqa_dir / "v1_avqa_r1+r2+r3_val_annotations.json",
        avqa_dir / "avqa_val_annotations.json",
        avqa_dir / "annotations.json",
        avqa_dir / "val_annotations.json",
    ]
    q_file = next((f for f in q_candidates if f.is_file()), None)
    ann_file = next((f for f in ann_candidates if f.is_file()), None)
    if (q_file is None or ann_file is None) and auto_download:
        _download_avqa_files(avqa_dir)
        q_file = next((f for f in q_candidates if f.is_file()), None)
        ann_file = next((f for f in ann_candidates if f.is_file()), None)
    if q_file is None or ann_file is None:
        return []

    try:
        with open(q_file, encoding="utf-8") as f:
            q_data = json.load(f)
        with open(ann_file, encoding="utf-8") as f:
            ann_data = json.load(f)
    except Exception as exc:
        logger.warning(f"Error reading local AVQA files: {exc}")
        return []

    q_list = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
    if isinstance(q_list, dict):
        q_list = [
            dict(v, question_id=k) if isinstance(v, dict) and "question_id" not in v else v
            for k, v in q_list.items()
        ]
    ann_list = ann_data.get("annotations", ann_data) if isinstance(ann_data, dict) else ann_data

    ann_map: dict[str, Any] = {}
    if isinstance(ann_list, list):
        for ann in ann_list:
            if isinstance(ann, dict):
                qid = ann.get("question_id", ann.get("id"))
                if qid is not None:
                    ann_map[str(qid)] = ann
    elif isinstance(ann_list, dict):
        ann_map = {str(k): v for k, v in ann_list.items()}

    images_dir = avqa_dir / "images"
    samples: list[dict[str, Any]] = []
    for q in q_list:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("question_id", q.get("id", len(samples))))
        ann = ann_map.get(qid, {})
        answers = ann.get("answers", ann.get("annotations", []))
        if isinstance(answers, (str, int, float)):
            answers = [answers]
        if not answers and "multiple_choice_answer" in ann:
            answers = [ann["multiple_choice_answer"]]
        elif not answers and "answer" in ann:
            answers = [ann["answer"]]
        if not answers:
            answers = q.get("answers", q.get("annotations", []))
            if isinstance(answers, (str, int, float)):
                answers = [answers]
        if not answers and "answer" in q:
            answers = [q["answer"]]

        img_name = q.get(
            "image_name",
            q.get(
                "image_path",
                q.get("image", q.get("image_id", q.get("picture", ""))),
            ),
        )
        if isinstance(img_name, int) or (isinstance(img_name, str) and img_name.isdigit()):
            coco_id = int(img_name)
            coco_name = f"COCO_val2017_{coco_id:012d}.jpg"
            coco_url = f"http://images.cocodataset.org/val2017/{coco_id:012d}.jpg"
        else:
            coco_name = str(img_name)
            coco_url = None

        img_path = None
        if img_name:
            cand_p = avqa_dir / coco_name
            cand_p_img = images_dir / coco_name
            if cand_p.is_file():
                img_path = str(cand_p)
            elif images_dir.is_dir() and cand_p_img.is_file():
                img_path = str(cand_p_img)
            elif Path(str(img_name)).is_file():
                img_path = str(img_name)
            elif coco_url:
                img_path = coco_url
            else:
                img_path = str(img_name)

        item: dict[str, Any] = {
            **q,
            "question_id": qid,
            "answers": answers,
        }
        if img_path is not None:
            item["image"] = img_path
        if coco_url:
            item["coco_url"] = coco_url
        samples.append(item)
    return samples


def _download_vllm_safety_files(vllm_dir: Path) -> None:
    """Auto-downloads VLLM safety benchmark dataset archive from Hugging Face if not present."""
    import zipfile

    from huggingface_hub import hf_hub_download

    vllm_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading safety_evaluation_benchmark_datasets.zip from Hugging Face...")
        zip_path = hf_hub_download(
            "PahaII/vllm_safety_evaluation",
            "safety_evaluation_benchmark_datasets.zip",
            repo_type="dataset",
        )
        logger.info(f"Extracting safety archive to {vllm_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(vllm_dir)
    except Exception as e:
        logger.warning(f"Could not auto-download VLLM safety benchmark: {e}")


def _load_local_vllm_safety(
    base_path: Path | str = "data/raw_datasets/vllm_safety", auto_download: bool = True
) -> list[dict[str, Any]]:
    """Loads full VLLM Safety Benchmark suites (challenging set + redteaming attacks)."""
    vllm_dir = Path(base_path)
    root = (
        vllm_dir / "safety_evaluation_benchmark_datasets"
        if (vllm_dir / "safety_evaluation_benchmark_datasets").is_dir()
        else vllm_dir
    )

    if not root.is_dir() and auto_download:
        _download_vllm_safety_files(vllm_dir)
        root = (
            vllm_dir / "safety_evaluation_benchmark_datasets"
            if (vllm_dir / "safety_evaluation_benchmark_datasets").is_dir()
            else vllm_dir
        )

    samples: list[dict[str, Any]] = []

    # 1. GPT4V challenging set (misleading attack, oodcv counterfactual, sketchy challenging)
    gpt4v_dir = root / "gpt4v_challenging_set"
    if gpt4v_dir.is_dir():
        for jf in sorted(gpt4v_dir.glob("*.json")):
            try:
                with open(jf, encoding="utf-8") as f:
                    d = json.load(f)
                items = d.get("data", d.get("annotation", d)) if isinstance(d, dict) else d
                if isinstance(items, dict):
                    items = list(items.values())
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    s = dict(item)
                    s["task"] = s.get("task", jf.stem)
                    s["question_id"] = f"{jf.stem}_{idx}"
                    if "text_answer" in s:
                        txt_ans = str(s["text_answer"]).strip()
                        if "answer" not in s:
                            s["answer"] = txt_ans
                        if "labels" not in s:
                            s["labels"] = [txt_ans]
                    img = s.get("image", s.get("image_path"))
                    if img:
                        img_p = gpt4v_dir / str(img)
                        if img_p.is_file():
                            s["image"] = str(img_p)
                    samples.append(s)
            except Exception as exc:
                logger.warning(f"Error loading {jf}: {exc}")

    # 2. Redteaming misleading attack (6 attack perturbation types or list/dict annotations)
    redteam_dir = root / "redteaming" / "misleading_attack"
    ann_file = redteam_dir / "annotation.json"
    if ann_file.is_file():
        try:
            with open(ann_file, encoding="utf-8") as f:
                ann_data = json.load(f)
            if isinstance(ann_data, list):
                for idx, item in enumerate(ann_data):
                    if isinstance(item, dict):
                        s = dict(item)
                        s["question_id"] = str(s.get("id", idx))
                        img = s.get("image", s.get("image_path"))
                        if img:
                            img_p = redteam_dir / str(img)
                            if img_p.is_file():
                                s["image"] = str(img_p)
                        samples.append(s)
            elif isinstance(ann_data, dict):
                attack_dirs = [d for d in sorted(redteam_dir.glob("*")) if d.is_dir()]
                if attack_dirs:
                    for ad in attack_dirs:
                        for img_name, labels in ann_data.items():
                            img_p = ad / img_name
                            if img_p.is_file():
                                samples.append(
                                    {
                                        "question_id": f"{ad.name}_{img_name}",
                                        "question": "Describe this image in detail.",
                                        "labels": labels if isinstance(labels, list) else [labels],
                                        "image": str(img_p),
                                        "task": ad.name,
                                    }
                                )
                else:
                    for img_name, labels in ann_data.items():
                        samples.append(
                            {
                                "question_id": f"misleading_{img_name}",
                                "question": "Describe this image in detail.",
                                "labels": labels if isinstance(labels, list) else [labels],
                                "image": str(redteam_dir / img_name),
                                "task": "misleading_attack",
                            }
                        )
        except Exception as exc:
            logger.warning(f"Error loading {ann_file}: {exc}")

    # Fallback to single json file if directory structure is flat
    if not samples:
        single_candidates = [
            root / "misleading-attack.json",
            root / "misleading_attack.json",
            root / "annotation.json",
            root / "annotations.json",
        ]
        target_file = next((f for f in single_candidates if f.is_file()), None)
        if target_file is not None:
            try:
                with open(target_file, encoding="utf-8") as f:
                    d = json.load(f)
                items = d.get("data", d.get("annotation", d)) if isinstance(d, dict) else d
                if isinstance(items, dict):
                    items = list(items.values())
                for idx, item in enumerate(items):
                    if isinstance(item, dict):
                        s = dict(item)
                        s["question_id"] = str(s.get("id", idx))
                        samples.append(s)
            except Exception as exc:
                logger.warning(f"Error loading {target_file}: {exc}")

    return samples


def load_hf_dataset(dataset_key: str, subset_size: int | None = None) -> Any:
    """Loads benchmark dataset from HuggingFace with custom filtering, merges, and local disk fallback."""
    from datasets import Dataset, load_dataset

    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Valid keys: {ALL_DATASET_KEYS}")

    cfg = DATASET_REGISTRY[dataset_key]
    repo = cfg["hf_repo"]
    config_name = cfg["config"]
    target_split = cfg["default_split"]

    logger.info(f"Loading dataset '{repo}' (config={config_name}, split={target_split})...")

    if dataset_key == "avqa":
        local_samples = _load_local_avqa()
        if local_samples:
            logger.info(f"Loaded {len(local_samples)} AVQA samples from local disk fallback.")
            ds = Dataset.from_list(local_samples)
            return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds
        try:
            ds = load_dataset(repo, split=target_split)
        except Exception:
            try:
                ds = load_dataset(repo, split="validation")
            except Exception:
                ds_dict = load_dataset(repo)
                ds = ds_dict[
                    target_split if target_split in ds_dict else next(iter(ds_dict.keys()))
                ]
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    elif dataset_key == "vllm-safety":
        local_samples = _load_local_vllm_safety()
        if local_samples:
            logger.info(
                f"Loaded {len(local_samples)} VLLM safety samples from local disk fallback."
            )
            ds = Dataset.from_list(local_samples)
            return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds
        try:
            ds = load_dataset(repo, split=target_split)
        except Exception:
            ds_dict = load_dataset(repo)
            ds = ds_dict[target_split if target_split in ds_dict else next(iter(ds_dict.keys()))]
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    elif dataset_key == "mmmu":
        try:
            ds = (
                load_dataset(repo, config_name, split=target_split)
                if config_name
                else load_dataset(repo, split=target_split)
            )
        except Exception:
            ds_dict = load_dataset(repo)
            ds = ds_dict[target_split if target_split in ds_dict else next(iter(ds_dict.keys()))]

        if "image_2" in ds.column_names:
            ds = ds.filter(lambda img2: img2 is None, input_columns=["image_2"])
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    elif dataset_key == "seedbench":
        ds = load_dataset(repo, split=target_split)
        if "data_type" in ds.column_names:
            ds = ds.filter(lambda dt: dt == "image", input_columns=["data_type"])
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    elif dataset_key == "gqa":
        inst_ds = load_dataset(repo, "testdev_balanced_instructions", split=target_split)
        img_ds = load_dataset(repo, "testdev_balanced_images", split=target_split)
        img_map = {x["id"]: x["image"] for x in img_ds}
        ds = inst_ds.map(lambda ex: {"image": img_map.get(ex.get("imageId"))})
        return ds.select(range(min(subset_size, len(ds)))) if subset_size else ds

    if config_name:
        ds = load_dataset(repo, config_name, split=target_split)
    else:
        ds = load_dataset(repo, split=target_split)

    if subset_size and len(ds) > subset_size:
        ds = ds.select(range(subset_size))
    return ds


__all__ = [
    "ALL_DATASET_KEYS",
    "DATASET_REGISTRY",
    "_load_local_avqa",
    "_load_local_vllm_safety",
    "load_hf_dataset",
]
