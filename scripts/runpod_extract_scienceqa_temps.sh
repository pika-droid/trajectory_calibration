#!/usr/bin/env bash
# =============================================================================
# RunPod: M3-LLaVA ScienceQA Feature Extraction for T in {0.3, 0.6, 1.0, 1.5}
# =============================================================================
# Targets:
#   /workspace/trajectory_calibration/data/features/m3_llava/temp_0.3/scienceqa.pt
#   /workspace/trajectory_calibration/data/features/m3_llava/temp_0.6/scienceqa.pt
#   /workspace/trajectory_calibration/data/features/m3_llava/temp_1.0/scienceqa.pt
#   /workspace/trajectory_calibration/data/features/m3_llava/temp_1.5/scienceqa.pt
#
# Requirements:
#   - RunPod pod with >= 1x A100/A40/A6000 GPU (40 GB+ VRAM recommended)
#   - Ubuntu 22.04 image (e.g. runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04)
#   - HF token (optional; mucai/llava-v1.5-7b-m3 and lmms-lab/ScienceQA are ungated)
#
# Usage:
#   bash scripts/runpod_extract_scienceqa_temps.sh
#   # Or specify custom temperatures:
#   TEMPS="0.3 0.6" bash scripts/runpod_extract_scienceqa_temps.sh
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/trajectory_calibration"
REPO_URL="https://github.com/pika-droid/trajectory_calibration.git"
M3_MODEL_ID="mucai/llava-v1.5-7b-m3"
TARGET_COUNT=2000
SAVE_INTERVAL=50
TEMPS="${TEMPS:-0.3 0.6 1.0 1.5}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== [1/6] System packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends \
    git git-lfs curl wget \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg
git lfs install

echo "=== [2/6] Repository check ==="
if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
else
    echo "Repo already present -- pulling latest..."
    git -C "${REPO_DIR}" pull --ff-only || true
fi
cd "${REPO_DIR}"

echo "=== [3/6] Python dependencies ==="
pip install --upgrade pip
pip install \
    "torch>=2.1.2" torchvision "numpy<2.0" "scipy>=1.10.0" "scikit-learn>=1.3.0" \
    "pandas>=2.0.0" "tqdm>=4.65.0" "matplotlib>=3.7.0" "pillow>=9.5.0" \
    "transformers>=4.38.0,<=4.41.2" "datasets" "accelerate<1.0.0" \
    "einops" "timm" "peft" "sentencepiece" "protobuf" "hf_transfer" \
    "huggingface-hub>=0.20.0" "sentence-transformers<3.0.0" "evaluate" "rouge-score"
pip install -e . --no-deps
pip install flash-attn --no-build-isolation || echo "flash-attn skipped"

echo "=== [4/6] Installing M3-LLaVA backbone ==="
LLAVA_SRC="${REPO_DIR}/llava_src"
if [ ! -d "${LLAVA_SRC}/llava" ]; then
    git clone --depth 1 https://github.com/mu-cai/matryoshka-mm.git "${LLAVA_SRC}"
fi
pip install -e "${LLAVA_SRC}" --no-deps || true

if [ -n "${HF_TOKEN}" ]; then
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

echo "=== [5/6] Verifying canonical manifest ==="
MANIFEST="${REPO_DIR}/data/canonical_manifest_all.json"
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest missing at ${MANIFEST}"
    exit 1
fi

echo "=== [6/6] Running Extractions for Temperatures: ${TEMPS} ==="
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for T in ${TEMPS}; do
    echo "-------------------------------------------------------------------------"
    echo ">>> Starting M3 ScienceQA Extraction at T=${T} (Test Split, N=2000)"
    echo "-------------------------------------------------------------------------"
    python scripts/extract_m3_scienceqa_2k.py \
        --model_path "${M3_MODEL_ID}" \
        --gen_temperature "${T}" \
        --target_count "${TARGET_COUNT}" \
        --save_interval "${SAVE_INTERVAL}"

    out_file="${REPO_DIR}/data/features/m3_llava/temp_${T}/scienceqa.pt"
    python3 - << PYEOF
import torch
p = "${out_file}"
data = torch.load(p, map_location="cpu")
print(f"Verified {p}: {len(data)} samples.")
print(f"Sample 0 question: {data[0].get('sample', {}).get('question')[:40]}...")
print(f"Sample 0 prediction: {data[0]['features'][576].get('answer')}")
PYEOF
done

echo "========================================================================="
echo "All ScienceQA temperature extractions completed successfully!"
echo "========================================================================="
