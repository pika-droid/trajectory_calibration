#!/usr/bin/env bash
# =============================================================================
# RunPod: VQAv2 Feature Extraction (M3-LLaVA & MQT-LLaVA, T in {0.0, 0.3, 0.6, 1.0, 1.5})
# =============================================================================
# Targets:
#   /workspace/trajectory_calibration/data/features/{m3_llava,mqt_llava}/temp_{0.0,0.3,0.6,1.0,1.5}/vqav2_5scale.pt
#
# Requirements:
#   - RunPod pod with >= 1x A100/A40/A6000 GPU (40 GB+ VRAM recommended)
#   - Ubuntu 22.04 image (e.g. runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04)
#   - HF token (optional; mucai/llava-v1.5-7b-m3, gordonhu/MQT-LLaVA-7b, and lmms-lab/vqav2 are public)
#
# Usage:
#   bash scripts/runpod_extract_vqav2_temps.sh
#   # Or customize architectures / temperatures:
#   ARCHS="mqt" TEMPS="0.3 0.6" bash scripts/runpod_extract_vqav2_temps.sh
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/trajectory_calibration"
REPO_URL="https://github.com/pika-droid/trajectory_calibration.git"
TARGET_COUNT=2000
SAVE_INTERVAL=50
ARCHS="${ARCHS:-m3 mqt}"
TEMPS="${TEMPS:-0.0 0.3 0.6 1.0 1.5}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== [1/6] Installing System Packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends \
    git git-lfs curl wget zip unzip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg
git lfs install

echo "=== [2/6] Setting Up Repository ==="
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
else
    echo "Repository already present -- pulling latest main..."
    git -C "${REPO_DIR}" pull --ff-only || true
fi
cd "${REPO_DIR}"

echo "=== [3/6] Python Dependencies ==="
pip install --upgrade pip
pip install \
    "torch>=2.1.2" torchvision "numpy<2.0" "scipy>=1.10.0" "scikit-learn>=1.3.0" \
    "pandas>=2.0.0" "tqdm>=4.65.0" "matplotlib>=3.7.0" "pillow>=9.5.0" \
    "transformers>=4.38.0,<=4.41.2" "datasets" "accelerate<1.0.0" \
    "einops" "timm" "peft" "sentencepiece" "protobuf" "hf_transfer" \
    "huggingface-hub>=0.20.0" "sentence-transformers<3.0.0" "evaluate" "rouge-score"
pip install -e . --no-deps
pip install flash-attn --no-build-isolation || echo "Notice: flash-attn skipped"

echo "=== [4/6] Installing LLaVA Backbone (matryoshka-mm) ==="
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

echo "=== [5/6] Verifying Canonical Manifest ==="
MANIFEST="${REPO_DIR}/data/canonical_manifest_all.json"
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: canonical_manifest_all.json missing at ${MANIFEST}"
    exit 1
fi

echo "=== [6/6] Running Extractions ==="
echo "Architectures: ${ARCHS}"
echo "Temperatures : ${TEMPS}"
echo "Target count : ${TARGET_COUNT}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for ARCH in ${ARCHS}; do
    for T in ${TEMPS}; do
        echo ""
        echo "========================================================================="
        echo ">>> Running ${ARCH^^} VQAv2 Extraction at T=${T} (N=${TARGET_COUNT})"
        echo "========================================================================="

        python scripts/extract_vqav2_2k.py \
            --arch "${ARCH}" \
            --gen_temperature "${T}" \
            --target_count "${TARGET_COUNT}" \
            --save_interval "${SAVE_INTERVAL}"

        # Verification check
        out_file="${REPO_DIR}/data/features/${ARCH}_llava/temp_${T}/vqav2_5scale.pt"
        python3 - << PYEOF
import torch
p = "${out_file}"
data = torch.load(p, map_location="cpu")
s0 = data[0].get("sample", {})
q0 = s0.get("question", "")
qid0 = str(data[0].get("question_id", ""))
fine_scale = 576 if "${ARCH}" == "m3" else 256
pred0 = data[0]["features"][fine_scale].get("answer")
acc = sum(1 for r in data if r.get("vqa_accuracy", 0) > 0.5) / len(data)

print(f"Verified : {p}")
print(f"Count    : {len(data)} / 2000")
print(f"Q0 ID    : {qid0} (expected 262148000)")
print(f"Q0 text  : '{q0}'")
print(f"Q0 pred  : '{pred0}'")
print(f"Fine Acc : {acc:.2%}")
assert len(data) == 2000, f"Expected 2000 samples, got {len(data)}"
assert qid0 == "262148000", f"Unexpected sample 0 QID {qid0}"
print("Status: 100% CANONICAL SAMPLE PARITY VERIFIED OK!")
PYEOF
    done
done

# -----------------------------------------------------------------------------
# Package Outputs for Easy Download
# -----------------------------------------------------------------------------
ZIP_FILE="${WORKSPACE}/vqav2_temps_all.zip"
TAR_FILE="${WORKSPACE}/vqav2_temps_all.tar.gz"

echo ""
echo "=== Packaging extracted VQAv2 feature files ==="
cd "${REPO_DIR}"
zip -r "${ZIP_FILE}" data/features/*/temp_*/vqav2_5scale.pt || true
tar -czvf "${TAR_FILE}" data/features/*/temp_*/vqav2_5scale.pt || true

echo ""
echo "========================================================================="
echo "ALL VQAV2 EXTRACTIONS COMPLETED AND VERIFIED!"
echo "Downloadable zip ready at    : ${ZIP_FILE}"
echo "Downloadable tarball ready at: ${TAR_FILE}"
echo "========================================================================="
