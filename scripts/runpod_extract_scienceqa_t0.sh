#!/usr/bin/env bash
# =============================================================================
# RunPod: M3-LLaVA ScienceQA Feature Extraction at T=0.0
# =============================================================================
# Target: /workspace/trajectory_calibration/data/features/m3_llava/temp_0.0/scienceqa.pt
#
# Requirements:
#   - RunPod pod with ≥1× A100/A40/A6000 GPU (40 GB+ VRAM recommended)
#   - Ubuntu 22.04 image (e.g. runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04)
#   - HF token with access to mucai/llava-v1.5-7b-m3 and lmms-lab/ScienceQA
#
# Usage:
#   HF_TOKEN=hf_xxxx bash runpod_extract_scienceqa_t0.sh
#
# Resume: re-running skips already-extracted samples automatically.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/trajectory_calibration"
REPO_URL="https://github.com/pika-droid/trajectory_calibration.git"

M3_MODEL_ID="mucai/llava-v1.5-7b-m3"
GEN_TEMPERATURE=0.0
TARGET_COUNT=2000
SAVE_INTERVAL=50

HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "=== [1/7] System packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends \
    git git-lfs curl wget \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg
git lfs install

# ---------------------------------------------------------------------------
# 2. Clone trajectory_calibration
# ---------------------------------------------------------------------------
echo "=== [2/7] Cloning trajectory_calibration ==="
if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
else
    echo "Repo already present — pulling latest..."
    git -C "${REPO_DIR}" pull --ff-only
fi
cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# 3. Install Python dependencies with pip
# ---------------------------------------------------------------------------
echo "=== [3/7] Installing Python dependencies ==="

pip install --upgrade pip

pip install \
    "torch>=2.1.2" torchvision \
    "numpy<2.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0" \
    "pandas>=2.0.0" \
    "tqdm>=4.65.0" \
    "matplotlib>=3.7.0" \
    "pillow>=9.5.0" \
    "transformers>=4.38.0,<=4.41.2" \
    "datasets" \
    "accelerate<1.0.0" \
    "einops" \
    "timm" \
    "peft" \
    "sentencepiece" \
    "protobuf" \
    "hf_transfer" \
    "huggingface-hub>=0.20.0" \
    "sentence-transformers<3.0.0" \
    "evaluate" \
    "rouge-score"

# Install the project package itself (src layout)
pip install -e . --no-deps

# flash-attn (optional, speeds up attention; skip if wheel unavailable)
pip install flash-attn --no-build-isolation || \
    echo "⚠  flash-attn not installed — extraction will work but be slower"

# bitsandbytes (optional, for quantization compat patches)
pip install bitsandbytes || true

# ---------------------------------------------------------------------------
# 4. Install the M3-LLaVA fork (matryoshka-mm → llava_src/)
# ---------------------------------------------------------------------------
echo "=== [4/7] Installing M3-LLaVA (matryoshka-mm) ==="
LLAVA_SRC="${REPO_DIR}/llava_src"

if [ ! -d "${LLAVA_SRC}/llava" ]; then
    git clone --depth 1 \
        https://github.com/mu-cai/matryoshka-mm.git \
        "${LLAVA_SRC}"
else
    echo "llava_src already present, skipping clone."
fi

# Editable install so `import llava` resolves correctly
# (llava_compat.py also prepends llava_src to sys.path at runtime)
pip install -e "${LLAVA_SRC}" --no-deps || \
    echo "⚠  Editable install of llava_src skipped — will rely on runtime sys.path patch"

# ---------------------------------------------------------------------------
# 5. HF authentication
# ---------------------------------------------------------------------------
echo "=== [5/7] Hugging Face authentication ==="
if [ -n "${HF_TOKEN}" ]; then
    python3 - << PYEOF
from huggingface_hub import login
login(token="${HF_TOKEN}", add_to_git_credential=False)
print("HF login OK.")
PYEOF
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    export HF_HUB_ENABLE_HF_TRANSFER=1
else
    echo "No HF_TOKEN set — proceeding with public download (mucai/llava-v1.5-7b-m3 and lmms-lab/ScienceQA are both ungated)."
fi

# ---------------------------------------------------------------------------
# 6. Verify canonical manifest
# ---------------------------------------------------------------------------
echo "=== [6/7] Verifying canonical manifest ==="
MANIFEST="${REPO_DIR}/data/canonical_manifest_all.json"
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: canonical_manifest_all.json not found at ${MANIFEST}"
    exit 1
fi

SCIENCEQA_COUNT=$(python3 -c "
import json
with open('${MANIFEST}') as f:
    m = json.load(f)
print(len(m.get('scienceqa', [])))
")
echo "Manifest OK — scienceqa entries: ${SCIENCEQA_COUNT}"

if [ "${SCIENCEQA_COUNT}" -lt "${TARGET_COUNT}" ]; then
    echo "ERROR: Manifest has only ${SCIENCEQA_COUNT} entries, need ${TARGET_COUNT}."
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. Run extraction
# ---------------------------------------------------------------------------
echo "=== [7/7] Running M3-LLaVA ScienceQA extraction (T=${GEN_TEMPERATURE}) ==="
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/extract_m3_scienceqa_2k.py \
    --model_path "${M3_MODEL_ID}" \
    --gen_temperature "${GEN_TEMPERATURE}" \
    --target_count "${TARGET_COUNT}" \
    --save_interval "${SAVE_INTERVAL}"

# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------
echo ""
echo "=== Verifying output ==="
python3 - << 'PYEOF'
import sys
from pathlib import Path

repo_root = Path("/workspace/trajectory_calibration")
sys.path.insert(0, str(repo_root / "src"))

from trajectory_calibration.utils.helpers import safe_torch_load

out_file = repo_root / "data/features/m3_llava/temp_0.0/scienceqa.pt"
if not out_file.exists():
    print(f"ERROR: Output not found: {out_file}")
    sys.exit(1)

records = safe_torch_load(out_file)
r0 = records[0]
print(f"Records   : {len(records)}")
print(f"Keys      : {list(r0.keys())}")
print(f"Scales    : {sorted(r0['features'].keys())}")
print(f"Temp      : {r0.get('temperature')}")

correct = sum(1 for r in records if r.get("vqa_accuracy", 0) > 0.5)
print(f"Accuracy  : {correct}/{len(records)} = {correct/len(records):.3%} (fine scale 576)")

status = "COMPLETE" if len(records) >= 2000 else f"PARTIAL ({len(records)}/2000 -- re-run to resume)"
print(f"Status    : {status}")
PYEOF

echo ""
echo "Output: ${REPO_DIR}/data/features/m3_llava/temp_0.0/scienceqa.pt"
