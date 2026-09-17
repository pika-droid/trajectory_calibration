#!/usr/bin/env bash
# ==============================================================================
# RunPod Script: Single-Pass MQT Top-Up to 2,000 Samples (ADR 0006)
# ==============================================================================
set -euo pipefail

echo "================================================================="
echo "Starting Single-Pass MQT Top-Up (1,000..1,999) on RunPod..."
echo "================================================================="

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

REPO_ROOT="/workspace/trajectory_calibration"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

# Ensure canonical manifest exists
if [ ! -f "data/canonical_manifest_all.json" ]; then
    echo "Canonical manifest not found! Generating data/canonical_manifest_all.json..."
    python3 scripts/build_canonical_manifest.py
fi

# Ensure MQT repository is cloned
if [ ! -d "/workspace/MQT-LLaVA" ]; then
    echo "Cloning gordonhu608/MQT-LLaVA..."
    git clone --depth 1 https://github.com/gordonhu608/MQT-LLaVA.git /workspace/MQT-LLaVA
fi

# Execute MQT top-up
echo "Executing MQT top-up across ai2d, chartqa, docvqa, and vqav2_5scale..."
python3 scripts/run_mqt_topup_2k.py \
    --model_path "gordonhu/MQT-LLaVA-7b" \
    --gen_temperature 0.0

echo "================================================================="
echo "MQT Top-Up to 2,000 samples completed successfully!"
echo "================================================================="
