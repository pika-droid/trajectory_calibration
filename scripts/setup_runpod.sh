#!/usr/bin/env bash
# Setup script for Trajectory Calibration environment on RunPod GPU containers

set -e

echo "=================================================================="
echo " Setting up Trajectory Calibration Environment on RunPod (CUDA)   "
echo "=================================================================="

# Enable fast HuggingFace downloads and expandable segments to prevent VRAM fragmentation
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Update package lists and install system utilities if needed
if command -v apt-get &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq git zip unzip curl wget
fi

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing Astral uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env" || export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install project dependencies with GPU and Dev extras into the container Python environment
echo "Installing project dependencies with GPU extras..."
uv pip install --system --break-system-packages -e ".[gpu,dev]" "transformers==4.41.2"

echo "=================================================================="
echo " Setup complete! Running quick PyTorch & CUDA sanity test...       "
echo "=================================================================="

python -c "
import torch, transformers, datasets
print('PyTorch Version:    ', torch.__version__)
print('CUDA Available:     ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:         ', torch.cuda.get_device_name(0))
    print('VRAM:               ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
print('Transformers:       ', transformers.__version__)
print('Datasets:           ', datasets.__version__)
"

echo "=================================================================="
echo " Ready to run extraction!                                          "
echo "=================================================================="
