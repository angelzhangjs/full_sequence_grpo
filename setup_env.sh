#!/usr/bin/env bash
set -euo pipefail

# Simple setup script to create/refresh the ltx-grpo conda env and install deps.
# Run from repo root: bash setup_env.sh

ENV_NAME="ltx-grpo-test"

# Ensure conda shell is available in non-interactive scripts
if [ -z "${CONDA_EXE:-}" ]; then
  echo ">>> Loading conda base shell"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo ">>> Creating/updating conda env: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.10 || true

echo ">>> Installing conda packages"
conda install -y -n "${ENV_NAME}" pip

CONDA_RUN="conda run -n ${ENV_NAME}"

echo ">>> Installing pip packages"
$CONDA_RUN python -m pip install --upgrade pip
$CONDA_RUN python -m pip install -r requirements.txt
# Pin critical versions for compatibility with LoRA and LTX-Video
$CONDA_RUN python -m pip install --no-cache-dir "transformers==4.48.0" "peft==0.17.1" "diffusers==0.36.0" \
    "imageio==2.37.2" "imageio-ffmpeg==0.6.0"

# Install torch with CUDA wheels (adjust index if needed)
$CONDA_RUN python -m pip install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cu121

echo ">>> Environment setup complete for ${ENV_NAME}"

