#!/usr/bin/env bash
set -euo pipefail

# Recreate (or refresh) the `cogvideo` conda environment for this repo.
#
# Usage:
#   bash setup/reinstall_cogvideo_env.sh                # recreates env "cogvideo"
#   bash setup/reinstall_cogvideo_env.sh myenv          # recreates env "myenv"
#
# Optional environment variables:
#   CONDA_EXE=/path/to/conda
#   PYTHON_VERSION=3.10
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#   RECREATE=1                 # 1 = remove env if it exists (default), 0 = keep env and just pip install/upgrade
#   WITH_WAN=0                 # 1 = also install Wan2.1 requirements
#   WITH_OPENSORA=0            # 1 = also install Open-Sora requirements
#
# Notes:
# - This script installs a "working superset" for CogVideoX + unified_grpo reward deps.
# - If you hit version conflicts, prefer pinning with a lockfile or per-subproject envs.

ENV_NAME="${1:-cogvideo}"
REPO_DIR="/home/ubuntu/angel-research"

CONDA_EXE="${CONDA_EXE:-/home/ubuntu/anaconda3/bin/conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
RECREATE="${RECREATE:-1}"
WITH_WAN="${WITH_WAN:-0}"
WITH_OPENSORA="${WITH_OPENSORA:-0}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "ERROR: conda not found at ${CONDA_EXE}"
  exit 1
fi

# shellcheck disable=SC1090
source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"

if [[ "${RECREATE}" == "1" ]]; then
  if "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[1/6] Removing existing conda env: ${ENV_NAME}"
    "${CONDA_EXE}" env remove -n "${ENV_NAME}" -y
  fi

  echo "[2/6] Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  "${CONDA_EXE}" create -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}" pip
else
  echo "[1/6] RECREATE=0: keeping existing env '${ENV_NAME}'"
fi

echo "[3/6] Activating env: ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "[4/6] Installing base system deps (ffmpeg) into env"
conda install -y -c conda-forge ffmpeg

echo "[5/6] Upgrading pip tooling"
python -m pip install --upgrade pip wheel
# Pin setuptools to a version that still provides pkg_resources (needed to install OpenAI CLIP from git).
python -m pip install --upgrade "setuptools==70.3.0"

echo "[6/6] Installing Python deps"

# Prevent user-site packages (~/.local) from shadowing this conda environment during installs.
export PYTHONNOUSERSITE=1

# 1) Install PyTorch first (CUDA wheels).
python -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" torch torchvision

# 2) Core training/inference stack used across CogVideoX + unified_grpo
python -m pip install --upgrade \
  diffusers accelerate transformers peft safetensors \
  numpy scipy pyyaml regex tqdm psutil \
  einops pillow opencv-python \
  imageio imageio-ffmpeg av

# Lightweight Wan2.1 deps needed for imports (even if WITH_WAN=0)
# - easydict: used by Wan2.1 config files
# - dashscope: imported by wan/utils/prompt_extend.py
python -m pip install --upgrade easydict dashscope

# Hugging Face CLI (huggingface-cli)
python -m pip install --upgrade "huggingface_hub[cli]"

# Tokenizer deps (CogVideoX uses SentencePiece; some conversions require protobuf).
python -m pip install --upgrade sentencepiece protobuf

# Tokenizer dependency: some HF tokenizers ship `tiktoken` BPE files.
python -m pip install --upgrade tiktoken

# 3) Reward-model helpers (DINOv2 hub load commonly needs timm)
python -m pip install --upgrade timm

# 4) CLIP (ensure `import clip` works)
# Prefer OpenAI CLIP repo; pip's `clip` packages are often unrelated.
# Install dependencies in-env first, then install CLIP itself without pulling a different torch build.
python -m pip install --upgrade ftfy wcwidth packaging regex tqdm
python -m pip install -U --force-reinstall --no-build-isolation --no-deps "git+https://github.com/openai/CLIP.git"

# 5) Install repo subproject requirements (best-effort)
if [[ -f "${REPO_DIR}/CogVideo/requirements.txt" ]]; then
  python -m pip install --upgrade -r "${REPO_DIR}/CogVideo/requirements.txt"
fi

if [[ "${WITH_WAN}" == "1" && -f "${REPO_DIR}/Wan2.1/requirements.txt" ]]; then
  # Wan2.1 uses FlashAttention for speed. This is a CUDA extension and may require
  # a matching CUDA/toolchain; install best-effort.
  python -m pip install --upgrade flash-attn --no-build-isolation || echo "WARN: flash-attn install failed (ok to ignore if you want slower SDPA attention)"
  # Wan2.1 requirements include optional heavy deps (e.g., flash_attn). If it fails,
  # rerun and install missing pieces manually for your GPU/driver.
  python -m pip install --upgrade -r "${REPO_DIR}/Wan2.1/requirements.txt"
fi

if [[ "${WITH_OPENSORA}" == "1" && -f "${REPO_DIR}/Open-Sora/requirements.txt" ]]; then
  python -m pip install --upgrade -r "${REPO_DIR}/Open-Sora/requirements.txt"
fi

echo ""
echo "✅ Done. Quick sanity checks:"
echo "  python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\""
echo "  python -c \"import clip; print('clip ok')\""
echo "  python -c \"import imageio; import av; print('video io ok')\""
echo ""
echo "Activate later with:"
echo "  source \"$(${CONDA_EXE} info --base)/etc/profile.d/conda.sh\" && conda activate \"${ENV_NAME}\""

