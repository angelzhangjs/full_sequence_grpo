#!/usr/bin/env bash
set -euo pipefail

# Setup script to reproduce the manual conda setup steps in a non-interactive way.
# Run from repo root: bash setup_env.sh
# Override env name if desired:
#   ENV_NAME=myenv bash setup_env.sh
#
# After this finishes, to activate the env in your current terminal:
#   source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
#   conda activate ltx-grpo

ENV_NAME="${ENV_NAME:-ltx-grpo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Torch install settings (override if needed):
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup_env.sh
#   TORCH_VERSION=2.5.1+cu121 TORCHVISION_VERSION=0.20.1+cu121 bash setup_env.sh
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"

# Ensure conda is available in non-interactive scripts
load_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  echo ">>> conda not found on PATH; trying common install locations..."

  local candidates=()
  # Allow override from environment
  if [ -n "${ANACONDA_PREFIX:-}" ]; then
    candidates+=("${ANACONDA_PREFIX}")
  fi
  candidates+=(
    "$HOME/anaconda3"
    "$HOME/miniconda3"
    "/opt/conda"
  )

  local prefix=""
  for prefix in "${candidates[@]}"; do
    if [ -f "${prefix}/etc/profile.d/conda.sh" ]; then
      echo ">>> Loading conda from: ${prefix}"
      # shellcheck disable=SC1091
      source "${prefix}/etc/profile.d/conda.sh"
      break
    fi
  done

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not installed or not discoverable." >&2
    echo "Install Anaconda first, then re-run this script:" >&2
    echo "  cd /home/ubuntu/angel-research/full_sequence_grpo" >&2
    echo "  bash install_anaconda.sh" >&2
    echo "  source ~/.bashrc" >&2
    exit 1
  fi
}

load_conda

echo ">>> Creating/updating conda env: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" || true

echo ">>> Installing conda packages"
conda install -y -n "${ENV_NAME}" pip

CONDA_RUN="conda run -n ${ENV_NAME}"

echo ">>> Upgrading pip"
$CONDA_RUN python -m pip install --upgrade pip

echo ">>> Installing PyTorch wheels"
echo "    index-url:  ${TORCH_INDEX_URL}"
echo "    torch:      ${TORCH_VERSION}"
echo "    torchvision:${TORCHVISION_VERSION}"

set +e
$CONDA_RUN python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX_URL}"
TORCH_INSTALL_RC=$?
set -e

if [ "${TORCH_INSTALL_RC}" -ne 0 ]; then
  echo ""
  echo "WARNING: Failed to install torch==${TORCH_VERSION} from ${TORCH_INDEX_URL}"
  echo "This usually means that CUDA wheel index does not host that version."
  echo "Falling back to the latest torch/torchvision pair available on cu121..."
  echo ""

  # cu121 wheel index currently provides up to torch 2.5.1+cu121.
  $CONDA_RUN python -m pip install \
    "torch==2.5.1+cu121" \
    "torchvision==0.20.1+cu121" \
    --index-url "https://download.pytorch.org/whl/cu121"
fi

echo ">>> Installing repo requirements"
# Avoid re-installing torch/torchvision from requirements.txt (they may be pinned
# to versions not present on the chosen CUDA wheel index).
REQ_TMP="$(mktemp)"
grep -vE '^(torch|torchvision)==|^torch==|^torchvision==' requirements.txt > "${REQ_TMP}"
$CONDA_RUN python -m pip install -r "${REQ_TMP}"
rm -f "${REQ_TMP}"

echo ">>> Verifying installs"
$CONDA_RUN python -c "import torch; print(f'✓ torch {torch.__version__} CUDA:{torch.cuda.is_available()}')"
$CONDA_RUN python -c "from ltx_video.ltx_video.inference import create_ltx_video_pipeline; print('✓ LTX-Video OK')"
$CONDA_RUN python -c "import google.generativeai as genai; print('✓ google-generativeai OK')"
$CONDA_RUN python -c "import clip; print('✓ CLIP OK')"
$CONDA_RUN python -c "from reward_functions import reward_function; print('✓ Reward functions OK')"

echo ">>> Environment setup complete for ${ENV_NAME}"
echo ""
echo "To activate in this terminal:"
echo "  source /home/ubuntu/anaconda3/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"

