#!/usr/bin/env bash
set -euo pipefail

# Recreate the current "ltx" conda environment packages in a reproducible bash script.
#
# This script mirrors the live `ltx` environment as closely as possible:
# - creates/reuses a conda env
# - installs the base conda packages exported from that env
# - installs the pip packages with the same pinned versions
# - reinstalls the local editable packages for this repo
#
# Usage:
#   bash setup/ltx.sh
#
# Common overrides:
#   ENV_NAME=ltx bash setup/ltx.sh
#   RECREATE=1 bash setup/ltx.sh
#   DOWNLOAD_ASSETS=1 bash setup/ltx.sh

REPO_ROOT="/home/ubuntu/angel-research"
repo_root="/home/ubuntu/angel-neurips"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "${repo_root}"

ENV_NAME="${ENV_NAME:-ltx}"
RECREATE="${RECREATE:-0}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"
CONDA_CHANNEL="${CONDA_CHANNEL:-defaults}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl#sha256=c59be18fa934e132e5a405bcca6673af1d9d09a0036a9e081133bc7b7fc2992e}"

info() { echo -e "\n==> $*\n"; }
die() { echo "ERROR: $*" 1>&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

ensure_conda_on_path() {
  if have conda; then
    return 0
  fi
  local candidates=(
    "${HOME}/anaconda3/bin/conda"
    "${HOME}/miniconda3/bin/conda"
    "/opt/conda/bin/conda"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -x "${c}" ]]; then
      export PATH="$(dirname "${c}"):${PATH}"
      return 0
    fi
  done
  return 1
}

activate_conda() {
  ensure_conda_on_path || die "conda not found on PATH"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

create_or_replace_env() {
  local conda_pkgs=(
    "python=3.10.20"
    "pip=26.0.1"
    "wheel=0.46.3"
    "setuptools=80.10.2"
    "packaging=25.0"
    "bzip2=1.0.8"
    "ca-certificates=2025.12.2"
    "ld_impl_linux-64=2.44"
    "libexpat=2.7.4"
    "libffi=3.4.4"
    "libgcc=15.2.0"
    "libgcc-ng=15.2.0"
    "libgomp=15.2.0"
    "libnsl=2.0.0"
    "libstdcxx=15.2.0"
    "libstdcxx-ng=15.2.0"
    "libuuid=1.41.5"
    "libxcb=1.17.0"
    "libzlib=1.3.1"
    "ncurses=6.5"
    "openssl=3.5.5"
    "pthread-stubs=0.3"
    "readline=8.3"
    "sqlite=3.51.2"
    "tk=8.6.15"
    "tzdata=2026a"
    "xorg-libx11=1.8.12"
    "xorg-libxau=1.0.12"
    "xorg-libxdmcp=1.1.5"
    "xorg-xorgproto=2024.1"
    "xz=5.8.2"
    "zlib=1.3.1"
  )

  if conda_env_exists; then
    if [[ "${RECREATE}" == "1" || "${RECREATE}" == "true" ]]; then
      info "Removing existing env '${ENV_NAME}'"
      conda env remove -n "${ENV_NAME}" -y
    else
      info "Env '${ENV_NAME}' already exists; reusing it"
      return 0
    fi
  fi

  info "Creating env '${ENV_NAME}'"
  conda create -y -n "${ENV_NAME}" -c "${CONDA_CHANNEL}" "${conda_pkgs[@]}"
}

install_python_packages() {
  info "Activating env '${ENV_NAME}'"
  conda activate "${ENV_NAME}"

  info "Installing PyTorch from ${TORCH_INDEX_URL}"
  python -m pip install --no-input --index-url "${TORCH_INDEX_URL}" \
    "torch==2.5.1+cu121" \
    "torchvision==0.20.1+cu121"

  info "Installing FlashAttention wheel"
  python -m pip install --no-input "${FLASH_ATTN_WHEEL}"

  info "Installing OpenAI CLIP"
  python -m pip install --no-input --no-build-isolation \
    "git+https://github.com/openai/CLIP.git@ded190a052fdf4585bd685cee5bc96e0310d2c93"

  info "Installing pinned pip packages"
  python -m pip install --no-input \
    "accelerate==1.13.0" \
    "deepspeed" \
    "anyio==4.12.1" \
    "av==17.0.0" \
    "certifi==2026.2.25" \
    "charset-normalizer==3.4.6" \
    "diffusers==0.37.0" \
    "einops==0.8.2" \
    "exceptiongroup==1.3.1" \
    "filelock==3.20.0" \
    "fsspec==2025.12.0" \
    "ftfy==6.3.1" \
    "h11==0.16.0" \
    "hf-xet==1.4.2" \
    "httpcore==1.0.9" \
    "httpx==0.28.1" \
    "huggingface-hub==0.36.2" \
    "idna==3.11" \
    "imageio==2.37.3" \
    "imageio-ffmpeg==0.6.0" \
    "importlib-metadata==8.7.1" \
    "jinja2==3.1.6" \
    "markupsafe==3.0.2" \
    "mpmath==1.3.0" \
    "networkx==3.4.2" \
    "numpy==2.2.6" \
    "nvidia-cublas-cu12==12.1.3.1" \
    "nvidia-cuda-cupti-cu12==12.1.105" \
    "nvidia-cuda-nvrtc-cu12==12.1.105" \
    "nvidia-cuda-runtime-cu12==12.1.105" \
    "nvidia-cudnn-cu12==9.1.0.70" \
    "nvidia-cufft-cu12==11.0.2.54" \
    "nvidia-curand-cu12==10.3.2.106" \
    "nvidia-cusolver-cu12==11.4.5.107" \
    "nvidia-cusparse-cu12==12.1.0.106" \
    "nvidia-nccl-cu12==2.21.5" \
    "nvidia-nvjitlink-cu12==12.9.86" \
    "nvidia-nvtx-cu12==12.1.105" \
    "opencv-python==4.13.0.92" \
    "peft==0.18.1" \
    "pillow==12.0.0" \
    "psutil==7.2.2" \
    "pyyaml==6.0.3" \
    "regex==2026.2.28" \
    "requests==2.32.5" \
    "safetensors==0.7.0" \
    "scipy==1.15.3" \
    "sentencepiece==0.2.1" \
    "sympy==1.13.1" \
    "timm==1.0.25" \
    "tokenizers==0.21.4" \
    "tqdm==4.67.3" \
    "transformers==4.51.3" \
    "triton==3.1.0" \
    "typing-extensions==4.15.0" \
    "urllib3==2.6.3" \
    "wcwidth==0.6.0" \
    "zipp==3.23.0"

  info "Installing local editable packages from this repo"
  python -m pip install --no-input -e "${REPO_ROOT}/ltx_video[inference]" --no-deps
  python -m pip install --no-input -e "${REPO_ROOT}" --no-deps
}

sanity_check() {
  info "Sanity check"
  conda activate "${ENV_NAME}"
  python - <<'PY'
import sys
import torch
import diffusers
import transformers
import accelerate
import clip
import ltx_video
import unified_grpo

print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("clip:", getattr(clip, "__version__", "ok"))
print("ltx_video: ok")
print("unified_grpo: ok")
PY
}

maybe_download_assets() {
  if [[ "${DOWNLOAD_ASSETS}" == "1" || "${DOWNLOAD_ASSETS}" == "true" ]]; then
    info "Pre-downloading LTX model assets"
    bash "${REPO_ROOT}/setup/download_ltx_assets.sh"
  fi
}

main() {
  activate_conda

  info "Config"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "ENV_NAME=${ENV_NAME}"
  echo "RECREATE=${RECREATE}"
  echo "DOWNLOAD_ASSETS=${DOWNLOAD_ASSETS}"
  echo "TORCH_INDEX_URL=${TORCH_INDEX_URL}"

  create_or_replace_env
  install_python_packages
  sanity_check
  maybe_download_assets

  info "Done"
  echo "Activate with: conda activate ${ENV_NAME}"
}

main "$@"
