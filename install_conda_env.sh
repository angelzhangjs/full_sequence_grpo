#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash install_conda_env.sh [env_name] [python_version]
# Defaults: env_name="myenv", python_version="3.11"
# MINICONDA_DIR can be overridden via env var (default: "$HOME/miniconda").

ENV_NAME="${1:-myenv}"
PY_VERSION="${2:-3.11}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda}"
MINICONDA_SH="${MINICONDA_SH:-$HOME/miniconda.sh}"

echo "Target env: ${ENV_NAME}"
echo "Python version: ${PY_VERSION}"
echo "Miniconda dir: ${MINICONDA_DIR}"

# Install Miniconda if missing.
if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
  echo "Miniconda not found, installing to ${MINICONDA_DIR}..."
  curl -fsSLo "${MINICONDA_SH}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "${MINICONDA_SH}" -b -p "${MINICONDA_DIR}"
  rm -f "${MINICONDA_SH}"
else
  echo "Miniconda already installed at ${MINICONDA_DIR}"
fi

# Make conda available in this shell.
if ! command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
fi

echo "Creating environment ${ENV_NAME} with Python ${PY_VERSION} (idempotent)..."
conda create -y -n "${ENV_NAME}" python="${PY_VERSION}" >/dev/null 2>&1 || true
conda activate "${ENV_NAME}"

echo "Environment ready. To use later, run:"
echo "  source \"${MINICONDA_DIR}/etc/profile.d/conda.sh\" && conda activate \"${ENV_NAME}\""


