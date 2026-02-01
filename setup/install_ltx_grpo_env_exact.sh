#!/usr/bin/env bash
set -euo pipefail

# Recreate the exact ltx-grpo conda environment from conda_requirement.txt.
#
# Usage:
#   bash install_ltx_grpo_env_exact.sh [env_name]
#
# Examples:
#   bash install_ltx_grpo_env_exact.sh              # creates env 'ltx-grpo'
#   bash install_ltx_grpo_env_exact.sh ltx-grpo     # same
#   bash install_ltx_grpo_env_exact.sh ltx-grpo-new # creates env 'ltx-grpo-new'
#
# Notes:
# - This uses `conda env export` YAML (stored in conda_requirement.txt), including pip deps.
# - It assumes conda is installed at /home/ubuntu/anaconda3. Override CONDA_EXE if needed.

ENV_NAME="${1:-ltx-grpo}"
ROOT_DIR="/home/ubuntu/angel-research/full_sequence_grpo"
ENV_FILE="${ROOT_DIR}/conda_requirement.txt"

CONDA_EXE="${CONDA_EXE:-/home/ubuntu/anaconda3/bin/conda}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "ERROR: conda not found at ${CONDA_EXE}."
  echo "Set CONDA_EXE=/path/to/conda or install conda first."
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found."
  exit 1
fi

echo "[1/3] Using env file: ${ENV_FILE}"
echo "[2/3] Creating/updating env: ${ENV_NAME}"

# If env exists, remove it to guarantee an exact recreation.
if "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Env '${ENV_NAME}' already exists; removing it to recreate exactly..."
  "${CONDA_EXE}" env remove -n "${ENV_NAME}" -y
fi

# Create env from exported YAML; override name for convenience.
"${CONDA_EXE}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"

echo "[3/3] Done."
echo "To activate:"
echo "  source \"$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh\" && conda activate \"${ENV_NAME}\""

