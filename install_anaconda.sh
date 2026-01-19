#!/usr/bin/env bash
set -euo pipefail

# Install Anaconda (non-interactive) on Linux x86_64.
# Based on the manual steps you provided:
#   wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
#   bash Anaconda3-2024.02-1-Linux-x86_64.sh
#   source ~/.bashrc
#   conda --version

ANACONDA_INSTALLER_URL_DEFAULT="https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh"
ANACONDA_INSTALLER_DEFAULT="Anaconda3-2024.02-1-Linux-x86_64.sh"
ANACONDA_PREFIX_DEFAULT="$HOME/anaconda3"

INSTALLER_URL="${ANACONDA_INSTALLER_URL:-$ANACONDA_INSTALLER_URL_DEFAULT}"
INSTALLER_FILE="${ANACONDA_INSTALLER_FILE:-$ANACONDA_INSTALLER_DEFAULT}"
PREFIX="${ANACONDA_PREFIX:-$ANACONDA_PREFIX_DEFAULT}"

echo "==> Anaconda installer URL:  ${INSTALLER_URL}"
echo "==> Installer file:         ${INSTALLER_FILE}"
echo "==> Install prefix:         ${PREFIX}"
echo ""

if command -v conda >/dev/null 2>&1; then
  echo "==> conda already on PATH:"
  conda --version
  echo "If you want to reinstall anyway, set a different ANACONDA_PREFIX and re-run."
  exit 0
fi

if [[ -d "${PREFIX}" ]]; then
  echo "==> Prefix already exists at ${PREFIX}."
  echo "Remove it or set ANACONDA_PREFIX to a different path before installing."
  exit 1
fi

echo "==> Downloading installer..."
if command -v wget >/dev/null 2>&1; then
  wget -O "${INSTALLER_FILE}" "${INSTALLER_URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L "${INSTALLER_URL}" -o "${INSTALLER_FILE}"
else
  echo "ERROR: Need wget or curl to download the installer." >&2
  exit 1
fi

echo "==> Running installer (batch mode)..."
# -b: batch (no prompts), -p: install prefix
bash "${INSTALLER_FILE}" -b -p "${PREFIX}"

echo "==> Initializing conda for bash..."
"${PREFIX}/bin/conda" init bash

echo "==> Reloading shell config (~/.bashrc) for this session..."
# shellcheck disable=SC1090
source "$HOME/.bashrc" || true

echo "==> Verifying conda..."
if command -v conda >/dev/null 2>&1; then
  conda --version
  echo ""
  echo "✅ Anaconda installed successfully."
  echo "Tip: open a new terminal or run: source ~/.bashrc"
else
  echo "WARNING: conda is not on PATH yet in this shell."
  echo "Open a new terminal or run: source ~/.bashrc"
  echo "You can also use it directly: ${PREFIX}/bin/conda --version"
  exit 2
fi

