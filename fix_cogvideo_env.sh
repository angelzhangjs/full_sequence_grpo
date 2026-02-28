#!/usr/bin/env bash
set -euo pipefail

# Fix an existing CogVideo conda env so CogVideoX pipelines import correctly.
#
# This script intentionally does NOT touch your Open-Sora / LTX environments.
#
# Usage:
#   bash fix_cogvideo_env.sh
#
# Overrides:
#   ENV_NAME=cogvideo bash fix_cogvideo_env.sh

ENV_NAME="${ENV_NAME:-cogvideo}"

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

if ! ensure_conda_on_path; then
  die "conda not found. Common fix: export PATH=\"$HOME/anaconda3/bin:$PATH\""
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}" || die "conda env '${ENV_NAME}' not found."

echo "==> Fixing conda env '${ENV_NAME}'"

conda run -n "${ENV_NAME}" python -m pip install --no-input --upgrade \
  "diffusers>=0.35.2" \
  "peft>=0.18.0" \
  "transformers>=4.47.2,<4.52.0" \
  regex

# If Open-Sora / ColossalAI were installed here previously, remove them to avoid pin conflicts.
conda run -n "${ENV_NAME}" python -m pip uninstall -y opensora colossalai >/dev/null 2>&1 || true

echo "==> Sanity check imports"
conda run -n "${ENV_NAME}" python - <<'PY'
import diffusers, transformers, peft, regex
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("regex:", regex.__version__)
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXPipeline, CogVideoXVideoToVideoPipeline
print("CogVideoX pipelines: ok")
PY

echo ""
echo "✅ Done. Activate with:"
echo "  source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${ENV_NAME}"

