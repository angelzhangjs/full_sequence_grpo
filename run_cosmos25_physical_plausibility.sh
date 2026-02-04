#!/usr/bin/env bash
set -euo pipefail

# Cosmos-Predict-2.5: generate a video for every prompt in origin_grpo/physical_plausibility.txt
# Skips blank lines and lines starting with '#'.
#
# Uses cosmos-predict2.5/scripts/diffusers_inference.py (tyro CLI).
#
# Default model_id/revision match the script defaults:
#   model_id = nvidia/Cosmos-Predict2.5-2B
#   revision = diffusers/base/post-trained
#
# Example:
#   ./run_cosmos25_physical_plausibility.sh
#   COSMOS_MODEL_ID="nvidia/Cosmos-Predict2.5-14B" ./run_cosmos25_physical_plausibility.sh

cd /home/ubuntu/angel-research/full_sequence_grpo

PROMPT_FILE="origin_grpo/physical_plausibility.txt"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="cosmos25_outputs/physical_plausibility_${RUN_ID}"
mkdir -p "${OUT_DIR}"

# -------- Config (override via env vars) --------
: "${COSMOS_MODEL_ID:=nvidia/Cosmos-Predict2.5-2B}"
: "${COSMOS_REVISION:=diffusers/base/post-trained}"
: "${COSMOS_NUM_FRAMES:=93}"     # cosmos script default (2Video)
: "${COSMOS_NUM_STEPS:=36}"
: "${COSMOS_SEED:=26}"
: "${COSMOS_DEVICE:=cuda}"

# Negative prompt: if you want to override, set COSMOS_NEGATIVE_PROMPT (otherwise script default is used)
: "${COSMOS_NEGATIVE_PROMPT:=}"

i=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  prompt="$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "${prompt}" ]] && continue
  [[ "${prompt}" == \#* ]] && continue

  slug="$(echo "${prompt}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' \
    | cut -c1-64)"
  [[ -z "${slug}" ]] && slug="prompt"

  PDIR="${OUT_DIR}/p$(printf '%03d' "${i}")_${slug}"
  mkdir -p "${PDIR}"
  printf "%s\n" "${prompt}" > "${PDIR}/prompt.txt"

  OUT_MP4="${PDIR}/final.mp4"

  echo
  echo "[${i}] ${prompt}"
  echo "  -> ${OUT_MP4}"

  if [[ -n "${COSMOS_NEGATIVE_PROMPT}" ]]; then
    python cosmos-predict2.5/scripts/diffusers_inference.py \
      --output-path "${OUT_MP4}" \
      --model-id "${COSMOS_MODEL_ID}" \
      --revision "${COSMOS_REVISION}" \
      --prompt "${prompt}" \
      --negative-prompt "${COSMOS_NEGATIVE_PROMPT}" \
      --num-output-frames "${COSMOS_NUM_FRAMES}" \
      --num-steps "${COSMOS_NUM_STEPS}" \
      --seed "${COSMOS_SEED}" \
      --device "${COSMOS_DEVICE}"
  else
    python cosmos-predict2.5/scripts/diffusers_inference.py \
      --output-path "${OUT_MP4}" \
      --model-id "${COSMOS_MODEL_ID}" \
      --revision "${COSMOS_REVISION}" \
      --prompt "${prompt}" \
      --num-output-frames "${COSMOS_NUM_FRAMES}" \
      --num-steps "${COSMOS_NUM_STEPS}" \
      --seed "${COSMOS_SEED}" \
      --device "${COSMOS_DEVICE}"
  fi

  i=$((i + 1))
done < "${PROMPT_FILE}"

echo
echo "Done. Outputs in: ${OUT_DIR}"

