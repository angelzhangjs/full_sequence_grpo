#!/usr/bin/env bash
set -euo pipefail

# Run CogVideoX (diffusers) through every prompt in origin_grpo/physical_plausibility.txt.
# Skips blank lines and lines starting with '#'.
#
# Uses CogVideo's provided inference entrypoint:
#   CogVideo/inference/cli_demo.py
#
# By default this targets the HF model repo "zai-org/CogVideoX-2b":
#   https://huggingface.co/zai-org/CogVideoX-2b
#
# If you prefer THUDM weights, override MODEL_PATH accordingly (e.g., THUDM/CogVideoX-2b).
#
# Example:
#   MODEL_PATH="zai-org/CogVideoX-2b" ./origin_grpo/run_cogvideox_physical_plausibility.sh

cd /home/ubuntu/angel-research/full_sequence_grpo

PROMPT_FILE="origin_grpo/physical_plausibility.txt"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="cogvideox_outputs/physical_plausibility_${RUN_ID}"
mkdir -p "${OUT_DIR}"

# -------- Config (override via env vars) --------
: "${MODEL_PATH:=zai-org/CogVideoX-2b}"
: "${GENERATE_TYPE:=t2v}"
: "${NUM_FRAMES:=81}"
: "${FPS:=16}"
: "${NUM_INFERENCE_STEPS:=50}"
: "${GUIDANCE_SCALE:=6.0}"
: "${SEED:=26}"

echo "CogVideoX runner"
echo "  prompts: ${PROMPT_FILE}"
echo "  out:     ${OUT_DIR}"
echo "  model:   ${MODEL_PATH}"
echo "  type:    ${GENERATE_TYPE}"
echo "  frames:  ${NUM_FRAMES} @ ${FPS} fps"
echo "  steps:   ${NUM_INFERENCE_STEPS}"
echo "  cfg:     ${GUIDANCE_SCALE}"
echo "  seed:    ${SEED}"
echo

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

  SAVE_FILE="${PDIR}/final.mp4"

  echo
  echo "[${i}] ${prompt}"

  python CogVideo/inference/cli_demo.py \
    --prompt "${prompt}" \
    --model_path "${MODEL_PATH}" \
    --generate_type "${GENERATE_TYPE}" \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --output_path "${SAVE_FILE}"

  i=$((i + 1))
done < "${PROMPT_FILE}"

echo
echo "Done. Outputs in: ${OUT_DIR}"

