#!/usr/bin/env bash
set -euo pipefail

# Run baseline (no-GRPO) inference for every non-empty line in prompt.txt.
#
# Usage:
#   bash run_baseline_prompts.sh
#
# Optional overrides:
#   PROMPT_FILE=prompt.txt bash run_baseline_prompts.sh
#   OUT_ROOT=baseline_batch_$(date +%Y%m%d_%H%M%S) bash run_baseline_prompts.sh
#   PIPELINE_CONFIG=ltx_video/configs/ltxv-2b-0.9.8-distilled.yaml bash run_baseline_prompts.sh
#   PYTHON=/home/ubuntu/anaconda3/envs/ltx-grpo/bin/python bash run_baseline_prompts.sh

cd /home/ubuntu/angel-research/full_sequence_grpo

PROMPT_FILE="${PROMPT_FILE:-prompt.txt}"
# Match pipeline.py defaults:
#   config_path = "configs/ltxv-2b-0.9.5.yaml"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-ltx_video/configs/ltxv-2b-0.9.5.yaml}"
PYTHON="${PYTHON:-python}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
# Save all baseline outputs under baseline/
OUT_ROOT="${OUT_ROOT:-baseline/baseline_prompts_${RUN_ID}}"

# Match pipeline.py defaults:
#   SEED=2026, height=512, width=768, num_frames=81, frame_rate=16
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-768}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FRAME_RATE="${FRAME_RATE:-16}"
SEED="${SEED:-2026}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "ERROR: prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PIPELINE_CONFIG}" ]]; then
  echo "ERROR: pipeline config not found: ${PIPELINE_CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "Prompt file:    ${PROMPT_FILE}"
echo "Config:         ${PIPELINE_CONFIG}"
echo "Output root:    ${OUT_ROOT}"
echo "Params:         ${WIDTH}x${HEIGHT} frames=${NUM_FRAMES} fps=${FRAME_RATE} seed=${SEED}"
echo ""

idx=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  prompt="$(echo "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "${prompt}" ]] && continue

  slug="$(echo "${prompt}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g' | cut -c1-48)"
  [[ -z "${slug}" ]] && slug="prompt"

  out_dir="${OUT_ROOT}/p$(printf '%03d' "${idx}")_${slug}"
  mkdir -p "${out_dir}"
  printf "%s\n" "${prompt}" > "${out_dir}/prompt.txt"

  echo "================================================================================"
  echo "[${idx}] Baseline inference"
  echo "Prompt: ${prompt}"
  echo "Out:    ${out_dir}"
  echo "================================================================================"

  "${PYTHON}" ltx_video/run_inference.py \
    --pipeline_config "${PIPELINE_CONFIG}" \
    --prompt "${prompt}" \
    --output_path "${out_dir}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --frame_rate "${FRAME_RATE}" \
    --seed "${SEED}"

  idx=$((idx + 1))
done < "${PROMPT_FILE}"

echo ""
echo "✅ Done. Baseline outputs saved under: ${OUT_ROOT}"

