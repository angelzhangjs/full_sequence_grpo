#!/usr/bin/env bash
set -euo pipefail

# HunyuanVideo-1.5: generate a video for every prompt in origin_grpo/physical_plausibility.txt
# Skips blank lines and lines starting with '#'.
#
# Uses HunyuanVideo/sample_video.py (argparse via hyvideo.config.parse_args).
#
# You MUST set HUANYUAN_MODEL_BASE to the folder that contains the HunyuanVideo weights.
# In sample_video.py it is read as: --model-base /path/to/models_root
#
# Example:
#   export HUANYUAN_MODEL_BASE="/abs/path/to/HunyuanVideo/models"
#   ./run_hunyuan15_physical_plausibility.sh

cd /home/ubuntu/angel-research/full_sequence_grpo

PROMPT_FILE="origin_grpo/physical_plausibility.txt"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="hunyuan15_outputs/physical_plausibility_${RUN_ID}"
mkdir -p "${OUT_DIR}"

: "${HUANYUAN_MODEL_BASE:=}"

# -------- Config (override via env vars) --------
: "${HUANYUAN_VIDEO_H:=720}"
: "${HUANYUAN_VIDEO_W:=1280}"
: "${HUANYUAN_VIDEO_LENGTH:=129}"
: "${HUANYUAN_INFER_STEPS:=50}"
: "${HUANYUAN_SEED:=26}"
: "${HUANYUAN_CFG_SCALE:=}"              # if empty, sample_video.py default applies
: "${HUANYUAN_EMBEDDED_CFG_SCALE:=6.0}"
: "${HUANYUAN_FLOW_SHIFT:=7.0}"
: "${HUANYUAN_EXTRA_FLAGS:=--flow-reverse --use-cpu-offload}"

if [[ -z "${HUANYUAN_MODEL_BASE}" ]]; then
  echo "ERROR: HUANYUAN_MODEL_BASE is not set."
  echo "Set it to the folder containing HunyuanVideo weights, e.g.:"
  echo "  export HUANYUAN_MODEL_BASE=\"/abs/path/to/HunyuanVideo/models\""
  exit 2
fi

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

  echo
  echo "[${i}] ${prompt}"

  # sample_video.py writes mp4s into --save-path; we isolate each prompt in its own folder.
  CMD=(python HunyuanVideo/sample_video.py
    --model-base "${HUANYUAN_MODEL_BASE}"
    --save-path "${PDIR}"
    --prompt "${prompt}"
    --video-size "${HUANYUAN_VIDEO_H}" "${HUANYUAN_VIDEO_W}"
    --video-length "${HUANYUAN_VIDEO_LENGTH}"
    --infer-steps "${HUANYUAN_INFER_STEPS}"
    --seed "${HUANYUAN_SEED}"
    --embedded-cfg-scale "${HUANYUAN_EMBEDDED_CFG_SCALE}"
    --flow-shift "${HUANYUAN_FLOW_SHIFT}"
  )

  if [[ -n "${HUANYUAN_CFG_SCALE}" ]]; then
    CMD+=(--cfg-scale "${HUANYUAN_CFG_SCALE}")
  fi

  # shellcheck disable=SC2206
  CMD+=(${HUANYUAN_EXTRA_FLAGS})

  "${CMD[@]}"

  # Convenience: copy the latest mp4 in the folder to final.mp4
  latest_mp4="$(ls -1t "${PDIR}"/*.mp4 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_mp4}" ]]; then
    cp -f "${latest_mp4}" "${PDIR}/final.mp4"
    echo "Saved: ${PDIR}/final.mp4"
  else
    echo "WARN: no mp4 found under ${PDIR}/*.mp4 (check Hunyuan logs above)"
  fi

  i=$((i + 1))
done < "${PROMPT_FILE}"

echo
echo "Done. Outputs in: ${OUT_DIR}"

