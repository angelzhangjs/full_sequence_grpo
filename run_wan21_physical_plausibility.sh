#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/angel-research/full_sequence_grpo

# REQUIRED: set to your downloaded checkpoint folder.
# Example:
#   export CKPT_DIR="/path/to/Wan2.1-T2V-1.3B"
: "${CKPT_DIR:=}"

# Prompts (one per line; blank lines and lines starting with '#' are ignored)
PROMPT_FILE="origin_grpo/physical_plausibility.txt"

# Output
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="wan21_outputs/physical_plausibility_${RUN_ID}"
mkdir -p "${OUT_DIR}"

# Wan2.1 args (override via env vars)
: "${TASK:=t2v-1.3B}"
: "${SIZE:=832*480}"
: "${FRAME_NUM:=81}"            # should be 4n+1
: "${SAMPLE_SOLVER:=unipc}"     # unipc | dpm++
: "${SAMPLE_STEPS:=50}"
: "${SAMPLE_SHIFT:=8}"         # suggested 8-12 for t2v-1.3B
: "${GUIDE_SCALE:=6}"          # suggested 6 for t2v-1.3B
: "${BASE_SEED:=26}"
: "${OFFLOAD_MODEL:=True}"
: "${USE_T5_CPU:=1}"           # 1 => pass --t5_cpu, 0 => don't

if [[ -z "${CKPT_DIR}" ]]; then
  echo "ERROR: CKPT_DIR is not set."
  echo "Set it to your Wan2.1 checkpoint folder, e.g.:"
  echo "  export CKPT_DIR=\"/path/to/Wan2.1-T2V-1.3B\""
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

  SAVE_FILE="${PDIR}/final.mp4"

  echo
  echo "[${i}] ${prompt}"

  if [[ "${USE_T5_CPU}" == "1" ]]; then
    python Wan2.1/generate.py \
      --task "${TASK}" \
      --size "${SIZE}" \
      --frame_num "${FRAME_NUM}" \
      --ckpt_dir "${CKPT_DIR}" \
      --offload_model "${OFFLOAD_MODEL}" \
      --t5_cpu \
      --sample_solver "${SAMPLE_SOLVER}" \
      --sample_steps "${SAMPLE_STEPS}" \
      --sample_shift "${SAMPLE_SHIFT}" \
      --sample_guide_scale "${GUIDE_SCALE}" \
      --base_seed "${BASE_SEED}" \
      --prompt "${prompt}" \
      --save_file "${SAVE_FILE}"
  else
    python Wan2.1/generate.py \
      --task "${TASK}" \
      --size "${SIZE}" \
      --frame_num "${FRAME_NUM}" \
      --ckpt_dir "${CKPT_DIR}" \
      --offload_model "${OFFLOAD_MODEL}" \
      --sample_solver "${SAMPLE_SOLVER}" \
      --sample_steps "${SAMPLE_STEPS}" \
      --sample_shift "${SAMPLE_SHIFT}" \
      --sample_guide_scale "${GUIDE_SCALE}" \
      --base_seed "${BASE_SEED}" \
      --prompt "${prompt}" \
      --save_file "${SAVE_FILE}"
  fi

  i=$((i + 1))
done < "${PROMPT_FILE}"

echo
echo "Done. Outputs in: ${OUT_DIR}"

