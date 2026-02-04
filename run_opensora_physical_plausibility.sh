#!/usr/bin/env bash
set -euo pipefail

# Run Open-Sora (scripts/diffusion/inference.py) over every prompt in:
#   origin_grpo/physical_plausibility.txt
#
# Lines that are blank or start with '#' are ignored.
#
# NOTE: Open-Sora requires model/AE/text encoder checkpoints. This script expects you to
# provide local paths (or HF repo IDs if Open-Sora supports them in your env).
#
# Minimal example (you must set the checkpoint paths that exist on your machine):
#   export OPENSORA_MODEL_CKPT="/abs/path/to/Open_Sora_v2.safetensors"
#   export OPENSORA_AE_CKPT="/abs/path/to/hunyuan_vae.safetensors"
#   export OPENSORA_T5_DIR="/abs/path/to/t5-v1_1-xxl"
#   export OPENSORA_CLIP_DIR="/abs/path/to/clip-vit-large-patch14"
#   ./run_opensora_physical_plausibility.sh

cd /home/ubuntu/angel-research/full_sequence_grpo

PROMPT_FILE="origin_grpo/physical_plausibility.txt"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="opensora_outputs/physical_plausibility_${RUN_ID}"
mkdir -p "${OUT_DIR}"

# ---------------- Config (override via env vars) ----------------
# Open-Sora uses a positional CONFIG argument + "--key value" overrides.
: "${OPENSORA_CONFIG:=Open-Sora/configs/diffusion/inference/256px.py}"

# Sampling overrides (these map to cfg.sampling_option.* and cfg.fps_save, cfg.seed)
: "${RESOLUTION:=256px}"     # 256px or 768px (should match config family)
: "${ASPECT_RATIO:=16:9}"    # 16:9 | 9:16 | 1:1
: "${NUM_FRAMES:=81}"
: "${NUM_STEPS:=50}"
: "${GUIDANCE:=7.5}"
: "${FPS_SAVE:=16}"
: "${SEED:=26}"

# Checkpoint / component paths (REQUIRED unless your config already points to valid local paths)
: "${OPENSORA_MODEL_CKPT:=}"
: "${OPENSORA_AE_CKPT:=}"
: "${OPENSORA_T5_DIR:=}"
: "${OPENSORA_CLIP_DIR:=}"

if [[ -z "${OPENSORA_MODEL_CKPT}" || -z "${OPENSORA_AE_CKPT}" || -z "${OPENSORA_T5_DIR}" || -z "${OPENSORA_CLIP_DIR}" ]]; then
  echo "ERROR: Missing Open-Sora checkpoints."
  echo "Please export these env vars to valid paths:"
  echo "  OPENSORA_MODEL_CKPT=/abs/path/to/Open_Sora_v2.safetensors"
  echo "  OPENSORA_AE_CKPT=/abs/path/to/hunyuan_vae.safetensors"
  echo "  OPENSORA_T5_DIR=/abs/path/to/t5-v1_1-xxl"
  echo "  OPENSORA_CLIP_DIR=/abs/path/to/clip-vit-large-patch14"
  exit 2
fi

echo "Open-Sora runner"
echo "  prompts:   ${PROMPT_FILE}"
echo "  out:       ${OUT_DIR}"
echo "  config:    ${OPENSORA_CONFIG}"
echo "  res/ar:    ${RESOLUTION} @ ${ASPECT_RATIO}"
echo "  frames:    ${NUM_FRAMES}"
echo "  steps:     ${NUM_STEPS}"
echo "  guidance:  ${GUIDANCE}"
echo "  fps_save:  ${FPS_SAVE}"
echo "  seed:      ${SEED}"
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

  echo
  echo "[${i}] ${prompt}"

  # Open-Sora writes outputs under:
  #   ${save_dir}/video_${resolution}/*.mp4
  python Open-Sora/scripts/diffusion/inference.py "${OPENSORA_CONFIG}" \
    --prompt "${prompt}" \
    --save_dir "${PDIR}" \
    --seed "${SEED}" \
    --fps_save "${FPS_SAVE}" \
    --sampling_option.resolution "${RESOLUTION}" \
    --sampling_option.aspect_ratio "${ASPECT_RATIO}" \
    --sampling_option.num_frames "${NUM_FRAMES}" \
    --sampling_option.num_steps "${NUM_STEPS}" \
    --sampling_option.guidance "${GUIDANCE}" \
    --model.from_pretrained "${OPENSORA_MODEL_CKPT}" \
    --ae.from_pretrained "${OPENSORA_AE_CKPT}" \
    --t5.from_pretrained "${OPENSORA_T5_DIR}" \
    --clip.from_pretrained "${OPENSORA_CLIP_DIR}"

  # Convenience: copy the first produced mp4 to ${PDIR}/final.mp4 if present.
  first_mp4="$(ls -1 "${PDIR}"/video_*/*.mp4 2>/dev/null | head -n 1 || true)"
  if [[ -n "${first_mp4}" ]]; then
    cp -f "${first_mp4}" "${PDIR}/final.mp4"
    echo "Saved: ${PDIR}/final.mp4"
  else
    echo "WARN: no mp4 found under ${PDIR}/video_*/*.mp4 (check Open-Sora logs above)"
  fi

  i=$((i + 1))
done < "${PROMPT_FILE}"

echo
echo "Done. Outputs in: ${OUT_DIR}"

