#!/usr/bin/env bash
set -euo pipefail

# Rename mp4 files to VideoAlign convention: {video_id}__{seed}.mp4
#
# By default:
# - video_id = current filename stem (basename without .mp4)
# - seed = $SEED (default: 42)
# - dry-run only unless APPLY=1
#
# Usage:
#   bash rename_videoalign_videos.sh /home/ubuntu/angel-research/videoalign_cogvideo_benchmark
#
# Overrides:
#   SEED=0 bash rename_videoalign_videos.sh ...
#   APPLY=1 SEED=42 bash rename_videoalign_videos.sh ...

DIR="${1:-/home/ubuntu/angel-research/videoalign_cogvideo_benchmark}"
SEED="${SEED:-42}"
APPLY="${APPLY:-0}"

if [[ ! -d "${DIR}" ]]; then
  echo "ERROR: dir not found: ${DIR}" 1>&2
  exit 1
fi

shopt -s nullglob
files=("${DIR}"/*.mp4)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .mp4 files found in: ${DIR}"
  exit 0
fi

echo "Directory: ${DIR}"
echo "Seed: ${SEED}"
if [[ "${APPLY}" == "1" ]]; then
  echo "Mode: APPLY"
else
  echo "Mode: DRY RUN (set APPLY=1 to rename)"
fi
echo ""

for src in "${files[@]}"; do
  base="$(basename "${src}")"
  stem="${base%.mp4}"

  # If already matches "*__*.mp4", skip.
  if [[ "${stem}" == *"__"* ]]; then
    echo "SKIP (already named): ${base}"
    continue
  fi

  dst="${DIR}/${stem}__${SEED}.mp4"
  dst_base="$(basename "${dst}")"

  if [[ -e "${dst}" ]]; then
    echo "ERROR: target exists, refusing to overwrite: ${dst_base}" 1>&2
    exit 1
  fi

  echo "${base}  ->  ${dst_base}"
  if [[ "${APPLY}" == "1" ]]; then
    mv -n -- "${src}" "${dst}"
  fi
done

