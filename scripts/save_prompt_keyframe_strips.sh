#!/usr/bin/env bash
# After baseline + GRPO mp4 exist under a batch prompt folder, write two PNG strips:
#   <prompt_dir>/baseline_keyframes.png
#   <prompt_dir>/grpo_keyframes.png
# Each strip tiles K frames at aligned timestamps (default K=2).
#
# Usage:
#   save_prompt_keyframe_strips.sh <prompt_output_dir> [num_frames]
#
# Skips quietly if videos or compare script are missing. Intended for batch wrappers (|| true).

set -euo pipefail

OD="${1:?usage: $0 <prompt_output_dir> [num_frames]}"
K="${2:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPARE_SH="${REPO_ROOT}/scripts/compare_two_videos_keyframes.sh"

BASE="${OD}/baseline/baseline.mp4"
GRPO_V=""
for v in "${OD}/grpo"/*_grpo.mp4; do
  [[ -f "$v" ]] || continue
  GRPO_V="$v"
  break
done

[[ -f "${COMPARE_SH}" ]] || exit 0
[[ -f "${BASE}" && -n "${GRPO_V}" ]] || exit 0

bash "${COMPARE_SH}" "${BASE}" "${GRPO_V}" --out-dir "${OD}" --frames "${K}" --separate
