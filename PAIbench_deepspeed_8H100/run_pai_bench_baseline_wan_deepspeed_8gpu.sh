#!/bin/bash
# PAI-bench baseline video runner for Wan2.1.
# Reads a TSV file with columns: video_id, prompt_en
# Exports baseline videos as <video_id>__<seed>.mp4 under OUTPUT_ROOT/videos.
#
# Note: despite the naming symmetry with the GRPO scripts, baseline generation itself
# uses Wan's native generation entrypoint rather than Accelerate/DeepSpeed training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Wan2.1:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PROMPTS_FILE="${PROMPTS_FILE:-/home/ubuntu/angel-neurips/full_sequence_grpo/cosmos_predict2_bench_video_prompts.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/wan_pai_baseline}"
VIDEOS_DIR="${OUTPUT_ROOT}/videos"
RUNS_DIR="${OUTPUT_ROOT}/runs"
MANIFEST_PATH="${OUTPUT_ROOT}/prompt_manifest.tsv"

MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B}"
WAN_TASK="${WAN_TASK:-t2v-1.3B}"
WAN_SIZE="${WAN_SIZE:-832*480}"
NUM_FRAMES="${NUM_FRAMES:-33}"
SEED="${SEED:-42}"
TARGET_DURATION_S="${TARGET_DURATION_S:-4}"
SAVE_SNAPSHOTS="${SAVE_SNAPSHOTS:-0}"
VIDEO_SNAPSHOT_SH="${REPO_ROOT}/scripts/video_to_snapshot.sh"

if [[ "${PROMPTS_FILE}" != /* ]]; then
    PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
fi

if [[ ! -f "${PROMPTS_FILE}" ]]; then
    echo "ERROR: Prompt TSV not found: ${PROMPTS_FILE}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${VIDEOS_DIR}" "${RUNS_DIR}"
printf "video_id\tprompt_en\tvideo_path\n" > "${MANIFEST_PATH}"

TOTAL_PROMPTS=$(python3 - <<'PY' "${PROMPTS_FILE}"
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    print(sum(1 for _ in reader))
PY
)

echo "======================================================================"
echo "PAI-BENCH WAN BASELINE"
echo "======================================================================"
echo "Prompts TSV:       ${PROMPTS_FILE}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "Videos output dir: ${VIDEOS_DIR}"
echo "Total prompts:     ${TOTAL_PROMPTS}"
echo ""

WAN_CKPT_DIR="${WAN_CKPT_DIR:-}"
if [[ -z "$WAN_CKPT_DIR" ]]; then
  if [[ -d "$MODEL_PATH" ]]; then
    WAN_CKPT_DIR="$MODEL_PATH"
  else
    echo "Downloading Wan checkpoint from HuggingFace: $MODEL_PATH ..."
    WAN_CKPT_DIR=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${MODEL_PATH}'))
")
    export WAN_CKPT_DIR
    echo "  -> $WAN_CKPT_DIR"
  fi
fi

AUTO_YES="${AUTO_YES:-0}"
if [[ "${AUTO_YES}" == "1" || "${AUTO_YES}" == "true" || ! -t 0 ]]; then
    echo "Non-interactive run detected (or AUTO_YES set). Continuing without prompt."
else
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

PROMPT_IDX=0
while IFS=$'\t' read -r VIDEO_ID PROMPT_EN _REST || [[ -n "${VIDEO_ID:-}" ]]; do
    if [[ "${VIDEO_ID}" == "video_id" ]]; then
        continue
    fi
    if [[ -z "${VIDEO_ID//[[:space:]]/}" ]]; then
        continue
    fi

    PROMPT_IDX=$((PROMPT_IDX + 1))
    SAFE_VIDEO_ID="$(printf '%s' "${VIDEO_ID}" | tr '/:' '__')"
    SAMPLE_DIR="${RUNS_DIR}/${SAFE_VIDEO_ID}"
    BASELINE_DIR="${SAMPLE_DIR}/baseline"
    mkdir -p "${BASELINE_DIR}"
    printf "%s\n" "${PROMPT_EN}" > "${SAMPLE_DIR}/prompt.txt"
    printf "%s\n" "${VIDEO_ID}" > "${SAMPLE_DIR}/video_id.txt"

    echo ""
    echo "======================================================================"
    echo "Sample ${PROMPT_IDX}/${TOTAL_PROMPTS}"
    echo "video_id: ${VIDEO_ID}"
    echo "======================================================================"
    echo "${PROMPT_EN}"
    echo ""
    echo "Step 1/1: Generating Wan baseline..."

    (cd "${REPO_ROOT}/Wan2.1" && python generate.py \
        --task "${WAN_TASK}" \
        --ckpt_dir "${WAN_CKPT_DIR}" \
        --prompt "${PROMPT_EN}" \
        --size "${WAN_SIZE}" \
        --frame_num "${NUM_FRAMES}" \
        --save_file "${BASELINE_DIR}/baseline.mp4" \
        --base_seed "${SEED}") || {
        echo "WARNING: Baseline failed for video_id=${VIDEO_ID}, continuing..."
        continue
    }

    if [[ -f "${BASELINE_DIR}/baseline.mp4" ]] && command -v ffmpeg >/dev/null 2>&1; then
        in_file="${BASELINE_DIR}/baseline.mp4"
        in_dur="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${in_file}" 2>/dev/null || true)"
        if [[ -n "${in_dur}" ]]; then
            read -r PTS_SCALE TARGET_FPS < <(python3 - <<PY
dur=float("${in_dur}")
target=float("${TARGET_DURATION_S}")
frames=int("${NUM_FRAMES}")
print(target/dur, frames/target)
PY
)
        else
            PTS_SCALE="1.0"
            TARGET_FPS="$(python3 - <<PY
frames=int("${NUM_FRAMES}")
target=float("${TARGET_DURATION_S}")
print(frames/target)
PY
)"
        fi
        tmp_out="${BASELINE_DIR}/baseline_${TARGET_DURATION_S}s.mp4"
        ffmpeg -y -i "${in_file}" -vf "setpts=${PTS_SCALE}*PTS" -r "${TARGET_FPS}" -vsync cfr -an -movflags +faststart "${tmp_out}" \
            && mv -f "${tmp_out}" "${in_file}"
    fi

    if [[ ! -f "${BASELINE_DIR}/baseline.mp4" ]]; then
        echo "WARNING: Expected baseline video not found: ${BASELINE_DIR}/baseline.mp4" >&2
        continue
    fi

    DST_VIDEO="${VIDEOS_DIR}/${VIDEO_ID}__${SEED}.mp4"
    cp -f "${BASELINE_DIR}/baseline.mp4" "${DST_VIDEO}"

    if [[ "${SAVE_SNAPSHOTS}" == "1" && -f "${VIDEO_SNAPSHOT_SH}" ]]; then
        bash "${VIDEO_SNAPSHOT_SH}" "${DST_VIDEO}" || true
    fi

    printf "%s\t%s\t%s\n" "${VIDEO_ID}" "${PROMPT_EN}" "${DST_VIDEO}" >> "${MANIFEST_PATH}"
    echo "Exported baseline video: ${DST_VIDEO}"
done < "${PROMPTS_FILE}"

echo ""
echo "======================================================================"
echo "PAI-BENCH WAN BASELINE COMPLETE"
echo "======================================================================"
echo "Videos:   ${VIDEOS_DIR}"
echo "Manifest: ${MANIFEST_PATH}"
echo ""
