#!/bin/bash
# PAI-bench baseline video runner for CogVideoX.
# Reads a TSV file with columns: video_id, prompt_en
# Exports baseline videos as <video_id>__<seed>.mp4 under OUTPUT_ROOT/videos.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PROMPTS_FILE="${PROMPTS_FILE:-/home/ubuntu/angel-neurips/full_sequence_grpo/cosmos_predict2_bench_video_prompts.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/cogvideox_pai_baseline}"
VIDEOS_DIR="${OUTPUT_ROOT}/videos"
RUNS_DIR="${OUTPUT_ROOT}/runs"
MANIFEST_PATH="${OUTPUT_ROOT}/prompt_manifest.tsv"

MODEL_PATH="${MODEL_PATH:-THUDM/CogVideoX-2b}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_FRAMES="${NUM_FRAMES:-32}"
FPS="${FPS:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
SEED="${SEED:-42}"
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
echo "PAI-BENCH COGVIDEOX BASELINE"
echo "======================================================================"
echo "Prompts TSV:       ${PROMPTS_FILE}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "Videos output dir: ${VIDEOS_DIR}"
echo "Total prompts:     ${TOTAL_PROMPTS}"
echo ""

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
    echo "Step 1/1: Generating CogVideoX baseline..."

    pushd "${REPO_ROOT}/CogVideo" >/dev/null
    python inference/cli_demo.py \
        --prompt "${PROMPT_EN}" \
        --model_path "${MODEL_PATH}" \
        --generate_type "t2v" \
        --num_frames "${NUM_FRAMES}" \
        --fps "${FPS}" \
        --guidance_scale "${GUIDANCE_SCALE}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --seed "${SEED}" \
        --output_path "../${BASELINE_DIR}/baseline.mp4" || {
            echo "WARNING: Baseline failed for video_id=${VIDEO_ID}, continuing..."
            popd >/dev/null
            continue
        }
    popd >/dev/null

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
echo "PAI-BENCH COGVIDEOX BASELINE COMPLETE"
echo "======================================================================"
echo "Videos:   ${VIDEOS_DIR}"
echo "Manifest: ${MANIFEST_PATH}"
echo ""
