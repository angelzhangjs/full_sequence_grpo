#!/bin/bash
# PAI-bench batch GRPO runner for Wan2.1 with Accelerate + DeepSpeed ZeRO-3 over 8 GPUs.
# Reads a TSV file with columns: video_id, prompt_en
# Exports final videos as <video_id>__<seed>.mp4 under OUTPUT_ROOT/videos for PAI-style evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Wan2.1:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PROMPTS_FILE="${PROMPTS_FILE:-pai_bench_text_only/cosmos_predict2_bench_video_prompts.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/wan_pai_grpo}"
VIDEOS_DIR="${OUTPUT_ROOT}/videos"
RUNS_DIR="${OUTPUT_ROOT}/runs"
MANIFEST_PATH="${OUTPUT_ROOT}/prompt_manifest.tsv"

MODEL_TYPE="${MODEL_TYPE:-wan}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B}"
WAN_TASK="${WAN_TASK:-t2v-1.3B}"
WAN_SIZE="${WAN_SIZE:-832*480}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-15}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-6}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
NUM_FRAMES="${NUM_FRAMES:-33}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
UNFREEZE_PERCENTAGE="${UNFREEZE_PERCENTAGE:-0.20}"
USE_LORA="${USE_LORA:-1}"
LORA_BLOCKS="${LORA_BLOCKS:-}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"
REWARD_DEBUG="${REWARD_DEBUG:-0}"
CLIP_NUM_FRAMES="${CLIP_NUM_FRAMES:-0}"
CLIP_AGGREGATION="${CLIP_AGGREGATION:-video_mean_pool}"
ADAPTIVE_PHYSICS_HIDDEN_DIM="${ADAPTIVE_PHYSICS_HIDDEN_DIM:-32}"
PHYSICS_CATEGORY_OVERRIDE="${PHYSICS_CATEGORY_OVERRIDE:-}"
PHYSICS_HANDCRAFTED_W_MOTION="${PHYSICS_HANDCRAFTED_W_MOTION:-0.35}"
PHYSICS_HANDCRAFTED_W_CATEGORY="${PHYSICS_HANDCRAFTED_W_CATEGORY:-0.65}"
WAN_T5_CPU="${WAN_T5_CPU:-0}"

if [[ "$WAN_SIZE" == *"*"* ]]; then
  WIDTH="${WIDTH:-${WAN_SIZE%%\**}}"
  HEIGHT="${HEIGHT:-${WAN_SIZE#*\*}}"
else
  WIDTH="${WIDTH:-832}"
  HEIGHT="${HEIGHT:-480}"
fi

NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/configs/accelerate_zero3_8xa100.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero3_8xa100.json}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29531}"

# PAI export runs are usually interested in final MP4s, not debug artifacts.
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-0}"

echo "======================================================================"
echo "PAI-BENCH WAN GRPO - DEEPSPEED ZERO-3 - ${NUM_GPUS} GPU"
echo "======================================================================"
echo "Prompts TSV:       ${PROMPTS_FILE}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "Videos output dir: ${VIDEOS_DIR}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "DeepSpeed config:  ${DEEPSPEED_CONFIG}"
echo ""

if [[ "$MODEL_TYPE" != "wan" ]]; then
    echo "ERROR: MODEL_TYPE='$MODEL_TYPE' is not supported by this script. Use MODEL_TYPE=wan." >&2
    exit 2
fi

if [[ "${PROMPTS_FILE}" != /* ]]; then
    PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: Prompt TSV not found: $PROMPTS_FILE" >&2
    exit 1
fi

if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
    echo "ERROR: Accelerate config not found: $ACCELERATE_CONFIG" >&2
    exit 1
fi

if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    echo "ERROR: DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2
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

echo "Total prompts: ${TOTAL_PROMPTS}"
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
    # Skip header
    if [[ "${VIDEO_ID}" == "video_id" ]]; then
        continue
    fi
    # Skip empty rows
    if [[ -z "${VIDEO_ID//[[:space:]]/}" ]]; then
        continue
    fi

    PROMPT_IDX=$((PROMPT_IDX + 1))
    SAFE_VIDEO_ID="$(printf '%s' "${VIDEO_ID}" | tr '/:' '__')"
    SAMPLE_DIR="${RUNS_DIR}/${SAFE_VIDEO_ID}"
    SAMPLE_GRPO_DIR="${SAMPLE_DIR}/grpo"
    mkdir -p "${SAMPLE_DIR}"
    printf "%s\n" "${PROMPT_EN}" > "${SAMPLE_DIR}/prompt.txt"

    echo ""
    echo "======================================================================"
    echo "Sample ${PROMPT_IDX}/${TOTAL_PROMPTS}"
    echo "video_id: ${VIDEO_ID}"
    echo "======================================================================"
    echo "${PROMPT_EN}"
    echo ""
    echo "Step 1/1: Running GRPO training with Accelerate + DeepSpeed..."

    cmd=(accelerate launch
        --config_file "$ACCELERATE_CONFIG"
        --num_processes "$NUM_GPUS"
        --main_process_port "$MAIN_PROCESS_PORT"
        -m unified_grpo.run
        --use-accelerate
        --distributed-backend deepspeed
        --model-type "$MODEL_TYPE"
        --model-path "$MODEL_PATH"
        --prompt "$PROMPT_EN"
        --reward-backend "$REWARD_BACKEND"
        --gradient-checkpointing
        --height "$HEIGHT"
        --width "$WIDTH"
        --num-frames "$NUM_FRAMES"
        --guidance-scale "$GUIDANCE_SCALE"
        --num-inference-steps "$NUM_INFERENCE_STEPS"
        --num-grpo-steps "$NUM_GRPO_STEPS"
        --num-rollouts "$NUM_ROLLOUTS"
        --lr "$LR"
        --seed "$SEED"
        --unfreeze-percentage "$UNFREEZE_PERCENTAGE"
        --output-dir "$SAMPLE_GRPO_DIR"
        --clip-num-frames "$CLIP_NUM_FRAMES"
        --clip-aggregation "$CLIP_AGGREGATION"
        --adaptive-physics-hidden-dim "$ADAPTIVE_PHYSICS_HIDDEN_DIM"
        --physics-handcrafted-w-motion "$PHYSICS_HANDCRAFTED_W_MOTION"
        --physics-handcrafted-w-category "$PHYSICS_HANDCRAFTED_W_CATEGORY"
    )

    if [[ "${REWARD_DEBUG}" == "1" || "${REWARD_DEBUG}" == "true" ]]; then
        cmd+=(--reward-debug)
    fi
    if [[ -n "${PHYSICS_CATEGORY_OVERRIDE}" ]]; then
        cmd+=(--physics-category-override "$PHYSICS_CATEGORY_OVERRIDE")
    fi
    if [[ "${WAN_T5_CPU}" == "1" || "${WAN_T5_CPU}" == "true" ]]; then
        cmd+=(--wan-t5-cpu)
    fi
    if [[ "$USE_LORA" == "1" ]]; then
        cmd+=(--use-lora --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA")
        [[ -n "$LORA_BLOCKS" ]] && cmd+=(--lora-blocks "$LORA_BLOCKS")
    fi
    if [[ "${SAVE_DENOISING_STRIP}" == "1" ]]; then
        cmd+=(
            --save-denoising-strip-png
            --save-denoising-step-snapshots
        )
    fi

    "${cmd[@]}" || {
        echo "WARNING: GRPO failed for video_id=${VIDEO_ID}, continuing..."
        continue
    }

    SRC_VIDEO="${SAMPLE_GRPO_DIR}/wan_grpo.mp4"
    DST_VIDEO="${VIDEOS_DIR}/${VIDEO_ID}__${SEED}.mp4"

    if [[ ! -f "${SRC_VIDEO}" ]]; then
        echo "WARNING: Expected output video not found: ${SRC_VIDEO}" >&2
        continue
    fi

    cp -f "${SRC_VIDEO}" "${DST_VIDEO}"
    printf "%s\t%s\t%s\n" "${VIDEO_ID}" "${PROMPT_EN}" "${DST_VIDEO}" >> "${MANIFEST_PATH}"
    echo "Exported PAI video: ${DST_VIDEO}"
done < "${PROMPTS_FILE}"

echo ""
echo "======================================================================"
echo "PAI-BENCH WAN GRPO COMPLETE"
echo "======================================================================"
echo "Videos:   ${VIDEOS_DIR}"
echo "Manifest: ${MANIFEST_PATH}"
echo ""
