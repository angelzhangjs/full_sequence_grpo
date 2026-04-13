#!/bin/bash
# Batch processing: Run GRPO for all prompts in a single prompt file using Wan2.1
# with Accelerate + DeepSpeed ZeRO-3 over 8 GPUs.
#
# This intentionally leaves `run_all_prompts_wan.sh` untouched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Wan2.1:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SAVE_SNAPSHOTS="${SAVE_SNAPSHOTS:-1}"
VIDEO_SNAPSHOT_SH="${REPO_ROOT}/scripts/video_to_snapshot.sh"
SAVE_KEYFRAME_STRIP="${SAVE_KEYFRAME_STRIP:-1}"
KEYFRAME_STRIP_FRAMES="${KEYFRAME_STRIP_FRAMES:-2}"
KEYFRAME_STRIP_SH="${REPO_ROOT}/scripts/save_prompt_keyframe_strips.sh"
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-1}"
DENOISING_SNAPSHOT_STRIDE="${DENOISING_SNAPSHOT_STRIDE:-5}"

PROMPTS_FILE="${PROMPTS_FILE:-total.txt}"

MODEL_TYPE="${MODEL_TYPE:-wan}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B}"
WAN_TASK="${WAN_TASK:-t2v-1.3B}"
WAN_SIZE="${WAN_SIZE:-832*480}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-15}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-6}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

if [[ "$WAN_SIZE" == *"*"* ]]; then
  WIDTH="${WIDTH:-${WAN_SIZE%%\**}}"
  HEIGHT="${HEIGHT:-${WAN_SIZE#*\*}}"
else
  WIDTH="${WIDTH:-832}"
  HEIGHT="${HEIGHT:-480}"
fi

NUM_FRAMES="${NUM_FRAMES:-33}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
UNFREEZE_PERCENTAGE="${UNFREEZE_PERCENTAGE:-0.20}"
USE_LORA="${USE_LORA:-1}"
LORA_BLOCKS="${LORA_BLOCKS:-}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"
REWARD_DEBUG="${REWARD_DEBUG:-1}"
CLIP_NUM_FRAMES="${CLIP_NUM_FRAMES:-0}"
CLIP_AGGREGATION="${CLIP_AGGREGATION:-video_mean_pool}"
ADAPTIVE_PHYSICS_HIDDEN_DIM="${ADAPTIVE_PHYSICS_HIDDEN_DIM:-32}"
PHYSICS_CATEGORY_OVERRIDE="${PHYSICS_CATEGORY_OVERRIDE:-}"
PHYSICS_HANDCRAFTED_W_MOTION="${PHYSICS_HANDCRAFTED_W_MOTION:-0.35}"
PHYSICS_HANDCRAFTED_W_CATEGORY="${PHYSICS_HANDCRAFTED_W_CATEGORY:-0.65}"
WAN_T5_CPU="${WAN_T5_CPU:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/configs/accelerate_zero3_8xa100.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero3_8xa100.json}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

echo "======================================================================"
echo "BATCH GRPO TRAINING - WAN2.1 - DEEPSPEED ZER0-3 - ${NUM_GPUS} GPU"
echo "======================================================================"
echo "Prompts: ${PROMPTS_FILE}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "DeepSpeed config: ${DEEPSPEED_CONFIG}"
echo ""

if [[ "$MODEL_TYPE" != "wan" ]]; then
    echo "ERROR: MODEL_TYPE='$MODEL_TYPE' is not supported by this script. Use MODEL_TYPE=wan." >&2
    exit 2
fi

if [[ "${PROMPTS_FILE}" != /* ]]; then
    PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPTS_FILE" >&2
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

PROMPT_FILES=("$PROMPTS_FILE")
TOTAL_FILES=1
TOTAL_PROMPTS_IN_FILE=$(grep -cve '^[[:space:]]*$' "$PROMPTS_FILE" || true)

echo "Using prompt file: $(basename "$PROMPTS_FILE")"
echo "  - prompts: $TOTAL_PROMPTS_IN_FILE"
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

MODEL_NAME="${MODEL_PATH##*/}"
MODEL_NAME="${MODEL_NAME//\//-}"
BATCH_TIMESTEP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="./${MODEL_NAME}_grpo_ds_${BATCH_TIMESTEP}"
mkdir -p "$BATCH_DIR"

echo ""
echo "Batch output: $BATCH_DIR"
echo "======================================================================"
echo ""

TOTAL_PROMPTS_PROCESSED=0
FILE_INDEX=0

for PROMPTS_FILE in "${PROMPT_FILES[@]}"; do
    FILE_INDEX=$((FILE_INDEX + 1))
    FILE_BASE=$(basename "$PROMPTS_FILE" .txt)
    TIMESTEP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_TOP_REL="${BATCH_DIR}/${FILE_BASE}_${TIMESTEP}"
    mkdir -p "$OUTPUT_TOP_REL"

    TOTAL_PROMPTS="$(grep -cve '^[[:space:]]*$' "${PROMPTS_FILE}" || true)"
    echo ""
    echo "######################################################################"
    echo "FILE $FILE_INDEX/$TOTAL_FILES: $PROMPTS_FILE -> $OUTPUT_TOP_REL"
    echo "######################################################################"
    echo ""

    LINE_NUM=0
    PROMPT_IDX=0
    while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
        LINE_NUM=$((LINE_NUM + 1))
        if [[ -z "${PROMPT//[[:space:]]/}" ]]; then
            continue
        fi

        PROMPT_IDX=$((PROMPT_IDX + 1))
        echo ""
        echo "======================================================================"
        echo "Prompt $PROMPT_IDX/$TOTAL_PROMPTS (file: $FILE_BASE)"
        echo "======================================================================"
        echo "$PROMPT"
        echo ""

        PROMPT_SHORT=$(echo "$PROMPT" | head -c 50 | tr -cd '[:alnum:] ' | tr ' ' '-')
        PROMPT_ID="p$(printf '%03d' $PROMPT_IDX)"
        OUTPUT_DIR_REL="${OUTPUT_TOP_REL}/${PROMPT_ID}_${PROMPT_SHORT}"
        OUTPUT_DIR="$(python3 - "$OUTPUT_DIR_REL" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
        mkdir -p "$OUTPUT_DIR"
        echo "$PROMPT" > "$OUTPUT_DIR/prompt.txt"

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
            --prompt "$PROMPT"
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
            --output-dir "$OUTPUT_DIR/grpo"
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
                --denoising-step-snapshot-stride "$DENOISING_SNAPSHOT_STRIDE"
            )
        fi

        "${cmd[@]}" || {
            echo "WARNING: GRPO failed for prompt $PROMPT_IDX, continuing..."
            continue
        }

        echo "GRPO complete for prompt $PROMPT_IDX"
        if [[ "${SAVE_SNAPSHOTS}" == "1" && -f "${VIDEO_SNAPSHOT_SH}" ]]; then
            for v in "$OUTPUT_DIR/grpo"/*_grpo.mp4; do
                [[ -f "$v" ]] || continue
                bash "${VIDEO_SNAPSHOT_SH}" "$v" || true
            done
        fi
        if [[ "${SAVE_KEYFRAME_STRIP}" == "1" && -f "${KEYFRAME_STRIP_SH}" ]]; then
            bash "${KEYFRAME_STRIP_SH}" "$OUTPUT_DIR" "${KEYFRAME_STRIP_FRAMES}" || true
        fi
        echo ""
        TOTAL_PROMPTS_PROCESSED=$((TOTAL_PROMPTS_PROCESSED + 1))
    done < "$PROMPTS_FILE"
done

echo ""
echo "======================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "======================================================================"
echo ""
echo "Processed $TOTAL_PROMPTS_PROCESSED prompts across $TOTAL_FILES file(s)"
echo "Output: $BATCH_DIR/"
echo ""
