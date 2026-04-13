#!/bin/bash
# Batch processing: Run GRPO for all prompts in a single prompt file using Wan2.1.
# Creates one output folder per prompt file: {filename}_{timestamp}, with per-prompt subdirs (baseline + GRPO).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Wan2.1:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

SAVE_SNAPSHOTS="${SAVE_SNAPSHOTS:-1}"
VIDEO_SNAPSHOT_SH="${REPO_ROOT}/scripts/video_to_snapshot.sh"
# Two PNG strips per prompt (baseline vs GRPO), K frames each (aligned timestamps).
SAVE_KEYFRAME_STRIP="${SAVE_KEYFRAME_STRIP:-1}"
KEYFRAME_STRIP_FRAMES="${KEYFRAME_STRIP_FRAMES:-6}"
KEYFRAME_STRIP_SH="${REPO_ROOT}/scripts/save_prompt_keyframe_strips.sh"
# Enable denoising trajectory strip + step snapshots by default for WAN.
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-1}"
DENOISING_SNAPSHOT_STRIDE="${DENOISING_SNAPSHOT_STRIDE:-5}"
TARGET_DURATION_S="${TARGET_DURATION_S:-4}"

# Single prompt file (one prompt per line). Default: New Year physics prompt set.
PROMPTS_FILE="${PROMPTS_FILE:-origin_grpo/total.txt}"

echo "======================================================================"
echo "BATCH GRPO TRAINING - WAN2.1 - PROMPTS FROM ${PROMPTS_FILE}"
echo "======================================================================"
echo ""
# Wan2.1 configuration (t2v-1.3B supported sizes: 480*832, 832*480)
MODEL_TYPE="${MODEL_TYPE:-wan}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B}"
WAN_TASK="${WAN_TASK:-t2v-1.3B}"
# Wan 1.3B supported sizes: 480*832 or 832*480 (width*height)
WAN_SIZE="${WAN_SIZE:-832*480}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-15}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-6}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
# Parse WAN_SIZE for run.py (width and height). Format: 832*480 -> WIDTH=832, HEIGHT=480
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
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"  # image_clip | xclip | qwen | hybrid_video | adaptive_physics
REWARD_DEBUG="${REWARD_DEBUG:-1}"               # 1 -> pass --reward-debug
CLIP_NUM_FRAMES="${CLIP_NUM_FRAMES:-0}"         # used by image_clip / adaptive_physics
CLIP_AGGREGATION="${CLIP_AGGREGATION:-video_mean_pool}"
ADAPTIVE_PHYSICS_HIDDEN_DIM="${ADAPTIVE_PHYSICS_HIDDEN_DIM:-32}"
PHYSICS_CATEGORY_OVERRIDE="${PHYSICS_CATEGORY_OVERRIDE:-}"   # optional explicit category label
PHYSICS_HANDCRAFTED_W_MOTION="${PHYSICS_HANDCRAFTED_W_MOTION:-0.35}"
PHYSICS_HANDCRAFTED_W_CATEGORY="${PHYSICS_HANDCRAFTED_W_CATEGORY:-0.65}"
RUN_BASELINE="${RUN_BASELINE:-1}"

if [[ "$MODEL_TYPE" != "wan" ]]; then
    echo "❌ MODEL_TYPE='$MODEL_TYPE' is not supported by this script. Use MODEL_TYPE=wan."
    exit 2
fi

if [[ "${PROMPTS_FILE}" != /* ]]; then
    PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPTS_FILE" >&2
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

# Batch output directory: <model_name>_grpo_<timestep>
MODEL_NAME="${MODEL_PATH##*/}"
MODEL_NAME="${MODEL_NAME//\//-}"
BATCH_TIMESTEP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="./${MODEL_NAME}_grpo_${BATCH_TIMESTEP}"
mkdir -p "$BATCH_DIR"

# Resolve Wan checkpoint dir: if MODEL_PATH is not an existing dir, download from HuggingFace
WAN_CKPT_DIR="${WAN_CKPT_DIR:-}"
if [[ -z "$WAN_CKPT_DIR" ]]; then
  if [[ -d "$MODEL_PATH" ]]; then
    WAN_CKPT_DIR="$MODEL_PATH"
  else
    echo "Downloading Wan checkpoint from HuggingFace: $MODEL_PATH ..."
    WAN_CKPT_DIR=$(python3 -c "
from pathlib import Path
from huggingface_hub import snapshot_download
p = snapshot_download('${MODEL_PATH}')
print(p)
")
    export WAN_CKPT_DIR
    echo "  -> $WAN_CKPT_DIR"
  fi
fi

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

        # Step 1: Wan2.1 baseline (optional)
        if [[ "${RUN_BASELINE}" == "1" ]]; then
            echo "Step 1/2: Generating baseline (Wan2.1 generate.py)..."
            mkdir -p "$OUTPUT_DIR/baseline"
            (cd "${REPO_ROOT}/Wan2.1" && python generate.py \
                --task "$WAN_TASK" \
                --ckpt_dir "$WAN_CKPT_DIR" \
                --prompt "$PROMPT" \
                --size "$WAN_SIZE" \
                --frame_num "$NUM_FRAMES" \
                --save_file "${OUTPUT_DIR}/baseline/baseline.mp4" \
                --base_seed "$SEED") || {
                echo "⚠️ Baseline failed for prompt $PROMPT_IDX (Wan generate.py). Continuing to GRPO..."
            }
            if [[ -f "$OUTPUT_DIR/baseline/baseline.mp4" ]] && command -v ffmpeg >/dev/null 2>&1; then
                in_file="${OUTPUT_DIR}/baseline/baseline.mp4"
                in_dur="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${in_file}" 2>/dev/null || true)"
                if [[ -n "${in_dur}" ]]; then
                    read -r PTS_SCALE TARGET_FPS < <(python - <<PY
dur=float("${in_dur}")
target=float("${TARGET_DURATION_S}")
frames=int("${NUM_FRAMES}")
print(target/dur, frames/target)
PY
)
                else
                    PTS_SCALE="1.0"
                    TARGET_FPS="$(python - <<PY
frames=int("${NUM_FRAMES}")
target=float("${TARGET_DURATION_S}")
print(frames/target)
PY
)"
                fi
                tmp_out="${OUTPUT_DIR}/baseline/baseline_${TARGET_DURATION_S}s.mp4"
                ffmpeg -y -i "${in_file}" -vf "setpts=${PTS_SCALE}*PTS" -r "${TARGET_FPS}" -vsync cfr -an -movflags +faststart "${tmp_out}" \
                    && mv -f "${tmp_out}" "${in_file}"
            fi
            if [[ -f "$OUTPUT_DIR/baseline/baseline.mp4" ]]; then
                echo "✅ Baseline saved: $OUTPUT_DIR/baseline/baseline.mp4"
                if [[ "${SAVE_SNAPSHOTS}" == "1" && -f "${VIDEO_SNAPSHOT_SH}" ]]; then
                    bash "${VIDEO_SNAPSHOT_SH}" "$OUTPUT_DIR/baseline/baseline.mp4" || true
                fi
            fi
            echo ""
        fi

        # Step 2: GRPO
        echo "Step 2/2: Running GRPO training..."
        cmd=(python "${REPO_ROOT}/unified_grpo/run.py"
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
            echo "⚠️ GRPO failed for prompt $PROMPT_IDX, continuing..."
            continue
        }
        echo "✅ GRPO complete for prompt $PROMPT_IDX"
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
echo "BATCH PROCESSING COMPLETE! 🎉"
echo "======================================================================"
echo ""
echo "Processed $TOTAL_PROMPTS_PROCESSED prompts across $TOTAL_FILES file(s)"
echo "Output: $BATCH_DIR/"
echo ""
echo "Structure:"
echo "  <model_name>_grpo_YYYYMMDD_HHMMSS/"
echo "  ├── falling_YYYYMMDD_HHMMSS/"
echo "  │   ├── p001_*/"
echo "  │   │   ├── baseline/baseline.mp4 (+ *_snapshot.png)"
echo "  │   │   ├── grpo/wan_grpo.mp4 (+ *_snapshot.png)"
echo "  │   │   ├── baseline_keyframes.png, grpo_keyframes.png (${KEYFRAME_STRIP_FRAMES} frames each, if SAVE_KEYFRAME_STRIP=1)"
echo "  │   │   └── prompt.txt"
echo "  │   └── ..."
echo "  └── ..."
echo "======================================================================"
