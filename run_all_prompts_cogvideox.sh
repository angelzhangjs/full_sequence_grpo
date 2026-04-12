#!/bin/bash
# Batch processing: Run GRPO for all prompts in a single prompt file.
# Creates one output folder per prompt file: {filename}_{timestamp}, with per-prompt subdirs (baseline + GRPO).

set -e

cd "$(dirname "$0")"
REPO_ROOT="${PWD}"
# Extract middle-frame PNG next to each mp4 (baseline + GRPO). Set SAVE_SNAPSHOTS=0 to skip.
SAVE_SNAPSHOTS="${SAVE_SNAPSHOTS:-1}"
VIDEO_SNAPSHOT_SH="${REPO_ROOT}/scripts/video_to_snapshot.sh"
SAVE_KEYFRAME_STRIP="${SAVE_KEYFRAME_STRIP:-1}"
KEYFRAME_STRIP_FRAMES="${KEYFRAME_STRIP_FRAMES:-2}"
KEYFRAME_STRIP_SH="${REPO_ROOT}/scripts/save_prompt_keyframe_strips.sh"
# One wide PNG of every denoising step during GRPO (VAE decode each step).
# Enabled by default; set SAVE_DENOISING_STRIP=0 to disable.
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-1}"

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Prevent user-site packages (~/.local) from shadowing the conda env.
export PYTHONNOUSERSITE=1

# Single prompt file (one prompt per line). Default: total.txt (500 prompts on main branch).
PROMPTS_FILE="${PROMPTS_FILE:-total.txt}"

echo "======================================================================"
echo "BATCH GRPO TRAINING - PROMPTS FROM ${PROMPTS_FILE}"
echo ""

# NOTE:
# - This script is currently configured for CogVideoX baseline generation (CogVideo/inference/cli_demo.py)
# - unified_grpo/run.py in this repo currently supports CogVideoX adapter end-to-end.
MODEL_TYPE="${MODEL_TYPE:-cogvideox}"
MODEL_PATH="${MODEL_PATH:-THUDM/CogVideoX-2b}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-15}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
UNFREEZE_PERCENTAGE="${UNFREEZE_PERCENTAGE:-0.20}"
USE_LORA="${USE_LORA:-1}"         # 1 -> add --use-lora, 0 -> full/partial finetune (no LoRA)
LORA_BLOCKS="${LORA_BLOCKS:-last}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"  # image_clip | xclip | qwen | hybrid_video | adaptive_physics
RUN_BASELINE="${RUN_BASELINE:-1}"              # 1 -> baseline mp4, 0 -> skip baseline

if [[ "$MODEL_TYPE" != "cogvideox" ]]; then
    echo "❌ MODEL_TYPE='$MODEL_TYPE' is not supported by this script right now."
    exit 2
fi

if [[ "${PROMPTS_FILE}" != /* ]]; then
    PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPTS_FILE"
    exit 1
fi

PROMPT_FILES=("$PROMPTS_FILE")
TOTAL_FILES=1
n=$(grep -cve '^[[:space:]]*$' "$PROMPTS_FILE" || true)
echo "Using prompt file: $(basename "$PROMPTS_FILE")"
echo "  - prompts: $n"
echo ""

# Non-interactive mode:
# - If stdin is not a TTY (e.g. running under `conda run`, CI), auto-continue.
# - Or set AUTO_YES=1 to skip the prompt explicitly.

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

# Create batch output directory: <model_name>_grpo_<timestep>
# Model name = last component of MODEL_PATH (e.g. THUDM/CogVideoX-2b -> CogVideoX-2b), sanitized for dirname
MODEL_NAME="${MODEL_PATH##*/}"
MODEL_NAME="${MODEL_NAME//\//-}"
BATCH_TIMESTEP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="./${MODEL_NAME}_grpo_${BATCH_TIMESTEP}"
mkdir -p "$BATCH_DIR"

echo ""
echo "Batch output: $BATCH_DIR"
echo "======================================================================"
echo ""

TOTAL_PROMPTS_PROCESSED=0
FILE_INDEX=0

# Loop over prompt file(s)
for PROMPTS_FILE in "${PROMPT_FILES[@]}"; do
    FILE_INDEX=$((FILE_INDEX + 1))
    FILE_BASE=$(basename "$PROMPTS_FILE" .txt)
    # Output folder for this file: filename + current timestamp
    TIMESTEP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_TOP="${BATCH_DIR}/${FILE_BASE}_${TIMESTEP}"
    mkdir -p "$OUTPUT_TOP"

    TOTAL_PROMPTS=$(grep -cve '^[[:space:]]*$' "$PROMPTS_FILE" || true)
    echo ""
    echo "######################################################################"
    echo "FILE $FILE_INDEX/$TOTAL_FILES: $PROMPTS_FILE -> $OUTPUT_TOP"
    echo "######################################################################"
    echo ""

    # Loop over each prompt in this file
    LINE_NUM=0
    while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
        LINE_NUM=$((LINE_NUM + 1))

        # Skip empty lines
        if [[ -z "${PROMPT//[[:space:]]/}" ]]; then
            continue
        fi
        echo ""
        echo "======================================================================"
        echo "Prompt $LINE_NUM/$TOTAL_PROMPTS (file: $FILE_BASE)"
        echo "======================================================================"
        echo "$PROMPT"
        echo ""

        # Create output directory for this prompt under this file's folder
        PROMPT_SHORT=$(echo "$PROMPT" | head -c 50 | tr -cd '[:alnum:] ' | tr ' ' '-')
        OUTPUT_DIR="${OUTPUT_TOP}/p$(printf '%03d' $LINE_NUM)_${PROMPT_SHORT}"

        mkdir -p "$OUTPUT_DIR"

        # ======================================================================
        # Step 1: Generate Baseline
        # ======================================================================
        if [[ "$RUN_BASELINE" == "1" ]]; then
            echo "Step 1/2: Generating baseline..."

            mkdir -p "$OUTPUT_DIR/baseline"

            pushd CogVideo >/dev/null

            python inference/cli_demo.py \
                --prompt "$PROMPT" \
                --model_path "$MODEL_PATH" \
                --generate_type "t2v" \
                --num_frames 32 \
                --fps 8 \
                --guidance_scale 7.5 \
                --num_inference_steps "$NUM_INFERENCE_STEPS" \
                --seed "$SEED" \
                --output_path "../$OUTPUT_DIR/baseline/baseline.mp4" || {
                    echo "⚠️ Baseline failed for prompt $LINE_NUM, skipping..."
                    popd >/dev/null
                    continue
                }

            popd >/dev/null

            echo "✅ Baseline complete"
            if [[ "${SAVE_SNAPSHOTS}" == "1" && -f "$OUTPUT_DIR/baseline/baseline.mp4" && -f "${VIDEO_SNAPSHOT_SH}" ]]; then
                bash "${VIDEO_SNAPSHOT_SH}" "$OUTPUT_DIR/baseline/baseline.mp4" || true
            fi
            echo ""
        else
            echo "Step 1/2: Skipping baseline (RUN_BASELINE=0)"
        fi

        # ======================================================================
        # Step 2: Run GRPO Training
        # ======================================================================
        echo "Step 2/2: Running GRPO training..."

        cmd=(python "/home/ubuntu/angel-research/unified_grpo/run.py"
            --model-type "$MODEL_TYPE"
            --model-path "$MODEL_PATH"
            --prompt "$PROMPT"
            --reward-backend "$REWARD_BACKEND"
            --gradient-checkpointing
            --height 480
            --width 720
            --num-frames 32
            --guidance-scale 7.5
            --num-inference-steps "$NUM_INFERENCE_STEPS"
            --num-grpo-steps "$NUM_GRPO_STEPS"
            --num-rollouts "$NUM_ROLLOUTS"
            --lr "$LR"
            --seed "$SEED"
            --unfreeze-percentage "$UNFREEZE_PERCENTAGE"
            --output-dir "$OUTPUT_DIR/grpo"
        )

        if [[ "$USE_LORA" == "1" ]]; then
            cmd+=(--use-lora --lora-blocks "$LORA_BLOCKS" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA")
        fi

        if [[ "${SAVE_DENOISING_STRIP}" == "1" ]]; then
            cmd+=(--save-denoising-strip-png --save-denoising-step-snapshots)
        fi

        "${cmd[@]}" || {
            echo "⚠️ GRPO failed for prompt $LINE_NUM, continuing..."
            continue
        }

        echo "✅ GRPO complete for prompt $LINE_NUM"
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

        # Save prompt text for reference
        echo "$PROMPT" > "$OUTPUT_DIR/prompt.txt"

        TOTAL_PROMPTS_PROCESSED=$((TOTAL_PROMPTS_PROCESSED + 1))

    done < "$PROMPTS_FILE"
done

# ======================================================================
# Summary
# ======================================================================
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
echo "  └── total_YYYYMMDD_HHMMSS/   (or <promptfile_basename>_YYYYMMDD_HHMMSS/)"
echo "      ├── p001_*/"
echo "      │   ├── baseline/baseline.mp4 (+ baseline_snapshot.png)"
echo "      │   ├── grpo/cogvideox_grpo.mp4 (+ cogvideox_grpo_snapshot.png)"
echo "      │   ├── baseline_keyframes.png, grpo_keyframes.png (${KEYFRAME_STRIP_FRAMES} frames each, if SAVE_KEYFRAME_STRIP=1)"
echo "      │   └── prompt.txt"
echo "      └── ..."
echo ""
echo "Compare baseline vs GRPO for each prompt!"
echo "======================================================================"
