#!/bin/bash
# Batch processing: Run GRPO for LTX on physics prompt suites.
# Default: every *.txt under basic_physics_prompts_ltx/ (sorted), one batch subfolder per file.
# Override with PROMPTS_FILE=/path/to/one.txt for a single file.
# Optional: PROMPTS_DIR=other/dir to glob *.txt elsewhere (when PROMPTS_FILE is unset).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Prevent user-site packages (~/.local) from shadowing the conda env.
export PYTHONNOUSERSITE=1

SAVE_SNAPSHOTS="${SAVE_SNAPSHOTS:-1}"
VIDEO_SNAPSHOT_SH="${REPO_ROOT}/scripts/video_to_snapshot.sh"
SAVE_KEYFRAME_STRIP="${SAVE_KEYFRAME_STRIP:-1}"
KEYFRAME_STRIP_FRAMES="${KEYFRAME_STRIP_FRAMES:-5}"
KEYFRAME_STRIP_SH="${REPO_ROOT}/scripts/save_prompt_keyframe_strips.sh"
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"

# Single file mode: set PROMPTS_FILE=path/to/prompts.txt (one prompt per line).
# Default: unset PROMPTS_FILE and use all *.txt in PROMPTS_DIR (basic_physics_prompts_ltx).
PROMPTS_FILE="${PROMPTS_FILE:-}"
PROMPTS_DIR="${PROMPTS_DIR:-basic_physics_prompts_ltx}"

echo "======================================================================"
echo "BATCH GRPO TRAINING - LTX"
echo "======================================================================"
echo ""

# LTX configuration defaults (can be overridden via env vars)
MODEL_TYPE="${MODEL_TYPE:-ltx}"
MODEL_PATH="${MODEL_PATH:-Lightricks/LTX-Video}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-15}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-720}"
NUM_FRAMES="${NUM_FRAMES:-32}"
FPS="${FPS:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
UNFREEZE_PERCENTAGE="${UNFREEZE_PERCENTAGE:-0.20}"
USE_LORA="${USE_LORA:-1}"         # 1 -> add --use-lora, 0 -> full/partial finetune (no LoRA)
# Default to ALL blocks for LTX. ("last" requires total block count and can fail on some impls.)
LORA_BLOCKS="${LORA_BLOCKS:-last}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"  # image_clip | xclip | qwen
RUN_BASELINE="${RUN_BASELINE:-1}"              # 1 -> baseline mp4, 0 -> skip baseline
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
# Baseline pipeline config for LTX's reference inference script
LTX_PIPELINE_CONFIG="${LTX_PIPELINE_CONFIG:-${REPO_ROOT}/ltx_video/configs/ltxv-2b-0.9.6-dev.yaml}"

if [[ "$MODEL_TYPE" != "ltx" ]]; then
    echo "❌ MODEL_TYPE='$MODEL_TYPE' is not supported by this script. Use MODEL_TYPE=ltx."
    exit 2
fi

if [[ -n "$PROMPTS_FILE" ]]; then
    if [[ "${PROMPTS_FILE}" != /* ]]; then
        PROMPTS_FILE="${REPO_ROOT}/${PROMPTS_FILE}"
    fi
    if [[ ! -f "$PROMPTS_FILE" ]]; then
        echo "ERROR: Prompt file not found: $PROMPTS_FILE" >&2
        exit 1
    fi
    PROMPT_FILES=("$PROMPTS_FILE")
    echo "Using single prompt file: $PROMPTS_FILE"
else
    if [[ "${PROMPTS_DIR}" != /* ]]; then
        PROMPTS_DIR="${REPO_ROOT}/${PROMPTS_DIR}"
    fi
    if [[ ! -d "$PROMPTS_DIR" ]]; then
        echo "ERROR: PROMPTS_DIR not found: $PROMPTS_DIR" >&2
        echo "       Set PROMPTS_DIR or PROMPTS_FILE=path/to/prompts.txt" >&2
        exit 1
    fi
    shopt -s nullglob
    _ltx_pf=("${PROMPTS_DIR}"/*.txt)
    shopt -u nullglob
    if [[ ${#_ltx_pf[@]} -eq 0 ]]; then
        echo "ERROR: No .txt files in $PROMPTS_DIR" >&2
        exit 1
    fi
    readarray -t PROMPT_FILES < <(printf '%s\n' "${_ltx_pf[@]}" | sort)
    echo "Using LTX physics suite: ${PROMPTS_DIR}/*.txt"
    echo "  - files: ${#PROMPT_FILES[@]}"
    for _f in "${PROMPT_FILES[@]}"; do
        echo "    - $(basename "$_f")"
    done
fi

TOTAL_FILES=${#PROMPT_FILES[@]}
TOTAL_LINES_ALL=0
for _f in "${PROMPT_FILES[@]}"; do
    TOTAL_LINES_ALL=$((TOTAL_LINES_ALL + $(grep -cve '^[[:space:]]*$' "$_f" || true)))
done
echo "  - total non-empty prompt lines (all files): $TOTAL_LINES_ALL"
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
# Model name = last component of MODEL_PATH (e.g. Lightricks/LTX-Video -> LTX-Video), sanitized for dirname
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
    OUTPUT_TOP_REL="${BATCH_DIR}/${FILE_BASE}_${TIMESTEP}"
    mkdir -p "$OUTPUT_TOP_REL"

    TOTAL_PROMPTS="$(grep -cve '^[[:space:]]*$' "${PROMPTS_FILE}" || true)"
    echo ""
    echo "######################################################################"
    echo "FILE $FILE_INDEX/$TOTAL_FILES: $PROMPTS_FILE -> $OUTPUT_TOP_REL"
    echo "######################################################################"
    echo ""

    # Loop over each prompt in this file
    LINE_NUM=0
    PROMPT_IDX=0
    while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
        LINE_NUM=$((LINE_NUM + 1))

        # Skip empty lines
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

        # Create output directory for this prompt under this file's folder
        PROMPT_SHORT=$(echo "$PROMPT" | head -c 50 | tr -cd '[:alnum:] ' | tr ' ' '-')
        PROMPT_ID="p$(printf '%03d' $PROMPT_IDX)"
        OUTPUT_DIR_REL="${OUTPUT_TOP_REL}/${PROMPT_ID}_${PROMPT_SHORT}"
        # Use an absolute output dir so baseline + GRPO always write under the same tree
        OUTPUT_DIR="$(python3 - "$OUTPUT_DIR_REL" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
        mkdir -p "$OUTPUT_DIR"

        # Save prompt text for reference
        echo "$PROMPT" > "$OUTPUT_DIR/prompt.txt"

        # ======================================================================
        # Step 1/2: Run LTX baseline (pretrained model)
        # ======================================================================
        if [[ "${RUN_BASELINE}" == "1" ]]; then
            echo "Step 1/2: Generating baseline video (LTX inference.py)..."
            mkdir -p "$OUTPUT_DIR/baseline"

            PROMPT="$PROMPT" \
            OUTPUT_DIR="$OUTPUT_DIR/baseline" \
            SEED="$SEED" \
            HEIGHT="$HEIGHT" \
            WIDTH="$WIDTH" \
            NUM_FRAMES="$NUM_FRAMES" \
            FPS="$FPS" \
            NEGATIVE_PROMPT="$NEGATIVE_PROMPT" \
            LTX_PIPELINE_CONFIG="$LTX_PIPELINE_CONFIG" \
            bash "${REPO_ROOT}/unified_grpo/baseline/ltx_baseline.sh" || {
                echo "⚠️ Baseline failed for prompt $PROMPT_IDX, continuing to GRPO..."
            }

            if [[ -f "$OUTPUT_DIR/baseline/baseline.mp4" ]]; then
                echo "✅ Baseline saved: $OUTPUT_DIR/baseline/baseline.mp4"
                if [[ "${SAVE_SNAPSHOTS}" == "1" && -f "${VIDEO_SNAPSHOT_SH}" ]]; then
                    bash "${VIDEO_SNAPSHOT_SH}" "$OUTPUT_DIR/baseline/baseline.mp4" || true
                fi
            fi
            echo ""
        fi

        # ======================================================================
        # Step 2: Run GRPO Training
        # ======================================================================
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
        )
        if [[ "${SAVE_CHECKPOINTS}" == "1" ]]; then
            cmd+=(--save-checkpoint-dir "$OUTPUT_DIR/grpo/checkpoint")
        fi

        if [[ "$USE_LORA" == "1" ]]; then
            cmd+=(--use-lora --lora-blocks "$LORA_BLOCKS" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA")
        fi

        if [[ "${SAVE_DENOISING_STRIP}" == "1" ]]; then
            cmd+=(--save-denoising-strip-png --save-denoising-step-snapshots)
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
echo "  └── <stem>_YYYYMMDD_HHMMSS/   # one folder per .txt (e.g. bouncing_, falling_, ...)"
echo "      ├── p001_*/"
echo "      │   ├── baseline/baseline.mp4 (+ *_snapshot.png)"
echo "      │   ├── grpo/ltx_grpo.mp4 (+ *_snapshot.png)"
echo "      │   ├── baseline_keyframes.png, grpo_keyframes.png (${KEYFRAME_STRIP_FRAMES} frames each, if SAVE_KEYFRAME_STRIP=1)"
echo "      │   ├── grpo/checkpoint/"
echo "      │   └── prompt.txt"
echo "      └── ..."
echo ""
echo "Compare baseline vs GRPO for each prompt!"
echo "======================================================================"