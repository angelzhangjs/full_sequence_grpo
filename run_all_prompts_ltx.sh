#!/bin/bash
# Batch processing: Run GRPO for ALL prompts in a file (LTX).
# Creates one output folder per prompt (baseline + GRPO).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Prevent user-site packages (~/.local) from shadowing the conda env.
export PYTHONNOUSERSITE=1

echo "======================================================================"
echo "BATCH GRPO TRAINING - LTX - ALL PROMPTS"
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
REWARD_BACKEND="${REWARD_BACKEND:-clip_dino}"  # clip_dino | qwen
RUN_BASELINE="${RUN_BASELINE:-1}"              # 1 -> baseline mp4, 0 -> skip baseline
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
# Baseline pipeline config for LTX's reference inference script
LTX_PIPELINE_CONFIG="${LTX_PIPELINE_CONFIG:-${REPO_ROOT}/ltx_video/configs/ltxv-2b-0.9.6-dev.yaml}"

if [[ "$MODEL_TYPE" != "ltx" ]]; then
    echo "❌ MODEL_TYPE='$MODEL_TYPE' is not supported by this script. Use MODEL_TYPE=ltx."
    exit 2
fi

# Input file
PROMPTS_FILE="${PROMPTS_FILE:-origin_grpo/newyear_prompts.txt}"
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "❌ PROMPTS_FILE not found: ${PROMPTS_FILE}" >&2
  exit 1
fi
# Count non-empty prompts for nicer progress reporting.
TOTAL_PROMPTS="$(grep -cve '^[[:space:]]*$' "${PROMPTS_FILE}" || true)"

echo "Found $TOTAL_PROMPTS prompts in $PROMPTS_FILE"
echo "This will take approximately $((TOTAL_PROMPTS * 15)) minutes"
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

# Create batch output directory
BATCH_DIR="./batch_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH_DIR"

echo ""
echo "Batch output: $BATCH_DIR"
echo "======================================================================"
echo ""

# Loop over each prompt
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
    echo "Prompt $PROMPT_IDX/$TOTAL_PROMPTS"
    echo "======================================================================"
    echo "$PROMPT"
    echo ""
    
    # Create output directory for this prompt
    # Sanitize prompt for directory name (first 50 chars, safe characters)
    PROMPT_SHORT=$(echo "$PROMPT" | head -c 50 | tr -cd '[:alnum:] ' | tr ' ' '-')
    PROMPT_ID="p$(printf '%03d' $PROMPT_IDX)"
    OUTPUT_DIR_REL="$BATCH_DIR/${PROMPT_ID}_${PROMPT_SHORT}"
    # Use an absolute output dir so baseline + GRPO always write under the same `batch_grpo_...` tree,
    # regardless of any `cd` inside called scripts.
    OUTPUT_DIR="$(python3 - "$OUTPUT_DIR_REL" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
    
    mkdir -p "$OUTPUT_DIR"

    # Save prompt text for reference (early, so even failures keep prompt.txt)
    echo "$PROMPT" > "$OUTPUT_DIR/newyear_prompts.txt"

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

    if [[ "$USE_LORA" == "1" ]]; then
        cmd+=(--use-lora --lora-blocks "$LORA_BLOCKS" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA")
    fi

    "${cmd[@]}" || {
            echo "⚠️ GRPO failed for prompt $PROMPT_IDX, continuing..."
            continue
        }
    
    echo "✅ GRPO complete for prompt $PROMPT_IDX"
    echo ""
    
done < "$PROMPTS_FILE"

# ======================================================================
# Summary
# ======================================================================
echo ""
echo "======================================================================"
echo "BATCH PROCESSING COMPLETE! 🎉"
echo "======================================================================"
echo ""
echo "Processed $LINE_NUM prompts"
echo "Output: $BATCH_DIR/"
echo ""
echo "Structure:"
echo "  batch_grpo_YYYYMMDD_HHMMSS/"
echo "  ├── p001_*/"
echo "  │   ├── baseline/baseline.mp4"
echo "  │   ├── grpo/ltx_grpo.mp4"
echo "  │   ├── prompt.txt"
echo "  │   └── newyear_prompts.txt"
echo "  ├── p002_*/"
echo "  │   └── ..."
echo "  └── ..."
echo ""
echo "Compare baseline vs GRPO for each prompt!"
echo "======================================================================"