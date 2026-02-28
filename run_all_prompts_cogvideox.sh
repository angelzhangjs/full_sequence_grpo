#!/bin/bash
# Batch processing: Run GRPO for ALL prompts in origin_grpo/newyear_prompts.txt
# Creates one output folder per prompt (baseline + GRPO).

set -e

cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Prevent user-site packages (~/.local) from shadowing the conda env.
export PYTHONNOUSERSITE=1

echo "======================================================================"
echo "BATCH GRPO TRAINING - ALL NEW YEAR PROMPTS"
echo "======================================================================"
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
REWARD_BACKEND="${REWARD_BACKEND:-clip_dino}"  # clip_dino | qwen
RUN_BASELINE="${RUN_BASELINE:-1}"              # 1 -> baseline mp4, 0 -> skip baseline

if [[ "$MODEL_TYPE" != "cogvideox" ]]; then
    echo "❌ MODEL_TYPE='$MODEL_TYPE' is not supported by this script right now."
    exit 2
fi

# Input file
PROMPTS_FILE="${PROMPTS_FILE:-origin_grpo/action_prompts.txt}"
TOTAL_PROMPTS=$(wc -l < "$PROMPTS_FILE")

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
while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
    LINE_NUM=$((LINE_NUM + 1))
    
    # Skip empty lines
    if [ -z "$PROMPT" ]; then
        continue
    fi 
    echo ""
    echo "======================================================================"
    echo "Prompt $LINE_NUM/$TOTAL_PROMPTS"
    echo "======================================================================"
    echo "$PROMPT"
    echo ""
    
    # Create output directory for this prompt
    # Sanitize prompt for directory name (first 50 chars, safe characters)
    PROMPT_SHORT=$(echo "$PROMPT" | head -c 50 | tr -cd '[:alnum:] ' | tr ' ' '-')
    OUTPUT_DIR="$BATCH_DIR/p$(printf '%03d' $LINE_NUM)_${PROMPT_SHORT}"
    
    mkdir -p "$OUTPUT_DIR"
    
    # ======================================================================
    # Step 1: Generate Baseline
    # ======================================================================
    if [[ "$RUN_BASELINE" == "1" ]]; then
        echo "Step 1/2: Generating baseline..."
    
        # Create baseline directory first!
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

    "${cmd[@]}" || {
            echo "⚠️ GRPO failed for prompt $LINE_NUM, continuing..."
            continue
        }
    
    echo "✅ GRPO complete for prompt $LINE_NUM"
    echo ""
    
    # Save prompt text for reference
    echo "$PROMPT" > "$OUTPUT_DIR/prompt.txt"
    
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
echo "  │   ├── grpo/cogvideox_grpo.mp4"
echo "  │   └── prompt.txt"
echo "  ├── p002_*/"
echo "  │   └── ..."
echo "  └── ..."
echo ""
echo "Compare baseline vs GRPO for each physics scenario!"
echo "======================================================================"
