#!/bin/bash
# Batch processing: Run GRPO for ALL prompts in action_prompts.txt
# Creates comparison for each physics scenario

set -e

cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================================================"
echo "BATCH GRPO TRAINING - ALL ACTION PROMPTS"
echo "======================================================================"
echo ""

# Input file
PROMPTS_FILE="origin_grpo/action_prompts.txt"
TOTAL_PROMPTS=$(wc -l < "$PROMPTS_FILE")

echo "Found $TOTAL_PROMPTS prompts in $PROMPTS_FILE"
echo "This will take approximately $((TOTAL_PROMPTS * 15)) minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
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
    echo "Step 1/2: Generating baseline..."
    
    # Create baseline directory first!
    mkdir -p "$OUTPUT_DIR/baseline"
    
    cd CogVideo
    
    python inference/cli_demo.py \
        --prompt "$PROMPT" \
        --model_path THUDM/CogVideoX-2b \
        --generate_type "t2v" \
        --num_frames 32 \
        --fps 8 \
        --guidance_scale 7.5 \
        --num_inference_steps 50 \
        --seed 42 \
        --output_path "../$OUTPUT_DIR/baseline/baseline.mp4" || {
            echo "⚠️ Baseline failed for prompt $LINE_NUM, skipping..."
            cd ..
            continue
        }
    
    cd ..
    
    echo "✅ Baseline complete"
    echo ""
    
    # ======================================================================
    # Step 2: Run GRPO Training
    # ======================================================================
    echo "Step 2/2: Running GRPO training..."
    
    ./run_unified_grpo.sh \
        --model-type cogvideox \
        --model-path THUDM/CogVideoX-2b \
        --prompt "$PROMPT" \
        --height 480 \
        --width 720 \
        --num-frames 32 \
        --guidance-scale 7.5 \
        --num-inference-steps 50 \
        --num-grpo-steps 10 \
        --num-rollouts 2 \
        --lr 1e-4 \
        --seed 42 \
        --unfreeze-percentage 0.20 \
        --use-lora \
        --lora-rank 4 \
        --lora-alpha 8 \
        --output-dir "$OUTPUT_DIR/grpo" || {
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
echo "  │   ├── grpo/cogvideox/final_grpo.mp4"
echo "  │   └── prompt.txt"
echo "  ├── p002_*/"
echo "  │   └── ..."
echo "  └── ..."
echo ""
echo "Compare baseline vs GRPO for each physics scenario!"
echo "======================================================================"
