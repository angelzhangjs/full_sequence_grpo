#!/bin/bash
# Complete pipeline: Baseline + GRPO Training
# Usage: bash run.sh

set -e  # Exit on error

cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Prevent user-site packages (~/.local) from shadowing the conda env.
export PYTHONNOUSERSITE=1

echo "======================================================================"
echo "FULL PIPELINE: BASELINE + GRPO TRAINING"
echo "======================================================================"
echo ""

# Configuration
# Using physics-based prompt from action_prompts.txt (line 21 - dropping ball)
PROMPT="A small rock falls into shallow water; it drops, splashes, and then rests."
OUTPUT_DIR="./cogvideox_comparison_$(date +%Y%m%d_%H%M%S)"

echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_DIR"
echo ""

# ======================================================================
# STEP 1: Generate Baseline (Pretrained Model)
# ======================================================================
echo "======================================================================"
echo "STEP 1/2: Generating Baseline Video (Pretrained CogVideoX-2B)"
echo "======================================================================"
echo ""

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
    --output_path "../$OUTPUT_DIR/baseline/baseline.mp4"

cd ..

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Baseline generation complete!"
    echo ""
else
    echo ""
    echo "❌ Baseline failed!"
    exit 1
fi

# ======================================================================
# STEP 2: Run GRPO Training
# ======================================================================
echo "======================================================================"
echo "STEP 2/2: Running GRPO Training"
echo "======================================================================"
echo ""

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
    --output-dir "$OUTPUT_DIR/grpo"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GRPO training complete!"
    echo ""
else
    echo ""
    echo "❌ GRPO training failed!"
    exit 1
fi

# ======================================================================
# Summary
# ======================================================================
echo "======================================================================"
echo "PIPELINE COMPLETE! 🎉"
echo "======================================================================"
echo ""
echo "📂 Output: $OUTPUT_DIR/"
echo ""
echo "Files:"
echo "  📹 Baseline (pretrained):  $OUTPUT_DIR/baseline/*.mp4"
echo "  📹 Trained (GRPO+LoRA):   $OUTPUT_DIR/grpo/cogvideox/final_grpo.mp4"
echo "  📄 Training log:          $OUTPUT_DIR/grpo/cogvideox/training_log_*.txt"
echo ""
echo "Compare baseline vs trained to see GRPO improvement!"
echo "======================================================================"
