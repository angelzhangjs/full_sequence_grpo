#!/bin/bash
# Complete Pipeline: GRPO Training + Inference with same prompt
# Usage: bash pipeline.sh

set -e  # Exit on error

echo "======================================================================"
echo "FULL PIPELINE: GRPO TRAINING + INFERENCE"
echo "======================================================================"
echo ""

# Change to the full_sequence_grpo directory
cd /home/ubuntu/angel-research/full_sequence_grpo

# Read prompt from prompt.txt
PROMPT=$(cat prompt.txt | head -1)
echo "📝 Using prompt from prompt.txt:"
echo "   \"$PROMPT\""
echo ""

# ======================================================================
# STEP 1: Run GRPO Training
# ======================================================================
echo "======================================================================"
echo "STEP 1/2: Running GRPO Training"
echo "======================================================================"
echo ""

python pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GRPO Training complete!"
    echo ""
else
    echo ""
    echo "❌ GRPO Training failed!"
    exit 1
fi

# ======================================================================
# STEP 2: Run Inference with same prompt (for comparison)
# ======================================================================
echo "======================================================================"
echo "STEP 2/2: Running Standard Inference (for comparison)"
echo "======================================================================"
echo ""
echo "Running inference with:"
echo "  Prompt: \"$PROMPT\""
echo "  Config: ltxv-2b-0.9.8-distilled.yaml"
echo "  Frames: 125"
echo "  Seed: 42"
echo ""

cd ltx_video

python run_inference.py \
    --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml \
    --prompt "$PROMPT" \
    --output_path ../baseline \
    --height 512 \
    --width 768 \
    --num_frames 125 \
    --frame_rate 25 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Inference complete!"
    echo ""
else
    echo ""
    echo "❌ Inference failed!"
    exit 1
fi

# ======================================================================
# Organize outputs
# ======================================================================
cd /home/ubuntu/angel-research/full_sequence_grpo

echo "======================================================================"
echo "Organizing outputs..."
echo "======================================================================"

# Create grpo directory if it doesn't exist
mkdir -p grpo

# Move GRPO trained video to grpo folder
if [ -f outputs/final_video_*.mp4 ]; then
    mv outputs/final_video_*.mp4 grpo/
    echo "✅ Moved GRPO video to grpo/"
fi

# Move training log to grpo folder
if [ -f training_log_*.txt ]; then
    mv training_log_*.txt grpo/
    echo "✅ Moved training log to grpo/"
fi

echo ""

# ======================================================================
# Summary
# ======================================================================
echo "======================================================================"
echo "PIPELINE COMPLETE! 🎉"
echo "======================================================================"
echo ""
echo "📂 Output locations:"
echo "  1. GRPO trained video:  grpo/final_video_*.mp4"
echo "  2. Baseline video:      baseline/video_output_*.mp4"
echo "  3. Training log:        grpo/training_log_*.txt"
echo ""
echo "Full paths:"
echo "  GRPO:     /home/ubuntu/angel-research/full_sequence_grpo/grpo/"
echo "  Baseline: /home/ubuntu/angel-research/full_sequence_grpo/baseline/"
echo ""
echo "Compare them to see the improvement from GRPO training!"
echo "======================================================================"

