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

# Per-run output folder name (shared with pipeline.py via RUN_ID)
RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RUN_ID
GRPO_OUTDIR="grpo${RUN_ID}"

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
    --pipeline_config configs/ltxv-2b-0.9.8-distilled-no-enhancer.yaml \
    --prompt "$PROMPT" \
    --output_path "../${GRPO_OUTDIR}/baseline" \
    --height 512 \
    --width 768 \
    --num_frames 80 \
    --frame_rate 16 \
    --seed 2026

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

# pipeline.py already writes outputs into ${GRPO_OUTDIR}/
echo "GRPO outputs are in: ${GRPO_OUTDIR}/"

echo ""

# ======================================================================
# Summary
# ======================================================================
echo "======================================================================"
echo "PIPELINE COMPLETE! 🎉"
echo "======================================================================"
echo ""
echo "📂 Output location:"
echo "  ${GRPO_OUTDIR}/"
echo ""
echo "Files:"
echo "  📹 Baseline (pretrained):  ${GRPO_OUTDIR}/baseline/video_output_*.mp4"
echo "  📹 Trained (GRPO+LoRA):   ${GRPO_OUTDIR}/final_video_*.mp4"
echo "  📄 Training log:          ${GRPO_OUTDIR}/training_log_*.txt"
echo "  📁 Rollout videos:        ${GRPO_OUTDIR}/intermediate_rollout/"
echo ""
echo "Both videos are in the SAME folder for easy comparison!"
echo ""
echo "Full path:"
echo "  /home/ubuntu/angel-research/full_sequence_grpo/${GRPO_OUTDIR}/"
echo "======================================================================"

