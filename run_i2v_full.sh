#!/bin/bash
# Image-to-Video Generation Script for LTX-Video (Full Model - Better I2V)
# Usage: ./run_i2v_full.sh <image_path> <prompt> [num_frames] [seed]

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <image_path> <prompt> [num_frames] [seed]"
    echo ""
    echo "This uses the FULL (non-distilled) model for better image preservation in I2V"
    echo ""
    echo "Examples:"
    echo "  $0 input.jpg \"A woman smiling and looking at the camera\" 125 42"
    echo "  $0 sather_gate.jpg \"Hair gently blowing in the breeze\" 97 2026"
    echo ""
    exit 1
fi

IMAGE_PATH="$1"
PROMPT="$2"
NUM_FRAMES="${3:-49}"  # Default: 49 frames (2 seconds at 25fps, cleaner for latent math)
SEED="${4:-42}"         # Default: 42

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

echo "========================================"
echo "LTX-Video I2V - FULL MODEL (Better Image Preservation)"
echo "========================================"
echo "Input Image: $IMAGE_PATH"
echo "Prompt: $PROMPT"
echo "Frames: $NUM_FRAMES"
echo "Seed: $SEED"
echo "Model: 2B-dev (Full, 40 steps)"
echo "Strategy: Image conditioning (better I2V support)"
echo "========================================"
echo ""

# Convert relative path to absolute BEFORE changing directory
if [[ "$IMAGE_PATH" != /* ]]; then
    IMAGE_PATH="$(cd "$(dirname "$IMAGE_PATH")" && pwd)/$(basename "$IMAGE_PATH")"
fi

echo "Resolved image path: $IMAGE_PATH"
echo ""

cd /home/ubuntu/angel-research/full_sequence_grpo/ltx_video

python run_inference.py \
    --pipeline_config configs/ltxv-2b-0.9.6-dev.yaml \
    --conditioning_media_paths "$IMAGE_PATH" \
    --conditioning_start_frames 0 \
    --conditioning_strengths 0.9 \
    --prompt "$PROMPT" \
    --output_path ../outputs/i2v \
    --height 480 \
    --width 704 \
    --num_frames "$NUM_FRAMES" \
    --frame_rate 25 \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Image-to-video generation complete!"
    echo "Output saved to: outputs/i2v/"
    echo "Full path: /home/ubuntu/angel-research/full_sequence_grpo/outputs/i2v/"
else
    echo ""
    echo "❌ Generation failed"
    exit 1
fi

