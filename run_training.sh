#!/bin/bash
# Training script with optimized memory settings

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate ltx-grpo

# Check GPU memory
echo "Checking GPU memory..."
nvidia-smi

# Run training
echo ""
echo "Starting training with optimized memory settings..."
echo "Note: If you see OOM errors, manually run: pkill -9 python"
echo ""
python pipeline.py

echo ""
echo "Training complete!"

###===================================================================
echo "Generating baseline video (no GRPO training)..."
echo ""

# Run baseline generation
python generate_baseline.py

echo ""
echo "✅ Baseline generation complete!"
echo "Compare outputs/baseline_video_*.mp4 with outputs/final_video_*.mp4"

