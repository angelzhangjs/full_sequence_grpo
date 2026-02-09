#!/bin/bash
# Quick run script - just: bash run.sh

# Go to script directory
cd "$(dirname "$0")"

# Add to PYTHONPATH so imports work
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Training Mode Selection (Choose ONE)
# ==============================================================================
# 
# OPTION A: LoRA on ALL Blocks (RECOMMENDED for 40GB GPU!)
#   - Memory: ~30GB (fits in 40GB!)
#   - Trains: Tiny adapters in ALL layers (~10M params, 0.5%)
#   - Speed: Fast
#   - Quality: Excellent ⭐⭐⭐
#   Flags: --use-lora --lora-rank 16 --lora-alpha 32
#
# OPTION B: LoRA on SPECIFIC Blocks (NEW! Ultra Memory-Efficient!)
#   - Memory: ~26GB (even safer for 40GB!)
#   - Trains: Tiny adapters in SELECTED layers (~350K params, 0.02%)
#   - Speed: Fastest
#   - Quality: Great ⭐⭐
#   Flags: --use-lora --lora-rank 16 --lora-alpha 32 --lora-blocks "29"
#
# OPTION C: Unfrozen Blocks (Needs 48GB+ VRAM)
#   - Memory: ~45GB (OOM on 40GB!)
#   - Trains: Entire transformer blocks (~250M params)
#   - Speed: Slower
#   - Quality: Best ⭐⭐⭐
#   Flags: --train-blocks "22,23,24,25,26,27,28,29"
#
# TIP: Option B (LoRA on block 29) is perfect for prototyping!
# ==============================================================================

# Run unified GRPO
./run_unified_grpo.sh \
    --model-type cogvideox \
    --model-path THUDM/CogVideoX-2b \
    --prompt "A ball bouncing up a staircase, hitting each step sequentially" \
    --height 480 \
    --width 720 \
    --num-frames 49 \
    --guidance-scale 7.5 \
    --num-inference-steps 50 \
    --num-grpo-steps 5 \
    --num-rollouts 1 \
    --lr 1e-4 \
    --seed 42 \
    --use-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-blocks "29" \
    --output-dir ./cogvideox_physics_grpo_lora_output

# ===========================================================================
# Training Mode Examples:
# ===========================================================================
# ✅ LoRA on ALL blocks: --use-lora --lora-rank 16 --lora-alpha 32 (no --lora-blocks)
# ✅ LoRA on block 29 ONLY (CURRENT): --lora-blocks "29"
# ✅ LoRA on last 3 blocks: --lora-blocks "27,28,29"
# ✅ Unfrozen blocks (48GB+): Remove LoRA flags, add --train-blocks "22,23,24,25,26,27,28,29"
# ===========================================================================
