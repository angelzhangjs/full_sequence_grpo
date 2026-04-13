#!/bin/bash
# Unified GRPO Training Script
# Supports multiple video models with full argument control

set -e
export PYTHONNOUSERSITE=1

# ============================================================================
# Display Help
# ============================================================================

show_help() {
    cat << EOF
Unified GRPO Training Script

Usage: $0 [OPTIONS]

Required:
  --model-type TYPE         Model type: cogvideox, ltx, hunyuan, wan, opensora

Optional:
  --model-path PATH         HuggingFace model path (default: auto-selected)
  --prompt TEXT             Text prompt (default: "A ball bouncing up staircase")
  --height INT              Video height (default: 480)
  --width INT               Video width (default: 720)
  --num-frames INT          Number of frames (default: 49)
  --guidance-scale FLOAT    CFG scale (default: 6.0)
  --num-inference-steps INT Total denoising steps (default: 40)
  --num-grpo-steps INT      Last N steps for GRPO (default: 25)
  --num-rollouts INT        Rollouts per step (default: 3)
  --lr FLOAT                Learning rate (default: 1e-4)
  --seed INT                Random seed (default: 2026)
  --train-blocks STR        Comma-separated block indices (default: auto)
  --unfreeze-percentage NUM Percentage of blocks to unfreeze (0.0-1.0, default: 0.25)
  --output-dir PATH         Output directory (default: ./grpo_output)
  --use-deepspeed          Launch with Accelerate + DeepSpeed ZeRO-3
  --accelerate-config PATH Accelerate YAML config (default: configs/accelerate_zero3_8xa100.yaml)
  --deepspeed-config PATH  DeepSpeed JSON config (default: configs/deepspeed_zero3_8xa100.json)
  --mixed-precision MODE   Accelerate mixed precision: no, bf16, fp16 (default: bf16)
  
  LoRA Options (recommended for 40GB GPU!):
  --use-lora                Enable LoRA training (default: disabled)
  --lora-rank INT           LoRA rank (default: 16)
  --lora-alpha INT          LoRA alpha (default: 32)
  --lora-blocks STR         Comma-separated block indices for LoRA (default: all blocks)

Examples:
  # CogVideoX with defaults:
  $0 --model-type cogvideox

  # CogVideoX with custom prompt:
  $0 --model-type cogvideox --prompt "A cat playing"

  # LTX-Video with custom settings:
  $0 --model-type ltx --num-grpo-steps 30 --num-rollouts 4 --lr 5e-6

  # Full customization:
  $0 --model-type cogvideox \\
     --model-path THUDM/CogVideoX-2b \\
     --prompt "Water pouring into glass" \\
     --num-grpo-steps 20 \\
     --num-rollouts 5 \\
     --lr 1e-5 \\
     --output-dir ./my_output

EOF
}

# ============================================================================
# Parse Arguments
# ============================================================================

# Defaults
MODEL_TYPE=""
MODEL_PATH=""
PROMPT="A ball bouncing up a staircase, hitting each step sequentially"
HEIGHT=480
WIDTH=720
NUM_FRAMES=49
GUIDANCE_SCALE=6.0
NUM_INFERENCE_STEPS=40
NUM_GRPO_STEPS=25
NUM_ROLLOUTS=3
LR=1e-4
SEED=2026
TRAIN_BLOCKS=""
UNFREEZE_PERCENTAGE=0.20
OUTPUT_DIR="./grpo_output"

# LoRA defaults
USE_LORA=""
LORA_RANK=16
LORA_ALPHA=32
LORA_BLOCKS=""
USE_DEEPSPEED=""
ACCELERATE_CONFIG="configs/accelerate_zero3_8xa100.yaml"
DEEPSPEED_CONFIG="configs/deepspeed_zero3_8xa100.json"
MIXED_PRECISION="bf16"

# Parse
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --num-inference-steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --num-grpo-steps)
            NUM_GRPO_STEPS="$2"
            shift 2
            ;;
        --num-rollouts)
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --train-blocks)
            TRAIN_BLOCKS="$2"
            shift 2
            ;;
        --unfreeze-percentage)
            UNFREEZE_PERCENTAGE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use-lora)
            USE_LORA="true"
            shift 1
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lora-blocks)
            LORA_BLOCKS="$2"
            shift 2
            ;;
        --use-deepspeed)
            USE_DEEPSPEED="true"
            shift 1
            ;;
        --accelerate-config)
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        --deepspeed-config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required
if [[ -z "$MODEL_TYPE" ]]; then
    echo "Error: --model-type is required!"
    echo ""
    show_help
    exit 1
fi

# ============================================================================
# Build Python Command
# ============================================================================

LAUNCHER="python"
ENTRYPOINT="-m unified_grpo.run"
if [[ "$USE_DEEPSPEED" == "true" ]]; then
    LAUNCHER="accelerate launch --config_file $ACCELERATE_CONFIG"
fi

CMD="$LAUNCHER $ENTRYPOINT \
    --model-type $MODEL_TYPE \
    --prompt \"$PROMPT\" \
    --height $HEIGHT \
    --width $WIDTH \
    --num-frames $NUM_FRAMES \
    --guidance-scale $GUIDANCE_SCALE \
    --num-inference-steps $NUM_INFERENCE_STEPS \
    --num-grpo-steps $NUM_GRPO_STEPS \
    --num-rollouts $NUM_ROLLOUTS \
    --lr $LR \
    --seed $SEED \
    --output-dir $OUTPUT_DIR"

if [[ "$USE_DEEPSPEED" == "true" ]]; then
    CMD="$CMD --use-accelerate --distributed-backend deepspeed --deepspeed-config $DEEPSPEED_CONFIG --mixed-precision $MIXED_PRECISION"
fi

# Add optional arguments
if [[ -n "$MODEL_PATH" ]]; then
    CMD="$CMD --model-path $MODEL_PATH"
fi

if [[ -n "$TRAIN_BLOCKS" ]]; then
    CMD="$CMD --train-blocks $TRAIN_BLOCKS"
fi

# Add unfreeze percentage (always pass, has default)
CMD="$CMD --unfreeze-percentage $UNFREEZE_PERCENTAGE"

# Add LoRA arguments
if [[ "$USE_LORA" == "true" ]]; then
    CMD="$CMD --use-lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA"
fi

if [[ -n "$LORA_BLOCKS" ]]; then
    CMD="$CMD --lora-blocks $LORA_BLOCKS"
fi

# ============================================================================
# Display Configuration
# ============================================================================

echo "Configuration:"
echo "  Model Type: $MODEL_TYPE"
[[ -n "$MODEL_PATH" ]] && echo "  Model Path: $MODEL_PATH"
echo "  Prompt: $PROMPT"
echo "  Resolution: ${WIDTH}×${HEIGHT}, ${NUM_FRAMES} frames"
echo "  GRPO Steps: $NUM_GRPO_STEPS"
echo "  Rollouts: $NUM_ROLLOUTS"
echo "  Learning Rate: $LR"
echo "  Seed: $SEED"
if [[ "$USE_DEEPSPEED" == "true" ]]; then
    echo "  DeepSpeed: Enabled"
    echo "    Accelerate config: $ACCELERATE_CONFIG"
    echo "    DeepSpeed config: $DEEPSPEED_CONFIG"
    echo "    Mixed precision: $MIXED_PRECISION"
fi

# Display LoRA settings
if [[ "$USE_LORA" == "true" ]]; then
    echo "  LoRA: Enabled ✓"
    echo "    Rank: $LORA_RANK"
    echo "    Alpha: $LORA_ALPHA"
    if [[ -n "$LORA_BLOCKS" ]]; then
        echo "    Blocks: $LORA_BLOCKS (block-specific)"
    else
        echo "    Blocks: ALL"
    fi
else
    echo "  LoRA: Disabled"
    [[ -n "$TRAIN_BLOCKS" ]] && echo "  Training Blocks: $TRAIN_BLOCKS"
fi

echo ""

# ============================================================================
# Execute
# ============================================================================

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "✅ Training complete!"
