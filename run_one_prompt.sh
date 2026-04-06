#!/bin/bash
# Single prompt: (optional) Baseline + GRPO Training via `unified_grpo/run.py`
# Usage:
#   bash run_one_prompt.sh
# Environment overrides (examples):
#   PROMPT="A leaf floats down..." MODEL_TYPE=cogvideox USE_LORA=1 bash run_one_prompt.sh

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
# If PROMPT is not set, default to the first line of origin_grpo/newyear_prompts.txt (if present).
PROMPTS_FILE="${PROMPTS_FILE:-origin_grpo/newyear_prompts.txt}"
if [[ -z "${PROMPT:-}" && -f "$PROMPTS_FILE" ]]; then
  PROMPT="$(head -n 1 "$PROMPTS_FILE" || true)"
fi
PROMPT="${PROMPT:-A leaf floats down in a loose spiral instead of a straight line.}"
MODEL_TYPE="${MODEL_TYPE:-cogvideox}"   # NOTE: unified_grpo/run.py currently supports cogvideox end-to-end in this repo.
MODEL_PATH="${MODEL_PATH:-THUDM/CogVideoX-2b}"      # optional: local dir or HF id
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
NUM_GRPO_STEPS="${NUM_GRPO_STEPS:-10}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
OUTPUT_DIR="./${MODEL_TYPE}_comparison_$(date +%Y%m%d_%H%M%S)"
USE_LORA="${USE_LORA:-1}"   # 1 -> add --use-lora, 0 -> full/partial finetune (no LoRA)
LORA_BLOCKS="${LORA_BLOCKS:-last}"              # "last" uses --unfreeze-percentage as percentage-of-blocks
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
UNFREEZE_PERCENTAGE="${UNFREEZE_PERCENTAGE:-0.20}"
REWARD_BACKEND="${REWARD_BACKEND:-image_clip}"   # image_clip | xclip | qwen
RUN_BASELINE="${RUN_BASELINE:-1}"               # 1 -> also save baseline mp4 (CogVideo cli), 0 -> skip baseline
SAVE_DENOISING_STRIP="${SAVE_DENOISING_STRIP:-0}"  # 1 -> grpo/denoising_trajectory_strip.png (one wide PNG of all steps)

echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_DIR"
echo ""

# Basic sanity: this script is wired for CogVideoX baseline + unified_grpo runner.
if [[ "$MODEL_TYPE" != "cogvideox" ]]; then
  echo "❌ MODEL_TYPE='$MODEL_TYPE' is not supported by this script right now."
  echo "   - Baseline generation is CogVideoX-specific (CogVideo/inference/cli_demo.py)."
  echo "   - unified_grpo/run.py in this repo currently creates only the CogVideoX adapter end-to-end."
  exit 2
fi

# ======================================================================
# STEP 1: Generate Baseline (optional)
# ======================================================================
if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "======================================================================"
  echo "STEP 1/2: Generating baseline (pretrained)"
  echo "======================================================================"
  echo ""

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
    --output_path "../$OUTPUT_DIR/baseline/baseline.mp4"
  popd >/dev/null

  echo ""
  echo "✅ Baseline saved: $OUTPUT_DIR/baseline/baseline.mp4"
  echo ""
else
  echo "Skipping baseline generation (RUN_BASELINE=0)."
fi

# ======================================================================
# STEP 2: Run GRPO Training (unified_grpo/run.py)
# ======================================================================
echo "======================================================================"
echo "STEP 2/2: Running GRPO Training"
echo "======================================================================"
echo ""

#
# IMPORTANT: don't put `# comments` on lines that are continued with `\` — it breaks the command and your flags
# won't be passed (leading to defaults like --num-grpo-steps 25).
#
mkdir -p "$OUTPUT_DIR/grpo"

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

# LoRA: default ON. If enabled, apply LoRA to selected blocks via --lora-blocks.
if [[ "$USE_LORA" == "1" ]]; then
  cmd+=(--use-lora --lora-blocks "$LORA_BLOCKS" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA")
fi

if [[ "${SAVE_DENOISING_STRIP}" == "1" ]]; then
  cmd+=(--save-denoising-strip-png --save-denoising-step-snapshots)
fi

echo "Running:"
printf '%q ' "${cmd[@]}"
echo ""
echo ""

"${cmd[@]}"

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
echo "  📹 Baseline (pretrained):  $OUTPUT_DIR/baseline/baseline.mp4"
echo "  📹 Trained (GRPO):         $OUTPUT_DIR/grpo/${MODEL_TYPE}_grpo.mp4"
echo "  📄 Training log:           $OUTPUT_DIR/grpo/training_log_*.txt"
echo ""
echo "Compare baseline vs trained to see GRPO improvement!"
echo "======================================================================"
